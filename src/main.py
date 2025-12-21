#!/usr/bin/env python
import asyncio
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Protocol, Tuple

import jsonlines
import pandas as pd
from pandas import DataFrame
import typer
from google.cloud import translate_v2 as translate_v2
from googletrans import Translator
from datasets import Dataset, DatasetDict, DownloadMode, Sequence, Value
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm


class AsyncTranslator(Protocol):
    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List["TranslationResult"]: ...


@dataclass
class TranslationResult:
    text: str


class TokenBucket:
    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self._rate = rate
        self._capacity = 1
        self._tokens = 1.0
        self._lock = asyncio.Lock()
        self._last_ts: Optional[float] = None

    async def acquire(self, tokens: int = 1) -> None:
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                if self._last_ts is None:
                    self._last_ts = now
                elapsed = now - self._last_ts
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity,
                        self._tokens + (elapsed * self._rate),
                    )
                    self._last_ts = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_for = (tokens - self._tokens) / self._rate

        await asyncio.sleep(wait_for)


VALID_COLUMN_TYPES = {"string", "list[string]"}


class CloudTranslator:
    def __init__(self, api_key: str):
        if not api_key.strip():
            raise ValueError("Google API key must be provided.")
        self.client = translate_v2.Client(api_key=api_key)

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        return await asyncio.to_thread(self._translate_sync, texts, src, dest)

    def _translate_sync(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        results = self.client.translate(
            texts,
            source_language=src,
            target_language=dest,
            format_="text",
        )
        if isinstance(texts, str):
            results = [results]
        return [
            TranslationResult(item.get("translatedText", ""))
            for item in results
        ]


def create_translator(
    google_api_key: Optional[str], proxy: Optional[str]
) -> AsyncTranslator:
    if google_api_key:
        return CloudTranslator(google_api_key)

    translator_args = {}
    if proxy:
        translator_args["proxy"] = proxy
    return Translator(**translator_args)


def is_file_path(path: str) -> bool:
    """Returns True if the given path appears to be a file path (contains a filename)."""
    p = Path(path)
    return p.suffix != "" or (p.name != "" and "." in p.name)


def load_protected_words(protected_words_arg: Optional[str]) -> List[str]:
    """Load protected words from either a comma-separated string or a file."""
    if not protected_words_arg:
        return []

    if protected_words_arg.startswith("@"):
        file_path = Path(protected_words_arg[1:])
        if not file_path.exists():
            raise FileNotFoundError(
                f"Protected words file not found: {file_path}"
            )
        return [
            line.strip()
            for line in file_path.read_text().splitlines()
            if line.strip()
        ]
    else:
        return [
            word.strip()
            for word in protected_words_arg.split(",")
            if word.strip()
        ]


def normalize_column_types(
    column_types: Optional[Iterable[str]],
) -> List[str]:
    if not column_types:
        return []
    normalized = []
    for entry in column_types:
        for raw in entry.split(","):
            value = raw.strip().lower()
            if value:
                normalized.append(value)
    invalid = sorted(
        {value for value in normalized if value not in VALID_COLUMN_TYPES}
    )
    if invalid:
        raise ValueError(
            f"Unsupported column type(s): {', '.join(invalid)}. "
            f"Supported types: {', '.join(sorted(VALID_COLUMN_TYPES))}."
        )
    return sorted(set(normalized))


def _series_is_string(series: pd.Series) -> bool:
    if pd.api.types.is_string_dtype(series):
        return True
    if series.dtype != object:
        return False
    values = series.dropna()
    if values.empty:
        return False
    return all(isinstance(value, str) for value in values)


def _series_is_list_of_strings(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    for value in values:
        if not isinstance(value, (list, tuple)):
            return False
        if not all(isinstance(item, str) for item in value):
            return False
    return True


def select_columns_from_df(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
) -> List[str]:
    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found in DataFrame: {missing}")
        selected = list(columns)
    else:
        selected = list(df.columns)

    if column_type_filters:
        type_set = set(column_type_filters)
        filtered = []
        for col in selected:
            series = df[col]
            is_string = _series_is_string(series)
            is_list_string = _series_is_list_of_strings(series)
            if "string" in type_set and is_string:
                filtered.append(col)
            elif "list[string]" in type_set and is_list_string:
                filtered.append(col)
        selected = filtered
    return selected


def replace_protected_words(
    text: str, protected_words: List[str]
) -> Tuple[str, Dict[str, str]]:
    """
    Replaces protected words/phrases with UUID-based placeholders.
    Returns modified text and placeholder mapping.
    """
    placeholders = {}
    for phrase in protected_words:
        token = f"__PROTECTED_{uuid.uuid4().hex}__"
        placeholders[token] = phrase
        text = re.sub(re.escape(phrase), token, text, flags=re.IGNORECASE)
    return text, placeholders


def restore_protected_words(
    translated_text: str, placeholders: Dict[str, str]
) -> str:
    """Restores protected words using regex matching."""
    for placeholder, original in placeholders.items():
        pattern = re.compile(
            r"\s*".join(map(re.escape, placeholder)), re.IGNORECASE
        )
        translated_text = pattern.sub(original, translated_text)
    return translated_text


async def process_batch(
    batch: List[Tuple[int, str, str]],
    translator: AsyncTranslator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    max_retries: int = 3,
    rate_limiter: Optional[TokenBucket] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Process a batch of translations."""
    processed_texts = []
    placeholders_list = []
    for _, _, text in batch:
        modified, ph = replace_protected_words(text, protected_words)
        processed_texts.append(modified)
        placeholders_list.append(ph)

    translations = []
    for attempt in range(max_retries):
        try:
            if rate_limiter:
                await rate_limiter.acquire()
            translations = await translator.translate(
                processed_texts, src=source_lang, dest=target_lang
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                await asyncio.sleep((2**attempt) + random.uniform(0, 1))
            else:
                # If it fails after max_retries, raise the exception
                raise e

    successes = []
    failures = []
    for (row_idx, col_name, original_text), translation_obj, ph in zip(
        batch, translations, placeholders_list
    ):
        if translation_obj is None:
            # API returned no translation object
            failures.append(
                {
                    "original_index": row_idx,
                    "column": col_name,
                    "original_text": original_text,
                    "error": "No translation object returned",
                }
            )
            continue

        translated = restore_protected_words(translation_obj.text, ph)
        # Simple heuristic for "failed" translation, but no API error
        if (
            not translated.strip()
            or translated.strip() == original_text.strip()
        ):
            failures.append(
                {
                    "original_index": row_idx,
                    "column": col_name,
                    "original_text": original_text,
                    "translated_text": translated,
                }
            )
        else:
            successes.append(
                {
                    "original_index": row_idx,
                    "column": col_name,
                    "translated_text": translated,
                }
            )
    return successes, failures


async def process_texts(
    items: List[Tuple[int, str, str]],
    translator: AsyncTranslator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    save_dir: Path,
    file_format: str,
    batch_size: int = 20,
    max_concurrency: int = 10,
    checkpoint_step: int = 100,
    max_retries: int = 3,
    failure_retry_cycles: int = 1,
    rate_limiter: Optional[TokenBucket] = None,
) -> None:
    """
    Orchestrate concurrent processing with checkpointing and failure cycles.
    """
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    existing = merge_checkpoints(checkpoint_dir, file_format)
    skip_set = {(idx, col) for idx, cols in existing.items() for col in cols}
    filtered_items = [(i, c, t) for i, c, t in items if (i, c) not in skip_set]

    all_failures = await _translate_in_batches(
        filtered_items,
        translator,
        source_lang,
        target_lang,
        protected_words,
        checkpoint_dir,
        batch_size,
        max_concurrency,
        checkpoint_step,
        max_retries,
        rate_limiter,
        file_format=file_format,
    )

    for cycle_idx in range(1, failure_retry_cycles + 1):
        if not all_failures:
            break

        failures_to_retry = []
        for fail in all_failures:
            idx = fail["original_index"]
            col = fail["column"]
            text = fail.get("original_text", "")
            if text.strip() and (idx, col) not in skip_set:
                failures_to_retry.append((idx, col, text))

        if not failures_to_retry:
            break

        print(f"\n=== Starting failure retry cycle {cycle_idx} ===")
        cycle_failures = await _translate_in_batches(
            failures_to_retry,
            translator,
            source_lang,
            target_lang,
            protected_words,
            checkpoint_dir,
            batch_size,
            max_concurrency,
            checkpoint_step,
            max_retries,
            rate_limiter,
            file_format=file_format,
            is_retry_cycle=True,
        )

        new_success_checkpoint = merge_checkpoints(checkpoint_dir, file_format)
        newly_translated = {
            (idx, col)
            for idx, cols in new_success_checkpoint.items()
            for col in cols
        }
        skip_set.update(newly_translated)
        all_failures = cycle_failures

    if all_failures:
        final_failures_path = (
            checkpoint_dir / f"translation_failures.{file_format}"
        )
        pd.DataFrame(all_failures).to_parquet(final_failures_path)
        print(
            f"Some items still failed after {failure_retry_cycles} retry cycles."
        )
        print(f"Saved those failures to: {final_failures_path}")


async def _translate_in_batches(
    items: List[Tuple[int, str, str]],
    translator: AsyncTranslator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    checkpoint_dir: Path,
    batch_size: int,
    max_concurrency: int,
    checkpoint_step: int,
    max_retries: int,
    rate_limiter: Optional[TokenBucket],
    file_format: str,
    is_retry_cycle: bool = False,
) -> List[Dict]:
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_desc = (
        "Translating (retry cycle)" if is_retry_cycle else "Translating"
    )
    progress = tqdm(total=len(items), desc=progress_desc)

    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    async def process_with_retry(batch):
        async with semaphore:
            batch_size_items = len(batch)
            try:
                successes, failures = await process_batch(
                    batch,
                    translator,
                    source_lang,
                    target_lang,
                    protected_words,
                    max_retries,
                    rate_limiter,
                )
                return successes, failures, batch_size_items
            except Exception as e:
                failures = [
                    {
                        "original_index": idx,
                        "column": col,
                        "error": str(e),
                        "original_text": txt,
                    }
                    for idx, col, txt in batch
                ]
                return [], failures, batch_size_items

    all_failures = []
    results_buffer = []
    checkpoint_counter = 0

    tasks = [
        asyncio.create_task(process_with_retry(batch))
        for batch in chunked(items, batch_size)
    ]

    for future in asyncio.as_completed(tasks):
        successes, failures, num_items = await future
        results_buffer.extend(successes)
        all_failures.extend(failures)
        progress.update(num_items)

        if len(results_buffer) >= checkpoint_step:
            checkpoint_counter += 1
            save_checkpoint(
                results_buffer,
                checkpoint_dir
                / f"checkpoint_{checkpoint_counter:04d}.{file_format}",
                file_format,
            )
            results_buffer = []

    if results_buffer:
        checkpoint_counter += 1
        save_checkpoint(
            results_buffer,
            checkpoint_dir
            / f"checkpoint_{checkpoint_counter:04d}.{file_format}",
            file_format,
        )

    progress.close()
    return all_failures


def detect_file_format(file_path: Path) -> Literal["csv", "parquet", "jsonl"]:
    """Detect the file format based on the file extension."""
    if file_path.suffix.lower() == ".csv":
        return "csv"
    elif file_path.suffix.lower() in (".parquet", ".pq"):
        return "parquet"
    elif file_path.suffix.lower() == ".jsonl":
        return "jsonl"
    elif file_path.is_dir():
        parquet_files = list(file_path.glob("*.parquet"))
        if parquet_files:
            return "parquet"
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def load_tabular_dataset(
    file_path: Path, file_format: Optional[str] = None
) -> pd.DataFrame:
    """Load dataset from CSV or Parquet file."""
    if file_format is None or file_format == "auto":
        file_format = detect_file_format(file_path)
    if file_format == "jsonl":
        with jsonlines.open(file_path, "r") as reader:
            return pd.DataFrame(reader)
    if file_format == "csv":
        return pd.read_csv(file_path)
    elif file_format == "parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def save_dataset(
    df: pd.DataFrame, file_path: Path, file_format: Optional[str] = None
):
    """Save dataset to CSV, Parquet or JSONL file."""
    if file_format is None or file_format == "auto":
        file_format = detect_file_format(file_path)
    if file_format == "csv":
        df.to_csv(file_path, index=False)
    elif file_format == "parquet":
        df.to_parquet(file_path, index=False)
    elif file_format == "jsonl":
        with jsonlines.open(file_path, "w") as writer:
            for _, row in df.iterrows():
                writer.write(row.to_dict())
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def save_checkpoint(data: List[Dict], path: Path, file_format: str):
    """Save checkpoint file with transaction safety."""
    if not data:
        return
    temp_path = path.with_suffix(".tmp")
    df = pd.DataFrame(data)
    save_dataset(df, temp_path, file_format)
    temp_path.rename(path)


def merge_checkpoints(
    checkpoint_dir: Path, file_format: str
) -> Dict[int, Dict[str, str]]:
    """Combine all checkpoint files into a single translation mapping."""
    merged = defaultdict(dict)
    for ckpt in checkpoint_dir.glob(f"checkpoint_*.{file_format}"):
        df = load_tabular_dataset(ckpt, file_format)
        for _, row in df.iterrows():
            if "translated_text" in row:
                merged[row["original_index"]][row["column"]] = row[
                    "translated_text"
                ]
    return merged


def select_columns_from_hf(
    dataset,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
) -> List[str]:
    if columns:
        missing = [col for col in columns if col not in dataset.column_names]
        if missing:
            raise ValueError(
                f"Column(s) not found in Hugging Face dataset: {missing}"
            )
        selected = list(columns)
    else:
        selected = list(dataset.column_names)

    if not column_type_filters:
        return selected

    type_set = set(column_type_filters)
    filtered = []
    for col in selected:
        feature = dataset.features.get(col)
        is_string = isinstance(feature, Value) and feature.dtype == "string"
        is_list_string = (
            isinstance(feature, Sequence)
            and isinstance(feature.feature, Value)
            and feature.feature.dtype == "string"
        )
        if "string" in type_set and is_string:
            filtered.append(col)
        elif "list[string]" in type_set and is_list_string:
            filtered.append(col)
    return filtered


def load_hf_splits(
    dataset_name: str,
    subset: Optional[str],
    splits: Optional[List[str]],
    cache_dir: Path,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs = {
        "name": subset,
        "cache_dir": str(cache_dir),
        "download_mode": DownloadMode.REUSE_CACHE_IF_EXISTS,
    }
    if splits:
        return {
            split: hf_load_dataset(dataset_name, split=split, **load_kwargs)
            for split in splits
        }

    loaded = hf_load_dataset(dataset_name, **load_kwargs)
    if isinstance(loaded, DatasetDict):
        return dict(loaded)
    if isinstance(loaded, Dataset):
        return {"data": loaded}
    raise ValueError("Unsupported Hugging Face dataset type.")


async def translate_dataframe(
    df: pd.DataFrame,
    save_dir: Path,
    source_lang: str,
    target_lang: str,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
    protected_words: List[str],
    file_format: str,
    output_file_format: str,
    translator: AsyncTranslator,
    rate_limiter: Optional[TokenBucket],
    batch_size: int = 20,
    max_concurrency: int = 10,
    checkpoint_step: int = 100,
    max_retries: int = 3,
    failure_retry_cycles: int = 1,
    only_failed: bool = False,
) -> None:
    if only_failed:
        failures_path = (
            save_dir / "checkpoints" / f"translation_failures.{file_format}"
        )
        if not failures_path.exists():
            raise FileNotFoundError(
                f"No failures file found at {failures_path}"
            )
        failures_df: DataFrame = load_tabular_dataset(
            failures_path, file_format
        )
        required_cols: list = [
            "original_index",
            "column",
            "original_text",
        ]
        if not all(col in failures_df.columns for col in required_cols):
            raise ValueError(
                f"Failures file missing required columns: {required_cols}"
            )
        if column_type_filters:
            eligible = select_columns_from_df(df, None, column_type_filters)
            columns = (
                [c for c in columns if c in eligible] if columns else eligible
            )
        if columns:
            failures_df = failures_df[failures_df["column"].isin(columns)]
        items = [
            (row["original_index"], row["column"], row["original_text"])
            for _, row in failures_df.iterrows()
        ]
    else:
        selected_columns = select_columns_from_df(
            df, columns, column_type_filters
        )
        if not selected_columns:
            raise ValueError(
                "No columns selected for translation after applying filters."
            )
        items = []
        for idx, row in df.iterrows():
            for col in selected_columns:
                text = row[col]
                if isinstance(text, str) and text.strip():
                    items.append((idx, col, text))

    columns_used = list({col for _, col, _ in items})
    if not columns_used:
        print("No items to translate.")
        return

    await process_texts(
        items=items,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        protected_words=protected_words,
        save_dir=save_dir,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        checkpoint_step=checkpoint_step,
        max_retries=max_retries,
        failure_retry_cycles=failure_retry_cycles,
        rate_limiter=rate_limiter,
        file_format=file_format,
    )

    merged = merge_checkpoints(save_dir / "checkpoints", file_format)
    output_records = []
    for idx, row in df.iterrows():
        record = {"original_index": idx}
        for col in columns_used:
            record[f"original_{col}"] = row.get(col, "")
            record[f"translated_{col}"] = merged.get(idx, {}).get(col, "")
        output_records.append(record)

    final_path = save_dir / f"translated_dataset.{output_file_format}"
    save_dataset(pd.DataFrame(output_records), final_path, output_file_format)
    print(f"✅ Translation complete! Final dataset saved to {final_path}")


async def translate_dataset(
    input_path: Path,
    save_dir: Path,
    source_lang: str,
    target_lang: str,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
    protected_words: List[str],
    file_format: str,
    output_file_format: str,
    batch_size: int = 20,
    max_concurrency: int = 10,
    checkpoint_step: int = 100,
    max_retries: int = 3,
    failure_retry_cycles: int = 1,
    only_failed: bool = False,
    proxy: Optional[str] = None,
    google_api_key: Optional[str] = None,
    rate_limit_per_sec: Optional[float] = None,
):
    df = load_tabular_dataset(input_path, file_format)
    if columns is None and not column_type_filters:
        column_type_filters = ["string"]

    translator: AsyncTranslator = create_translator(
        google_api_key=google_api_key,
        proxy=proxy,
    )
    rate_limiter = None
    if rate_limit_per_sec is not None:
        if rate_limit_per_sec <= 0:
            raise ValueError("rate_limit_per_sec must be > 0")
        rate_limiter = TokenBucket(rate_limit_per_sec)

    await translate_dataframe(
        df=df,
        save_dir=save_dir,
        source_lang=source_lang,
        target_lang=target_lang,
        columns=columns,
        column_type_filters=column_type_filters,
        protected_words=protected_words,
        file_format=file_format,
        output_file_format=output_file_format,
        translator=translator,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        checkpoint_step=checkpoint_step,
        max_retries=max_retries,
        failure_retry_cycles=failure_retry_cycles,
        only_failed=only_failed,
    )


async def translate_hf_dataset(
    dataset_name: str,
    save_dir: Path,
    source_lang: str,
    target_lang: str,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
    protected_words: List[str],
    output_file_format: str,
    batch_size: int = 20,
    max_concurrency: int = 10,
    checkpoint_step: int = 100,
    max_retries: int = 3,
    failure_retry_cycles: int = 1,
    only_failed: bool = False,
    proxy: Optional[str] = None,
    google_api_key: Optional[str] = None,
    rate_limit_per_sec: Optional[float] = None,
    subset: Optional[str] = None,
    splits: Optional[List[str]] = None,
) -> None:
    hf_cache_dir = save_dir / "hf_cache"
    datasets = load_hf_splits(dataset_name, subset, splits, hf_cache_dir)
    if columns is None and not column_type_filters:
        column_type_filters = ["string"]
    translator: AsyncTranslator = create_translator(
        google_api_key=google_api_key,
        proxy=proxy,
    )
    rate_limiter = None
    if rate_limit_per_sec is not None:
        if rate_limit_per_sec <= 0:
            raise ValueError("rate_limit_per_sec must be > 0")
        rate_limiter = TokenBucket(rate_limit_per_sec)

    for split_name, dataset in datasets.items():
        selected_columns = select_columns_from_hf(
            dataset, columns, column_type_filters
        )
        if not selected_columns and not only_failed:
            raise ValueError(
                f"No columns selected for split '{split_name}' after applying filters."
            )
        split_dir = save_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        df = dataset.to_pandas()
        await translate_dataframe(
            df=df,
            save_dir=split_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            columns=selected_columns,
            column_type_filters=None,
            protected_words=protected_words,
            file_format=output_file_format,
            output_file_format=output_file_format,
            translator=translator,
            rate_limiter=rate_limiter,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            checkpoint_step=checkpoint_step,
            max_retries=max_retries,
            failure_retry_cycles=failure_retry_cycles,
            only_failed=only_failed,
        )


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ..., help="Path to input dataset (CSV, Parquet, or JSONL)"
    ),
    save_dir: Path = typer.Argument(
        ..., help="Directory to save translated data"
    ),
    source_lang: str = typer.Argument(..., help="Source language code"),
    target_lang: str = typer.Argument(..., help="Target language code"),
    columns: Optional[List[str]] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Columns to translate (defaults to string columns). Can be multiple. Pass the --columns (-c) flag multiple times to specify multiple columns.",
    ),
    column_types: Optional[List[str]] = typer.Option(
        None,
        "--column-type",
        "-t",
        help="Filter columns by type (string, list[string]). Can be provided multiple times or comma-separated.",
    ),
    protected_words: Optional[str] = typer.Option(
        None,
        "--protected-words",
        "-p",
        help="Comma-separated list or @file.txt file with protected words/phrases. See docs for format.",
    ),
    file_format: str = typer.Option(
        "auto",
        "--file-format",
        "-f",
        help="File format (csv, parquet, jsonl, auto). If not specified, file format will be inferred from the input file path.",
    ),
    output_file_format: Optional[str] = typer.Option(
        "auto",
        "--output-file-format",
        help="Output file format (csv, parquet, jsonl, auto). If not specified, output format will be fallback to input file format.",
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", "-b", help="Number of texts per translation request"
    ),
    max_concurrency: int = typer.Option(
        1, "--max-concurrency", help="Maximum concurrent translation requests"
    ),
    checkpoint_step: int = typer.Option(
        100,
        "--checkpoint-step",
        help="Number of successful translations between checkpoints",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Maximum retry attempts per batch before marking as failed",
    ),
    failure_retry_cycles: int = typer.Option(
        3,
        "--max-failure-cycles",
        help="Number of full retry cycles for previously failed translations",
    ),
    only_failed: bool = typer.Option(
        False,
        "--only-failed",
        help="Process only previously failed translations from checkpoint directory",
    ),
    proxy: Optional[str] = typer.Option(
        None,
        "--proxy",
        help="Proxy URL to use for translation requests. Protocol must be specified. Example: http://<ip>:<port>",
    ),
    google_api_key: Optional[str] = typer.Option(
        None,
        "--google-api-key",
        help="Google Cloud Translation API key. When provided, the Cloud Translation API is used instead of the free endpoint.",
    ),
    rate_limit_per_sec: Optional[float] = typer.Option(
        None,
        "--rate-limit",
        help="Max translation requests per second. Token bucket is applied per batch.",
    ),
    hf_dataset: bool = typer.Option(
        False,
        "--hf",
        help="Treat input_path as a Hugging Face dataset name.",
    ),
    subset: Optional[str] = typer.Option(
        None,
        "--subset",
        "--config",
        help="Dataset subset/config name.",
    ),
    splits: Optional[List[str]] = typer.Option(
        None,
        "--split",
        "-s",
        help="Dataset split(s) to translate. Can be provided multiple times.",
    ),
):
    """
    Translate columns in a dataset with support for retrying failed items.
    """
    try:
        column_type_filters = normalize_column_types(column_types)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    if is_file_path(str(save_dir)):
        raise ValueError("save_dir must be a directory, not a file path.")

    protected = load_protected_words(protected_words)
    save_dir.mkdir(parents=True, exist_ok=True)

    if hf_dataset:
        if output_file_format == "auto":
            output_file_format = "parquet"
        if output_file_format not in {"csv", "parquet", "jsonl"}:
            raise typer.BadParameter(
                "output_file_format must be one of: csv, parquet, jsonl"
            )
        asyncio.run(
            translate_hf_dataset(
                dataset_name=str(input_path),
                save_dir=save_dir,
                source_lang=source_lang,
                target_lang=target_lang,
                columns=columns,
                column_type_filters=column_type_filters,
                protected_words=protected,
                output_file_format=output_file_format,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                checkpoint_step=checkpoint_step,
                max_retries=max_retries,
                failure_retry_cycles=failure_retry_cycles,
                only_failed=only_failed,
                proxy=proxy,
                google_api_key=google_api_key,
                rate_limit_per_sec=rate_limit_per_sec,
                subset=subset,
                splits=splits,
            )
        )
        return

    if file_format == "auto":
        file_format = detect_file_format(input_path)
    if output_file_format == "auto":
        output_file_format = file_format

    asyncio.run(
        translate_dataset(
            input_path=input_path,
            save_dir=save_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            columns=columns,
            column_type_filters=column_type_filters,
            protected_words=protected,
            file_format=file_format,
            output_file_format=output_file_format,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            checkpoint_step=checkpoint_step,
            max_retries=max_retries,
            failure_retry_cycles=failure_retry_cycles,
            only_failed=only_failed,
            proxy=proxy,
            google_api_key=google_api_key,
            rate_limit_per_sec=rate_limit_per_sec,
        )
    )


if __name__ == "__main__":
    app()
