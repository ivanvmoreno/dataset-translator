#!/usr/bin/env python
import asyncio
import json
import itertools
import importlib.metadata
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Set,
    Any,
)

import jsonlines
import pandas as pd
import typer
from google.api_core.client_options import ClientOptions
from google.cloud import translate_v2 as translate_v2
from googletrans import Translator
from huggingface_hub import HfApi
from datasets import Dataset, DatasetDict, DownloadMode, Sequence, Value
from datasets import load_dataset as hf_load_dataset
from tqdm.asyncio import tqdm


@dataclass
class TranslationResult:
    text: str


class AsyncTranslator(Protocol):
    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]: ...


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
OUTPUT_FILE_EXTENSIONS = {".csv", ".parquet", ".jsonl"}


def get_version() -> Optional[str]:
    try:
        return importlib.metadata.version("dataset-translator")
    except ImportError:
        return None


def ensure_output_root(save_dir: Path) -> None:
    if save_dir.exists() and not save_dir.is_dir():
        raise ValueError(f"Output path must be a directory: {save_dir}")
    if save_dir.suffix in OUTPUT_FILE_EXTENSIONS and not save_dir.exists():
        raise ValueError(
            f"Output path must be a directory, not a file: {save_dir}"
        )
    save_dir.mkdir(parents=True, exist_ok=True)


def write_translation_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_translation_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def normalize_hub_repo_id(repo_id: str) -> str:
    if "/" in repo_id:
        return repo_id
    api = HfApi()
    try:
        who = api.whoami()
    except Exception as exc:
        raise ValueError(
            "Hub upload requires authentication or a fully qualified "
            "repo id like `user/repo`."
        ) from exc
    username = who.get("name")
    if not username:
        raise ValueError(
            "Unable to resolve Hub username; pass `user/repo` explicitly."
        )
    return f"{username}/{repo_id}"


def upload_metadata_to_hub(repo_id: str, path: Path) -> None:
    if not path.exists():
        return
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )


def ensure_hub_repo(repo_id: str, private: bool) -> str:
    normalized = normalize_hub_repo_id(repo_id)
    api = HfApi()
    api.create_repo(
        repo_id=normalized,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    return normalized


def rename_splits_for_hub(dataset_dict: DatasetDict) -> DatasetDict:
    renamed = {}
    for split_name, dataset in dataset_dict.items():
        renamed[split_name.replace("-", "_")] = dataset
    return DatasetDict(renamed)


def sanitize_run_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-")
    return cleaned or "dataset"


def normalize_target_langs(target_lang: str) -> List[str]:
    langs = []
    for raw in target_lang.split(","):
        cleaned = raw.strip()
        if cleaned:
            langs.append(cleaned)
    return langs or [target_lang]


def normalize_source_lang(source_lang: Optional[str]) -> str:
    if not source_lang:
        return "auto"
    cleaned = source_lang.strip()
    return cleaned or "auto"


def build_run_dir(
    save_dir: Path, label: str, source_lang: Optional[str], target_lang: str
) -> Path:
    run_name = (
        f"{sanitize_run_label(label)}__"
        f"{normalize_source_lang(source_lang)}_to_{target_lang}"
    )
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def list_checkpoint_files(
    checkpoint_root: Path, file_format: str
) -> List[Path]:
    batches_dir = checkpoint_root / "batches"
    batches = list(batches_dir.glob(f"checkpoint_*.{file_format}"))
    files = {path.resolve() for path in batches}
    return sorted(files, key=lambda p: str(p))


def next_checkpoint_index(checkpoint_dir: Path, file_format: str) -> int:
    max_index = 0
    for path in checkpoint_dir.glob(f"checkpoint_*.{file_format}"):
        match = re.search(r"checkpoint_(\d+)", path.stem)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index


def prepare_checkpoint_dirs(save_dir: Path) -> Tuple[Path, Path, Path]:
    checkpoint_root = save_dir / "checkpoints"
    batches_dir = checkpoint_root / "batches"
    failures_dir = checkpoint_root / "failures"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_root, batches_dir, failures_dir


def resolve_hf_cache_dir(save_dir: Path) -> Path:
    return save_dir.parent / "hf_cache"


def detect_file_format(path: Path) -> str:
    if path.is_dir():
        for ext, fmt in (
            (".parquet", "parquet"),
            (".pq", "parquet"),
            (".jsonl", "jsonl"),
            (".csv", "csv"),
        ):
            if list(path.glob(f"*{ext}")):
                return fmt
        raise ValueError(f"Could not detect file format in directory: {path}")
    if path.suffix == ".csv":
        return "csv"
    if path.suffix in (".parquet", ".pq"):
        return "parquet"
    if path.suffix == ".jsonl":
        return "jsonl"
    raise ValueError(f"Could not detect file format: {path}")


def load_dataset(path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(path)
    if file_format == "parquet":
        return pd.read_parquet(path)
    if file_format == "jsonl":
        with jsonlines.open(path, "r") as reader:
            return pd.DataFrame(list(reader))
    raise ValueError(f"Unknown format {file_format}")


def save_dataset(df: pd.DataFrame, path: Path, file_format: str) -> None:
    if file_format == "csv":
        df.to_csv(path, index=False)
        return
    if file_format == "parquet":
        df.to_parquet(path, index=False)
        return
    if file_format == "jsonl":
        with jsonlines.open(path, "w") as writer:
            writer.write_all(df.to_dict("records"))
        return
    raise ValueError(f"Unknown format {file_format}")


class CloudTranslate:
    """Async wrapper for Google Cloud Translate SDK."""

    def __init__(self):
        self.client = translate_v2.Client()

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        return await asyncio.to_thread(self._translate_sync, texts, src, dest)

    def _translate_sync(
        self, texts: List[str], src: Optional[str], dest: str
    ) -> List[TranslationResult]:
        src_lang = normalize_source_lang(src)
        request = {"target_language": dest, "format_": "text"}
        if src_lang != "auto":
            request["source_language"] = src_lang
        results = self.client.translate(texts, **request)
        if isinstance(texts, str):
            results = [results]
        return [
            TranslationResult(item.get("translatedText", ""))
            for item in results
        ]


class GoogleTranslate:
    """Async wrapper for the synchronous googletrans library."""

    def __init__(self, proxy: Optional[str] = None):
        self.translator = Translator()
        self.proxy = proxy

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        if asyncio.iscoroutinefunction(self.translator.translate):
            results = await self.translator.translate(texts, src=src, dest=dest)
        else:
            results = await asyncio.to_thread(
                self._translate_sync, texts, src, dest
            )

        if not isinstance(results, list):
            results = [results]

        return [TranslationResult(r.text) for r in results]

    def _translate_sync(self, texts: List[str], src: str, dest: str) -> Any:
        src_lang = normalize_source_lang(src)
        return self.translator.translate(texts, src=src_lang, dest=dest)


def create_translator(
    use_cloud_api: bool, proxy: Optional[str]
) -> AsyncTranslator:
    if use_cloud_api:
        return CloudTranslate()
    return GoogleTranslate(proxy=proxy)


def is_file_path(path: str) -> bool:
    p = Path(path)
    return p.suffix != "" or (p.name != "" and "." in p.name)


def load_protected_words(protected_words_arg: Optional[str]) -> List[str]:
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


def normalize_column_types(column_types: Optional[Iterable[str]]) -> List[str]:
    if not column_types:
        return []
    normalized = []
    for entry in column_types:
        for raw in entry.split(","):
            value = raw.strip().lower()
            if value:
                normalized.append(value)
    invalid = sorted({v for v in normalized if v not in VALID_COLUMN_TYPES})
    if invalid:
        raise ValueError(f"Unsupported types: {', '.join(invalid)}")
    return sorted(set(normalized))


def normalize_list_arg(values: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not values:
        return None
    normalized = []
    for entry in values:
        for raw in entry.split(","):
            cleaned = raw.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized or None


def replace_protected_words(
    text: str, protected_words: List[str]
) -> Tuple[str, Dict[str, str]]:
    placeholders = {}
    for phrase in protected_words:
        token = f"__PROTECTED_{uuid.uuid4().hex}__"
        placeholders[token] = phrase
        text = re.sub(re.escape(phrase), token, text, flags=re.IGNORECASE)
    return text, placeholders


def restore_protected_words(
    translated_text: str, placeholders: Dict[str, str]
) -> str:
    for placeholder, original in placeholders.items():
        # Handle potential spaces added by translator around tokens
        pattern = re.compile(
            r"\s*".join(map(re.escape, placeholder)), re.IGNORECASE
        )
        translated_text = pattern.sub(original, translated_text)
    return translated_text


def batched(iterable, n):
    if hasattr(itertools, "batched"):
        return itertools.batched(iterable, n)
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


async def process_batch(
    batch: List[Tuple[int, str, str]],
    translator: AsyncTranslator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    max_retries: int,
    rate_limiter: Optional[TokenBucket] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Process a single batch of text."""

    successes = []
    failures = []

    # Separate empty strings (no need to translate) from actual content
    items_to_translate = []
    metadata_to_translate = (
        []
    )  # (row_idx, col_name, original_text, placeholders)

    for row_idx, col_name, text in batch:
        # Optimization: Skip API call for empty/whitespace strings
        if not text.strip():
            successes.append(
                {
                    "original_index": row_idx,
                    "column": col_name,
                    "translated_text": text,  # Return as is
                }
            )
            continue

        modified, ph = replace_protected_words(text, protected_words)
        items_to_translate.append(modified)
        metadata_to_translate.append((row_idx, col_name, text, ph))

    if not items_to_translate:
        return successes, failures

    # Perform translation only on valid items
    translations = []
    for attempt in range(max_retries):
        try:
            if rate_limiter:
                await rate_limiter.acquire()

            translations = await translator.translate(
                items_to_translate, src=source_lang, dest=target_lang
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(sleep_time)
            else:
                for r_idx, c_name, orig, _ in metadata_to_translate:
                    failures.append(
                        {
                            "original_index": r_idx,
                            "column": c_name,
                            "original_text": orig,
                            "error": str(e),
                        }
                    )
                return successes, failures

    for (row_idx, col_name, original_text, ph), translation_obj in zip(
        metadata_to_translate, translations
    ):
        if not translation_obj:
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

        if not translated.strip() and original_text.strip():
            failures.append(
                {
                    "original_index": row_idx,
                    "column": col_name,
                    "original_text": original_text,
                    "translated_text": translated,
                    "error": "Empty translation returned",
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


async def save_checkpoint_async(data: List[Dict], path: Path, file_format: str):
    if not data:
        return
    await asyncio.to_thread(save_checkpoint, data, path, file_format)


def save_checkpoint(data: List[Dict], path: Path, file_format: str):
    temp_path = path.with_suffix(".tmp")
    df = pd.DataFrame(data)
    if file_format == "csv":
        df.to_csv(temp_path, index=False)
    elif file_format == "parquet":
        df.to_parquet(temp_path, index=False)
    elif file_format == "jsonl":
        with jsonlines.open(temp_path, "w") as writer:
            writer.write_all(data)
    temp_path.rename(path)


async def process_stream(
    items_generator: Iterable[Tuple[int, str, str]],
    total_items: int,
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_desc = "Translating (Retry)" if is_retry_cycle else "Translating"

    progress = tqdm(
        total=total_items, desc=progress_desc, position=1, leave=False
    )

    async def run_batch(batch_items):
        async with semaphore:
            try:
                s, f = await process_batch(
                    batch_items,
                    translator,
                    source_lang,
                    target_lang,
                    protected_words,
                    max_retries,
                    rate_limiter,
                )
                return s, f, len(batch_items)
            except Exception as e:
                f = [
                    {
                        "original_index": idx,
                        "column": col,
                        "original_text": txt,
                        "error": str(e),
                    }
                    for idx, col, txt in batch_items
                ]
                return [], f, len(batch_items)

    pending_tasks: Set[asyncio.Task] = set()
    results_buffer = []
    all_failures = []
    checkpoint_counter = next_checkpoint_index(checkpoint_dir, file_format)

    batch_iterator = batched(items_generator, batch_size)

    for batch in batch_iterator:
        task = asyncio.create_task(run_batch(batch))
        pending_tasks.add(task)

        if len(pending_tasks) >= max_concurrency * 2:
            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for t in done:
                successes, failures, count = await t
                results_buffer.extend(successes)
                all_failures.extend(failures)
                progress.update(count)

                if len(results_buffer) >= checkpoint_step:
                    checkpoint_counter += 1
                    await save_checkpoint_async(
                        results_buffer,
                        checkpoint_dir
                        / f"checkpoint_{checkpoint_counter:04d}.{file_format}",
                        file_format,
                    )
                    results_buffer = []

    if pending_tasks:
        done, _ = await asyncio.wait(pending_tasks)
        for t in done:
            successes, failures, count = await t
            results_buffer.extend(successes)
            all_failures.extend(failures)
            progress.update(count)

    if results_buffer:
        checkpoint_counter += 1
        await save_checkpoint_async(
            results_buffer,
            checkpoint_dir
            / f"checkpoint_{checkpoint_counter:04d}.{file_format}",
            file_format,
        )

    progress.close()
    return all_failures


async def process_texts(
    items: Iterable[Tuple[int, str, str]],
    translator: AsyncTranslator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    save_dir: Path,
    file_format: str,
    batch_size: int,
    max_concurrency: int,
    checkpoint_step: int,
    max_retries: int,
    failure_retry_cycles: int = 0,
    rate_limiter: Optional[TokenBucket] = None,
) -> List[Dict]:
    ensure_output_root(save_dir)
    if target_lang is None:
        if source_lang is None:
            raise ValueError(
                "target_lang is required when source_lang is omitted"
            )
        target_lang = source_lang
        source_lang = "auto"
    if source_lang is None:
        source_lang = "auto"
    source_lang = normalize_source_lang(source_lang)
    checkpoint_root, checkpoint_batches_dir, checkpoint_failures_dir = (
        prepare_checkpoint_dirs(save_dir)
    )
    failures = await process_stream(
        items_generator=items,
        total_items=len(items),
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        protected_words=protected_words,
        checkpoint_dir=checkpoint_batches_dir,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        checkpoint_step=checkpoint_step,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
        file_format=file_format,
    )
    if failures:
        final_fail_path = (
            checkpoint_failures_dir / f"translation_failures.{file_format}"
        )
        if file_format == "parquet":
            pd.DataFrame(failures).to_parquet(final_fail_path)
        else:
            pd.DataFrame(failures).to_csv(
                final_fail_path.with_suffix(".csv"), index=False
            )
    return failures


def generate_translation_tasks(
    dataset: Any, columns: List[str], skip_set: Set[Tuple[int, str]]
) -> Iterable[Tuple[int, str, str]]:
    if isinstance(dataset, (Dataset, Sequence)):
        for i in range(len(dataset)):
            try:
                row = dataset[i]
                for col in columns:
                    if (i, col) in skip_set:
                        continue
                    text = row.get(col)
                    if isinstance(text, str):
                        yield (i, col, text)
            except Exception:
                continue

    elif isinstance(dataset, pd.DataFrame):
        for i, row in dataset.iterrows():
            for col in columns:
                if (i, col) in skip_set:
                    continue
                text = row.get(col)
                if isinstance(text, str):
                    yield (i, col, text)


def select_columns_from_df(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
) -> List[str]:
    available = list(df.columns)
    if columns:
        missing = [c for c in columns if c not in available]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        selected = columns
    else:
        selected = available

    if column_type_filters:
        type_set = set(column_type_filters)
        filtered = []
        for col in selected:
            series = df[col].dropna()
            if series.empty:
                continue

            is_str = pd.api.types.is_string_dtype(series) or all(
                isinstance(x, str) for x in series.head(100)
            )
            is_list_str = all(
                isinstance(x, list) and all(isinstance(item, str) for item in x)
                for x in series.head(100)
            )
            if "string" in type_set and is_str:
                filtered.append(col)
            elif "list[string]" in type_set and is_list_str:
                filtered.append(col)
        selected = filtered
    return selected


def select_columns_from_hf(
    dataset: Dataset,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
) -> List[str]:
    available = dataset.column_names
    if columns:
        missing = [c for c in columns if c not in available]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        selected = columns
    else:
        selected = available

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


def apply_translations_to_hf_dataset(
    dataset: Dataset,
    final_merged: Dict[int, Dict[str, str]],
    columns: List[str],
    replace_columns: bool,
) -> Dataset:
    def apply_record(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
        merged = final_merged.get(idx)
        if replace_columns:
            if merged:
                for col in columns:
                    if col in merged:
                        record[col] = merged[col]
            return record

        updates = {}
        for col in columns:
            updates[f"translated_{col}"] = merged.get(col) if merged else None
        return {**record, **updates}

    return dataset.map(
        apply_record,
        with_indices=True,
        load_from_cache_file=False,
    )


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
            s: hf_load_dataset(dataset_name, split=s, **load_kwargs)
            for s in splits
        }

    loaded = hf_load_dataset(dataset_name, **load_kwargs)
    if isinstance(loaded, DatasetDict):
        return dict(loaded)
    if isinstance(loaded, Dataset):
        return {"data": loaded}
    raise ValueError("Unsupported HF Dataset structure")


def merge_checkpoints(
    checkpoint_root: Path, file_format: str
) -> Dict[int, Dict[str, str]]:
    """Loads all checkpoints into a dictionary for fast lookup during reconstruction."""
    merged = defaultdict(dict)
    files = list_checkpoint_files(checkpoint_root, file_format)
    if not files:
        return merged

    for ckpt in files:
        if file_format == "jsonl":
            with jsonlines.open(ckpt, "r") as reader:
                for row in reader:
                    merged[row["original_index"]][row["column"]] = row.get(
                        "translated_text", ""
                    )
        elif file_format == "csv":
            df = pd.read_csv(ckpt)
            for _, row in df.iterrows():
                merged[row["original_index"]][row["column"]] = row[
                    "translated_text"
                ]
        elif file_format == "parquet":
            df = pd.read_parquet(ckpt)
            for _, row in df.iterrows():
                merged[row["original_index"]][row["column"]] = row[
                    "translated_text"
                ]
    return merged


async def orchestrate_translation(
    dataset: Any,
    dataset_length: int,
    save_dir: Path,
    source_lang: str,
    target_lang: str,
    columns: List[str],
    protected_words: List[str],
    file_format: str,
    output_file_format: str,
    translator: AsyncTranslator,
    rate_limiter: Optional[TokenBucket],
    batch_size: int,
    max_concurrency: int,
    checkpoint_step: int,
    max_retries: int,
    failure_retry_cycles: int,
    only_failed: bool,
    replace_columns: bool,
    output_mode: str = "file",
    output_basename: str = "translated_dataset",
):
    checkpoint_root, checkpoint_batches_dir, checkpoint_failures_dir = (
        prepare_checkpoint_dirs(save_dir)
    )

    existing = merge_checkpoints(checkpoint_root, file_format)
    skip_set = {(idx, col) for idx, cols in existing.items() for col in cols}

    if not only_failed:
        task_gen = generate_translation_tasks(dataset, columns, skip_set)

        total_tasks = dataset_length * len(columns) - len(skip_set)

        failures = await process_stream(
            items_generator=task_gen,
            total_items=max(0, total_tasks),
            translator=translator,
            source_lang=source_lang,
            target_lang=target_lang,
            protected_words=protected_words,
            checkpoint_dir=checkpoint_batches_dir,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            checkpoint_step=checkpoint_step,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
            file_format=file_format,
        )
    else:
        fail_path = (
            checkpoint_failures_dir / f"translation_failures.{file_format}"
        )
        if not fail_path.exists():
            print("No failures file found.")
            failures = []
        else:
            if file_format == "jsonl":
                with jsonlines.open(fail_path) as r:
                    failures = list(r)
            else:
                failures = pd.read_parquet(fail_path).to_dict("records")

    for cycle in range(1, failure_retry_cycles + 1):
        if not failures:
            break

        retry_items = []
        for f in failures:
            idx, col, txt = (
                f["original_index"],
                f["column"],
                f.get("original_text", ""),
            )
            if (idx, col) not in skip_set and txt:
                retry_items.append((idx, col, txt))

        if not retry_items:
            break

        print(f"\n=== Failure Retry Cycle {cycle} ===")
        cycle_failures = await process_stream(
            items_generator=retry_items,
            total_items=len(retry_items),
            translator=translator,
            source_lang=source_lang,
            target_lang=target_lang,
            protected_words=protected_words,
            checkpoint_dir=checkpoint_batches_dir,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            checkpoint_step=checkpoint_step,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
            file_format=file_format,
            is_retry_cycle=True,
        )

        new_merged = merge_checkpoints(checkpoint_root, file_format)
        skip_set = {
            (idx, col) for idx, cols in new_merged.items() for col in cols
        }
        failures = cycle_failures

    if failures:
        final_fail_path = (
            checkpoint_failures_dir / f"translation_failures.{file_format}"
        )
        (
            pd.DataFrame(failures).to_parquet(final_fail_path)
            if file_format == "parquet"
            else pd.DataFrame(failures).to_csv(
                final_fail_path.with_suffix(".csv")
            )
        )

    print("Merging translations into final dataset...")
    final_merged = merge_checkpoints(checkpoint_root, file_format)
    if output_mode != "file":
        if isinstance(dataset, Dataset):
            return apply_translations_to_hf_dataset(
                dataset, final_merged, columns, replace_columns
            )
        return None

    output_path = save_dir / f"{output_basename}.{output_file_format}"

    if output_file_format == "jsonl":
        writer = jsonlines.open(output_path, "w")
    elif output_file_format == "csv":
        import csv

        f_obj = open(output_path, "w", newline="", encoding="utf-8")
        dummy_row = (
            dataset[0]
            if isinstance(dataset, (Dataset, Sequence))
            else dataset.iloc[0].to_dict()
        )
        if replace_columns:
            fieldnames = list(dummy_row.keys())
        else:
            fieldnames = (
                ["original_index"]
                + list(dummy_row.keys())
                + [f"translated_{c}" for c in columns]
            )
        writer = csv.DictWriter(f_obj, fieldnames=fieldnames)
        writer.writeheader()
    else:
        writer = None

    buffer = []
    CHUNK_SIZE = 10000

    if isinstance(dataset, (Dataset, Sequence)):
        iterator = range(len(dataset))
        getter = lambda i: dataset[i]
    else:
        iterator = range(len(dataset))
        getter = lambda i: dataset.iloc[i].to_dict()

    for i in tqdm(iterator, desc="Writing Output"):
        row = getter(i)
        record = dict(row)

        if replace_columns:
            if i in final_merged:
                for col in columns:
                    if col in final_merged[i]:
                        record[col] = final_merged[i][col]
        else:
            record = {"original_index": i, **record}
            if i in final_merged:
                for col in columns:
                    if col in final_merged[i]:
                        record[f"translated_{col}"] = final_merged[i][col]

        if output_file_format == "jsonl":
            writer.write(record)
        elif output_file_format == "csv":
            writer.writerow(record)
        elif output_file_format == "parquet":
            buffer.append(record)
            if len(buffer) >= CHUNK_SIZE:
                pass

    if output_file_format == "jsonl":
        writer.close()
    elif output_file_format == "csv":
        f_obj.close()
    elif output_file_format == "parquet":
        if buffer:
            pd.DataFrame(buffer).to_parquet(output_path)
        else:
            pass

    print(f"✅ Complete. Saved to {output_path}")


async def translate_dataset_file(
    input_path: Path,
    save_dir: Path,
    source_lang: str,
    target_lang: str,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
    protected_words: List[str],
    file_format: str,
    output_file_format: str,
    replace_columns: bool = False,
    **kwargs,
):
    ensure_output_root(save_dir)
    kwargs.pop("use_cloud_api", None)
    kwargs.pop("google_api_key", None)
    if file_format == "jsonl":
        with jsonlines.open(input_path, "r") as r:
            df = pd.DataFrame(list(r))
    elif file_format == "csv":
        df = pd.read_csv(input_path)
    elif file_format == "parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unknown format {file_format}")

    selected_cols = select_columns_from_df(df, columns, column_type_filters)
    if not selected_cols:
        print("No columns to translate.")
        return

    target_langs = normalize_target_langs(target_lang)
    overall = tqdm(total=len(target_langs), desc="Overall Progress", position=0)
    translator = kwargs.get("translator")
    rate_limiter = kwargs.get("rate_limiter")
    output_basename = "translated_dataset"
    for lang in target_langs:
        run_dir = build_run_dir(save_dir, input_path.stem, source_lang, lang)
        await orchestrate_translation(
            dataset=df,
            dataset_length=len(df),
            save_dir=run_dir,
            source_lang=source_lang,
            target_lang=lang,
            columns=selected_cols,
            protected_words=protected_words,
            file_format=file_format,
            output_file_format=output_file_format,
            replace_columns=replace_columns,
            **kwargs,
        )
        metadata = {
            "input_path": str(input_path),
            "output_basename": output_basename,
            "source_lang": source_lang,
            "target_lang": lang,
            "columns": selected_cols,
            "column_type_filters": column_type_filters,
            "protected_words": protected_words,
            "file_format": file_format,
            "output_file_format": output_file_format,
            "translator": (
                translator.__class__.__name__ if translator else None
            ),
            "rate_limit_per_sec": (
                getattr(rate_limiter, "_rate", None) if rate_limiter else None
            ),
            "translator_version": get_version(),
        }
        write_translation_metadata(
            run_dir / "translation_metadata.json", metadata
        )
        overall.update(1)
    overall.close()


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
    replace_columns: bool = False,
    **kwargs,
):
    if column_type_filters is None and columns is None:
        column_type_filters = ["string"]

    proxy = kwargs.pop("proxy", None)
    use_cloud_api = kwargs.pop("use_cloud_api", False)
    kwargs.pop("google_api_key", None)
    rate_limit_per_sec = kwargs.pop("rate_limit_per_sec", None)
    translator = kwargs.pop(
        "translator", create_translator(use_cloud_api, proxy)
    )
    rate_limiter = kwargs.pop(
        "rate_limiter",
        TokenBucket(rate_limit_per_sec) if rate_limit_per_sec else None,
    )
    await translate_dataset_file(
        input_path=input_path,
        save_dir=save_dir,
        source_lang=source_lang,
        target_lang=target_lang,
        columns=columns,
        column_type_filters=column_type_filters,
        protected_words=protected_words,
        file_format=file_format,
        output_file_format=output_file_format,
        replace_columns=replace_columns,
        translator=translator,
        rate_limiter=rate_limiter,
        **kwargs,
    )


async def translate_hf_dataset_entry(
    dataset_name: str,
    save_dir: Path,
    source_lang: str,
    target_lang: str,
    columns: Optional[List[str]],
    column_type_filters: Optional[List[str]],
    protected_words: List[str],
    output_file_format: str,
    subset: Optional[str],
    splits: Optional[List[str]],
    hf_cache_dir: Optional[Path],
    replace_columns: bool = False,
    merge_translated_subsets: bool = False,
    push_to_hub: Optional[str] = None,
    hub_private: bool = False,
    **kwargs,
):
    ensure_output_root(save_dir)
    use_cloud_api = kwargs.pop("use_cloud_api", False)
    translator = kwargs.get("translator")
    rate_limiter = kwargs.get("rate_limiter")
    hf_cache = hf_cache_dir or resolve_hf_cache_dir(save_dir)
    datasets_dict = load_hf_splits(dataset_name, subset, splits, hf_cache)

    target_langs = normalize_target_langs(target_lang)
    total_jobs = len(target_langs) * len(datasets_dict)

    overall = tqdm(total=total_jobs, desc="Overall Progress", position=0)
    dataset_root = save_dir / sanitize_run_label(dataset_name)
    dataset_root.mkdir(parents=True, exist_ok=True)
    subset_label = subset or sanitize_run_label(dataset_name)
    merged_sources: List[Tuple[str, Path]] = []
    if push_to_hub and not merge_translated_subsets:
        if len(target_langs) > 1 and "{lang}" not in push_to_hub:
            raise ValueError(
                "When pushing multiple target languages without "
                "`--merge-translated-subsets`, include `{lang}` in "
                "`--push-to-hub` (e.g. `myuser/ds-{lang}`)."
            )
    for lang in target_langs:
        translated_subset = f"{subset_label}-{lang}"
        subset_dir = dataset_root / sanitize_run_label(translated_subset)
        subset_dir.mkdir(parents=True, exist_ok=True)
        merged_sources.append((lang, subset_dir))
        existing_meta = read_translation_metadata(
            subset_dir / "translation_metadata.json"
        )
        existing_dataset = (subset_dir / "dataset_dict.json").exists()
        if existing_meta and existing_dataset:
            if (
                existing_meta.get("dataset_name") == dataset_name
                and existing_meta.get("subset") == subset
                and existing_meta.get("source_lang") == source_lang
                and existing_meta.get("target_lang") == lang
                and existing_meta.get("columns") == columns
                and existing_meta.get("column_type_filters")
                == column_type_filters
                and existing_meta.get("replace_columns") == replace_columns
            ):
                overall.update(len(datasets_dict))
                continue
        translated_splits: Dict[str, Dataset] = {}
        for split_name, dataset in datasets_dict.items():
            print(f"Processing split: {split_name}")
            split_dir = subset_dir / "checkpoints" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            selected_cols = select_columns_from_hf(
                dataset, columns, column_type_filters
            )
            if not selected_cols:
                print(f"Skipping {split_name} (no matching columns)")
                overall.update(1)
                continue

            translated_dataset = await orchestrate_translation(
                dataset=dataset,
                dataset_length=len(dataset),
                save_dir=split_dir,
                source_lang=source_lang,
                target_lang=lang,
                columns=selected_cols,
                protected_words=protected_words,
                file_format="jsonl",
                output_file_format=output_file_format,
                replace_columns=replace_columns,
                output_mode="hf_dataset",
                **kwargs,
            )
            if translated_dataset is not None:
                translated_splits[split_name] = translated_dataset
            overall.update(1)

        if translated_splits:
            translated_dict = DatasetDict(translated_splits)
            translated_dict.save_to_disk(subset_dir)
            metadata = {
                "dataset_name": dataset_name,
                "subset": subset,
                "translated_subset": translated_subset,
                "source_lang": source_lang or "auto",
                "target_lang": lang,
                "splits": sorted(translated_splits.keys()),
                "columns": columns,
                "column_type_filters": column_type_filters,
                "protected_words": protected_words,
                "translator": (
                    translator.__class__.__name__ if translator else None
                ),
                "translator_version": get_version() or "development",
            }
            write_translation_metadata(
                subset_dir / "translation_metadata.json", metadata
            )
            if push_to_hub:
                repo_id = push_to_hub.replace("{lang}", lang)
                repo_id = ensure_hub_repo(repo_id, hub_private)
                translated_dict.push_to_hub(repo_id, private=hub_private)
                upload_metadata_to_hub(
                    repo_id, subset_dir / "translation_metadata.json"
                )
    overall.close()

    if merge_translated_subsets and merged_sources:
        merged_splits: Dict[str, Dataset] = {}
        merged_dir = dataset_root / "merged"
        merged_meta = read_translation_metadata(
            merged_dir / "translation_metadata.json"
        )
        if merged_meta:
            if (
                merged_meta.get("dataset_name") == dataset_name
                and merged_meta.get("subset") == subset
                and merged_meta.get("split_naming") == "<split>-<lang>"
                and set(merged_meta.get("languages", [])) == set(target_langs)
                and merged_meta.get("include_original") is True
                and set(merged_meta.get("original_splits", []))
                == set(datasets_dict.keys())
            ):
                if push_to_hub:
                    if "{lang}" in push_to_hub:
                        raise ValueError(
                            "Remove `{lang}` from `--push-to-hub` when using "
                            "`--merge-translated-subsets`."
                        )
                    repo_id = ensure_hub_repo(push_to_hub, hub_private)
                    merged_dict = DatasetDict.load_from_disk(merged_dir)
                    hub_dict = rename_splits_for_hub(merged_dict)
                    hub_dict.push_to_hub(repo_id, private=hub_private)
                    upload_metadata_to_hub(
                        repo_id,
                        merged_dir / "translation_metadata.json",
                    )
                return
        for split_name, dataset in datasets_dict.items():
            merged_splits[split_name] = dataset
        for lang, subset_dir in merged_sources:
            loaded = DatasetDict.load_from_disk(subset_dir)
            for split_name, dataset in loaded.items():
                merged_splits[f"{split_name}-{lang}"] = dataset
        merged_dict = DatasetDict(merged_splits)
        merged_dict.save_to_disk(merged_dir)
        merge_metadata = {
            "dataset_name": dataset_name,
            "subset": subset,
            "merged_dir": str(merged_dir),
            "split_naming": "<split>-<lang>",
            "languages": target_langs,
            "include_original": True,
            "original_splits": sorted(datasets_dict.keys()),
            "source_subsets": [str(path) for _, path in merged_sources],
        }
        write_translation_metadata(
            merged_dir / "translation_metadata.json", merge_metadata
        )
        if push_to_hub:
            if "{lang}" in push_to_hub:
                raise ValueError(
                    "Remove `{lang}` from `--push-to-hub` when using "
                    "`--merge-translated-subsets`."
                )
            repo_id = ensure_hub_repo(push_to_hub, hub_private)
            hub_dict = rename_splits_for_hub(merged_dict)
            hub_dict.push_to_hub(repo_id, private=hub_private)
            upload_metadata_to_hub(
                repo_id, merged_dir / "translation_metadata.json"
            )


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ..., help="Path to input dataset OR HF dataset name"
    ),
    save_dir: Path = typer.Argument(..., help="Directory to save output"),
    source_lang: Optional[str] = typer.Argument(
        None, help="Source language code (optional; defaults to auto-detect)"
    ),
    target_lang: Optional[str] = typer.Argument(
        None, help="Target language code"
    ),
    columns: Optional[List[str]] = typer.Option(
        None, "--columns", "-c", help="Specific columns to translate"
    ),
    column_types: Optional[List[str]] = typer.Option(
        None,
        "--column-type",
        "-t",
        help="Filter by type (string, list[string])",
    ),
    protected_words: Optional[str] = typer.Option(
        None, "--protected-words", "-p", help="Words to preserve"
    ),
    file_format: str = typer.Option(
        "auto",
        "--file-format",
        "-f",
        help="Input format (csv, parquet, jsonl, auto)",
    ),
    output_file_format: str = typer.Option(
        "auto", "--output-file-format", help="Output format"
    ),
    replace_columns: bool = typer.Option(
        False,
        "--replace-columns",
        help="Replace translated columns in-place (no extra columns)",
    ),
    batch_size: int = typer.Option(20, "--batch-size", "-b"),
    max_concurrency: int = typer.Option(10, "--max-concurrency"),
    checkpoint_step: int = typer.Option(100, "--checkpoint-step"),
    max_retries: int = typer.Option(3, "--max-retries"),
    failure_retry_cycles: int = typer.Option(3, "--max-failure-cycles"),
    only_failed: bool = typer.Option(False, "--only-failed"),
    proxy: Optional[str] = typer.Option(None, "--proxy"),
    use_cloud_api: bool = typer.Option(
        False, "--use-cloud-api", help="Use Google Cloud Translate API"
    ),
    rate_limit_per_sec: Optional[float] = typer.Option(None, "--rate-limit"),
    hf_dataset: bool = typer.Option(
        False, "--hf", help="Treat input_path as Hugging Face dataset name"
    ),
    subset: Optional[str] = typer.Option(None, "--subset", "--config"),
    splits: Optional[List[str]] = typer.Option(None, "--split", "-s"),
    hf_cache_dir: Optional[Path] = typer.Option(
        None, "--hf-cache-dir", help="Shared HF cache directory"
    ),
    merge_translated_subsets: bool = typer.Option(
        False,
        "--merge-translated-subsets",
        help="Merge per-language HF subsets into a single dataset root",
    ),
    push_to_hub: Optional[str] = typer.Option(
        None,
        "--push-to-hub",
        help=(
            "Push translated HF dataset(s) to the Hub. Use `{lang}` in the "
            "repo ID for per-language outputs."
        ),
    ),
    hub_private: bool = typer.Option(
        False,
        "--hub-private",
        help="Create/push the Hub repo as private (HF only).",
    ),
):
    ensure_output_root(save_dir)
    protected = load_protected_words(protected_words)
    columns = normalize_list_arg(columns)
    splits = normalize_list_arg(splits)
    filters = (
        normalize_column_types(column_types)
        if column_types
        else ["string"] if not columns else None
    )
    if target_lang is None:
        if source_lang and "," in source_lang:
            print(
                "No target_lang provided; treating source_lang as "
                "comma-separated targets and using auto-detect for source."
            )
            target_lang = source_lang
            source_lang = None
        else:
            raise ValueError(
                "target_lang is required (e.g. "
                "`dataset-translator <input> <out> en es,fr`)."
            )

    translator = create_translator(use_cloud_api, proxy)
    rate_limiter = (
        TokenBucket(rate_limit_per_sec) if rate_limit_per_sec else None
    )

    common_kwargs = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "columns": columns,
        "column_type_filters": filters,
        "protected_words": protected,
        "translator": translator,
        "rate_limiter": rate_limiter,
        "replace_columns": replace_columns,
        "use_cloud_api": use_cloud_api,
        "batch_size": batch_size,
        "max_concurrency": max_concurrency,
        "checkpoint_step": checkpoint_step,
        "max_retries": max_retries,
        "failure_retry_cycles": failure_retry_cycles,
        "only_failed": only_failed,
    }

    if hf_dataset:
        if output_file_format == "auto":
            output_file_format = "jsonl"
        asyncio.run(
            translate_hf_dataset_entry(
                dataset_name=str(input_path),
                save_dir=save_dir,
                output_file_format=output_file_format,
                subset=subset,
                splits=splits,
                hf_cache_dir=hf_cache_dir,
                merge_translated_subsets=merge_translated_subsets,
                push_to_hub=push_to_hub,
                hub_private=hub_private,
                **common_kwargs,
            )
        )
    else:
        if file_format == "auto":
            if input_path.suffix == ".csv":
                file_format = "csv"
            elif input_path.suffix in (".parquet", ".pq"):
                file_format = "parquet"
            elif input_path.suffix == ".jsonl":
                file_format = "jsonl"
            else:
                raise ValueError("Could not detect file format")

        if output_file_format == "auto":
            output_file_format = file_format

        asyncio.run(
            translate_dataset_file(
                input_path=input_path,
                save_dir=save_dir,
                file_format=file_format,
                output_file_format=output_file_format,
                **common_kwargs,
            )
        )


if __name__ == "__main__":
    app()
