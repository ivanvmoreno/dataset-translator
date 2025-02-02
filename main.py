#!/usr/bin/env python
import asyncio
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import typer
from googletrans import Translator
from tqdm import tqdm


def load_protected_words(protected_words_arg: Optional[str]) -> List[str]:
    """Load protected words from either a comma-separated string or a file."""
    if not protected_words_arg:
        return []

    if protected_words_arg.startswith("@"):
        file_path = Path(protected_words_arg[1:])
        if not file_path.exists():
            raise FileNotFoundError(f"Protected words file not found: {file_path}")
        return [
            line.strip() for line in file_path.read_text().splitlines() if line.strip()
        ]
    else:
        return [word.strip() for word in protected_words_arg.split(",") if word.strip()]


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


def restore_protected_words(translated_text: str, placeholders: Dict[str, str]) -> str:
    """Restores protected words using regex matching."""
    for placeholder, original in placeholders.items():
        pattern = re.compile(r"\s*".join(map(re.escape, placeholder)), re.IGNORECASE)
        translated_text = pattern.sub(original, translated_text)
    return translated_text


async def process_batch(
    batch: List[Tuple[int, str, str]],
    translator: Translator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    max_retries: int = 3,
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
        if not translated.strip() or translated.strip() == original_text.strip():
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
    translator: Translator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    save_dir: Path,
    batch_size: int = 20,
    max_concurrency: int = 10,
    checkpoint_step: int = 100,
    max_retries: int = 3,
    failure_retry_cycles: int = 1,
) -> None:
    """
    Orchestrate concurrent processing with checkpointing and failure cycles.
    """
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    existing = merge_checkpoints(checkpoint_dir)
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
            is_retry_cycle=True,
        )

        new_success_checkpoint = merge_checkpoints(checkpoint_dir)
        newly_translated = {
            (idx, col) for idx, cols in new_success_checkpoint.items() for col in cols
        }
        skip_set.update(newly_translated)
        all_failures = cycle_failures

    if all_failures:
        final_failures_path = checkpoint_dir / "translation_failures.parquet"
        pd.DataFrame(all_failures).to_parquet(final_failures_path)
        print(f"Some items still failed after {failure_retry_cycles} retry cycles.")
        print(f"Saved those failures to: {final_failures_path}")


async def _translate_in_batches(
    items: List[Tuple[int, str, str]],
    translator: Translator,
    source_lang: str,
    target_lang: str,
    protected_words: List[str],
    checkpoint_dir: Path,
    batch_size: int,
    max_concurrency: int,
    checkpoint_step: int,
    max_retries: int,
    is_retry_cycle: bool = False,
) -> List[Dict]:
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_desc = "Translating (retry cycle)" if is_retry_cycle else "Translating"
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
                checkpoint_dir / f"checkpoint_{checkpoint_counter:04d}.parquet",
            )
            results_buffer = []

    if results_buffer:
        checkpoint_counter += 1
        save_checkpoint(
            results_buffer,
            checkpoint_dir / f"checkpoint_{checkpoint_counter:04d}.parquet",
        )

    progress.close()
    return all_failures


def save_checkpoint(data: List[Dict], path: Path):
    """Save checkpoint file with transaction safety."""
    if not data:
        return
    temp_path = path.with_suffix(".tmp")
    pd.DataFrame(data).to_parquet(temp_path)
    temp_path.rename(path)


def merge_checkpoints(checkpoint_dir: Path) -> Dict[int, Dict[str, str]]:
    """
    Combine all checkpoint files into a single translation mapping.
    """
    merged = defaultdict(dict)
    for ckpt in checkpoint_dir.glob("checkpoint_*.parquet"):
        df = pd.read_parquet(ckpt)
        for _, row in df.iterrows():
            if "translated_text" in row:
                merged[row["original_index"]][row["column"]] = row["translated_text"]
    return merged


async def translate_dataset(
    input_path: Path,
    save_dir: Path,
    source_lang: str,
    target_lang: str,
    columns: Optional[List[str]],
    protected_words: List[str],
    batch_size: int = 20,
    max_concurrency: int = 10,
    checkpoint_step: int = 100,
    max_retries: int = 3,
    failure_retry_cycles: int = 1,
    only_failed: bool = False,
):
    """Main translation workflow with --only-failed support."""
    df = pd.read_parquet(input_path)

    if only_failed:
        failures_path = save_dir / "checkpoints" / "translation_failures.parquet"
        if not failures_path.exists():
            raise FileNotFoundError(f"No failures file found at {failures_path}")
        failures_df = pd.read_parquet(failures_path)
        required_cols = ["original_index", "column", "original_text"]
        if not all(col in failures_df.columns for col in required_cols):
            raise ValueError(f"Failures file missing required columns: {required_cols}")
        if columns:
            failures_df = failures_df[failures_df["column"].isin(columns)]
        items = [
            (row["original_index"], row["column"], row["original_text"])
            for _, row in failures_df.iterrows()
        ]
    else:
        if not columns:
            raise ValueError("Columns must be specified when not using --only-failed")
        items = []
        for idx, row in df.iterrows():
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")
                text = row[col]
                if isinstance(text, str) and text.strip():
                    items.append((idx, col, text))

    columns_used = list({col for _, col, _ in items})
    if not columns_used:
        print("No items to translate.")
        return

    await process_texts(
        items=items,
        translator=Translator(),
        source_lang=source_lang,
        target_lang=target_lang,
        protected_words=protected_words,
        save_dir=save_dir,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        checkpoint_step=checkpoint_step,
        max_retries=max_retries,
        failure_retry_cycles=failure_retry_cycles,
    )

    merged = merge_checkpoints(save_dir / "checkpoints")
    output_records = []
    for idx, row in df.iterrows():
        record = {"original_index": idx}
        for col in columns_used:
            record[f"original_{col}"] = row.get(col, "")
            record[f"translated_{col}"] = merged.get(idx, {}).get(col, "")
        output_records.append(record)

    final_path = save_dir / "translated_dataset.parquet"
    pd.DataFrame(output_records).to_parquet(final_path)
    print(f"✅ Translation complete! Final dataset saved to {final_path}")


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Path to input Parquet dataset"),
    save_dir: Path = typer.Argument(..., help="Directory to save translated data"),
    source_lang: str = typer.Option(
        "en", "--source-lang", "-s", help="Source language code"
    ),
    target_lang: str = typer.Option(
        "es", "--target-lang", "-t", help="Target language code"
    ),
    columns: Optional[List[str]] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Columns to translate (required unless --only-failed). Can be multiple. Pass the --columns (-c) flag multiple times to specify multiple columns.",
    ),
    protected_words: Optional[str] = typer.Option(
        None,
        "--protected-words",
        "-p",
        help="Comma-separated list or @file.txt of protected words",
    ),
    batch_size: int = typer.Option(
        20, "--batch-size", "-b", help="Number of texts per translation request"
    ),
    max_concurrency: int = typer.Option(
        20, "--max-concurrency", help="Maximum concurrent translation requests"
    ),
    checkpoint_step: int = typer.Option(
        500,
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
):
    """
    Translate columns in a Parquet dataset with support for retrying failed items.
    """
    if not only_failed and not columns:
        raise typer.BadParameter(
            "You must specify --columns unless using --only-failed"
        )

    protected = load_protected_words(protected_words)
    save_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        translate_dataset(
            input_path=input_path,
            save_dir=save_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            columns=columns,
            protected_words=protected,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            checkpoint_step=checkpoint_step,
            max_retries=max_retries,
            failure_retry_cycles=failure_retry_cycles,
            only_failed=only_failed,
        )
    )


if __name__ == "__main__":
    app()
