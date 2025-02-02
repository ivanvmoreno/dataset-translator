# `dataset-translator`

A robust CLI tool for translating text columns in datasets using Google Translate, with support for protected words, retries, and checkpoint recovery.

## Features

- **⚡️ Asynchronous**
  - Leverages Python’s asyncio for concurrent translation of text batches.
- **📦 Batch Processing**
  - Translates texts in batches to improve API efficiency.
- **💾 Checkpointing**
  - Saves completed translations periodically to prevent data loss during long-running tasks. Supports resuming from the last checkpoint.
- **🔄 Retry Mechanism**
  - Automatically retries failed translation batches with exponential backoff.
- **🛡️ Protected Words**
  - Preserves specific terms/phrases from being translated.
- **🚑 Failure Handling**
  - Supports re-processing of previously failed translations using a dedicated "only-failed" mode.

## Installation

```bash
pip install dataset-translator
```

## Usage

```bash
dataset-translator ./data.parquet ./output \
  --columns review_text --source-lang en --target-lang es \
  --protected-words "<think>,</think>"
```

### Key Options

| Option | Description |
|--------|-------------|
| `--columns \| -c` | Columns to translate (multiple allowed). Required unless using `--only-failed`. You can pass this flag multiple times for several columns. |
| `--source-lang \| -s` | Source language code (default: `en`). |
| `--target-lang \| -t` | Target language code (default: `es`). |
| `--protected-words \| -p` | Comma-separated list or `@file.txt` of protected words. |
| `--file-format \| -f` | File format to use: `csv`, `parquet`, or `auto` (automatic detection; default: `auto`). |
| `--batch-size \| -b` | Number of texts per translation request (default: `20`). |
| `--max-concurrency` | Maximum concurrent translation requests (default: `20`). |
| `--checkpoint-step` | Number of successful translations between checkpoints (default: `500`). |
| `--max-retries` | Maximum retry attempts per batch before marking as failed (default: `3`). |
| `--max-failure-cycles` | Number of full retry cycles for previously failed translations (default: `3`). |
| `--only-failed` | Process only previously failed translations from the checkpoint directory (default: `False`). |
