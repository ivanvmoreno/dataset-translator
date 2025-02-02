# `dataset-translator`

A robust CLI tool for translating text columns in datasets using Google Translate, with support for protected words, retries, and checkpoint recovery.

## Features

- **Asynchronous:**
  - Leverages Python’s asyncio for concurrent translation of text batches.
- **Batch Processing:**
  - Translates texts in batches to improve API efficiency.
- **Checkpointing:**
  - Saves completed translations periodically to prevent data loss during long-running tasks. Supports resuming from the last checkpoint.
- **Retry Mechanism:**
  - Automatically retries failed translation batches with exponential backoff.
- **Protected Words:**
  - Preserves specific terms/phrases from being translated.
- **Failure Handling:**
  - Supports re-processing of previously failed translations using a dedicated "only-failed" mode.

## Installation

```bash
pip install dataset-translator
```

## Usage

```bash
parquet-translator ./data.parquet ./output \
  --columns review_text --source-lang en --target-lang es \
  --protected-words "<think>,</think>"
```

### Key Options

| Option | Description |
|--------|-------------|
| `--columns` | Columns to translate (multiple allowed) |
| `--source-lang` | Source language code (default: en) |
| `--target-lang` | Target language code (default: es) |
| `--protected-words` | Protected terms (comma-separated or @file.txt) |