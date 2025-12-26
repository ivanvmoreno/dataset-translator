# `dataset-translator`

![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![PyPI version](https://badge.fury.io/py/dataset-translator.svg?icon=si%3Apython)](https://pypi.org/project/dataset-translator/)
![License](https://img.shields.io/badge/license-MIT-blue)
![Tests](img/tests-badge.svg)

A robust CLI tool for translating text columns in datasets using Google Translate, with support for protected words, retries, and checkpoint recovery. Works with both the unofficial Google Translate backend (free) and the Google Cloud Translation API.

> [!TIP]
> Check the [prioritized backlog](https://github.com/users/ivanvmoreno/projects/1/views/3?system_template=iterative_development) for ideas to contribute, or to get an idea of what's coming next!

## Features

- **📄 Supports multiple input and output formats**
  - Supports `CSV`, `Parquet`, `JSONL` inputs and outputs, with automatic format detection.
- **⚡️ Asynchronous**
  - Leverages Python’s `asyncio` for concurrent translation of text batches.
- **📦 Batch Processing**
  - Translates texts in batches to improve API efficiency.
- **💾 Checkpointing**
  - Saves completed translations periodically to prevent data loss during long-running tasks. Supports resuming from the last checkpoint.
- **🌍 Multi-target Translation**
  - Translate to multiple target languages in a single run.
- **🔄 Retry Mechanism**
  - Automatically retries failed translation batches with exponential backoff.
- **🛡️ Protected Words**
  - Preserves specific terms/phrases from being translated.
- **🚑 Failure Handling**
  - Supports re-processing of previously failed translations using a dedicated "only-failed" mode.
- **🧭 Auto Source Detection**
  - Omit the source language to auto-detect it (per batch).
- **🧩 Schema Control**
  - Filter by column types and optionally replace columns in-place.
- **🤗 Hugging Face Datasets**
  - Translate datasets from the Hub with support for subsets/configs, splits, and column type filters.
- **🧾 Translation Metadata**
  - Writes `translation_metadata.json` alongside outputs for reproducibility and traceability.
- **🧰 HF Subset Management**
  - Saves translated subsets as `<subset>-<lang>` and can merge them into a unified dataset.
- **☁️ Hub Uploads**
  - Optionally push translated HF datasets to the Hugging Face Hub.
- **🌐 Proxy Support**
  - Supports HTTP/HTTPS proxies for network requests.

## ✋ Important Notes and Limitations

- This project is not affiliated with Google.
- This project supports two backends: the unofficial web API via [py-googletrans](https://github.com/ssut/py-googletrans#how-does-this-library-work) and the official Google Cloud Translation API. The CLI is designed for feature parity across both.
- To use Google Cloud Translation API, pass `--use-cloud-api` and ensure the Cloud Translation API is enabled for your project and credentials.
- Maximum length per text is `15,000` characters for the unofficial Google Translate backend.
- When using the unofficial backend, your IP may be at risk of being blocked by Google if you abuse the service. Use responsibly (or consider using a proxy; see `--proxy` option).

## Installation

```bash
> pip install -U dataset-translator
> dataset-translator --help
```

## Usage

```bash
> dataset-translator <path_to_dataset> ./output eu \
  -c instruction -c output
```

Multi-target example (comma-separated targets):

```bash
> dataset-translator <path_to_dataset> ./output en es,fr,de \
  -c instruction -c output
```

You can omit the source language to auto-detect it:

```bash
> dataset-translator <path_to_dataset> ./output es \
  -c instruction -c output
```

### Output Layout

Each run creates a dedicated subdirectory under `save_dir` to prevent collisions:

- `<save_dir>/<dataset>__<source>_to_<target>/<dataset>_<target>.<format>` (use `auto` when source is auto-detected)
- Checkpoints: `checkpoints/batches/checkpoint_XXXX.<format>`
- Failures: `checkpoints/failures/translation_failures.csv` (or `.parquet` for parquet inputs)
- Translation metadata: `translation_metadata.json`

### Key Options

The `target_lang` positional argument can be a single language code or a comma-separated list for multi-target output.
If `source_lang` is omitted, it defaults to auto-detection.

| Option | Description |
|--------|-------------|
| `--columns \| -c` | Columns to translate (multiple allowed). Defaults to string columns. You can pass this flag multiple times for several columns. |
| `--column-type \| -t` | Filter columns by type (`string`, `list[string]`). Can be provided multiple times or comma-separated. |
| `--protected-words \| -p` | Comma-separated list or `@file.txt` of protected words. |
| `--file-format \| -f` | File format (`csv`, `parquet`, `jsonl`, `auto`). If not specified, file format will be inferred from the input file path. (default: `auto`). |
| `--output-file-format` | Output file format (`csv`, `parquet`, `jsonl`, `auto`). If not specified, output format will be fallback to input file format. (default: `auto`). |
| `--replace-columns` | Replace translated columns in-place to keep the output schema identical to the input. |
| `--batch-size \| -b` | Number of texts per translation request (default: `20`). |
| `--max-concurrency` | Maximum concurrent translation requests (default: `10`). |
| `--checkpoint-step` | Number of successful translations between checkpoints (default: `100`). |
| `--max-retries` | Maximum retry attempts per batch before marking as failed (default: `3`). |
| `--max-failure-cycles` | Number of full retry cycles for previously failed translations (default: `3`). |
| `--only-failed` | Process only previously failed translations from the checkpoint directory (default: `False`). |
| `--rate-limit` | Max translation requests per second (applied per batch). |
| `--proxy` | HTTP/HTTPS proxy URL. Protocol must be specified. (e.g., `http://<proxy_host>:<proxy_port>`). |
| `--use-cloud-api` | Use Google Cloud Translation API (auth via standard Google Cloud credentials). |
| `--hf-cache-dir` | Shared Hugging Face cache directory (defaults to `<save_dir>/../hf_cache`). |
| `--help` | Show help message and exit. |

### Hugging Face Datasets 🤗

Translate datasets from the Hub by passing `--hf` and using the dataset name in place of the input path.

Each translation run creates a new subset directory named `<subset>-<lang_code>` (or `<dataset_name>-<lang_code>` when no subset is provided) under `save_dir/<dataset_name>/`, saved as a Hugging Face dataset with translated splits.

Downloads are cached locally in a shared sibling directory (`<save_dir>/../hf_cache`) and reused on resume.

Each translated subset includes a `translation_metadata.json` file with the configuration used for reproducibility.

Checkpoints for each split are stored under `checkpoints/<split>` within the subset directory.

If `--merge-translated-subsets` is used, a unified dataset is written to `save_dir/<dataset_name>/merged/` containing the original splits plus `<split>-<lang>` translated splits.

When pushing merged datasets to the Hub, translated split names use underscores (`<split>_<lang>`) to satisfy Hub split naming rules.

```bash
> dataset-translator imdb ./output en es \
  --hf \
  --split train --split test \
  --column-type string
```

Use `--subset` (or `--config`) for dataset configurations, and `--columns` / `--column-type` to control which fields get translated (defaults to string columns).

Common HF options:

| Option | Description |
|--------|-------------|
| `--hf` | Treat the input path as a Hugging Face dataset name. |
| `--subset \| --config` | Dataset subset/config name. |
| `--split \| -s` | Split(s) to translate; can be provided multiple times. |
| `--merge-translated-subsets` | Merge per-language translated subsets into a single dataset root with `<split>-<lang>` split names, keeping the original splits intact. |
| `--push-to-hub` | Push translated HF dataset(s) to the Hub. Use `{lang}` placeholder in the repo ID name template for per-language outputs (omit `{lang}` when using `--merge-translated-subsets`). Missing repos are created automatically. If you omit the namespace, the logged-in user is used. |
| `--hub-private` | Create/push the Hub repo as private (HF only). |

## Supported Languages

Here is the list of languages supported by the Google Translate backends.

| Code     | Language                 |
|----------|--------------------------|
| af       | Afrikaans                |
| sq       | Albanian                 |
| am       | Amharic                  |
| ar       | Arabic                   |
| hy       | Armenian                 |
| as       | Assamese                 |
| ay       | Aymara                   |
| az       | Azerbaijani              |
| bm       | Bambara                  |
| eu       | Basque                   |
| be       | Belarusian               |
| bn       | Bengali                  |
| bho      | Bhojpuri                 |
| bs       | Bosnian                  |
| bg       | Bulgarian                |
| ca       | Catalan                  |
| ceb      | Cebuano                  |
| ny       | Chichewa                 |
| zh-CN    | Chinese (Simplified)     |
| zh-TW    | Chinese (Traditional)    |
| co       | Corsican                 |
| hr       | Croatian                 |
| cs       | Czech                    |
| da       | Danish                   |
| fa-AF    | Dari                     |
| dv       | Dhivehi                  |
| doi      | Dogri                    |
| nl       | Dutch                    |
| en       | English                  |
| eo       | Esperanto                |
| et       | Estonian                 |
| ee       | Ewe                      |
| tl       | Filipino                 |
| fi       | Finnish                  |
| fr       | French                   |
| fy       | Frisian                  |
| gl       | Galician                 |
| ka       | Georgian                 |
| de       | German                   |
| el       | Greek                    |
| gn       | Guarani                  |
| gu       | Gujarati                 |
| ht       | Haitian Creole           |
| ha       | Hausa                    |
| haw      | Hawaiian                 |
| iw       | Hebrew                   |
| hi       | Hindi                    |
| hmn      | Hmong                    |
| hu       | Hungarian                |
| is       | Icelandic                |
| ig       | Igbo                     |
| ilo      | Ilocano                  |
| id       | Indonesian               |
| ga       | Irish                    |
| it       | Italian                  |
| ja       | Japanese                 |
| jw       | Javanese                 |
| kn       | Kannada                  |
| kk       | Kazakh                   |
| km       | Khmer                    |
| rw       | Kinyarwanda              |
| gom      | Konkani                  |
| ko       | Korean                   |
| kri      | Krio                     |
| ku       | Kurdish (Kurmanji)       |
| ckb      | Kurdish (Sorani)         |
| ky       | Kyrgyz                   |
| lo       | Lao                      |
| la       | Latin                    |
| lv       | Latvian                  |
| ln       | Lingala                  |
| lt       | Lithuanian               |
| lg       | Luganda                  |
| lb       | Luxembourgish            |
| mk       | Macedonian               |
| mai      | Maithili                 |
| mg       | Malagasy                 |
| ms       | Malay                    |
| ms-Arab  | Malay (Jawi)             |
| ml       | Malayalam                |
| mt       | Maltese                  |
| mi       | Maori                    |
| mr       | Marathi                  |
| mni-Mtei | Meiteilon (Manipuri)     |
| lus      | Mizo                     |
| mn       | Mongolian                |
| my       | Myanmar (Burmese)        |
| ne       | Nepali                   |
| bm-Nkoo  | NKo                      |
| no       | Norwegian                |
| or       | Odia (Oriya)             |
| om       | Oromo                    |
| ps       | Pashto                   |
| fa       | Persian                  |
| pl       | Polish                   |
| pt       | Portuguese (Brazil)      |
| pt-PT    | Portuguese (Portugal)    |
| pa       | Punjabi (Gurmukhi)       |
| pa-Arab  | Punjabi (Shahmukhi)      |
| qu       | Quechua                  |
| ro       | Romanian                 |
| ru       | Russian                  |
| sm       | Samoan                   |
| sa       | Sanskrit                 |
| gd       | Scots Gaelic             |
| nso      | Sepedi                   |
| sr       | Serbian                  |
| st       | Sesotho                  |
| sn       | Shona                    |
| sd       | Sindhi                   |
| si       | Sinhala                  |
| sk       | Slovak                   |
| sl       | Slovenian                |
| so       | Somali                   |
| es       | Spanish                  |
| su       | Sundanese                |
| sw       | Swahili                  |
| sv       | Swedish                  |
| tg       | Tajik                    |
| ta       | Tamil                    |
| tt       | Tatar                    |
| te       | Telugu                   |
| th       | Thai                     |
| ti       | Tigrinya                 |
| ts       | Tsonga                   |
| tr       | Turkish                  |
| tk       | Turkmen                  |
| ak       | Twi                      |
| uk       | Ukrainian                |
| ur       | Urdu                     |
| ug       | Uyghur                   |
| uz       | Uzbek                    |
| vi       | Vietnamese               |
| cy       | Welsh                    |
| xh       | Xhosa                    |
| yi       | Yiddish                  |
| yo       | Yoruba                   |
| zu       | Zulu                     |

[Source](https://github.com/ssut/py-googletrans/issues/408#issuecomment-2246262832)
