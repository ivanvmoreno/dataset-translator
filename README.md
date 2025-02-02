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
dataset-translator <path_to_dataset> ./output en eu \
  -c instruction -c output
```

### Key Options

| Option | Description |
|--------|-------------|
| `--columns \| -c` | Columns to translate (multiple allowed). Required unless using `--only-failed`. You can pass this flag multiple times for several columns. |
| `--protected-words \| -p` | Comma-separated list or `@file.txt` of protected words. |
| `--file-format \| -f` | File format to use: `csv`, `parquet`, or `auto` (automatic detection; default: `auto`). |
| `--batch-size \| -b` | Number of texts per translation request (default: `1`). |
| `--max-concurrency` | Maximum concurrent translation requests (default: `1`). |
| `--checkpoint-step` | Number of successful translations between checkpoints (default: `500`). |
| `--max-retries` | Maximum retry attempts per batch before marking as failed (default: `3`). |
| `--max-failure-cycles` | Number of full retry cycles for previously failed translations (default: `3`). |
| `--only-failed` | Process only previously failed translations from the checkpoint directory (default: `False`). |

### Supported Languages

Here is the list of languages that are supported (free of restrictions, without subscription) by the service at `translate.googleapis.com`:

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