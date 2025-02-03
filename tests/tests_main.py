# tests/test_translation.py

import asyncio
import re
import shutil

import pandas as pd
import pytest
from typer.testing import CliRunner

from src.main import (
    load_protected_words,
    replace_protected_words,
    restore_protected_words,
    process_batch,
    process_texts,
    detect_file_format,
    load_dataset,
    save_dataset,
    save_checkpoint,
    merge_checkpoints,
    translate_dataset,
    app,
)


# --- Dummy translator for testing asynchronous functions ---
class DummyTranslation:
    def __init__(self, text):
        self.text = text


class DummyTranslator:
    async def translate(self, texts, src, dest):
        # For testing purposes, simply “translate” by reversing the string.
        # You can also simulate failures by raising exceptions conditionally.
        await asyncio.sleep(0.01)  # simulate network latency
        return [DummyTranslation(text[::-1]) for text in texts]


# --- Fixtures for temporary directories and files ---
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path / "test_dir"


@pytest.fixture
def sample_dataframe(tmp_path):
    df = pd.DataFrame(
        {"id": [1, 2], "text": ["Hello world", "Testing translation"]}
    )
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)
    return df, file_path


@pytest.fixture
def dummy_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


# --- Tests for helper functions ---


def test_load_protected_words_comma():
    # Test comma-separated list input.
    input_str = "alpha, beta, gamma"
    result = load_protected_words(input_str)
    assert result == ["alpha", "beta", "gamma"]


def test_load_protected_words_file(tmp_path):
    # Create a temporary file with protected words.
    words = ["foo", "bar", "baz"]
    file_path = tmp_path / "protected.txt"
    file_path.write_text("\n".join(words))
    result = load_protected_words("@" + str(file_path))
    assert result == words


def test_replace_and_restore_protected_words():
    text = "This is a secret message with a SAFE word."
    protected = ["secret", "safe"]
    # Replace
    modified_text, placeholders = replace_protected_words(text, protected)
    # Ensure that none of the original protected words appear in the modified text.
    for word in protected:
        assert re.search(re.escape(word), modified_text, re.IGNORECASE) is None
    # Simulate a “translation” that leaves placeholders unchanged (or with extra spaces)
    translated_text = modified_text.replace("__", " __ ")
    # Restore
    restored = restore_protected_words(translated_text, placeholders)
    # Check that the original words are back in the text.
    for word in protected:
        assert word.lower() in restored.lower()


def test_detect_file_format(tmp_path):
    # Create dummy files to test file format detection.
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2")
    parquet_file = tmp_path / "data.parquet"
    # Create a small dataframe and save as parquet
    df = pd.DataFrame({"a": [1, 2]})
    df.to_parquet(parquet_file)

    # Test detection based on file extension.
    assert detect_file_format(csv_file) == "csv"
    assert detect_file_format(parquet_file) == "parquet"

    # Test detection for a directory (when parquet files exist).
    temp_dir = tmp_path / "data_dir"
    temp_dir.mkdir()
    shutil.copy(str(parquet_file), str(temp_dir / "dummy.parquet"))
    assert detect_file_format(temp_dir) == "parquet"


def test_load_and_save_dataset(tmp_path):
    # Create a DataFrame, save it as CSV and then load it back.
    df = pd.DataFrame({"col": [1, 2, 3]})
    csv_path = tmp_path / "test_dataset.csv"
    save_dataset(df, csv_path, "csv")
    loaded_df = load_dataset(csv_path, "csv")
    pd.testing.assert_frame_equal(df, loaded_df)

    # Do the same for parquet.
    parquet_path = tmp_path / "test_dataset.parquet"
    save_dataset(df, parquet_path, "parquet")
    loaded_df = load_dataset(parquet_path, "parquet")
    pd.testing.assert_frame_equal(df, loaded_df)


def test_save_checkpoint_and_merge(dummy_checkpoint_dir):
    # Create dummy checkpoint data.
    data = [
        {"original_index": 0, "column": "text", "translated_text": "olleH"},
        {"original_index": 1, "column": "text", "translated_text": "dlroW"},
    ]
    checkpoint_path = dummy_checkpoint_dir / "checkpoint_0001.csv"
    save_checkpoint(data, checkpoint_path, "csv")
    assert checkpoint_path.exists()

    # Now merge checkpoints.
    merged = merge_checkpoints(dummy_checkpoint_dir, "csv")
    # merged is a dict mapping original_index to dicts of column: translated_text.
    assert 0 in merged
    assert merged[0]["text"] == "olleH"


# --- Tests for asynchronous translation functions ---


@pytest.mark.asyncio
async def test_process_batch_success():
    # Create a dummy batch: list of tuples (index, column, text)
    batch = [(0, "text", "Hello"), (1, "text", "World")]
    protected = []  # No protected words for this test.
    translator = DummyTranslator()
    successes, failures = await process_batch(
        batch,
        translator,
        source_lang="en",
        target_lang="es",
        protected_words=protected,
        max_retries=2,
    )
    # Since DummyTranslator “translates” by reversing the string, expect:
    assert len(successes) == 2
    for item in successes:
        # Check that the translated text is indeed reversed.
        orig_text = next(
            txt for idx, col, txt in batch if idx == item["original_index"]
        )
        assert item["translated_text"] == orig_text[::-1]
    assert len(failures) == 0


@pytest.mark.asyncio
async def test_process_texts_checkpointing(tmp_path):
    # Create a dummy dataset of items.
    items = [(i, "text", f"Message {i}") for i in range(25)]
    translator = DummyTranslator()
    protected = []
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Run the processing (note: using a small checkpoint_step to force writing checkpoints)
    await process_texts(
        items=items,
        translator=translator,
        source_lang="en",
        target_lang="es",
        protected_words=protected,
        save_dir=tmp_path,
        file_format="csv",
        batch_size=5,
        max_concurrency=3,
        checkpoint_step=10,
        max_retries=1,
        failure_retry_cycles=0,
    )
    # Check that some checkpoint files were created.
    ckpts = list((tmp_path / "checkpoints").glob("checkpoint_*.csv"))
    assert len(ckpts) >= 2


# --- Integration tests for the full workflow ---


@pytest.mark.asyncio
async def test_translate_dataset_integration(tmp_path, monkeypatch):
    # Prepare a small CSV dataset.
    df = pd.DataFrame({"text": ["Hello world", "Test message"]})
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    # Create a temporary directory for saving translations.
    save_dir = tmp_path / "output"
    save_dir.mkdir()

    # To simulate translation, we use our DummyTranslator.
    # Monkeypatch the Translator creation inside translate_dataset.
    def dummy_translator_init(*args, **kwargs):
        return DummyTranslator()

    monkeypatch.setattr(
        "src.main.Translator", lambda **kwargs: DummyTranslator()
    )

    # Call the main translation workflow.
    await translate_dataset(
        input_path=input_csv,
        save_dir=save_dir,
        source_lang="en",
        target_lang="es",
        columns=["text"],
        protected_words=[],
        file_format="csv",
        batch_size=1,
        max_concurrency=1,
        checkpoint_step=1,
        max_retries=1,
        failure_retry_cycles=0,
        only_failed=False,
        proxy=None,
    )

    # Check that the final translated dataset exists.
    final_path = save_dir / "translated_dataset.csv"
    assert final_path.exists()
    translated_df = pd.read_csv(final_path)
    # Our dummy translation reverses the string.
    expected = df["text"].apply(lambda s: s[::-1]).tolist()
    actual = translated_df["translated_text"].tolist()
    assert actual == expected


def test_main_cli(tmp_path, monkeypatch):
    # Test the Typer CLI command using CliRunner.
    runner = CliRunner()
    # Create a dummy CSV file.
    df = pd.DataFrame({"text": ["Hello", "World"]})
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)
    save_dir = tmp_path / "translated"
    save_dir.mkdir()

    # Monkeypatch the Translator so that it uses our DummyTranslator.
    monkeypatch.setattr(
        "src.main.Translator", lambda **kwargs: DummyTranslator()
    )

    # Run the CLI. Note: We need to supply --columns because --only-failed is False.
    result = runner.invoke(
        app,
        [
            str(input_csv),
            str(save_dir),
            "en",
            "es",
            "--columns",
            "text",
            "--batch-size",
            "1",
            "--max-concurrency",
            "1",
            "--checkpoint-step",
            "1",
            "--max-retries",
            "1",
            "--max-failure-cycles",
            "0",
        ],
    )
    # Check for successful execution (exit code 0) and confirmation message.
    assert result.exit_code == 0
    assert "Translation complete" in result.output
    # Verify that the final translated dataset was written.
    final_file = save_dir / "translated_dataset.csv"
    assert final_file.exists()


# --- Additional tests for error conditions ---


def test_main_no_columns(tmp_path):
    # When --only-failed is False and no columns are provided, main should raise an error.
    runner = CliRunner()
    df = pd.DataFrame({"text": ["Hello"]})
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)
    save_dir = tmp_path / "translated"
    save_dir.mkdir()
    result = runner.invoke(
        app,
        [
            str(input_csv),
            str(save_dir),
            "en",
            "es",
            # No --columns argument provided.
        ],
    )
    # Expect a nonzero exit code and error message about missing columns.
    assert result.exit_code != 0
    assert "must specify --columns" in result.output
