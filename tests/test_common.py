"""Tests for src/common.py that don't require a GPU or downloading VoxCPM2.

load_model() itself is intentionally not tested here -- it imports the real
voxcpm package and downloads weights on first use, which doesn't belong in a
fast unit-test suite.
"""

import numpy as np

from common import AUDIO_DIR, LOG_DIR, ROOT, TEST_SENTENCES, controlled_text, generate_and_save
from text_data import MENTOR_SENTENCES


def test_test_sentences_is_the_shared_mentor_sentences():
    # TEST_SENTENCES is kept only as a backwards-compatible alias -- make sure
    # it hasn't drifted from the single source of truth in text_data.py.
    assert TEST_SENTENCES is MENTOR_SENTENCES


def test_audio_and_log_dirs_are_under_results():
    assert AUDIO_DIR == ROOT / "results" / "audio"
    assert LOG_DIR == ROOT / "results" / "logs"


def test_controlled_text_wraps_control_instruction_in_parens():
    result = controlled_text("hello world", control="be calm")
    assert result == "(be calm)hello world"


def test_controlled_text_uses_default_control_when_not_given():
    result = controlled_text("hello world")
    assert result.startswith("(")
    assert result.endswith("hello world")


class _FakeTTSModel:
    sample_rate = 16000


class _FakeModel:
    tts_model = _FakeTTSModel()

    def generate(self, text, **kwargs):
        return np.zeros(10, dtype="float32")


def test_generate_and_save_writes_wav_and_appends_log(tmp_path, monkeypatch):
    written = {}

    def fake_write(path, data, samplerate):
        written["path"] = path
        written["samplerate"] = samplerate

    monkeypatch.setattr("common.sf.write", fake_write)

    out_path = tmp_path / "out.wav"
    log_path = tmp_path / "log.tsv"

    elapsed = generate_and_save(_FakeModel(), "hello world", out_path, log_path)

    assert elapsed >= 0
    assert written["path"] == out_path
    assert written["samplerate"] == 16000

    log_contents = log_path.read_text(encoding="utf-8")
    assert "out.wav" in log_contents
    assert "hello world" in log_contents
