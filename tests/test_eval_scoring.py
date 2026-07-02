"""Tests for the scoring logic in src/eval/run_wer.py and
src/eval/run_speaker_similarity.py -- uses small fixture text/vectors, not
real model inference (no Whisper or Resemblyzer weights needed).
"""

import numpy as np

from run_speaker_similarity import cosine_similarity
from run_wer import compute_wer


def test_compute_wer_is_zero_for_exact_match():
    assert compute_wer("hello world", "hello world") == 0.0


def test_compute_wer_ignores_case_and_punctuation():
    assert compute_wer("Hello, world!", "hello world") == 0.0


def test_compute_wer_counts_substitutions():
    # 1 wrong word out of 2 -> WER 0.5
    assert compute_wer("hello world", "hello there") == 0.5


def test_compute_wer_counts_deletions():
    # "world" missing entirely -> 1 error out of 2 reference words
    assert compute_wer("hello world", "hello") == 0.5


def test_cosine_similarity_identical_unit_vectors_is_one():
    v = np.array([0.6, 0.8], dtype="float32")
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0], dtype="float32")
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_opposite_unit_vectors_is_negative_one():
    a = np.array([1.0, 0.0], dtype="float32")
    b = np.array([-1.0, 0.0], dtype="float32")
    assert cosine_similarity(a, b) == -1.0
