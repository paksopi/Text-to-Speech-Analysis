"""Tests for src/eval/manifest.py -- the manifest drives both run_wer.py and
run_speaker_similarity.py, so an inconsistency here silently breaks those
evaluations without raising an error (they just skip missing files).
"""

from manifest import AUDIO_DIR, COMPARISON_DIR, MANIFEST
from text_data import MENTOR_SENTENCES, SHORT_SENTENCES

VALID_LANGS = {"en", "ms", "id"}


def test_manifest_entries_have_expected_shape():
    for entry in MANIFEST:
        assert len(entry) == 6
        path, model, lang, expected_text, is_cloned, anchor = entry
        assert path.suffix == ".wav"
        assert isinstance(model, str) and model
        assert lang in VALID_LANGS
        assert isinstance(expected_text, str) and expected_text
        assert isinstance(is_cloned, bool)
        assert anchor is None or anchor.suffix == ".wav"


def test_cloned_entries_always_have_an_anchor():
    for path, _model, _lang, _text, is_cloned, anchor in MANIFEST:
        if is_cloned:
            assert anchor is not None, f"{path.name} is marked cloned but has no anchor"
        else:
            assert anchor is None, f"{path.name} is not cloned but has an anchor set"


def test_manifest_paths_are_unique():
    paths = [str(entry[0]) for entry in MANIFEST]
    assert len(paths) == len(set(paths)), "duplicate path in MANIFEST"


def test_manifest_paths_live_under_audio_or_comparison_dir():
    for path, *_ in MANIFEST:
        assert AUDIO_DIR in path.parents or COMPARISON_DIR in path.parents


def test_voxcpm2_entries_use_the_mentor_sentences():
    voxcpm2_entries = [e for e in MANIFEST if e[1] == "voxcpm2"]
    assert len(voxcpm2_entries) == 6  # baseline + cloned, x3 languages
    lang_map = {"en": "en", "ms": "bm", "id": "id"}  # manifest lang -> text_data key
    for _path, _model, lang, expected_text, _is_cloned, _anchor in voxcpm2_entries:
        assert expected_text == MENTOR_SENTENCES[lang_map[lang]]


def test_non_voxcpm2_entries_use_the_short_sentences():
    # mms_tts/ms is the one exception: SHORT_SENTENCES has no "ms" entry, so
    # run_mms_tts.py generates that clip from the full BM mentor sentence instead.
    for _path, model, lang, expected_text, _is_cloned, _anchor in MANIFEST:
        if model == "mms_tts" and lang == "ms":
            assert expected_text == MENTOR_SENTENCES["bm"]
        elif model != "voxcpm2":
            assert expected_text == SHORT_SENTENCES[lang]
