"""text_data.py is the single source of truth for every test sentence used
across generation, comparison, and eval scripts. These tests just guard
against typos/regressions in that one shared file -- no GPU or model needed.
"""

import os

from text_data import F5TTS_REF_TEXT, MENTOR_SENTENCES, PARLER_DESCRIPTION, SHORT_SENTENCES


def test_mentor_sentences_cover_all_three_languages():
    assert set(MENTOR_SENTENCES) == {"en", "bm", "id"}
    for lang, text in MENTOR_SENTENCES.items():
        assert isinstance(text, str)
        assert text == text.strip(), f"{lang} mentor sentence has leading/trailing whitespace"
        assert len(text) > 0


def test_short_sentences_cover_en_and_id():
    assert set(SHORT_SENTENCES) == {"en", "id"}
    for lang, text in SHORT_SENTENCES.items():
        assert isinstance(text, str)
        assert text == text.strip(), f"{lang} short sentence has leading/trailing whitespace"
        assert len(text) > 0


def test_short_sentence_shares_opening_with_mentor_sentence():
    # The short comparison-script sentences are meant to be a truncated
    # version of the longer VoxCPM2 mentor sentences (they diverge partway
    # through -- the mentor version uses an em dash + more clauses). This
    # guards against the *opening* silently drifting apart between the two.
    for lang in SHORT_SENTENCES:
        common = os.path.commonprefix([MENTOR_SENTENCES[lang], SHORT_SENTENCES[lang]])
        assert len(common) > 60, f"{lang}: short/mentor sentences share only {len(common)} leading chars"


def test_f5tts_and_parler_auxiliary_text_present():
    assert isinstance(F5TTS_REF_TEXT, str) and F5TTS_REF_TEXT
    assert isinstance(PARLER_DESCRIPTION, str) and PARLER_DESCRIPTION
