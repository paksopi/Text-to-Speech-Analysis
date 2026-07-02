"""Speaker-similarity score for every cloned output vs. its anchor clip.

Uses Resemblyzer's speaker-verification embedding (a d-vector style encoder)
and reports cosine similarity between the anchor's embedding and the cloned
output's embedding. This is the objective analogue of "does the clone
actually sound like the anchor voice" -- 1.0 is a perfect match, ~0.0 is an
unrelated voice, real same-speaker recordings typically land in the
0.75-0.95 range depending on recording conditions.
"""

import json
from pathlib import Path

import numpy as np

from manifest import MANIFEST

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "results" / "eval" / "speaker_similarity_results.json"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Resemblyzer embeddings are unit-normalized, so dot product == cosine similarity."""
    return float(np.dot(a, b))


def main():
    from resemblyzer import VoiceEncoder, preprocess_wav  # heavy import (torch) -- deferred for testability

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoder = VoiceEncoder()

    embedding_cache = {}

    def embed(path: Path):
        key = str(path)
        if key not in embedding_cache:
            wav = preprocess_wav(path)
            embedding_cache[key] = encoder.embed_utterance(wav)
        return embedding_cache[key]

    results = []
    for path, model_name, lang, _, is_cloned, anchor in MANIFEST:
        if not is_cloned or anchor is None:
            continue
        if not path.exists() or not anchor.exists():
            print(f"[skip] {path} or {anchor} not found")
            continue

        anchor_emb = embed(anchor)
        clone_emb = embed(path)
        similarity = cosine_similarity(anchor_emb, clone_emb)

        result = {
            "file": path.name,
            "model": model_name,
            "lang": lang,
            "anchor": anchor.name,
            "cosine_similarity": round(similarity, 4),
        }
        results.append(result)
        print(f"[{model_name}/{lang}] {path.name} vs {anchor.name}: similarity={similarity:.4f}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {len(results)} results to {OUT_PATH}")


if __name__ == "__main__":
    main()
