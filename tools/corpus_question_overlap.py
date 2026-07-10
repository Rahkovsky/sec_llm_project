#!/usr/bin/env python3
"""Do the 260 gold questions live in the same semantic neighborhood as the
corpus, or somewhere the corpus is thin? Projects a random sample of the
actual indexed passage embeddings (pulled from Chroma, not re-embedded) and
the 260 questions' embeddings (via embed_query, the real query-time path)
into 2D with PCA -> t-SNE, then plots both.

    python tools/corpus_question_overlap.py [--config config/local.yaml] [--sample 4000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.embeddings.factory import build_embedder
from sec_llm_project.sec_intel.index.store import SECIndex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("--out", default="out/seminar/img")
    ap.add_argument("--sample", type=int, default=4000, help="random corpus chunks to plot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    print("loading embedder + index ...", flush=True)
    embedder = build_embedder(cfg.embedding)
    index = SECIndex.load(cfg.index.path, expected=embedder.info)
    coll = index._chroma_collection  # noqa: SLF001 — read-only analysis, not a public API change

    total = coll.count()
    rng = random.Random(args.seed)
    print(f"corpus has {total:,} chunks; sampling {args.sample:,} ...", flush=True)
    all_ids = coll.get(include=[])["ids"]
    sample_ids = rng.sample(all_ids, min(args.sample, len(all_ids)))
    got = coll.get(ids=sample_ids, include=["embeddings", "metadatas"])
    corpus_vecs = np.asarray(got["embeddings"], dtype=np.float32)
    forms = [m.get("filing_type", "?") for m in got["metadatas"]]
    print(f"pulled {len(corpus_vecs):,} stored corpus embeddings, dim={corpus_vecs.shape[1]}")

    cases = json.loads(Path(args.cases).read_text())
    questions = [c["question"] for c in cases]
    print(f"embedding {len(questions)} test questions (embed_query path) ...", flush=True)
    q_vecs = np.asarray([embedder.embed_query(q) for q in questions], dtype=np.float32)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    combined = np.vstack([corpus_vecs, q_vecs])
    print("PCA -> 50d ...", flush=True)
    reduced = PCA(n_components=50, random_state=args.seed).fit_transform(combined)
    print("t-SNE -> 2d (this is the slow step) ...", flush=True)
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed,
             max_iter=1000).fit_transform(reduced)
    corpus_xy, q_xy = xy[: len(corpus_vecs)], xy[len(corpus_vecs):]

    # Nearest-corpus-point distance per question: a cheap, honest number
    # alongside the picture — "does the plot's impression hold up numerically?"
    from sklearn.neighbors import NearestNeighbors
    nn_corpus = NearestNeighbors(n_neighbors=1).fit(corpus_vecs)
    d_q_to_corpus, _ = nn_corpus.kneighbors(q_vecs)
    nn_corpus2 = NearestNeighbors(n_neighbors=2).fit(corpus_vecs)
    d_corpus_to_corpus, _ = nn_corpus2.kneighbors(corpus_vecs)  # [:,0] is self (d=0)
    summary = {
        "n_corpus_sample": len(corpus_vecs),
        "n_questions": len(q_vecs),
        "mean_question_to_nearest_corpus_dist": round(float(d_q_to_corpus.mean()), 4),
        "mean_corpus_to_nearest_corpus_dist": round(float(d_corpus_to_corpus[:, 1].mean()), 4),
    }
    print(json.dumps(summary, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_question_overlap.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=160)
    ax.scatter(corpus_xy[:, 0], corpus_xy[:, 1], s=5, alpha=0.35, color="#94A3B8",
              label=f"corpus sample (n={len(corpus_vecs):,})")
    ax.scatter(q_xy[:, 0], q_xy[:, 1], s=22, alpha=0.9, color="#DC2626",
              edgecolors="white", linewidths=0.4, label=f"test questions (n={len(q_vecs)})")
    ax.set_title("Do the test questions land inside the corpus, or off to the side?",
                 fontsize=12.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=10.5, frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(out / "corpus_question_tsne.png", facecolor="white")
    print(f"wrote {out}/corpus_question_tsne.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
