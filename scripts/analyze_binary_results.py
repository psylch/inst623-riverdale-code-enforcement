"""Post-hoc analysis of Gemma 4 binary verification results.

Reads the (98, 9) score/answer matrices from `client_gemma4_binary.npz`
(or reconstructs them from `client_gemma4_binary_stream.jsonl`), joins
with multi-label ground truth, computes per-class AUC and recall/precision,
compares against CLIP separability results, and saves all figures to
`reports/figures/`.

Safe to run incrementally: if the binary run is still in progress, this
script will read whatever is in the stream file so far and report
per-class stats on the partial data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.client_data import CLIENT_CLASSES, CLIENT_DISPLAY, FOLDER_KEYWORD_MAP, build_client_catalogue


def load_binary_from_stream(stream_path: Path, N: int, K: int):
    """Rebuild (scores, answers, parse_ok) from the JSONL stream."""
    scores = np.full((N, K), np.nan, dtype=np.float32)
    answers = np.full((N, K), -1, dtype=np.int8)
    parse_ok = np.zeros((N, K), dtype=bool)
    done = np.zeros((N, K), dtype=bool)

    if not stream_path.exists():
        raise FileNotFoundError(stream_path)

    with stream_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            i, j = int(r.get("i", -1)), int(r.get("j", -1))
            if not (0 <= i < N and 0 <= j < K):
                continue
            done[i, j] = True
            parse_ok[i, j] = bool(r.get("parse_ok", False))
            ans = r.get("answer")
            conf = r.get("confidence")
            if ans == "yes":
                answers[i, j] = 1
                scores[i, j] = (conf if isinstance(conf, int) else 50) / 100.0
            elif ans == "no":
                answers[i, j] = 0
                scores[i, j] = 1.0 - ((conf if isinstance(conf, int) else 50) / 100.0)
            else:
                answers[i, j] = -1
                scores[i, j] = 0.5
    return scores, answers, parse_ok, done


def build_multilabel_truth(df) -> np.ndarray:
    root = REPO / "data" / "client-data"
    hash2labels: dict[str, set[str]] = defaultdict(set)
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        for keyword, canonical in FOLDER_KEYWORD_MAP.items():
            if keyword in f.parent.name:
                hash2labels[h].add(canonical)
                break

    path2hash = {
        str(f.resolve()): hashlib.md5(f.read_bytes()).hexdigest()
        for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
    }

    K = len(CLIENT_CLASSES)
    Y = np.zeros((len(df), K), dtype=np.int8)
    for i, p in enumerate(df["path"].tolist()):
        h = path2hash[str(Path(p).resolve())]
        for lab in hash2labels[h]:
            Y[i, CLIENT_CLASSES.index(lab)] = 1
    return Y


def per_class_binary_stats(scores: np.ndarray, Y_multi: np.ndarray, done: np.ndarray) -> list[dict]:
    """For each class: AUC, recall@0.5, precision@0.5, total n_pos, n_neg."""
    stats = []
    for j, cls in enumerate(CLIENT_CLASSES):
        col = scores[:, j]
        ok = done[:, j]
        pos_mask = (Y_multi[:, j] == 1) & ok
        neg_mask = (Y_multi[:, j] == 0) & ok
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            stats.append({"class": cls, "n_pos": n_pos, "n_neg": n_neg, "auc": float("nan"),
                          "recall_at_0.5": float("nan"), "precision_at_0.5": float("nan")})
            continue
        y_lbl = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        sc = np.concatenate([col[pos_mask], col[neg_mask]])
        auc = float(roc_auc_score(y_lbl, sc))
        # at threshold 0.5 (i.e. model said "yes" with conf >= 50)
        y_pred = (col >= 0.5)
        tp = int(((Y_multi[:, j] == 1) & y_pred & ok).sum())
        fp = int(((Y_multi[:, j] == 0) & y_pred & ok).sum())
        fn = int(((Y_multi[:, j] == 1) & ~y_pred & ok).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        stats.append({
            "class": cls, "n_pos": n_pos, "n_neg": n_neg,
            "auc": auc, "recall_at_0.5": recall, "precision_at_0.5": precision,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return stats


def jaccard_multilabel(y_pred_set: np.ndarray, y_true_set: np.ndarray) -> tuple[float, float]:
    """Per-image Jaccard + sample-averaged F1 over multi-label sets."""
    N = y_pred_set.shape[0]
    jac = []
    f1s = []
    for i in range(N):
        inter = np.logical_and(y_pred_set[i], y_true_set[i]).sum()
        union = np.logical_or(y_pred_set[i], y_true_set[i]).sum()
        tp = inter
        fp = y_pred_set[i].sum() - tp
        fn = y_true_set[i].sum() - tp
        jac.append(inter / union if union > 0 else 1.0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(jac)), float(np.mean(f1s))


def cascade_eval(
    clip_sim: np.ndarray,
    binary_scores: np.ndarray,
    Y_multi: np.ndarray,
    k_values: list[int],
):
    """Evaluate CLIP top-k candidate generation followed by Gemma binary filtering.

    Returns a list of {k, recall_any, jaccard, f1} records.
    """
    N, K = clip_sim.shape
    results = []
    for k in k_values:
        # for each image, the "candidates" are CLIP's top-k classes
        top_k_idx = np.argsort(-clip_sim, axis=1)[:, :k]
        y_pred_set = np.zeros((N, K), dtype=bool)
        for i in range(N):
            for j in top_k_idx[i]:
                if binary_scores[i, j] >= 0.5:
                    y_pred_set[i, j] = True
        jac, f1 = jaccard_multilabel(y_pred_set, Y_multi.astype(bool))
        # recall-any: did we catch any true label?
        hit_any = ((y_pred_set & Y_multi.astype(bool)).any(axis=1)).mean()
        results.append({
            "k": k,
            "jaccard": jac,
            "f1": f1,
            "recall_any": float(hit_any),
        })
    return results


def plot_binary_auc_compare(stats_binary: list[dict], clip_sum_path: Path, out_path: Path):
    clip_stats = json.loads(clip_sum_path.read_text())
    clip_by_class = {s["class"]: s for s in clip_stats}
    names = [CLIENT_DISPLAY[s["class"]].replace("\n", " ") for s in stats_binary]
    gem_auc = [s["auc"] for s in stats_binary]
    clip_auc = [clip_by_class[s["class"]]["auc"] for s in stats_binary]

    order = np.argsort(gem_auc)[::-1]
    names = [names[i] for i in order]
    gem_auc = [gem_auc[i] for i in order]
    clip_auc = [clip_auc[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(names))
    w = 0.38
    ax.barh(y_pos - w / 2, clip_auc, w, color="#4575b4", label="CLIP")
    ax.barh(y_pos + w / 2, gem_auc, w, color="#d73027", label="Gemma 4 binary")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlim(0.0, 1.0)
    ax.axvline(0.5, color="#888", linestyle="--", linewidth=1)
    ax.set_xlabel("ROC-AUC (multi-label positives vs rest)")
    ax.set_title("Per-class separability — CLIP vs Gemma 4 binary", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_client_catalogue()
    N, K = len(df), len(CLIENT_CLASSES)

    stream_path = REPO / "checkpoints" / "client_gemma4_binary_stream.jsonl"
    scores, answers, parse_ok, done = load_binary_from_stream(stream_path, N, K)
    print(f"[analyze] stream {stream_path.name}: {int(done.sum())}/{N*K} cells done")
    print(f"[analyze] parse ok on done cells: {int(parse_ok[done].sum())}/{int(done.sum())}")

    Y_multi = build_multilabel_truth(df)
    stats_binary = per_class_binary_stats(scores, Y_multi, done)
    print()
    print("=== Gemma 4 binary per-class ===")
    print(f"{'class':<25}  {'n_pos':>5}  {'n_neg':>5}  {'auc':>5}  {'recall':>7}  {'precision':>9}")
    for s in stats_binary:
        r = f"{s['recall_at_0.5']:.2f}" if np.isfinite(s['recall_at_0.5']) else "  nan"
        p = f"{s['precision_at_0.5']:.2f}" if np.isfinite(s['precision_at_0.5']) else "  nan"
        a = f"{s['auc']:.3f}" if np.isfinite(s['auc']) else " nan "
        print(f"  {s['class']:<23}  {s['n_pos']:>5}  {s['n_neg']:>5}  {a:>5}  {r:>7}  {p:>9}")

    summary_path = REPO / "checkpoints" / "gemma4_binary_summary.json"
    summary_path.write_text(json.dumps(stats_binary, indent=2))
    print(f"[analyze] saved {summary_path}")

    # only do the full eval if the run is complete
    if int(done.sum()) == N * K:
        print()
        print("=== cascade (CLIP top-k → Gemma binary verification) ===")
        sim_npz = np.load(REPO / "checkpoints" / "client_clip_similarity.npz")
        clip_sim = sim_npz["similarity"]
        cascade = cascade_eval(clip_sim, scores, Y_multi, k_values=[3, 5, 9])
        for r in cascade:
            print(f"  k={r['k']}  jaccard={r['jaccard']:.3f}  sample-F1={r['f1']:.3f}  recall_any={r['recall_any']:.3f}")

        cascade_path = REPO / "checkpoints" / "cascade_summary.json"
        cascade_path.write_text(json.dumps(cascade, indent=2))

        # full multi-label eval: model-only (no CLIP filter, threshold 0.5)
        y_pred_set = (scores >= 0.5)
        jac, f1 = jaccard_multilabel(y_pred_set, Y_multi.astype(bool))
        print()
        print(f"[standalone] Gemma binary (no CLIP filter, thresh 0.5): "
              f"jaccard={jac:.3f}  sample-F1={f1:.3f}")

        # AUC comparison plot
        clip_sum_path = REPO / "checkpoints" / "clip_separability_summary.json"
        if clip_sum_path.exists():
            out_path = REPO / "reports" / "figures" / "auc_clip_vs_gemma_binary.png"
            plot_binary_auc_compare(stats_binary, clip_sum_path, out_path)
            print(f"[analyze] saved {out_path}")
    else:
        missing = N * K - int(done.sum())
        print(f"[analyze] run still in progress — {missing} cells remain. Full eval deferred.")


if __name__ == "__main__":
    main()
