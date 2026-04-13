"""Data cleaning audit for proxy datasets.

Checks:
1. Corrupt / unreadable images
2. Duplicate images (perceptual hash)
3. Near-zero-variance images (blank / solid color)
4. Extreme aspect ratios
5. Tiny images (< 32x32)
6. Per-dataset and per-class statistics
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# ── BD3 label map (same as data.py) ──
BD3_MAP = {
    "major_crack": "structural_damage",
    "minor_crack": "structural_damage",
    "peeling": "exterior_deterioration",
    "spalling": "exterior_deterioration",
    "algae": "exterior_deterioration",
    "stain": "exterior_deterioration",
    "plain": "__excluded__",
}


def collect_all_images() -> list[dict]:
    """Collect all image paths with metadata."""
    rows = []

    # BD3
    bd3_base = ROOT / "BD3" / "BD3_original_dataset" / "train"
    if bd3_base.exists():
        for folder in sorted(bd3_base.iterdir()):
            if not folder.is_dir():
                continue
            label = BD3_MAP.get(folder.name, folder.name)
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append({"path": f, "label": label, "source": "BD3", "orig_class": folder.name})

    # TACO
    taco_ann = ROOT / "TACO" / "data" / "annotations.json"
    if taco_ann.exists():
        ann = json.loads(taco_ann.read_text())
        for img in ann["images"]:
            p = ROOT / "TACO" / "data" / img["file_name"]
            if p.exists():
                rows.append({"path": p, "label": "trash_debris", "source": "TACO", "orig_class": "trash"})

    # Roboflow datasets (original + broken-fence)
    for dataset_dir, label in [
        ("grass-weeds", "overgrown_vegetation"),
        ("aerial-dumping", "illegal_dumping"),
        ("broken-fence", "damaged_structures"),
    ]:
        base = ROOT / dataset_dir
        if not base.exists():
            continue
        for split_dir in ("train", "valid", "test"):
            d = base / split_dir
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append({"path": f, "label": label, "source": dataset_dir, "orig_class": label})

    # Garbage Object Detection (HuggingFace, extracted zips)
    garbage_base = ROOT / "garbage-object-detection"
    if garbage_base.exists():
        for split_dir in ("train", "valid", "test"):
            d = garbage_base / split_dir
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append({"path": f, "label": "trash_debris", "source": "garbage-object-detection", "orig_class": "garbage"})

    # Building Surface Defect Detection (HuggingFace, YOLO format)
    bsd_base = ROOT / "building-surface-defect" / "images"
    if bsd_base.exists():
        for split_dir in ("train", "val", "test"):
            d = bsd_base / split_dir
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append({"path": f, "label": "exterior_deterioration", "source": "building-surface-defect", "orig_class": "building_defect"})

    return rows


def check_image(row: dict) -> dict:
    """Run all checks on a single image. Returns issue dict."""
    path = row["path"]
    issues = []

    # 1. Try to open
    try:
        img = Image.open(path)
        img.verify()  # verify integrity
        img = Image.open(path)  # reopen after verify
        img.load()  # force full decode
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return {"path": str(path), "issues": [f"corrupt: {e}"], **{k: row[k] for k in ("label", "source", "orig_class")}}

    w, h = img.size
    info = {
        "path": str(path),
        "width": w,
        "height": h,
        "format": img.format,
        "mode": img.mode,
        "file_size_kb": path.stat().st_size / 1024,
        **{k: row[k] for k in ("label", "source", "orig_class")},
    }

    # 2. Tiny images
    if w < 32 or h < 32:
        issues.append(f"tiny: {w}x{h}")

    # 3. Extreme aspect ratio (> 5:1 or < 1:5)
    ar = max(w, h) / max(min(w, h), 1)
    if ar > 5:
        issues.append(f"extreme_aspect_ratio: {ar:.1f}")

    # 4. Near-zero variance (blank/solid)
    try:
        thumb = img.convert("L").resize((64, 64))
        arr = np.array(thumb, dtype=np.float32)
        std = arr.std()
        if std < 3.0:
            issues.append(f"low_variance: std={std:.1f}")
    except Exception:
        pass

    # 5. Perceptual hash (for duplicate detection later)
    try:
        thumb = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.array(thumb).flatten()
        avg = pixels.mean()
        phash = "".join("1" if p > avg else "0" for p in pixels)
        info["phash"] = phash
    except Exception:
        info["phash"] = None

    # 6. File hash (exact duplicate)
    info["md5"] = hashlib.md5(path.read_bytes()).hexdigest()

    info["issues"] = issues
    return info


def find_duplicates(results: list[dict]) -> dict:
    """Find exact and perceptual duplicates."""
    # Exact duplicates (same MD5)
    md5_groups = defaultdict(list)
    for r in results:
        if "md5" in r:
            md5_groups[r["md5"]].append(r["path"])
    exact_dupes = {k: v for k, v in md5_groups.items() if len(v) > 1}

    # Perceptual duplicates (same phash, across different sources)
    phash_groups = defaultdict(list)
    for r in results:
        if r.get("phash"):
            phash_groups[r["phash"]].append((r["path"], r["source"]))
    # Only flag cross-source perceptual dupes
    perceptual_dupes = {}
    for k, v in phash_groups.items():
        sources = set(s for _, s in v)
        if len(v) > 1 and len(sources) > 1:
            perceptual_dupes[k] = [(p, s) for p, s in v]

    return {"exact": exact_dupes, "perceptual_cross_source": perceptual_dupes}


def generate_report(results: list[dict], duplicates: dict) -> str:
    """Generate a markdown cleaning report."""
    total = len(results)
    corrupt = [r for r in results if any("corrupt" in i for i in r.get("issues", []))]
    tiny = [r for r in results if any("tiny" in i for i in r.get("issues", []))]
    extreme_ar = [r for r in results if any("extreme_aspect" in i for i in r.get("issues", []))]
    low_var = [r for r in results if any("low_variance" in i for i in r.get("issues", []))]
    clean = [r for r in results if not r.get("issues")]

    # Per-source stats
    source_stats = defaultdict(lambda: {"total": 0, "issues": 0, "widths": [], "heights": [], "sizes_kb": []})
    for r in results:
        s = source_stats[r["source"]]
        s["total"] += 1
        if r.get("issues"):
            s["issues"] += len(r["issues"])
        if "width" in r:
            s["widths"].append(r["width"])
            s["heights"].append(r["height"])
        if "file_size_kb" in r:
            s["sizes_kb"].append(r["file_size_kb"])

    # Per-label stats
    label_counts = Counter(r["label"] for r in results)

    lines = []
    lines.append("# Data Cleaning Report")
    lines.append("")
    lines.append(f"> Automated audit of {total} images across {len(source_stats)} proxy datasets.")
    lines.append(f"> Generated {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"| Metric | Count | % |")
    lines.append(f"|--------|-------|---|")
    lines.append(f"| Total images scanned | {total} | 100% |")
    lines.append(f"| Clean (no issues) | {len(clean)} | {len(clean)/total*100:.1f}% |")
    lines.append(f"| Corrupt / unreadable | {len(corrupt)} | {len(corrupt)/total*100:.1f}% |")
    lines.append(f"| Tiny (< 32x32) | {len(tiny)} | {len(tiny)/total*100:.1f}% |")
    lines.append(f"| Extreme aspect ratio (> 5:1) | {len(extreme_ar)} | {len(extreme_ar)/total*100:.1f}% |")
    lines.append(f"| Low variance (blank/solid) | {len(low_var)} | {len(low_var)/total*100:.1f}% |")
    lines.append(f"| Exact duplicates (MD5) | {sum(len(v) - 1 for v in duplicates['exact'].values())} dupes in {len(duplicates['exact'])} groups | — |")
    lines.append(f"| Cross-source perceptual dupes | {len(duplicates['perceptual_cross_source'])} groups | — |")
    lines.append("")

    # Per-source breakdown
    lines.append("## 2. Per-Dataset Breakdown")
    lines.append("")
    lines.append("| Dataset | Images | Issues | Resolution (median) | File Size (median) |")
    lines.append("|---------|--------|--------|--------------------|--------------------|")
    for src in sorted(source_stats):
        s = source_stats[src]
        med_w = int(np.median(s["widths"])) if s["widths"] else "—"
        med_h = int(np.median(s["heights"])) if s["heights"] else "—"
        med_sz = f'{np.median(s["sizes_kb"]):.0f} KB' if s["sizes_kb"] else "—"
        res = f"{med_w}x{med_h}" if s["widths"] else "—"
        lines.append(f"| {src} | {s['total']} | {s['issues']} | {res} | {med_sz} |")
    lines.append("")

    # Resolution distribution per source
    lines.append("## 3. Resolution Distribution")
    lines.append("")
    for src in sorted(source_stats):
        s = source_stats[src]
        if not s["widths"]:
            continue
        ws, hs = np.array(s["widths"]), np.array(s["heights"])
        lines.append(f"### {src}")
        lines.append(f"- Width:  min={ws.min()}, median={int(np.median(ws))}, max={ws.max()}")
        lines.append(f"- Height: min={hs.min()}, median={int(np.median(hs))}, max={hs.max()}")
        lines.append(f"- Unique resolutions: {len(set(zip(ws, hs)))}")
        # Top 5 resolutions
        res_counter = Counter(zip(ws.tolist(), hs.tolist()))
        lines.append(f"- Top resolutions: {', '.join(f'{w}x{h} ({c})' for (w,h), c in res_counter.most_common(5))}")
        lines.append("")

    # Label distribution
    lines.append("## 4. Label Distribution")
    lines.append("")
    lines.append("| Label | Count | % |")
    lines.append("|-------|-------|---|")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {label} | {count} | {count/total*100:.1f}% |")
    lines.append("")

    # Class imbalance ratio
    max_count = max(label_counts.values())
    min_count = min(v for k, v in label_counts.items() if k != "__excluded__")
    lines.append(f"**Imbalance ratio**: {max_count/min_count:.1f}:1 (largest class / smallest class)")
    lines.append("")

    # Flagged images detail
    lines.append("## 5. Flagged Images")
    lines.append("")
    if corrupt:
        lines.append(f"### Corrupt ({len(corrupt)})")
        lines.append("")
        for r in corrupt[:20]:
            lines.append(f"- `{Path(r['path']).name}` ({r['source']}): {r['issues']}")
        if len(corrupt) > 20:
            lines.append(f"- ... and {len(corrupt) - 20} more")
        lines.append("")

    if tiny:
        lines.append(f"### Tiny ({len(tiny)})")
        lines.append("")
        for r in tiny[:20]:
            lines.append(f"- `{Path(r['path']).name}` ({r['source']}): {r['width']}x{r['height']}")
        if len(tiny) > 20:
            lines.append(f"- ... and {len(tiny) - 20} more")
        lines.append("")

    if low_var:
        lines.append(f"### Low Variance ({len(low_var)})")
        lines.append("")
        for r in low_var[:20]:
            lines.append(f"- `{Path(r['path']).name}` ({r['source']}): {[i for i in r['issues'] if 'low_variance' in i]}")
        if len(low_var) > 20:
            lines.append(f"- ... and {len(low_var) - 20} more")
        lines.append("")

    if extreme_ar:
        lines.append(f"### Extreme Aspect Ratio ({len(extreme_ar)})")
        lines.append("")
        for r in extreme_ar[:20]:
            lines.append(f"- `{Path(r['path']).name}` ({r['source']}): {r['width']}x{r['height']}")
        if len(extreme_ar) > 20:
            lines.append(f"- ... and {len(extreme_ar) - 20} more")
        lines.append("")

    # Duplicates detail
    if duplicates["exact"]:
        lines.append(f"### Exact Duplicates ({len(duplicates['exact'])} groups)")
        lines.append("")
        for i, (md5, paths) in enumerate(list(duplicates["exact"].items())[:10]):
            lines.append(f"- Group {i+1} ({len(paths)} files): {', '.join(Path(p).name for p in paths[:5])}")
        if len(duplicates["exact"]) > 10:
            lines.append(f"- ... and {len(duplicates['exact']) - 10} more groups")
        lines.append("")

    if duplicates["perceptual_cross_source"]:
        lines.append(f"### Cross-Source Perceptual Duplicates ({len(duplicates['perceptual_cross_source'])} groups)")
        lines.append("")
        for i, (phash, entries) in enumerate(list(duplicates["perceptual_cross_source"].items())[:10]):
            desc = ", ".join(f"{Path(p).name} ({s})" for p, s in entries[:5])
            lines.append(f"- Group {i+1}: {desc}")
        if len(duplicates["perceptual_cross_source"]) > 10:
            lines.append(f"- ... and {len(duplicates['perceptual_cross_source']) - 10} more groups")
        lines.append("")

    # No issues found for a category
    all_issue_types = [corrupt, tiny, extreme_ar, low_var]
    if not any(all_issue_types) and not duplicates["exact"] and not duplicates["perceptual_cross_source"]:
        lines.append("No flagged images found. All images passed all checks.")
        lines.append("")

    # Recommendations
    lines.append("## 6. Recommendations")
    lines.append("")
    if corrupt:
        lines.append(f"- **Remove {len(corrupt)} corrupt images** — they will crash the training pipeline")
    if tiny:
        lines.append(f"- **Remove {len(tiny)} tiny images** — below minimum useful resolution for 224x224 training")
    if low_var:
        lines.append(f"- **Review {len(low_var)} low-variance images** — likely blank/solid, add noise or remove")
    if extreme_ar:
        lines.append(f"- **Review {len(extreme_ar)} extreme aspect ratio images** — may cause distortion after resize")
    if duplicates["exact"]:
        n_dupes = sum(len(v) - 1 for v in duplicates["exact"].values())
        lines.append(f"- **Deduplicate {n_dupes} exact duplicates** — keeping one copy per group")
    if duplicates["perceptual_cross_source"]:
        lines.append(f"- **Review {len(duplicates['perceptual_cross_source'])} cross-source perceptual duplicates** — same image may appear in multiple datasets")

    max_label = max(label_counts, key=label_counts.get)
    min_label = min((k for k in label_counts if k != "__excluded__"), key=label_counts.get)
    if max_count / min_count > 3:
        lines.append(f"- **Address class imbalance** — `{max_label}` has {max_count} imgs vs `{min_label}` with {min_count}. Consider oversampling minority or weighted loss")

    lines.append(f"- **Standardize resolution** — datasets have mixed resolutions; `RandomResizedCrop(224)` in augmentation pipeline handles this at training time")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Collecting images...", flush=True)
    images = collect_all_images()
    print(f"Found {len(images)} images across datasets.", flush=True)

    print("Running checks (this may take a few minutes)...", flush=True)
    results = []
    for i, row in enumerate(images):
        results.append(check_image(row))
        if (i + 1) % 500 == 0:
            print(f"  Checked {i+1}/{len(images)}", flush=True)

    print("Finding duplicates...", flush=True)
    duplicates = find_duplicates(results)

    print("Generating report...", flush=True)
    report = generate_report(results, duplicates)

    out_path = Path(__file__).resolve().parent.parent / "data-cleaning-report.md"
    out_path.write_text(report)
    print(f"Report written to {out_path}")

    # Also dump raw results as JSON for further analysis
    json_path = Path(__file__).resolve().parent.parent / "data" / "cleaning_results.json"
    serializable = []
    for r in results:
        sr = {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
        serializable.append(sr)
    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"Raw results saved to {json_path}")


if __name__ == "__main__":
    main()
