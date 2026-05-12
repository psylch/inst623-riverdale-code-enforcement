# Riverdale Code Enforcement — Code Usage Guide

A practical handoff for the Riverdale Park technical staff who will run, extend, or hand off this pipeline. Pair this document with `CLIENT_REPORT.md` (results & recommendations) and the slide deck.

GitHub: <https://github.com/psylch/inst623-riverdale-code-enforcement>

---

## 1. Requirements

- **Hardware:** Apple Silicon Mac (M-series), 16 GB+ RAM. The Gemma model runs through Apple's MLX framework and is not portable to Intel/Windows/Linux without code changes.
- **Software:** Python 3.12, [`uv`](https://github.com/astral-sh/uv) for dependency management, ~10 GB free disk for model weights and image caches.
- **First run:** Gemma 4 E4B-IT (4-bit) is downloaded on first invocation (~3 GB). CLIP ViT-B/32 is downloaded on first invocation (~600 MB).

## 2. Install

```bash
git clone https://github.com/psylch/inst623-riverdale-code-enforcement.git
cd inst623-riverdale-code-enforcement
uv sync                        # installs all dependencies into .venv
```

Verify the install:
```bash
uv run python -c "import mlx_vlm, open_clip; print('ok')"
```

## 3. Get the data

The inspector photos and the synthetic clean baseline are **not** in this repo (98 real-property photos kept out for privacy; 20 generated images for size). They live in a shared Google Drive folder:

📂 **Drive bundle**: <https://drive.google.com/open?id=1CBsvpWBzEDzC-kErvzHSxYFh1sldq-Yo>

The bundle has two top-level folders that map 1-to-1 onto your local `data/` tree:

| Drive folder | Extract into | Lives in repo? |
|---|---|---|
| `Riverdale-Handoff-Data/client-data/` (9 violation-code subfolders, 98 photos) | `data/client-data/` | ❌ photos kept out (privacy) |
| `Riverdale-Handoff-Data/synthetic/code-enforcement-compliant/` (20 PNG + manifest.md) | `data/synthetic/code-enforcement-compliant/` | ❌ regenerable from prompts |
| — | `data/client-data/manifest.csv` (image_id ↔ filename) | ✅ already in repo |
| — | `data/client-data/labeling/human/merged_3labelers.csv` | ✅ already in repo |
| — | `data/client-data/labeling/human/consensus_binary.csv` | ✅ already in repo |

Final directory layout after extracting the Drive bundle:

```
data/
├── client-data/
│   ├── manifest.csv                    # in repo
│   ├── Boarded Windows (§ 304.13 - 108.2)/   # from Drive
│   ├── Broken Windows (§ 304.13.1)/          # from Drive
│   ├── ...                                   # 12 more violation folders from Drive
│   └── labeling/human/                       # in repo
│       ├── merged_3labelers.csv
│       └── consensus_binary.csv
└── synthetic/
    └── code-enforcement-compliant/
        ├── compliant_01.png … compliant_20.png   # from Drive
        └── manifest.md                            # from Drive (per-image prompts)
```

After the bundle is in place, `scripts/run_clip_separability.py` and the rest of §4 will find every file they need.

## 4. Reproduce the headline numbers

Five commands, in order. The pipeline runs on **two image sets**: 98 violation photos (for recall/precision) and 20 synthetic clean houses (for false-alarm rate). Total wall time ≈ 60–90 min on M-series.

```bash
# --- 98 violation photos: produces recall, precision, cascade F1 ---

# Step 1 — CLIP per-class similarity on all 98 images
uv run python scripts/run_clip_separability.py
# writes: checkpoints/client_clip_similarity.npz

# Step 2 — Gemma binary verification: 98 images × 9 categories = 882 calls (~45 min, resumable)
uv run python scripts/run_gemma_binary.py
# writes: checkpoints/client_gemma4_binary_stream.jsonl  (per-call audit log)
#         checkpoints/client_gemma4_binary.npz           (final score matrix)

# --- 20 synthetic clean houses: produces false-alarm rate ---

# Step 3 — CLIP similarity on the clean baseline
uv run python scripts/run_clip_compliant.py
# writes: checkpoints/compliant_clip_sims.npz

# Step 4 — Gemma binary verification on the clean baseline (20 × 9 = 180 calls, ~10 min)
uv run python scripts/run_gemma_binary_compliant.py
# writes: checkpoints/compliant_gemma4_binary_stream.jsonl

# --- Aggregate ---

# Step 5 — Compute all metrics (per-class AUC/recall/precision/FPR, cascade F1@k, clean-house FPR)
uv run python scripts/analyze_binary_results.py

# Step 6 — Render the κ-tier performance table (CLIENT_REPORT §5.1)
uv run python scripts/report_by_tier.py
```

Expected headline numbers (matches CLIENT_REPORT §3):
- Catches real violations on **87 % of 98 known-violation images** (cascade k=5)
- **0 false alarms** across 20 clean houses × 9 categories = 180 cells
- **85 % precision** when the system says "yes" (cascade k=5, micro avg)

Step 6 produces the tier-grouped table in CLIENT_REPORT §5.1 — five rows (Almost perfect / Substantial / Moderate / Fair / Slight) with macro-averaged P/R/F1 and the deployment recommendation per tier.

## 5. Compute inter-annotator agreement (SOP Step 2)

When new labels are collected, the κ-gate (CLIENT_REPORT §9, Step 2–3) determines which deployment tier a category qualifies for.

```bash
uv run python scripts/compute_iaa.py
# Reads: data/client-data/labeling/human/merged_3labelers.csv
# Prints: per-category Cohen's κ (pairwise + mean), Landis & Koch tier,
#         and the deployment recommendation that follows from the tier.
```

To run on a different rater set, point at any CSV with the same wide layout (`image_id`, then `<RaterName>__<category>` columns valued -1 / 0 / 1, where -1 means *unsure*):

```bash
uv run python scripts/compute_iaa.py --csv path/to/new_merged.csv
```

Edit `RATERS` at the top of `scripts/compute_iaa.py` if the rater names differ.

## 6. Add a NEW violation category (SOP Step 4–5)

The system is config-driven: adding a category means editing **one file** (`src/client_data.py`) in five small places, putting images in a folder, and re-running the pipeline in §4.

### 6.1 Drop the photos in

```
data/client-data/<your folder name>/img_001.jpg, img_002.jpg, ...
```

The folder name should include the municipal code in `§ NNN.N` form (e.g. `§ 304.X - Cracked Driveway`). It is the substring match key into `FOLDER_KEYWORD_MAP` below.

### 6.2 Edit `src/client_data.py` — five places

| # | Constant | What to add |
|---|---|---|
| 1 | `CLIENT_CLASSES` | snake_case class id, e.g. `"cracked_driveway"` |
| 2 | `FOLDER_KEYWORD_MAP` | `"§ 304.X": "cracked_driveway"` — links folder name → class id |
| 3 | `CLIENT_DISPLAY` | short two-line label for plots: `"cracked_driveway": "Cracked\nDriveway"` |
| 4 | `CLIENT_CLIP_PROMPTS` | 3 natural-language paraphrases for CLIP, e.g. `"a photo of a driveway with large cracks"` |
| 5 | `CLIENT_DESCRIPTIONS` | One sentence ≤ 25 words, **visual not legal** (see SOP rules in CLIENT_REPORT §9 Step 4) |

### 6.3 Re-run

Re-run the violation-image half of the pipeline (the clean-house baseline only needs re-running if you want to verify your new class produces zero false alarms on it too — usually optional):

```bash
uv run python scripts/run_clip_separability.py     # picks up the new CLIP prompts
uv run python scripts/run_gemma_binary.py          # 98 + new images × (9+1) categories
uv run python scripts/analyze_binary_results.py    # per-class metrics + tier-grouped table
```

The new category appears in the per-class metrics table, which prints **AUC, recall, precision, and FPR** per class. Apply the SOP Step-6 thresholds directly from that row:

- **F1 ≥ 0.80 and FPR ≤ 5 %** → full automation tier
- **F1 0.50 – 0.80** → AI-assisted tier
- otherwise revise the prompt or fix the SOP and retest

FPR is reported as a fraction (e.g. `0.047` = 4.7 %). The 5 % gate corresponds to `fpr ≤ 0.05`.

## 7. Output formats

| Artifact | Path | Format |
|---|---|---|
| CLIP similarity (98 violation images) | `checkpoints/client_clip_similarity.npz` | npz with `similarity` key, float32 (98, 9) cosine sim |
| Gemma binary stream (98) | `checkpoints/client_gemma4_binary_stream.jsonl` | one JSON record per (image, class) call: `image_idx`, `class_idx`, `answer` ∈ {yes, no, unclear}, `confidence` 0–100, `rationale` |
| Gemma binary aggregate (98) | `checkpoints/client_gemma4_binary.npz` | npz with `scores`, `answers`, `parse_ok` (98, 9) |
| CLIP similarity (20 clean houses) | `checkpoints/compliant_clip_sims.npz` | same shape as above, (20, 9) |
| Gemma binary stream (20 clean) | `checkpoints/compliant_gemma4_binary_stream.jsonl` | same format as above |
| Per-class summary | `checkpoints/gemma4_binary_summary.json` | list of dicts: class, AUC, recall@0.5, precision@0.5, FPR@0.5, TP/FP/FN/TN counts |
| Clean-house summary | `checkpoints/compliant_summary.json` | total/per-class `n_yes` (false positives) out of 180 cells |
| Cascade summary | `checkpoints/cascade_summary.json` | per-k (3, 5, 9) sample-F1, Jaccard, recall-any |
| Human consensus | `data/client-data/labeling/human/consensus_binary.csv` | image_id × 9 classes ∈ {0, 1}, where `1` = ≥ 2/3 raters said yes |

## 8. Repository map

```
src/
  client_data.py             # category configuration — edit here to add a class
  zeroshot.py                # CLIP + Gemma zero-shot inference loops
  binary_prompt.py           # Gemma binary prompt template + parsing
  cache.py                   # checkpoint/resume helpers
scripts/
  run_clip_separability.py       # CLIP similarity on the 98 violation images
  run_gemma_binary.py            # Gemma binary verification on the 98 violation images
  run_clip_compliant.py          # CLIP similarity on the 20 clean baseline images
  run_gemma_binary_compliant.py  # Gemma binary verification on the 20 clean images
  analyze_binary_results.py      # per-class metrics + clean-house FPR + cascade F1@k
  report_by_tier.py              # κ-tier grouped P/R/F1 table (CLIENT_REPORT §5.1)
  compute_iaa.py                 # Cohen's κ across raters (SOP Step 2)
data/client-data/       # photos from Drive + manifest.csv + labeling/human/*.csv (in repo)
data/synthetic/         # 20 synthetic clean houses (from Drive)
checkpoints/            # cached model outputs (regenerable, gitignored)
handoff/                # CLIENT_REPORT.md, USAGE.md (this file), SYNTHETIC_NEGATIVES.md
```

## 9. Known limitations & next steps

See CLIENT_REPORT §10 for the full list. Short version:
- The 0 % false-alarm rate is validated on 20 *synthetic* clean houses (see [`SYNTHETIC_NEGATIVES.md`](SYNTHETIC_NEGATIVES.md) for how they were built and verified). Validating against 50–100 real city clean-house photos is the highest-value next step on the client side.
- The pipeline scores one photo at a time. Multi-angle aggregation per address is not implemented.
- No web UI — current entry points are CLI scripts. A REST API + upload form is a reasonable next engineering increment (CLIENT_REPORT §10.2 item 5).

## 10. Contact

Project lead: Chihao Li · UMD INST623 Spring 2026 cohort.
For issues: open a GitHub issue on the repo above, or contact the course liaison.
