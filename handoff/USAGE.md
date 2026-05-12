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

## 3. Reproduce the headline numbers (98 images, 9 categories)

Three commands, in order. Total wall time ≈ 60–90 min on M-series.

```bash
# Step 1 — CLIP screening over all (image, category) cells
uv run python scripts/run_clip_compliant.py
# writes: checkpoints/client_clip_scores.npy

# Step 2 — Gemma binary verification on the top-K CLIP candidates
uv run python scripts/run_gemma_binary_compliant.py
# writes: checkpoints/client_gemma_yes_aligned.npy  (and per-cell rationales)

# Step 3 — Aggregate metrics against human consensus
uv run python scripts/analyze_binary_results.py
# prints: per-class AUC / recall / precision, cascade F1 at k=1..9
```

Expected headline numbers (matches CLIENT_REPORT §3):
- Catches real violations on **87 % of 98 known-violation images** (cascade k=5)
- **0 false alarms** across 20 clean houses × 9 categories = 180 cells
- **85 % precision** when the system says "yes" (cascade k=5, micro avg)

## 4. Compute inter-annotator agreement (SOP Step 2)

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

## 5. Add a NEW violation category (SOP Step 4–5)

The system is config-driven: adding a category means editing **one file** (`src/client_data.py`) in five small places, putting images in a folder, and re-running the three scripts in §3.

### 5.1 Drop the photos in

```
data/client-data/<your folder name>/img_001.jpg, img_002.jpg, ...
```

The folder name should include the municipal code in `§ NNN.N` form (e.g. `§ 304.X - Cracked Driveway`). It is the substring match key into `FOLDER_KEYWORD_MAP` below.

### 5.2 Edit `src/client_data.py` — five places

| # | Constant | What to add |
|---|---|---|
| 1 | `CLIENT_CLASSES` | snake_case class id, e.g. `"cracked_driveway"` |
| 2 | `FOLDER_KEYWORD_MAP` | `"§ 304.X": "cracked_driveway"` — links folder name → class id |
| 3 | `CLIENT_DISPLAY` | short two-line label for plots: `"cracked_driveway": "Cracked\nDriveway"` |
| 4 | `CLIENT_CLIP_PROMPTS` | 3 natural-language paraphrases for CLIP, e.g. `"a photo of a driveway with large cracks"` |
| 5 | `CLIENT_DESCRIPTIONS` | One sentence ≤ 25 words, **visual not legal** (see SOP rules in CLIENT_REPORT §9 Step 4) |

### 5.3 Re-run

```bash
uv run python scripts/run_clip_compliant.py
uv run python scripts/run_gemma_binary_compliant.py
uv run python scripts/analyze_binary_results.py
```

The new category appears in the per-class metrics table, which prints **AUC, recall, precision, and FPR** per class. Apply the SOP Step-6 thresholds directly from that row:

- **F1 ≥ 0.80 and FPR ≤ 5 %** → full automation tier
- **F1 0.50 – 0.80** → AI-assisted tier
- otherwise revise the prompt or fix the SOP and retest

FPR is reported as a fraction (e.g. `0.047` = 4.7 %). The 5 % gate corresponds to `fpr ≤ 0.05`.

## 6. Output formats

| Artifact | Path | Format |
|---|---|---|
| CLIP per-cell scores | `checkpoints/client_clip_scores.npy` | float32, shape (N_images, N_classes) |
| Gemma binary answers | `checkpoints/client_gemma_yes_aligned.npy` | int8, shape (N_images, N_classes), -1 = not run, 0/1 = no/yes |
| Gemma rationales | `checkpoints/client_gemma_stream.jsonl` | one JSON record per (image, class) call with `answer`, `confidence`, `rationale` |
| Human consensus | `data/client-data/labeling/human/consensus_binary.csv` | (image_id, class₁..class₉) ∈ {0, 1}, where `1` = ≥2/3 raters said yes |

## 7. Repository map

```
src/
  client_data.py        # all category configuration (edit here to add a class)
  clip_baseline.py      # CLIP feature extraction + scoring
  binary_prompt.py      # Gemma binary prompt template + parsing
  zeroshot.py           # CLIP zero-shot inference loop
scripts/
  run_clip_compliant.py         # Stage 1: CLIP screening
  run_gemma_binary_compliant.py # Stage 2: Gemma verification
  analyze_binary_results.py     # Final metrics & cascade evaluation
  compute_iaa.py                # Inter-annotator agreement (SOP Step 2)
data/client-data/       # 98 violation images + 3-rater labels
data/synthetic/         # 20 synthetic clean houses (FPR baseline)
checkpoints/            # cached model outputs (regenerable)
handoff/                # CLIENT_REPORT, USAGE (this file), slide PDF
```

## 8. Known limitations & next steps

See CLIENT_REPORT §10 for the full list. Short version:
- The 0 % false-alarm rate is validated on 20 *synthetic* clean houses. Validating against 50–100 real city clean-house photos is the highest-value next step on the client side.
- The pipeline scores one photo at a time. Multi-angle aggregation per address is not implemented.
- No web UI — current entry points are CLI scripts. A REST API + upload form is a reasonable next engineering increment (CLIENT_REPORT §10.2 item 5).

## 9. Contact

Project lead: Chihao Li · UMD INST623 Spring 2026 cohort.
For issues: open a GitHub issue on the repo above, or contact the course liaison.
