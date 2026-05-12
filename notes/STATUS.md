# Riverdale Park AI Code Enforcement — Status & Exploration Context

*Last updated: 2026-05-04 (after end-to-end Stage 1 + Stage 2 evaluation)*

This document captures the full exploration context for the project as of today, so a teammate picking it up can understand **what we ran, why, what the numbers mean, and what's left to polish**. Hand this to the PPT teammate alongside `CLIENT_REPORT.md` and the `ppt/` deck.

---

## 1. The data and the people

### 1.1 Datasets

| Set | Files | Folder |
|---|---|---|
| Known-violation images | 98 | `data/client-data/{9 violation folders}/` |
| Synthetic compliant houses | 20 | `data/synthetic/code-enforcement-compliant/` |
| Total evaluation surface | 118 images × 9 categories = 1,062 cells | — |

The 98 violation images include 12 image_ids whose physical files are duplicated across folders (multi-violation images placed in multiple primary folders). Unique filenames = 86; image_id rows = 98. **Don't get confused by this** — the evaluation goes off image_id, and Gemma binary outputs are broadcast from filename → image_id (see `checkpoints/client_gemma_yes_aligned.npy`).

The 20 synthetic clean houses were generated via `baoyu-image-gen` (or equivalent text-to-image) using prompts in `data/synthetic/code-enforcement-compliant/manifest.md`. Style: documentary daytime exterior, mid-Atlantic suburban, all 9 violation classes explicitly absent. Verified by all three human annotators as all-no on every category.

### 1.2 The 9 violation categories

```
boarded_windows
broken_windows
damaged_roof_shingles
deteriorating_chimney
graffiti
inoperable_vehicle
junk_trash_accumulation
overgrown_vegetation
peeling_paint
```

These map to Riverdale Park municipal code sections — see folder names in `data/client-data/` for full code citations (§ 304.x etc.).

### 1.3 The labelers

- **labeler_1** = Fechi
- **labeler_2** = Niping (NOT Chihao — Chihao is the project lead, didn't label)
- **labeler_3** = Jake Sheehi

Source files in `data/client-data/labeling/human/`:
- `(Fechi) labeler_1.xlsx`
- `labeler_2.xlsx`
- `labeler_3 - Jake Sheehi.xlsx`

Each has 98 rows, 9 category columns, value vocab is `Y / N / ?`. Fechi has 1 blank cell at `img_029 × inoperable_vehicle` — treated as `?` in pipeline.

---

## 2. Methodology decisions made today

### 2.1 Unsure → no (binarization)

Earlier IAA analysis was 3-class (Y / N / ?). For all client-facing metrics we **collapse `?` → `no`** so that:

- Human and Gemma label spaces match (Gemma is binary y/n only)
- Metrics are conventional binary precision/recall
- The framing is conservative ("we err on the side of not flagging")

Caveat to note in any methodology paragraph: roughly 7% of cells were originally `?`. Treating them as `no` slightly biases the system toward fewer false alarms, which is the right direction for a code-enforcement screening tool.

### 2.2 Human consensus = ≥2/3 yes

Ground truth for Stage 2 evaluation:
```python
votes = Fechi[c] + Niping[c] + Jake[c]   # each 0 or 1 after ?→0
positive = (votes >= 2)
```

Resulting consensus has 276 positive / 606 negative cells across the 98 images. With the 20 clean houses appended (all zeros), the unified evaluation set has **276 positive / 786 negative** across 118 × 9 = 1,062 cells.

Saved to `data/client-data/labeling/human/consensus_binary.csv` (98 rows, 9 columns) — the 20 compliant rows are appended in-memory at evaluation time, not saved to this CSV.

### 2.3 Tier bucketing — by κ, not by F1

The professor's last-meeting feedback explicitly asked for stratification by **human inter-rater agreement** (Cohen's κ tier per Landis & Koch), not by AI F1. We had initially bucketed by F1 — wrong axis. Corrected.

Tiers used in client-facing slides:

| Tier | κ range | Categories (binary κ) |
|---|---|---|
| Almost perfect | ≥ 0.81 | inoperable_vehicle (0.928), graffiti (0.888) |
| Substantial | 0.61–0.80 | overgrown_vegetation (0.677), damaged_roof_shingles (0.640) |
| Moderate | 0.41–0.60 | junk_trash (0.601), boarded_windows (0.564), broken_windows (0.541) |
| Fair | 0.21–0.40 | peeling_paint (0.248) |
| Slight | < 0.21 | deteriorating_chimney (0.150) |

`junk_trash` is right on the moderate/substantial boundary (0.601 vs threshold 0.61) — we put it in moderate per Landis & Koch's strict cutoff.

### 2.4 The Stage 1 / Stage 2 framing

To make the story client-friendly, we split evaluation into:

- **Stage 1 (Screening)** — "does the system catch known problems and avoid false alarms on clean houses?"
  - Recall against folder-name primary category (98 images)
  - FPR on 20 clean houses

- **Stage 2 (Detail Review)** — "for each individual category, how accurate is the per-label decision?"
  - Per-category precision / recall / F1 vs human consensus
  - Stratified by human-agreement tier (the professor's ask)

This split also dictates how human labels are used: not for Stage 1 (folder name is the GT there), only for Stage 2.

---

## 3. The cascade architecture

### 3.1 Production pipeline (cascade k=5)

```
Photo
   ↓
[Stage 1] CLIP zero-shot (ViT-B/32, laion2b)
   ↓ outputs softmax over 9 classes per image
   ↓ keep top-5 candidates
   ↓
[Stage 2] Gemma 4 binary verifier (mlx-community/gemma-4-e4b-it-4bit)
   ↓ for each top-5 candidate, ask "does this image show {description}? yes/no, confidence, rationale"
   ↓
For categories NOT in top-5, output is forced "no"
   ↓
Final: 9-vector of binary flags per image
```

CLIP prompts: `src/client_data.py::CLIENT_CLIP_PROMPTS` (3 paraphrases per class, averaged).
Gemma prompt template: `src/binary_prompt.py::BINARY_PROMPT_TEMPLATE` (single visual question, JSON output, includes confidence 0-100 and rationale).
Gemma class descriptions: `src/client_data.py::CLIENT_DESCRIPTIONS`.

### 3.2 Why cascade (not just Gemma alone)

Gemma 9-way binary actually has the **highest F1** (0.76 micro) — better than cascade k=5 (0.72). But:

- **Gemma 9-way means 9 binary calls per image** = expensive (882 calls for 98 images)
- **Cascade with CLIP top-5 reduces this to ~5 calls per image** = ~half the inference cost
- **Cascade has structurally lower FPR** — the 4 non-top-k categories are forced no, so non-target false alarms are impossible

We pitched cascade k=5 as the production config to the client because it's the inference-efficient configuration with strong specificity. **For the report / pre, also be ready to discuss Gemma-9-way as the high-recall alternative** — both are valid, depending on whether the city wants to optimize for cost or recall.

`k=3` is the high-precision mode (micro-P 0.92) at the cost of recall (0.52). Useful framing: "if you only want flags you can publish without inspector review, use k=3."

---

## 4. What was actually run today

### 4.1 Inputs already on disk (pre-existing, didn't re-run)

- `checkpoints/client_gemma4_binary.npz` — Gemma binary on 98 images, computed Apr 13. Cache hit on rerun (parse 100%).
- `checkpoints/client_gemma4_binary_stream.jsonl` — full per-call audit log (882 lines, 86 unique filenames × 9).
- `checkpoints/client_clip_similarity.npz` — CLIP softmax matrix on 98 images (86 unique × 9).

### 4.2 New runs (today, 2026-05-04)

1. **`scripts/run_gemma_binary_compliant.py`** — Gemma binary on 20 synthetic clean houses (180 calls, ~18 min). Output: `checkpoints/compliant_gemma4_binary.npz` and `compliant_gemma4_binary_stream.jsonl`. Result: **0 / 180 yes**, parse 100%, mean confidence 96.

2. **`scripts/run_clip_compliant.py`** — CLIP similarity on the same 20 clean houses (uses MPS on M4). Output: `checkpoints/compliant_clip_sims.npz`. Used to compute cascade FPR (0% by construction since pure Gemma FPR is 0%).

3. **In-memory analysis** (no script saved, but reproducible from above + `consensus_binary.csv`):
   - Stage 1 primary recall: cascade k=5 = **86.7%** (85/98 hit folder primary)
   - Per-class precision/recall/F1 on 98 + 20 unified set
   - Macro/micro aggregates per κ tier
   - Specificity = 96.1% on cascade k=5 (755/786 negatives correctly judged no — the 31 FPs all come from cross-category mistakes within the 98 violation images, not from clean houses)

### 4.3 Files written today

- `data/synthetic/code-enforcement-compliant/compliant_01.png … compliant_20.png` (user-generated)
- `data/synthetic/code-enforcement-compliant/manifest.md` (user-written)
- `data/client-data/labeling/human/merged_3labelers.csv` — 3 raters × 9 cats per row, ?→0 collapsed and Niping rename applied
- `data/client-data/labeling/human/consensus_binary.csv` — ≥2/3 binary consensus, 98 rows
- `checkpoints/compliant_gemma4_binary.npz`
- `checkpoints/compliant_gemma4_binary_stream.jsonl`
- `checkpoints/compliant_clip_sims.npz`
- `checkpoints/client_gemma_yes_aligned.npy` — Gemma 98 × 9 in consensus image_id order (filename → image_id broadcast)
- `scripts/run_gemma_binary_compliant.py`
- `scripts/run_clip_compliant.py`
- `ppt/index.html` and `ppt/assets/motion.min.js` (the deck)
- `handoff/STATUS.md` (this file)
- `handoff/CLIENT_REPORT.md` (sibling file)

---

## 5. Headline numbers (single source of truth)

All from cascade k=5 on the unified 118-image evaluation, unless noted.

| Metric | Value | Notes |
|---|---|---|
| Stage 1 primary recall (cascade k=5) | 86.7% | 85/98 hit folder primary |
| Stage 1 primary recall (Gemma 9-way) | 88.8% | 87/98 |
| Stage 1 primary recall (cascade k=3) | 82.7% | 81/98 |
| Stage 1 FPR on 20 clean houses | 0.0% | 0/180 across all 9 categories |
| Stage 2 micro-precision | 0.85 | "when the system says yes, 85% are right" |
| Stage 2 micro-recall | 0.63 | |
| Stage 2 micro-F1 | 0.72 | |
| Stage 2 macro-F1 | 0.64 | |
| Specificity (TN rate) | 96.1% | 755/786 negative cells correctly judged no |
| Mean Gemma confidence on clean houses | 96 / 100 | not borderline |

### 5.1 Per-category numbers (cascade k=5, unified 118 images)

| Category | κ | Tier | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| inoperable_vehicle | 0.928 | almost perfect | 0.84 | 0.88 | 0.86 | 21 | 4 | 3 |
| graffiti | 0.888 | almost perfect | 0.88 | 0.88 | 0.88 | 15 | 2 | 2 |
| overgrown_vegetation | 0.677 | substantial | 0.95 | 0.75 | 0.84 | 42 | 2 | 14 |
| damaged_roof_shingles | 0.640 | substantial | 0.89 | 0.38 | 0.53 | 8 | 1 | 13 |
| junk_trash_accumulation | 0.601 | moderate | 0.64 | 0.72 | 0.68 | 18 | 10 | 7 |
| boarded_windows | 0.564 | moderate | 0.64 | 0.56 | 0.60 | 9 | 5 | 7 |
| broken_windows | 0.541 | moderate | **1.00** | 0.22 | 0.36 | 8 | 0 | 29 |
| peeling_paint | 0.248 | fair | 0.93 | 0.70 | 0.80 | 51 | 4 | 22 |
| deteriorating_chimney | 0.150 | slight | 0.25 | 0.14 | 0.18 | 1 | 3 | 6 |

### 5.2 Macro by tier

| Tier | macro-P | macro-R | macro-F1 |
|---|---|---|---|
| Almost perfect | 0.86 | 0.88 | 0.87 |
| Substantial | 0.92 | 0.57 | 0.69 |
| Moderate | 0.76 | 0.50 | 0.55 |
| Fair | 0.93 | 0.70 | 0.80 |
| Slight | 0.25 | 0.14 | 0.18 |

### 5.3 Anomalies / data points worth narrating

- **Peeling paint** has high F1 (0.80) but low κ (0.25). This is **prevalence-driven** — 73/98 positive cells = 75% prevalence, so even with noisy human labels, a model that says "yes" often gets credit. **Don't oversell peeling paint.** Treat it as fair-tier in slides.

- **Broken windows** has perfect precision (1.00) but tiny recall (0.22). The system is very conservative on this category. Useful framing: "every broken-windows flag we raise is correct — but the system misses many cases, so inspectors should still walk the property if context suggests damage."

- **Deteriorating chimney** is the strongest evidence for the "label-definition problem" claim. Both human IAA (0.15) and AI F1 (0.18) are slight-tier. Use this as the bridge to the "we need a city SOP" Future Work item.

- **Stage 1 FPR is suspiciously good (0%)**. The 20 clean houses are AI-generated, idealized, and probably easier than real-world clean street views. Don't claim 0% as the production number; claim it as the synthetic-baseline result with a clear ask for real clean-house data.

---

## 6. Comparison to alternative configurations

| Config | macro-F1 | micro-P | micro-R | micro-F1 | Specificity | Notes |
|---|---|---|---|---|---|---|
| **Cascade k=5 (production)** | **0.64** | **0.85** | **0.63** | **0.72** | **96.1%** | recommended |
| Cascade k=3 (high precision) | 0.61 | 0.92 | 0.52 | 0.67 | 98.3% | for "publishable without review" |
| Gemma 9-way (no CLIP filter) | 0.67 | 0.82 | 0.71 | 0.76 | 94.5% | best F1, 9× inference cost |
| CLIP top-3 only | 0.58 | 0.61 | 0.65 | 0.62 | — | proves CLIP alone over-flags |
| CLIP top-5 only | 0.52 | 0.46 | 0.82 | 0.59 | — | proves CLIP alone over-flags |

The CLIP-only rows are useful for the "CLIP for recall, Gemma for precision" architecture argument — they show that without Gemma's verification, CLIP top-k would flag 33-56% of categories on every image (including clean houses), which is unusable.

---

## 7. The methodology insight (slide-worthy)

**AI accuracy tracks human inter-rater agreement, category by category.**

Sorting by κ produces a near-monotonic F1 ranking (with peeling_paint as the prevalence-driven exception). This is the central narrative:

- When humans agree → AI works
- When humans disagree → AI struggles
- Therefore the bottleneck is **label definition**, not algorithmic capability

This is what justifies the κ-tiered reporting structure and the "we need an SOP from the city" Future Work item. It's also a quietly profound point: it shifts the conversation from "can AI do this?" to "have we defined this task well enough for any system — human or AI — to do it?"

---

## 8. What the deck (`ppt/index.html`) contains

12 slides, English, magazine-style horizontal swipe deck. Built with the `guizang-ppt-skill` template (single HTML file, WebGL background, Motion One animations, keyboard / scroll / touch / dot navigation, ESC for index view).

Theme: 🖋 Ink Classic (pure ink + warm cream).

| # | Slide | Theme class |
|---|---|---|
| 1 | Cover | hero dark |
| 2 | How it works (4-step pipeline) | light |
| 3 | What we tested (datasets) | dark |
| 4 | Three headline numbers | light |
| 5 | Why bucket by agreement (hero question) | hero light |
| 6 | Performance by tier (main results table) | light |
| 7 | Per-category detail (9-row table) | dark |
| 8 | Zero false alarms (big number + facts) | light |
| 9 | How to use the system (3 deployment modes + ops) | dark |
| 10 | How to scale to a new category (6-step pipeline) | light |
| 11 | Future work (2 columns: client side / team side) | dark |
| 12 | Closing | hero dark |

Known issues at handoff time:
- Some slides may overflow vertically on smaller browser windows — already tightened slides 6, 7, 9, 10, 11 once but the PPT-master teammate may want to push further.
- WebGL background does not survive PDF export — this is fine, the design works on a flat color too.
- Slide 6's tier rows are dense; could be split into 2 slides if the teammate prefers more breathing room.

---

## 9. Outstanding things the PPT-master teammate may want to address

1. **Visual polish.** The deck is functional but the typography spacing / section transitions could use a designer's eye. The skill template prioritizes content over flourish.
2. **Slide-level narrative tightening.** Each slide stands alone now; transitions between slides 5→6 (the "insight → results" handoff) and 9→10 (deployment → scaling) could be smoother.
3. **PDF / PPTX export.** The deck is a single HTML file. For a hand-in:
   - Quickest: Cmd+P → Save as PDF (landscape, no margins, background graphics ✅).
   - Best fidelity to PowerPoint: PDF → Adobe Acrobat Pro → Export as PPTX, then manually adjust 2-3 slides.
4. **Image enhancements.** Currently no images in the deck (data-only). The teammate may want to add: an architecture diagram for slide 2, a sample image grid for slide 3 (a few violation examples + a clean house), or a screenshot of the Gemma rationale for slide 8.
5. **Speaker notes.** The deck has none. The talk track is in `CLIENT_REPORT.md` (sibling file in this folder).

---

## 10. File map for the teammate

```
FinalProject/
├── handoff/
│   ├── STATUS.md            ← this file
│   └── CLIENT_REPORT.md     ← the talk-track / report-style summary
├── ppt/
│   ├── index.html           ← the 12-slide deck
│   ├── assets/motion.min.js
│   └── images/              ← currently empty, ready for additions
├── data/
│   ├── client-data/         ← 98 violation images organized by folder
│   │   └── labeling/
│   │       ├── human/                    ← 3 labelers' xlsx + merged + consensus CSVs
│   │       └── _ground_truth.csv         ← image_id → filename → folder primary category
│   └── synthetic/
│       └── code-enforcement-compliant/   ← 20 clean houses + manifest
├── checkpoints/
│   ├── client_gemma4_binary.npz          ← Gemma on 98 violations (Apr 13 cache)
│   ├── client_gemma_yes_aligned.npy      ← reordered to consensus image_id order
│   ├── compliant_gemma4_binary.npz       ← Gemma on 20 clean houses (today)
│   ├── client_clip_similarity.npz        ← CLIP on 98 violations
│   └── compliant_clip_sims.npz           ← CLIP on 20 clean houses (today)
├── scripts/
│   ├── run_gemma_binary.py               ← original 98-image runner
│   ├── run_gemma_binary_compliant.py     ← 20 clean houses runner (today)
│   ├── run_gemma_client.py               ← Gemma 9-way zero-shot (no CLIP filter)
│   └── run_clip_compliant.py             ← CLIP on 20 clean houses (today)
├── src/
│   ├── client_data.py                    ← CLIENT_CLASSES, CLIP prompts, Gemma descriptions, build_client_catalogue
│   ├── binary_prompt.py                  ← Gemma binary template + parser + verify loop
│   ├── zeroshot.py                       ← CLIP and Gemma loaders + zero-shot functions
│   └── ...
└── reports/
    └── (older milestone reports — superseded by handoff/CLIENT_REPORT.md)
```
