# Riverdale Code Enforcement — Zero-Shot AI Pipeline

Final-project work for the **INST623 AI Adoption Clinic** at the University of
Maryland. Client: **Town of Riverdale Park, Development Services** (Director
Ryan Chelton).

The project builds an image classification system for municipal code
enforcement: inspectors photograph property violations in the field, and the
system maps each photo to the relevant Riverdale Park municipal code section
(e.g., `§ 304.7 Damaged Roof Shingles`). Inspectors confirm or reject the
suggestion — the AI is a decision-support tool, not an autonomous enforcement
agent.

## What's in the repo

Two-stage **zero-shot cascade** that runs entirely on a laptop (Apple Silicon,
no training, no fine-tuning):

1. **Stage 1 — CLIP ViT-B/32 (LAION-2B)** generates per-class candidates from
   an image using a small set of natural-language prompts per violation code.
2. **Stage 2 — Gemma 4 E4B-IT (4-bit, MLX)** independently answers a binary
   "does this image show X?" for each candidate and returns a confidence score
   plus a short rationale.

On a first batch of 98 real inspector photos covering 9 Riverdale Park
violation codes (multi-label ground truth), the cascade reaches
**sample-F1 0.703 / top-3 recall 96.9% with zero training data**.

## Reports

- `reports/progress-report-2026-04-13.md` — consolidated English progress
  report written for the course instructor (Phase 1 + Phase 2 narrative)
- `reports/zeroshot-client-evaluation.md` — detailed Phase 1 report
  (Chinese): multi-class zero-shot baseline, debugging, and the pivot to
  per-class binary detection
- `reports/zeroshot-client-phase2.md` — detailed Phase 2 report (Chinese):
  CLIP separability, Gemma binary verification, CLIP × Gemma cascade

Each report is self-contained. Use `pandoc report.md -o report.pdf
--pdf-engine=typst` to convert to PDF (add `-V mainfont="Sarasa UI SC"` for
the Chinese ones).

## Quick start

```bash
git clone https://github.com/psylch/inst623-riverdale-code-enforcement.git
cd inst623-riverdale-code-enforcement

# Install Python deps (uv, Python 3.12)
uv sync

# Place client photos under data/client-data/<violation-code>/
# (The client dataset is not redistributed here for privacy reasons.)

# Stage 1: CLIP per-class separability + top-k recall (~2 min)
uv run python scripts/run_clip_separability.py

# Stage 2: Gemma 4 binary verification (98 × 9 = 882 calls, ~47 min, resumable)
uv run python scripts/run_gemma_binary.py

# Cascade evaluation + figures
uv run python scripts/analyze_binary_results.py
```

All experimental artifacts (prediction matrices, similarity matrices, audit
logs, figures) land under `checkpoints/` and `reports/figures/`. Model weights
(CLIP ViT-B/32 ≈ 600 MB, Gemma 4 E4B-IT 4-bit ≈ 4 GB) are downloaded from
Hugging Face on first run and cached locally.

## Repository layout

```
.
├── src/                    Python modules (client_data, zeroshot, binary_prompt,
│                           evaluate, cache, data, models, train, clip_baseline)
├── scripts/                Standalone runners with JSONL streaming + resume
├── notebooks/              Experiment notebooks (CLIP + Gemma on client data)
├── reports/                Source markdown reports + figures
│   └── figures/            CLIP separability, AUC comparison plots
├── results/                Earlier progress reports
├── technical-plan.md       Execution plan
├── dataset-status.md       Proxy dataset audit
└── data-cleaning-report.md Data cleaning audit from the proxy phase
```

## Acknowledgments

- **Ryan Chelton** (Director of Development Services, Town of Riverdale Park)
  for the client problem framing and patience through compliance review.
- The **INST623 AI Adoption Clinic** team: Chihao Li (Technical Lead), Niping
  (Eileen) Duan (Domain & Evaluation Lead), Fechi Iwudyke (Client Liaison &
  Ethics/Fairness), Jake Sheehi (Documentation).
- Google DeepMind for the open-weight **Gemma 4** models released April 2026,
  and the `mlx-vlm` project for Apple Silicon inference support.

## License

Code: MIT. Inspector photos are not included in this repository.
