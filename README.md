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

## Handoff documents

The two files in [`handoff/`](handoff/) are the canonical references:

- [`handoff/CLIENT_REPORT.md`](handoff/CLIENT_REPORT.md) — results, methodology, per-tier deployment recommendations, and the SOP for adding new violation categories.
- [`handoff/USAGE.md`](handoff/USAGE.md) — code usage guide: install, reproduce the headline numbers, compute inter-annotator agreement, and add a new violation category.
- [`handoff/SYNTHETIC_NEGATIVES.md`](handoff/SYNTHETIC_NEGATIVES.md) — how the 20 synthetic clean-house images (the 0 % false-alarm baseline) were generated and verified.

Start with `USAGE.md` if you want to run the pipeline; start with `CLIENT_REPORT.md` if you want to understand the results.

## Quick start

See [`handoff/USAGE.md`](handoff/USAGE.md) for the full guide. TL;DR:

```bash
git clone https://github.com/psylch/inst623-riverdale-code-enforcement.git
cd inst623-riverdale-code-enforcement
uv sync

# Reproduce the headline numbers (98 images × 9 categories, ~60–90 min)
uv run python scripts/run_clip_compliant.py
uv run python scripts/run_gemma_binary_compliant.py
uv run python scripts/analyze_binary_results.py
```

Inspector photos are not redistributed in this repo for privacy reasons. Place them under `data/client-data/<§ code> - <name>/` to run the pipeline on your own data.

## Repository layout

```
.
├── src/         Python modules (client_data, zeroshot, binary_prompt, evaluate, ...)
├── scripts/     Pipeline entry points: run_clip_compliant, run_gemma_binary_compliant,
│                analyze_binary_results, compute_iaa
└── handoff/     Client-facing docs: CLIENT_REPORT.md + USAGE.md + SYNTHETIC_NEGATIVES.md
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
