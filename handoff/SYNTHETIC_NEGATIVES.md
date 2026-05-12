# Synthetic Clean-House Baseline — How the 20 Negatives Were Built

The 0 % false-alarm number in `CLIENT_REPORT.md` §3 is measured against 20 *synthetic* clean-house photos. This document explains how they were produced and verified, so the same procedure can be repeated when you want to refresh the baseline or extend it (e.g. replace synthetic with 50–100 real city clean-house photos, per CLIENT_REPORT §10).

## Why a synthetic baseline

A code-enforcement screener has to be evaluated on **clean houses**, not just known violations — otherwise you cannot measure false-alarm rate. The client batch we received is all violation photos. To get a clean baseline quickly we generated 20 documentary-style images of compliant houses and had all three annotators confirm every category was *no*.

Synthetic is a stopgap: the next-step ask in CLIENT_REPORT §10.1 is for Riverdale Park to provide 50–100 real clean-house photos from city records. Once available, drop them into `data/synthetic/code-enforcement-compliant/` (or any folder you wire into the evaluation script) and the 0 % FP claim becomes empirically validated against real-world variance (shadows, wreaths, parked cars, etc.).

## How they were generated

**Tool**: [Codex CLI](https://github.com/openai/codex) with its built-in `imagegen` skill (text-to-image). No external API key required beyond the Codex setup.

**Shared style constraint** (applied to every prompt to match inspector field-photo aesthetics):

> Match real municipal inspector field photographs: handheld smartphone or inspection camera, sidewalk/curb perspective, documentary framing, realistic residential textures, natural lighting, slight lens distortion, mild sensor noise/compression, no cinematic grading, no glossy real-estate staging.

**Shared negative constraint** (asserted absence of every violation category we evaluate on):

> No visible code violations, no boarded windows, no peeling paint, no loose trash or debris, no graffiti, no broken windows, no watermarks, no readable house numbers or license plates, no people.

**Per-image scene prompts**: 20 prompts varying house style (colonial, ranch, craftsman, Tudor, Cape Cod, Victorian, mid-century split, etc.), season (spring, autumn, winter, after-rain), and time of day, while keeping the "no violations visible" constraint intact. The full scene list is preserved at `data/synthetic/code-enforcement-compliant/manifest.md` (next to the images, outside git because `data/` is gitignored).

## How they were verified

All three annotators (Fechi, Niping, Jake) independently labeled the 20 synthetic images across all 9 violation categories. **Every cell came back `no`** — 20 × 9 = 180 negative cells, no disagreement.

This is why the 0 / 180 false-alarm number in `CLIENT_REPORT.md` §3 / §7 is reported with full confidence: the ground truth is unanimous across all three raters.

## Reproducing or extending the baseline

To regenerate or grow the set:

1. Open Codex CLI, invoke the `imagegen` skill.
2. Use the shared style + negative constraints above on every prompt.
3. Write one scene prompt per image — vary house style, season, lighting. Keep prompts visually grounded, no people, no readable text.
4. Save outputs as `compliant_NN.png` under `data/synthetic/code-enforcement-compliant/`.
5. Have three independent raters confirm every category is *no* before counting the image toward the FP evaluation.
6. Re-run the pipeline (`USAGE.md` §3). The clean houses are auto-appended to the evaluation surface in `scripts/analyze_binary_results.py`.

If you swap in real city clean-house photos instead, the same step 5 (three-rater verification) and step 6 (re-run) still apply — only the generation step (1–4) is replaced.
