# Claude Sonnet Labeling — 3-Run Validated Truth

> Internal analysis · Chihao · 2026-04-27

We labeled all 98 client photos for 9 violation classes using Claude Sonnet, **three independent times**. Each run consisted of 10 sub-agents (one per 10-photo batch) operating from the same instruction file with no access to other runs' outputs. We then took per-cell majority across the 3 runs to build a high-confidence ground truth.

This doc reports the final 3-run truth, the inter-run agreement, what changed from the original v2 single-run pass, and a worked validation: two specific errors I had spot-checked by eye in run_a both got automatically corrected by the multi-run majority.

---

## TL;DR

1. **89.1% of cells were unanimous across all 3 runs.** Another 10.7% had a 2/3 majority. Only **2 of 882 cells** had a true 3-way split. Single-rater Sonnet is more stable than I expected, and the multi-rater check confirmed this empirically rather than by guess.
2. **Both v2 errors I spotted by eye got corrected automatically.** I had flagged `img_065/graffiti` (Arabic shop signage mis-classified as graffiti) and `img_041/boarded_windows` (a confidence-flip with no new info). In each case, run_b and run_c independently disagreed with run_a, and the majority truth was the right answer. This is the cleanest possible validation of the multi-rater approach.
3. **Sonnet still finds 2× more violations than the folder labels do.** The 3-run truth has 201 yes flags vs 98 folder primaries, and recovers the folder primary on 97 of 98 photos. Multi-violation is the norm: 61 of 98 photos have 2+ confirmed violations.

---

## 1. Headline numbers (3-run truth)

| metric | value |
|---|---|
| total images | 98 |
| total cells (image × class) | 882 |
| yes (truth) | **201** (22.8%) |
| no (truth) | **639** (72.4%) |
| unsure (truth) | **42** (4.8%) |
| yes (Gemma 4, comparison) | 238 |
| yes (folder labels) | 98 (one per photo) |

### Yes-per-image distribution

```
1 yes:  37 images   █████████████████████████████████████
2 yes:  29 images   █████████████████████████████
3 yes:  24 images   ████████████████████████
4 yes:   6 images   ██████
5 yes:   2 images   ██
```

No photo has zero confirmed violations. **62% of photos have 2 or more confirmed violations** — the multi-label structure is real, not noise.

### Unsure distribution

35 of 98 photos (35.7%) have at least one unsure cell — a follow-up photo from a different angle would close the gap.

---

## 2. 3-run agreement

| tier | count | share |
|---|---:|---:|
| **Unanimous** (3/3 same answer) | 786 | 89.1% |
| **Majority** (2/3 same answer) | 94 | 10.7% |
| **Disputed** (3-way split) | **2** | 0.2% |

The 2 disputed cells are `img_000 / broken_windows` (yes/no/unsure) and `img_015 / graffiti` (yes/unsure/no). They are flagged in the truth file as `tier: disputed` and treated as `unsure` for downstream analysis.

### Per-class agreement

| class | unanimous | majority | split |
|---|---:|---:|---:|
| inoperable_vehicle | 96 | 2 | 0 |
| graffiti | 95 | 2 | 1 |
| boarded_windows | 94 | 4 | 0 |
| deteriorating_chimney | 91 | 7 | 0 |
| overgrown_vegetation | 85 | 13 | 0 |
| broken_windows | 83 | 14 | 1 |
| peeling_paint | 82 | 16 | 0 |
| damaged_roof_shingles | 81 | 17 | 0 |
| junk_trash_accumulation | 79 | 19 | 0 |

The most variable classes (junk, paint, roof, broken_windows) are the same ones with the most unsure cells. They share one root cause: the call depends on a small visual detail (a trash pile vs. scattered debris, peeling paint vs. weathered surface, missing glass vs. dark interior) that sits near the model's decision boundary. The multi-run majority cleans up most of this; a few cells genuinely cannot be decided from a single photo.

---

## 3. Validation — two errors I caught by eye, both auto-corrected

When I spot-checked the original single-run (v2) by reading 9 photos manually, I flagged two specific cells as wrong. The 3-run process recovered the correct answer in both cases without my intervention.

| photo / class | what I said about v2 | run_a | run_b | run_c | 3-run truth |
|---|---|:---:|:---:|:---:|:---:|
| `img_065 / graffiti` | "Arabic shop signage is not graffiti — v2 wrong" | yes | **no** | **no** | **no** ✓ |
| `img_041 / boarded_windows` | "Confidence-flip with no new info — unstable" | yes | **no** | **no** | **no** ✓ |
| `img_041 / deteriorating_chimney` | (related — distant chimney) | yes | unsure | unsure | unsure |

This is the ideal validation pattern: single-rater errors are corrected by multi-rater consensus precisely because the errors are not reproducible — different runs of the same model on the same photo arrive at different answers, and the majority filters them out.

---

## 4. Per-class breakdown — Truth vs Gemma vs Folder

| class | folder | **truth yes** | **truth unsure** | Gemma yes | recall vs folder |
|---|---:|---:|---:|---:|---:|
| boarded_windows | 8 | 12 | 0 | 14 | 100% |
| broken_windows | 12 | 20 | 14 | 11 | 100% |
| damaged_roof_shingles | 7 | 20 | 16 | 14 | 100% |
| deteriorating_chimney | 3 | 4 | 9 | 4 | 100% |
| graffiti | 12 | 18 | 1 | 21 | 100% |
| inoperable_vehicle | 14 | 22 | 1 | 25 | 100% |
| junk_trash_accumulation | 13 | 29 | 0 | 39 | 100% |
| overgrown_vegetation | 14 | 35 | 0 | 50 | **93%** |
| peeling_paint | 15 | 41 | 1 | 60 | 100% |

Three observations:

- **Folder primaries are recovered on 97 of 98 photos** (every class except 1 photo in `overgrown_vegetation` where the visual evidence is borderline by our property-scale rule). Folders are correct as primary labels, just incomplete as multi-label labels.
- **Two classes need closer photos**: `damaged_roof_shingles` (16 unsures) and `broken_windows` (14 unsures). These are exactly the violations where the inspector's standard street-level shot doesn't show the relevant detail. A second photo (upward roof shot, window closeup) would resolve nearly all of these.
- **Gemma is more aggressive on `peeling_paint` (60) and `junk_trash` (39)** than the truth. The earlier "Gemma is texture-triggered" hypothesis is now confirmed — the truth is in between Gemma's permissive count and the folder's restrictive count.

---

## 5. The deployment context — why "unsure" is right

In production, the inspection workflow is multi-photo per property: an inspector walks around a house and snaps front, side, rear, and closeups. Each photo runs through the model independently, and the system aggregates per property. In that workflow, an `unsure` from photo A is not a problem — photo B (taken from a different angle of the same property) closes the gap.

Therefore:
- The model's job is to decide what *this* photo shows. Forcing a yes/no when the photo doesn't show the relevant feature creates noise that propagates downstream.
- For our 98-photo evaluation, we treat unsure as a third state in per-image agreement metrics, and as "abstain" in per-property metrics where missing photos can't be substituted.
- The model should NOT be penalized for being honest about visual ambiguity — and the deliverable to the client should include a recommendation about photo coverage (upward roof shots, closeups for window/chimney damage).

---

## 6. Co-occurrence — three signatures

Reading the table as: **when truth says yes to row class, what % of those photos also have yes for column class?**

| row \\ col | boarded | broken | roof | chimney | graffiti | vehicle | junk | overgrown | paint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boarded | 100 | 17 | 33 | 0 | 8 | 0 | 25 | 25 | **58** |
| broken | 10 | 100 | 25 | 0 | 20 | 0 | 25 | 15 | **60** |
| roof | 20 | 25 | 100 | 0 | 30 | 0 | 25 | 30 | **60** |
| chimney | 0 | 0 | 0 | 100 | 0 | 0 | 25 | 0 | 0 |
| graffiti | 6 | 22 | 33 | 0 | 100 | 0 | 17 | 33 | 17 |
| vehicle | 0 | 0 | 0 | 0 | 0 | 100 | 36 | **77** | 18 |
| junk | 10 | 17 | 17 | 3 | 10 | 28 | 100 | **59** | 28 |
| overgrown | 9 | 9 | 17 | 0 | 17 | **49** | **49** | 100 | 34 |
| paint | 17 | 29 | 29 | 0 | 7 | 10 | 20 | 29 | 100 |

Three signatures, cleaner than what we saw in v2:

1. **Abandoned-house signature** — `peeling_paint` co-occurs with `boarded` (58%), `roof` (60%), and `broken` (60%). When the building is decaying, paint goes too.
2. **Unmaintained-lot signature** — `inoperable_vehicle` ↔ `overgrown_vegetation` 77%, `junk_trash` ↔ `overgrown` 59%, `vehicle` ↔ `junk` 36%. Lots that are dumping grounds.
3. **Lone violations** — `chimney` and `graffiti` rarely cluster with others. Chimney shows up on otherwise-decent buildings; graffiti shows up on otherwise-decent surfaces.

The cluster percentages are tighter than the single-run v2 numbers because spurious cross-flags from individual runs were filtered out by the majority.

---

## 7. Schema iteration history (brief)

For completeness — we ran two schema versions before the 3-run validation:

- **v1 (yes/no only)** — single run; manual spot-check found ~5% borderline cells where forced yes/no was producing bias (small vines on walls, vehicle windows, blue tarps over windows).
- **v2 (yes/no/unsure)** — single run with explicit class-boundary rules added (broken_windows is buildings only; overgrown_vegetation is property-scale only; can't-see-it → unsure). 88.9% of cells unchanged from v1; the 11.1% that changed were exactly the borderline cells.
- **3-run on v2 schema (this analysis)** — multi-rater majority on the v2 schema. 96% stable vs single-run v2; the 4% delta is the noisy cells the multi-run filtered out.

Each step added confidence in a different way: v2 fixed systematic class-boundary errors, the 3-run filtered single-rater noise.

---

## 8. Open questions

1. **Is the 3-run truth ground-truth-quality for our purposes?** My current judgment: **yes for working / pipeline-development purposes**, with one caveat — all 3 runs are Sonnet, so any systematic Sonnet-specific bias is invisible. For the final client deliverable, combining this with the upcoming 3-person human consensus (Stream A) is the right move.
2. **Should we re-run the cascade evaluation now?** With multi-label truth in hand, we can compute Jaccard / sample-F1 / per-class precision-recall for CLIP and Gemma against the truth. The Phase 2 numbers were against folder labels and probably underestimated both models.
3. **What to do with the 42 unsure cells?** Recommended: drop them from per-image accuracy; report `model abstention rate per class` separately. This is the honest representation and aligns with the multi-photo-per-property workflow.
4. **Recommendation to client about photo coverage.** Our unsure pattern shows the existing photos under-cover roofs (16 unsures) and broken windows behind boards/distance (14 unsures). Inspector workflow should include at least one upward roof shot per property and one closeup of any visible window damage.
