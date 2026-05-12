# Weekly Report: First Round of Zero-Shot Evaluation on Real Client Data

> **Project**: AI Adoption Clinic — Riverdale Park Code Enforcement Image Classification
> **Student**: Chihao Li (Technical Lead)
> **Date**: 2026-04-13

---

## What we're doing

This week we got the first batch of real photos from the client. Following the zero-shot direction we set last week, I went straight to off-the-shelf foundation models and didn't train anything. On top of CLIP, which we'd already validated on proxy data, this week I added Google's Gemma 4 E4B-IT — a multimodal VLM Google open-sourced in April. Quantized to 4-bit MLX, it runs locally on my own M4 24 GB Mac.

---

## What the data we got this week looks like

98 photos, organized into folders by 14 official violation codes — but only 9 folders actually have images, the other 5 are empty. The classes are very unbalanced:

| Class | Photos |
|---|---:|
| Peeling Paint | 15 |
| Inoperable Vehicles | 14 |
| Long Grass / Overgrown Vegetation | 14 |
| Junk / Trash Accumulation | 13 |
| Broken Windows | 12 |
| Graffiti | 12 |
| Boarded Windows | 8 |
| Damaged Roof Shingles | 7 |
| Deteriorating Chimney | 3 |

The smallest class has only 3 photos. There's no way to do supervised training at this volume, so zero-shot is the only option.

While preparing the data I noticed three structural things about it:

**1. The data is fundamentally multi-label, but stored as single-label folders.** I ran MD5 on all 98 photos and found that 11 of them are byte-identical copies that the inspector filed under multiple categories at once. One photo of a burned-out building shows up in Graffiti, Long Grass, and Damaged Roof Shingles all at the same time. The folder structure looks single-label, but it's actually a flattened multi-label dataset.

**2. There are no "no violation" reference photos at all.** All 98 are violations. Not a single one represents the state of "the inspector went out, looked at it, and judged it compliant."

**3. The label semantics is "which code did the inspector cite," not "what's visible in the photo."** A photo of a deteriorating building might have peeling paint, boarded windows, and a damaged roof all at once, but the inspector might only have cited boarded_windows that day, so the ground truth only contains boarded_windows.

---

## Phase 1 vs Phase 2: same data, two ways of asking

Both phases happened this week. I first ran the most intuitively natural framing (multi-class classification), then noticed something weird and switched to a different framing (per-class binary). Same 98 photos, same CLIP, same Gemma — the only difference is how I'm asking the model the question.

**Phase 1: multi-class.** For each photo, the model ranks all 9 classes and we take top-1 / top-3 as the answer.
- CLIP ViT-B/32: encode each of the 9 class descriptions to text vectors, encode the image to an image vector, compute cosine similarity, softmax compresses the scores into probabilities that sum to 100%, argmax picks the highest.
- Gemma 4 E4B-IT: stuff all 9 candidate classes into a single prompt and ask it to return a ranked list.

Aggregate top-1 / top-3 hit rate (under multi-label ground truth):

| Model | Top-1 | Top-3 |
|---|:---:|:---:|
| CLIP ViT-B/32 | 80.6% | 94.9% |
| Gemma 4 E4B-IT | 85.7% | 96.9% |

The aggregate looks fine, but breaking it down per-class shows the problem:

| Class | n_pos | CLIP@1 | CLIP@3 | Gemma@1 | Gemma@3 |
|---|---:|---:|---:|---:|---:|
| boarded_windows | 10 | 100% | 100% | **20%** | 40% |
| broken_windows | 13 | 85% | 85% | 77% | 100% |
| damaged_roof_shingles | 10 | 30% | 40% | 70% | 70% |
| deteriorating_chimney | 3 | 100% | 100% | 67% | 67% |
| graffiti | 14 | 100% | 100% | 100% | 100% |
| inoperable_vehicle | 20 | 100% | 100% | 100% | 100% |
| junk_trash_accumulation | 14 | 86% | 93% | 86% | 100% |
| overgrown_vegetation | 24 | **4%** | 88% | **12%** | 88% |
| peeling_paint | 16 | **31%** | 100% | 88% | 100% |

Two strange things stand out.

First, Gemma's top-1 recall on boarded_windows is only 20%, but its free-text rationale clearly describes "weathered wood, peeling paint, broken shingles." It sees the right visual features. The problem is in softmax: 9 classes are competing for the same 100% probability budget, argmax can only pick one winner, so Gemma is forced to give boarded_windows up in favor of some more specific class.

Second, CLIP's top-1 on overgrown_vegetation is only 4%, and on peeling_paint it's 31% — but their top-3 numbers jump to 88% and 100%. So CLIP is actually "seeing" these classes; argmax is just pushing them down to rank 2 or 3.

Given that the data itself is multi-label, this kind of compression was almost inevitable.

**Phase 2: per-class binary.** Same two models, same 98 photos, but instead of "pick one out of 9," we ask "for each (image, class) pair, yes or no, independently?"
- CLIP: no more argmax. Keep the raw 98×9 similarity matrix and treat each column as "the model's standalone score for this one class."
- Gemma: I wrote a new prompt that asks one question at a time — "does this image show X?" — with a forced JSON output. For each of the 98 images I ran it on all 9 classes, for a total of 882 calls, run overnight. The inference script supports resume-from-crash, which mattered because the run got interrupted once when the MPS thermal throttled.
- Then I stitched them into a cascade: CLIP produces a top-k candidate list → Gemma does a binary check on each candidate → the union becomes the final prediction.

---

## Results

Gemma's per-class numbers under the binary framing:

| Class | n_pos | AUC | Recall@0.5 | Precision@0.5 |
|---|---:|---:|---:|---:|
| deteriorating_chimney | 3 | 1.000 | 100% | 75% |
| graffiti | 14 | 0.993 | 100% | 67% |
| inoperable_vehicle | 20 | 0.977 | 90% | 72% |
| junk_trash_accumulation | 14 | 0.922 | 100% | 36% |
| overgrown_vegetation | 24 | 0.904 | 100% | 48% |
| damaged_roof_shingles | 10 | 0.850 | 70% | 50% |
| peeling_paint | 16 | 0.796 | 100% | **27%** |
| broken_windows | 13 | 0.787 | 54% | 64% |
| boarded_windows | 10 | 0.764 | **60%** | 43% |

Compared directly against Phase 1 ranked: Gemma's recall on boarded_windows went from 20% → 60%. Same model, same data, no training — only the question format changed. The other classes that argmax had been compressing (broken_windows, damaged_roof_shingles) also mostly got their recall back. The cost is that several classes now show very low precision — peeling_paint at 27%, junk_trash at 36%, boarded_windows at 43% — which I'll come back to in its own section.

I also computed CLIP's per-class AUC under the same multi-label framing, so we can compare CLIP head-to-head with Gemma binary:

| Class | CLIP AUC | Gemma AUC | Stronger |
|---|---:|---:|---|
| broken_windows | **0.953** | 0.787 | CLIP (+0.17) |
| boarded_windows | **0.931** | 0.764 | CLIP (+0.17) |
| peeling_paint | **0.903** | 0.796 | CLIP (+0.11) |
| overgrown_vegetation | 0.709 | **0.904** | Gemma (+0.20) |
| damaged_roof_shingles | 0.775 | **0.850** | Gemma (+0.08) |
| chimney / graffiti / vehicle / junk_trash | ~1.0 | ~1.0 | tied |

The two models' strengths are almost completely disjoint. Gemma is clearly better on overgrown and roof — the two classes CLIP can't see well. CLIP is clearly better on the three window/paint classes. This complementarity is the real reason the cascade works at all.

The cascade's F1:

| Configuration | F1 | Predictions / image |
|---|:---:|:---:|
| Gemma binary alone (≡ k=9) | 0.633 | 2.43 |
| Cascade k=5 | 0.668 | 2.08 |
| Cascade k=3 | **0.703** | 1.60 |

Smaller k gives higher F1, which is the opposite of what I expected. CLIP naturally assigns very low similarity to classes like peeling_paint and overgrown_vegetation. At k=3 it filters those classes out before Gemma ever gets a chance to misfire on them. So what the cascade is actually doing is less like "CLIP filters, Gemma verifies" and more like "two independent models cross-checking each other."

Two figures below: the first is a visualization of the CLIP vs Gemma AUC comparison table above; the second is CLIP's per-class score distribution (orange = positives, gray = leave-one-out negatives from other classes), which makes it easy to see which classes CLIP can separate and which it can't.

![Per-class AUC: CLIP vs Gemma 4 binary](figures/auc_clip_vs_gemma_binary.png)

![CLIP per-class score separability](figures/clip_separability.png)

---

## The biggest current problem: we don't have any real negatives

This is the most important thing that came out of this week, and it's the reason every number above needs to be read with an asterisk.

All 98 photos are violations. Not a single one is a "the inspector went out and judged the property compliant" reference. But to compute precision, recall, AUC, you need both positives and negatives. The only workaround I have is: for class X, the "negatives" are simply the photos that aren't tagged X.

But these "negatives" are still positives for some other violation. When I compute precision for peeling_paint, the "negatives" are photos filed under graffiti, boarded_windows, broken_windows, and so on. Riverdale's inspector photos are almost all photos of deteriorating buildings, so these "negatives" probably do have real peeling paint in them — the inspector just didn't cite § 304.2 that day.

This explains why peeling_paint at cascade k=3 has 100% recall but only 40% precision. Gemma is most likely not making things up; it's identifying real violations that exist in the photo but aren't in the ground truth, and our metric records them as false positives.

In other words, the precision we're measuring right now is "how consistent the model is with the inspector's filing habits," not "how visually accurate the model's judgment is." Until we get real negatives, every number has to be read with this asterisk. This isn't a model problem and it's not a prompt problem — it can only be solved by getting more or better data from the client.

---

## Next steps

**Things I can do next week without depending on the client:**

1. Rewrite CLIP's prompt for damaged_roof_shingles. This class is the only Stage 1 bottleneck for the cascade — CLIP fails to put it in the top-5 for 6 out of 10 true positives. Try a more visually specific phrasing.
2. Tighten Gemma's peeling_paint binary prompt with qualifiers like "substantial / noticeable area" and see whether the FPR drops. If it doesn't, that's actually further evidence that the problem is in the ground truth, not in the model.
3. Wrap the cascade into a single demo-able module so I can show it live at the next sync with Ryan.

**Two things to ask the client for** (to be raised at the next sync with Ryan):

1. About 200 "no violation" photos — buildings the inspector visited and judged compliant. This is the only way to compute an operationally meaningful false positive rate.
2. Full multi-label re-annotation of the existing 98 photos — have inspectors mark every visible violation in each photo, not just the one they happened to cite that day. This would directly resolve the peeling_paint precision paradox.

Our previous data ask was the generic "give us more labeled data." This time the ask is specific: negatives for FPR, completed labels for precision. A specific request like this is more likely to clear Town Hall's compliance review.

**Fallback if neither comes through:**

Town Hall's compliance review has been the bottleneck on the data side, and I'm not confident Ryan can get either of these batches through. If they don't come through, there are two backup directions:

1. **Build our own negative pool.** Use the proxy datasets we'd already collected (street view, residential exteriors from Google Open Images), manually pick out clearly-compliant building photos, and assemble a "non-violation" reference set. This kind of negative won't share the distribution of real Riverdale inspector photos, but it would at least let us compute the order of magnitude of the false positive rate, which is better than having nothing.
2. **Re-annotate the existing 98 photos as multi-label ourselves.** The inspector's labels are "which code did they cite" and we can't change that, but what's actually visible in a photo is something we can re-annotate from the photos themselves. That makes the 9 classes serve as positives and negatives for each other at the annotation level — a photo filed under boarded_windows that also visibly has peeling paint would get tagged peeling_paint as well. Re-running precision against the re-annotated ground truth would directly resolve the question of whether the 27% peeling_paint precision is the model misfiring or the ground truth being incomplete.

Both directions are independent of any external data and are technically feasible. Concretely how to drive them, and who would do the work, is something to settle after the next sync with Ryan and a team alignment.

---

## Reproducibility

All code, scripts, and report sources are public at `https://github.com/psylch/inst623-riverdale-code-enforcement`. To reproduce from a clean machine:

```bash
# Clone the repository and enter the project root
git clone https://github.com/psylch/inst623-riverdale-code-enforcement.git
cd inst623-riverdale-code-enforcement

# Install Python dependencies (uses uv for Python 3.12)
uv sync

# Place client photos under data/client-data/ before running
# (one subdirectory per violation code, matching the folder names in the
#  official Riverdale Park taxonomy). The client dataset itself is not
#  redistributed in this repository for privacy reasons.

# CLIP separability + top-k analysis (~2 minutes)
uv run python scripts/run_clip_separability.py

# Gemma binary verification (~47 minutes, resumable)
uv run python scripts/run_gemma_binary.py
# Monitor progress in another terminal:
tail -f checkpoints/client_gemma4_binary_stream.jsonl

# Cascade evaluation and figures
uv run python scripts/analyze_binary_results.py
```

Every path above is relative to the repository root. All experimental artifacts (predictions, similarity matrices, audit logs, and figures) land under `checkpoints/` and `reports/figures/`. The full pipeline runs end-to-end on an Apple Silicon Mac with 24 GB unified memory; model weights are downloaded from Hugging Face on first run and cached locally, and no external compute is required.
