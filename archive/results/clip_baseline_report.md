# CLIP Zero-Shot Baseline — Results

> Model: `ViT-B/32` (OpenCLIP, `laion2b_s34b_b79k`)
> Test set: 891 images, 5 classes
> Training: **None** — pure zero-shot using text prompts

## Setup

Each violation class has 3 hand-written text prompts (averaged). For example:

- `structural_damage` → "a photo of cracks in a wall or building structure", "a photo of structural damage on a building surface", ...
- `overgrown_vegetation` → "a photo of overgrown grass and weeds", ...

Image and text embeddings are compared via cosine similarity → softmax → top-N predictions.

## Results

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | 78.5% |
| **Top-3 Accuracy** | 98.8% |
| **Macro F1** | 0.759 |

### Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| structural_damage | 0.535 | 0.950 | 0.685 | 120 |
| exterior_deterioration | 0.865 | 0.625 | 0.726 | 216 |
| trash_debris | 0.794 | 0.513 | 0.623 | 150 |
| overgrown_vegetation | 0.899 | 1.000 | 0.947 | 249 |
| illegal_dumping | 0.838 | 0.795 | 0.816 | 156 |

## Key Observations

### What worked well

1. **Top-3 accuracy at 98.8%** — the correct answer is almost always in the top 3 predictions. For our use case (inspector sees top-N suggestions and picks), this is operationally useful even without any training.

2. **Overgrown vegetation is near-perfect** (F1=0.947, recall=100%). CLIP's language-vision alignment handles this category extremely well — "grass and weeds" is visually distinctive and semantically unambiguous.

3. **Overall 78.5% top-1 is well above the expected 40-60%**. This suggests the proxy datasets map more cleanly to natural language descriptions than anticipated.

### What struggled

1. **Trash/debris has low recall (51.3%)**. TACO contains highly varied objects — batteries, bottle caps, broken glass — that don't all match "a photo of trash and litter." Many are small objects in cluttered scenes. Prompt engineering could help, but there's a ceiling on how much text alone can capture this visual diversity.

2. **Structural damage has low precision (53.5%)**. Many `exterior_deterioration` images (peeling, stains, algae) are being misclassified as structural damage. This makes sense — CLIP sees "damage on a building" and both categories match. The text prompts don't sufficiently distinguish between surface-level deterioration and structural cracks.

3. **Exterior deterioration has low recall (62.5%)** — the flip side of the above. Peeling paint and stains are being pulled toward structural_damage or other classes.

### Confusion pattern

The main confusion axis is `structural_damage` ↔ `exterior_deterioration`. These two categories share visual context (building surfaces, close-up shots) and differ only in defect severity — a distinction CLIP can't reliably make from text descriptions alone.

## Implications for Trained Models

- The 78.5% baseline sets a **high bar** — trained models need to substantially beat this to justify the training cost.
- The structural/exterior confusion should be much easier for a trained model to resolve, since it can learn fine-grained visual features (crack patterns vs surface stains).
- The trash recall problem may persist even with training, given the diversity of the TACO dataset.
- Top-3 at 98.8% suggests the 5-class taxonomy is well-separated in visual feature space — good news for the trained models.

## Practical Value

Even at 78.5% accuracy, this zero-shot model has immediate deployment value:
- **No training data needed** — can demonstrate to client on Day 1
- **Top-3 covers 98.8%** — inspector picks from suggestions, not from scratch
- Can serve as a **fallback** when the trained model encounters out-of-distribution images
