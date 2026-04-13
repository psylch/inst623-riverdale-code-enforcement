# Technical Progress Report — March 31, 2026

**Project**: AI-Assisted Code Enforcement for Town of Riverdale Park
**Author**: Chihao Li (Technical Lead)
**Status**: Pipeline operational, baseline complete, deep learning experiments in progress
**Repository**: https://github.com/psylch/code-enforcement-cv

---

## What We've Done

### 1. Proxy Dataset Assembly (Complete)

Since the client's real inspection photos aren't available yet, we assembled a proxy dataset from four public sources to prototype and validate our pipeline:

| Dataset | Source | Images | Violation Category |
|---------|--------|--------|--------------------|
| BD3 | Kaggle (Kottari et al., BuildSys '24) | 3,965 | structural_damage, exterior_deterioration |
| TACO | tacodataset.org | 1,500 | trash_debris |
| Grass-Weeds | Roboflow Universe | 2,486 | overgrown_vegetation |
| Aerial Dumping | Roboflow Universe | 1,555 | illegal_dumping |
| **Total** | | **~8,900** | **5 unified classes** |

The original datasets had different formats (class-folder, COCO object detection, Flickr). We unified everything into a single image classification catalogue with a consistent 80/10/10 stratified train/val/test split (7,124 / 891 / 891).

### 2. Full Experiment Pipeline (Complete)

We built a modular Python codebase (`src/`) that handles:

- **Data loading & unification** — maps raw datasets into 5 violation classes
- **CLIP zero-shot baseline** — multi-prompt text-image matching, no training needed
- **DINOv2 ViT-B/14 LP-FT** — linear probe → full fine-tune, state-of-the-art self-supervised ViT
- **EfficientNetV2-S LP-FT** — same procedure, strong supervised CNN baseline
- **Evaluation** — per-class precision/recall/F1, top-3 accuracy, confusion matrices
- **Checkpointing** — every stage saves results to disk, so crashed runs resume from the last checkpoint

The pipeline runs locally on an M4 Mac (24 GB, MPS acceleration).

### 3. CLIP Zero-Shot Baseline (Complete — Results Below)

We ran CLIP ViT-B/32 with hand-written violation descriptions as a zero-training baseline.

**Headline numbers:**

| Metric | Result |
|--------|--------|
| Top-1 Accuracy | **78.5%** |
| Top-3 Accuracy | **98.8%** |
| Macro F1 | **0.759** |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| structural_damage | 0.535 | 0.950 | 0.685 |
| exterior_deterioration | 0.865 | 0.625 | 0.726 |
| trash_debris | 0.794 | 0.513 | 0.623 |
| overgrown_vegetation | 0.899 | 1.000 | 0.947 |
| illegal_dumping | 0.838 | 0.795 | 0.816 |

**Key takeaways:**

- **78.5% with zero training** is strong. For our intended use case — inspector sees top-N suggestions and picks the right one — the 98.8% top-3 accuracy means the system is almost always helpful.
- **Vegetation detection is near-perfect** (F1 = 0.947). CLIP's language-vision alignment handles "overgrown grass and weeds" extremely well.
- **Main weakness**: CLIP confuses `structural_damage` with `exterior_deterioration` — both involve building surfaces, and text prompts alone can't reliably distinguish cracks from peeling paint. This is exactly where trained models should excel.
- **Trash recall is low** (51.3%) because TACO includes highly varied objects (batteries, broken glass, bottle caps) that don't all match generic "trash" descriptions.

### 4. DINOv2 & EfficientNetV2 Experiments (In Progress)

Both models are running through the LP-FT (Linear Probe → Full Fine-Tune) pipeline right now:

| Model | Phase | Status |
|-------|-------|--------|
| DINOv2 ViT-B/14 | Linear Probe (30 epochs, backbone frozen) | Running |
| DINOv2 ViT-B/14 | Full Fine-Tune (20 epochs, all layers) | Queued |
| EfficientNetV2-S | Linear Probe | Queued |
| EfficientNetV2-S | Full Fine-Tune | Queued |

DINOv2 takes longer than expected because it operates at 518×518 resolution (vs 384×384 for EfficientNetV2), and Vision Transformers are heavier on MPS than CNNs. We expect all four training runs to complete within the next few hours.

The checkpoint system means we don't lose progress if anything interrupts — each completed phase is saved and skipped on re-run.

---

## What's Next

1. **Finish model comparison** — once DINOv2 and EfficientNetV2 complete, compile the full comparison table with confusion matrices and training curves.

2. **Write up the experiment notebook** — all results get assembled into a single Jupyter notebook with EDA, baseline, model comparison, and discussion.

3. **Prepare for real data** — when Riverdale Park's inspection photos arrive, we swap them into the same pipeline with minimal changes. If their violation categories overlap with our proxy classes, we can combine datasets; otherwise we retrain from scratch on the real data.

4. **Fairness analysis** — Niping will evaluate error rates stratified by neighborhood once we have geographically-tagged data.

---

## Repository Structure

```
FinalProject/
├── src/
│   ├── data.py            # Dataset unification, splits
│   ├── clip_baseline.py   # CLIP zero-shot logic
│   ├── models.py          # DINOv2, EfficientNetV2 via timm
│   ├── train.py           # Training loop (LP-FT, early stopping)
│   ├── evaluate.py        # Metrics, confusion matrix, comparison
│   └── cache.py           # Checkpoint/resume system
├── data/raw/              # Downloaded proxy datasets
├── checkpoints/           # Saved model weights & predictions
├── notebooks/
│   └── experiment.ipynb   # Main experiment notebook
├── results/
│   └── clip_baseline_report.md
├── technical-plan.md
├── my-pipeline.md
└── research-for-teammates.md
```

---

## For Discussion

- The CLIP baseline at 78.5% raises an interesting question: **how much accuracy do we gain from training, and is it worth the compute?** If DINOv2 lands at 90%+, that's a clear win. If it's only 82%, we might argue that CLIP zero-shot plus prompt engineering is the more practical approach for a small municipal client.

- The `structural_damage` vs `exterior_deterioration` confusion may reflect a real ambiguity in the violation taxonomy — inspectors might also struggle with this boundary. Worth discussing with Ryan (client) whether these should be merged.
