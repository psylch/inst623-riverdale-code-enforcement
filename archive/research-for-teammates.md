# AI-Assisted Code Enforcement: Technical Research

> For team learning — covers datasets, model options, and key concepts we'll use in this project.
> Prepared by Chihao, March 2026

---

## 1. Why We Can Start Before Client Data Arrives

The Town of Riverdale Park has ~1,000 labeled images across 600-700 violation codes. We'll focus on the top 20-30 categories. While waiting for data access, we can:

- **Find similar public datasets** to prototype our pipeline
- **Choose the best model architecture** based on current research
- **Build the training pipeline** so we're ready to swap in real data

---

## 2. Similar Datasets We Can Use Now

No single "municipal code enforcement" dataset exists publicly, but we can combine several domain-specific ones:

### Building Defects (directly relevant)

| Dataset | Size | Classes | Why It's Useful |
|---------|------|---------|-----------------|
| **BD3** ([GitHub](https://github.com/Praveenkottari/BD3-Dataset)) | 3,965 images | 7 (crack, peeling, spalling, stain, etc.) | Covers building surface conditions similar to code violations |
| **Building Surface Defect Detection** ([HuggingFace](https://huggingface.co/datasets/xueaidezhouzhou/buildingsurfacedefectdetection)) | 7,354 images | 6 (cracks, spalling, rust, etc.) | Diverse indoor/outdoor conditions, like real inspections |
| **SDNET2018** ([Kaggle](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images)) | ~56,000 images | 2 (cracked/not cracked) | Large scale, good for backbone pre-training |

### Trash / Illegal Dumping (directly relevant)

| Dataset | Size | Classes | Why It's Useful |
|---------|------|---------|-----------------|
| **TACO** ([tacodataset.org](http://tacodataset.org/)) | 1,500 images, 9,823 objects | 60 waste classes | Real-world outdoor litter, COCO format |
| **Garbage Object Detection** ([HuggingFace](https://huggingface.co/datasets/keremberke/garbage-object-detection)) | 10,464 images | Multiple | Large, well-annotated garbage dataset |
| **Aerial Dumping Sites** ([Roboflow](https://universe.roboflow.com/object-detection-of-illegal-dumping-sites/aerial-dumping-sites)) | 1,555 images | Dumping detection | Specifically targets illegal dumping — closest to our use case |

### Vegetation / Overgrowth

| Dataset | Size | Why It's Useful |
|---------|------|-----------------|
| **Grass-Weeds** ([Roboflow](https://universe.roboflow.com/roboflow-100/grass-weeds)) | 2,486 images | Can detect "overgrown grass" violations |

### Scene Understanding (for pre-training)

| Dataset | Size | Why It's Useful |
|---------|------|-----------------|
| **Places365** ([MIT](http://places2.csail.mit.edu/)) | 1.8M images, 365 scene categories | Pre-trained models understand residential/urban contexts |

### Real-World Industry Players

These companies are doing exactly what we're building. Their approaches validate our direction:

| Company | What They Do | Key Numbers |
|---------|-------------|-------------|
| **[CityDetect](https://www.citydetect.com/)** (PASS AI) | Vehicle-mounted cameras + AI to flag violations (overgrown vegetation, illegal dumping, graffiti, vehicles on lawns, roof damage) | Dallas pilot: **95% accuracy**, 3,000+ violations flagged in 4 days. $13M funding. Active in Dallas, Stockton CA, Miami, Atlanta |
| **[Forerunner](https://www.withforerunner.com/)** | Governance-first code enforcement platform. Automates case management, generates violation notices citing exact code sections | Human-in-the-loop design, ML flags document inconsistencies |
| **[DataGrid](https://datagrid.com/)** | AI agents for violation processing and citation management | Focus on automated complaint processing |
| **InspectMind AI** | Converts jobsite photos + voice notes into inspection reports | Mentioned by @aigrant on X |

**Industry trends**: Shift from complaint-driven → camera-driven proactive enforcement. Privacy-first design (auto-blur faces/plates). Human always makes final call.

**Criticism worth noting**: A UC Strategies article flagged "17 U.S. Cities Are Letting AI Inspect Neighborhoods — But Nobody Knows If It Works" — efficacy validation is an open question, which makes our evaluation framework even more important.

---

## 3. Model Architecture Options — What's Changed Since 2024

Our proposal says ResNet-50 or EfficientNet. That's still a reasonable baseline, but the field has moved significantly. Here's what we should actually consider:

### The Big Picture: CNN vs Vision Transformer

- **CNNs** (ResNet, EfficientNet, ConvNeXt): Have built-in understanding of spatial structure. Work well out of the box on small datasets.
- **Vision Transformers** (ViT): Need huge datasets if trained from scratch. BUT with modern self-supervised pre-training (DINOv2, CLIP), they now beat CNNs even on small datasets.

**Key insight**: It's not about the architecture anymore — it's about **how the model was pre-trained**.

### Recommended Models (ranked for our ~1K image scenario)

| Rank | Model | Type | Why |
|------|-------|------|-----|
| 1 | **DINOv2 ViT-B/14** | Self-supervised ViT | Best general-purpose visual features; even a simple linear classifier on top works well |
| 2 | **SigLIP 2** | Vision-language model | Released Feb 2025; can do zero-shot classification using text descriptions of violations |
| 3 | **ConvNeXt V2** | Modern CNN | Best pure CNN; very stable on small datasets |
| 4 | **EfficientNetV2-S** | CNN | Great accuracy/speed tradeoff; production-friendly |
| 5 | **CLIP ViT-B/16** | Vision-language model | Free zero-shot baseline — classify using text like "a photo of overgrown grass" |

### What is CLIP Zero-Shot? (important concept)

CLIP can classify images using natural language descriptions **without any training**. You just give it:
- An image (e.g., an inspection photo)
- A list of text descriptions (e.g., "overgrown vegetation", "broken window", "abandoned vehicle")

It returns which description best matches the image. This gives us a **Day 1 baseline** before we train anything.

### What is DINOv2?

DINOv2 (Meta, 2023) is a vision model trained on 142M images without any labels (self-supervised). It learns to understand visual structure so well that its features work across domains — medical imaging, satellite photos, and yes, property inspections. You freeze the model and just train a small classifier on top.

**What experts on X/Twitter say about DINOv2:**
- @omarsar0 (ML researcher): Called it a "massive release" for self-supervised CV
- @NielsRogge (HuggingFace): Published tutorial calling it "state-of-the-art image classification at your fingertips"
- @OpenCVUniverse: DINOv3 (2025 update, 7B params) outperforms domain-specialized solutions without fine-tuning
- @kothasuhas (Stanford): Found that replaying generic pre-training data during fine-tuning improves performance 1.87x when fine-tuning data is scarce

**Benchmark results** (from Voxel51/Nyckel):

| Model | Food-101 | 10K Species | Scenes-365 |
|-------|----------|-------------|------------|
| DINOv2 ViT-B/14 | 93% | 70% | 53% |
| CLIP ViT-B/16 | 88% | 15% | 51% |
| ResNet-101 | — | — | 48% |
| ResNet-18 | 65% | 12-14% | 41% |

DINOv2 was the best model in ~20% of 117 benchmarked datasets. For our small-data scenario, linear probing on frozen DINOv2 features achieves 86.5% on ImageNet — often outperforming full fine-tuning on small datasets.

---

## 4. Key Concepts for Our Pipeline

### Transfer Learning (what we'll actually do)

Instead of training from scratch, we take a model pre-trained on millions of images and adapt it:

1. **Linear Probing**: Freeze the pre-trained model, only train a new classification head. Fast, works surprisingly well.
2. **Fine-tuning**: Unfreeze the whole model and train with a very small learning rate. Gets the best accuracy but risks overfitting on small data.
3. **LP-FT** (our recommended approach): Do linear probing first, then fine-tune. This avoids a known problem where fine-tuning can distort good pre-trained features.

### Multi-Label Classification

A single photo might show MULTIPLE violations (overgrown grass + vehicle on unimproved surface). We need:
- **Sigmoid** output per class (not softmax, which forces a single prediction)
- **Binary Cross-Entropy** loss (each class is an independent yes/no)
- **Per-class threshold tuning** on validation set

### Data Augmentation (critical for small datasets)

To make our ~1K images act like a larger dataset:
- **Basic**: Random crop, flip, rotation, color jitter
- **Advanced**: MixUp (blend two images), CutMix (paste patches between images), RandAugment
- **Domain-specific**: Weather effects (rain, fog), perspective changes, brightness variation — because inspectors photograph in all conditions

---

## 5. Tools We'll Use

| Tool | Purpose |
|------|---------|
| **PyTorch** | Deep learning framework |
| **timm** (PyTorch Image Models) | 800+ pre-trained models, training recipes, augmentation |
| **HuggingFace Transformers** | CLIP, DINOv2, SigLIP models |
| **Albumentations** | Image augmentation library |
| **scikit-learn** | Evaluation metrics, fairness analysis |

---

## 6. Related Academic Papers

- DINOv2: [arXiv 2304.07193](https://arxiv.org/html/2304.07193v2) — self-supervised visual features
- DINOv2 for few-shot medical segmentation: [arXiv 2403.03273](https://arxiv.org/abs/2403.03273) — shows DINOv2 works on small specialized datasets
- Deep Learning for Building Defect Detection using CNNs: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6720984/)
- CV-Based Safety Inspection in Public Housing: [IEEE](https://ieeexplore.ieee.org/document/10902946/) — deep learning for detecting obstructions/clutter in common areas
- Real-time building defect detection smartphone app: [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/stc.2751)

## 7. Suggested Reading

If you want to go deeper on any of these topics:

- [CNN vs Vision Transformers in 2025](https://aicompetence.org/vision-transformers-vs-cnns/) — accessible comparison
- [DINOv2 Fine-Tuning Tutorial](https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/) — hands-on guide
- [SigLIP 2 Fine-Tuning for Image Classification](https://huggingface.co/blog/prithivMLmods/siglip2-finetune-image-classification) — newest approach
- [LP-FT: Why Fine-Tuning Can Distort Features](https://snorkel.ai/blog/boost-foundation-model-results-with-linear-probing-fine-tuning/) — explains our training strategy
- [CutMix, MixUp, and RandAugment Explained](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/) — visual guide to augmentation
