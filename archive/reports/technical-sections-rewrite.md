# Sections 4.2–4.4 Rewrite (Updated Technical Approach)

> Replace the corresponding sections in the original Roles & Task Identification document.
> Changes reflect: client data unavailable due to compliance; proxy dataset strategy; updated model architecture decisions.

---

## 4.2 Proposed Solution

We propose the development of an image classification system that leverages modern vision foundation models to identify residential code violations from inspection photographs. The system will:

- Accept a photograph as input from an inspector's mobile device or desktop interface
- Process the image through a vision model trained on proxy datasets that approximate the Town's most common violation categories
- Return the top-N most likely violation types (e.g., top 3), ranked by confidence score, along with the applicable code chapter and section
- Present the output in a clear, inspector-readable format requiring confirmation or rejection of each suggestion
- Log inspector feedback (agree/disagree) to support future model retraining and continuous improvement

A critical constraint has emerged since the initial project plan: the Town of Riverdale Park's compliance review determined that the original inspection photographs cannot be shared outside Town Hall due to privacy obligations — some images were captured inside private residences as part of rental licensing. As a result, the team has pivoted to a **proxy dataset strategy**, constructing a training corpus from publicly available datasets that approximate the visual characteristics of the Town's most common violation types. This approach allows the team to build and validate the full classification pipeline independently, while designing the system so that the Town can retrain on their own data in the future without modifying the architecture.

The system is designed as a decision-support tool, not an autonomous enforcement agent. All final judgments remain with the human inspector. This design reflects Ryan Chelton's stated vision of a "human-in-the-loop" system where AI serves as an assistant rather than a decision-maker. The logging of inspector feedback also addresses his intuition that staff confirmations and rejections could help educate future recommendations.

## 4.3 Technical Approach

The technical pipeline consists of four main components:

### 1) Data Construction and Preprocessing

Because the client's labeled image-code pairs are unavailable due to compliance constraints, the team constructed a proxy training corpus from seven publicly available datasets, selected to cover Riverdale Park's five most common violation categories as documented in the Town's public code enforcement materials:

| Violation Type | Proxy Datasets Used | Images |
|---|---|---|
| Overgrown Grass & Weeds | Grass-Weeds (Roboflow) | 2,486 |
| Open Storage of Garbage & Rubbish | TACO + Aerial Dumping + Garbage Object Detection | 13,519 |
| Chipping, Peeling & Flaking Paint | BD3 + Building Surface Defect Detection | 11,318 |
| Damaged Accessory Structures | Broken Fence Detection | 1,297 |
| Missing Address Numbers | House Number Detection (Roboflow) + generative synthesis (planned) | TBD |

For the first four categories, suitable public datasets exist and have been downloaded. The fifth category — Missing Address Numbers — presents a unique challenge: no public dataset captures the specific scenario of a building exterior *lacking* a visible house number. The team plans to address this through a synthetic data construction approach: using the House Number Detection dataset (Roboflow, 493 images of buildings with visible house numbers) as positive samples, then leveraging generative models to produce corresponding negative samples (building facades with house numbers removed or absent). This constructed binary dataset (`address_number_present` / `address_number_missing`) will be integrated into the classification pipeline as an additional class.

The combined corpus currently totals approximately 28,620 images (4.3 GB) across the first four categories, with the Missing Address Numbers class to be added once the synthetic generation pipeline is validated. Images from diverse source formats (COCO, YOLO, raw classification folders) are unified under a consistent label taxonomy of six classes: `overgrown_vegetation`, `trash_debris`, `exterior_deterioration`, `structural_damage`, `damaged_structures`, and `missing_address_number` (pending). A data cleaning audit identified 74 exact duplicates and 100 cross-source perceptual duplicates, which are removed before training. To address a 9.9:1 class imbalance (trash_debris dominates), the overrepresented class is downsampled to approximately 3,000 images and weighted cross-entropy loss is applied during training. All images are resized to the input dimension required by each model backbone, normalized, and augmented using random cropping, horizontal flipping, and color jitter.

### 2) Model Architecture

Rather than the originally proposed ResNet/EfficientNet approach, the team adopted a three-model comparative framework informed by recent advances in vision foundation models:

- **CLIP ViT-B/32 (Zero-Shot Baseline).** OpenAI's Contrastive Language-Image Pretraining model is used as a training-free baseline. Each violation class is described by three hand-written text prompts (e.g., "a photo of overgrown grass and weeds"); image-text cosine similarity produces ranked predictions. This baseline requires no labeled training data and can be demonstrated to the client immediately.

- **DINOv2 ViT-B/14 (Primary Model).** Meta's self-supervised vision transformer produces highly transferable image features, particularly effective when fine-tuning data is limited (Oquab et al., 2024). The team employs a Linear Probing then Full Fine-Tuning (LP-FT) strategy: first training only a linear classification head on frozen DINOv2 features, then unfreezing the full model for end-to-end fine-tuning. This two-stage approach prevents catastrophic forgetting of the pretrained representations.

- **EfficientNetV2-S (Comparison/Fallback).** A well-established CNN architecture included as a controlled comparison against the transformer-based models and as a lightweight deployment fallback. Also trained using the LP-FT strategy via the timm library.

All models are framed as multi-class classification (softmax), not multi-label — based on analysis of the client's workflow, where each inspection photograph typically depicts a single primary violation. The architecture can be extended to multi-label (sigmoid) if future client data reveals that multi-violation images are common.

### 3) Evaluation Framework

Model performance is measured using per-class precision, recall, and F1-score, supplemented by top-3 accuracy (whether the correct violation type appears among the top 3 predictions). Top-3 accuracy is operationally critical: the system is designed so that inspectors review a short ranked list, not a single prediction. The team additionally computes calibration metrics to assess whether the model's confidence scores are reliable — poorly calibrated confidence would undermine inspector trust.

The CLIP zero-shot baseline has already been evaluated: **78.5% top-1 accuracy and 98.8% top-3 accuracy** on a held-out test set of 891 images. This establishes a strong reference point — trained models are expected to improve primarily on the structural damage / exterior deterioration confusion that CLIP's text-only understanding cannot resolve.

Where the Town provides address-level metadata for future evaluation on real data, the team will stratify performance metrics by neighborhood to evaluate for potential disparate impact.

### 4) Prototype Interface

A lightweight prototype will be developed as either a web-based interface or a RESTful API endpoint. The interface will accept an image upload and return the model's top-N predictions with confidence scores and code references. The prototype will require explicit inspector confirmation for each suggestion, and will log all interactions for future training data collection. The prototype is intended as a proof-of-concept rather than a production-grade system.

## 4.4 Feasibility Assessment

### 1) Data Availability

The client's original dataset of approximately 1,000 labeled image-code pairs is unavailable due to the Town's compliance review, which determined that inspection photographs — particularly those captured inside private residences — cannot be transferred outside Town Hall. In response, the team constructed a proxy training corpus of 28,620 images from seven publicly available datasets covering four of the five most common violation categories. While proxy data cannot perfectly replicate the visual characteristics of Riverdale Park's actual inspection photographs (e.g., camera angles, lighting conditions, local building styles), it is sufficient to validate the classification pipeline and demonstrate technical feasibility. The system is designed so that the Town can retrain on their own data in the future by simply replacing the training images without modifying the model architecture or evaluation framework.

### 2) Technical Resources

The team's Technical Lead has professional AI engineering experience at a large technology company, providing the technical foundation necessary for model development. All training and evaluation run locally on an Apple M4 Mac with 24 GB unified memory using PyTorch with MPS (Metal Performance Shaders) acceleration — no cloud GPU or university cluster access is required. The primary open-source tools used are PyTorch, the timm model library, OpenCLIP, and scikit-learn, all freely available and well-documented.

### 3) Organizational Capacity

Ryan Chelton expressed openness to a working prototype as the ideal deliverable, while noting that a thoroughly documented system would also be valuable. The pivot to proxy data, while not ideal, was discussed with the client and does not diminish the project's value — the pipeline, evaluation framework, and responsible AI analysis remain directly applicable once the Town is ready to train on their own data. The client's willingness to provide the Town's public violation code documentation has been essential to ensuring the proxy datasets are mapped to realistic enforcement categories.

### 4) Timeline Realism

The proxy dataset strategy has partially decoupled the team's technical work from the data access timeline that originally represented the highest scheduling risk. The CLIP zero-shot baseline is already complete with strong initial results (78.5% top-1 accuracy). DINOv2 and EfficientNetV2 training runs are in progress and expected to complete within the current project phase. The remaining work — model comparison analysis, fairness evaluation, prototype development, and documentation — is on track for delivery within the semester.

### 5) Constraints and Limitations

The primary constraints are: (1) the proxy datasets, while carefully selected, do not perfectly replicate the visual conditions of Riverdale Park inspection photographs, meaning model performance on real data may differ; (2) the Missing Address Numbers category requires synthetic data construction via generative models, which introduces an additional validation step to ensure the generated images are realistic enough for effective training; (3) the class imbalance in available proxy data (trash/debris images outnumber structural damage images by nearly 10:1) requires careful mitigation through downsampling and loss weighting; and (4) the system's demonstrated accuracy is based on proxy data evaluation and will need to be re-validated when the Town is ready to run on their own images.
