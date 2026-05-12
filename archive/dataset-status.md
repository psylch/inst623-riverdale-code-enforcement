# Dataset Status & Supplementary Sources

> Based on Riverdale Park's official Common Violations list + public data source research. Updated 2026-04-06.

---

## 1. Requirements: Top 5 Official Violation Types

Source: Riverdale Park Development Services Common Violations (see `context/Riverdale_Park_Public_Info.docx.pdf` Section 4)

| # | Violation Type | Description | Visual Identifiability |
|---|---------------|-------------|----------------------|
| 1 | Overgrown Grass & Weeds | Grass/weeds exceeding 10 inches in height | High |
| 2 | Open Storage of Garbage & Rubbish | Accumulation of garbage in yard or exterior areas | High |
| 3 | Chipping, Peeling & Flaking Paint | Deteriorated paint on exterior surfaces | Medium-High |
| 4 | Damaged Accessory Structures | Fences, garages, sheds, retaining walls in disrepair | Medium |
| 5 | Missing Address Numbers | House number absent, illegible, or non-compliant (min. 4-inch, contrasting color) | Medium |

Additional visually identifiable categories from the Town Code (lower priority):
- Ch. 66: Walls, Fences and Hedges
- Ch. 68: Vehicles - Permit Parking (vehicles on unimproved surfaces)

---

## 2. Current Proxy Datasets (already in technical-plan.md)

| Dataset | Size | Source | Violation Coverage | Format |
|---------|------|--------|-------------------|--------|
| BD3 | 3,965 imgs | [GitHub](https://github.com/Praveenkottari/BD3-Dataset) | Building surface defects (cracks, peeling, stains) → #3 Paint | Classification |
| TACO | 1,500 imgs | [tacodataset.org](http://tacodataset.org/) | Outdoor litter and debris → #2 Garbage | COCO |
| Aerial Dumping | 1,555 imgs | [Roboflow](https://universe.roboflow.com/object-detection-of-illegal-dumping-sites/aerial-dumping-sites) | Illegal dumping → #2 Garbage | YOLO/COCO |
| Grass-Weeds | 2,486 imgs | [Roboflow](https://universe.roboflow.com/roboflow-100/grass-weeds) | Overgrown vegetation → #1 Overgrown | YOLO/COCO |

**Current total: ~9,500 images covering 3 out of 5 official violation types**

### Coverage Gaps

| Official Violation Type | Status |
|------------------------|--------|
| #1 Overgrown Grass & Weeds | ✅ Grass-Weeds |
| #2 Garbage & Rubbish | ✅ TACO + Aerial Dumping |
| #3 Peeling & Flaking Paint | ✅ BD3 |
| #4 Damaged Accessory Structures | ❌ Not covered |
| #5 Missing Address Numbers | ❌ Not covered |

---

## 3. Supplementary Datasets

### 3.1 Filling the Gap: Damaged Accessory Structures (#4)

| Dataset | Size | Source | Classes | Recommendation |
|---------|------|--------|---------|---------------|
| **Broken Fence Detection** | ~2,200 imgs | [Roboflow](https://universe.roboflow.com/uji-thesis/broken-fence-detection) | Broken fence detection | ⭐⭐⭐ Most directly relevant |
| **Damaged Constructions** | ~84 MB, 5 classes | [Kaggle](https://www.kaggle.com/datasets/turkertuncer/damaged-constructions-image-dataset) | Debris, damaged building, damaged highway, non-damaged | ⭐⭐ Supplementary |
| Francesco/wall-damage | 461 imgs | [HuggingFace](https://huggingface.co/datasets/Francesco/wall-damage) | Wall damage with 4 severity levels | ⭐ Small but precise |

**Recommendation**: Download Broken Fence Detection as the primary source; use Damaged Constructions to supplement shed/garage damage scenarios. Label mapping:

```
Broken Fence Detection                              → damaged_structures
Damaged Constructions "debris" + "damaged building"  → damaged_structures
```

### 3.2 Filling the Gap: Missing Address Numbers (#5)

No dedicated "missing house number" dataset exists. We propose a **reverse-construction approach using SVHN**:

| Dataset | Size | Source | Role |
|---------|------|--------|------|
| **SVHN** (Stanford) | 630K+ imgs | [Stanford](http://ufldl.stanford.edu/housenumbers/) / [HuggingFace](https://huggingface.co/datasets/ufldl-stanford/svhn) | Positive samples (number present) |
| House Number Detection | 493 imgs | [Roboflow](https://universe.roboflow.com) (EMRULLAH KARACA) | Positive samples with building context |
| House Number Instance Seg | 606 imgs | [Roboflow](https://universe.roboflow.com) (SA-Co Gold) | Positive samples with segmentation masks |

**Construction approaches**:

1. **Approach A — Binary classifier**: Use SVHN full-format images (number visible = positive) + cropped plain wall regions (no number = negative) → train a `has_number / missing_number` classifier
2. **Approach B — Detection + threshold**: Train a YOLO/detection head on House Number Detection data; confidence below threshold → classify as missing
3. **Recommended: Approach A** — simpler, consistent with our whole-image classification pipeline

**Caveat**: SVHN images are captured from street-level views, which differ from an inspector's close-range photos of building exteriors. Effectiveness may need validation with real client data. Priority should be lower than #4.

### 3.3 Strengthening Existing Categories

Datasets mentioned in `research-for-teammates.md` but not yet included in the plan:

| Dataset | Size | Source | Maps to | Value |
|---------|------|--------|---------|-------|
| **Building Surface Defect Detection** | 7,354 imgs | [HuggingFace](https://huggingface.co/datasets/xueaidezhouzhou/buildingsurfacedefectdetection) | #3 Paint (cracks, spalling, rust) | 2x larger than BD3; includes indoor & outdoor |
| **Garbage Object Detection** | 10,464 imgs | [HuggingFace](https://huggingface.co/datasets/keremberke/garbage-object-detection) | #2 Garbage | 7x larger than TACO |
| **SDNET2018** | ~56,000 imgs | [Kaggle](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images) | #3/#4 (cracked/not cracked) | Large-scale; useful for pretraining or data mixing |

### 3.4 Low Priority: Vehicle on Unimproved Surface

No precise match exists among public datasets. Closest available:

| Dataset | Size | Source | Notes |
|---------|------|--------|-------|
| Illegal Parking | 2,752 imgs | [Roboflow](https://universe.roboflow.com/thesis-gsgcj/illegal-parking-qrvua) | General parking violations, not "parked on grass" |
| Istanbul Sidewalk Parking | 1K-10K imgs | [HuggingFace](https://huggingface.co/datasets/eneskarabulut/istanbul-sidewalk-parking-detection) | "Parked on wrong surface" — closest conceptual match |

**Conclusion**: Defer to real client data. Not worth investing in proxy data for this category.

---

## 4. Grok/X Research Findings (2026-04-06)

Searched X/Twitter via Grok for discussions and public platform releases. Key findings:

- **Zero relevant X/Twitter discussions**: The intersection of "municipal code enforcement" and "image dataset" has no public discourse on social media
- **No new 2024–2025 dedicated datasets on Roboflow/HuggingFace/Kaggle**: Public data for this domain remains scarce
- **All industry data is proprietary**: CityDetect (PASS AI) has 199K+ images / 39K+ parcels covering 100+ violation indicators, but none are public. NYU capstone project and IWDD@WACV 2026 competition data are also closed
- **Potential unlabeled source**: Mapillary Vistas (open street-level imagery) could serve as a future self-labeling candidate, but ROI is low at this stage

**Conclusion: The proxy datasets we've identified are the best publicly available options.**

---

## 5. Updated Dataset Overview

### Recommended Download List

| Priority | Dataset | Images | Disk Size | Violation Category | Status |
|----------|---------|--------|-----------|-------------------|--------|
| P0 | BD3 | 3,965 | 108 MB | #3 Peeling Paint | Downloaded |
| P0 | TACO | 1,500 | 2.6 GB | #2 Garbage | Downloaded |
| P0 | Aerial Dumping | 1,555 | 291 MB | #2 Garbage | Downloaded |
| P0 | Grass-Weeds | 2,486 | 115 MB | #1 Overgrown | Downloaded |
| P1 | Broken Fence Detection | 1,297 | 99 MB | #4 Damaged Structures | Downloaded |
| P1 | Garbage Object Detection | 10,464 | 409 MB | #2 Garbage (reinforcement) | Downloaded |
| P1 | Building Surface Defect | 7,353 | 865 MB | #3 Paint (reinforcement) | Downloaded |
| P2 | Damaged Constructions | ~500 | ~84 MB | #4 Damaged Structures (supplement) | Not downloaded |
| P2 | SVHN (full-format) | 630K+ | ~600 MB | #5 Missing Numbers (constructed) | Not downloaded |
| P3 | SDNET2018 | 56,000 | ~2 GB | Pretraining / data mixing | Not downloaded |
| | **Total (downloaded)** | **28,620** | **4.3 GB** | | |

### Unified Label Mapping (Updated)

```
# Original mappings
BD3 crack classes                → structural_damage
BD3 peeling/spalling             → exterior_deterioration
TACO (all classes)               → trash_debris
Grass-Weeds                      → overgrown_vegetation
Aerial Dumping                   → illegal_dumping

# New mappings
Broken Fence Detection           → damaged_structures
Damaged Constructions (damaged)  → damaged_structures
Building Surface Defect (all)    → exterior_deterioration
Garbage Object Detection (all)   → trash_debris
SVHN positive/negative           → address_number_present / address_number_missing
```

### Before vs. After

| | Original Plan | After Supplement |
|--|--------------|-----------------|
| Number of datasets | 4 | 7 (downloaded) + 2 optional |
| Total images | ~9,500 | 28,620 (4.3 GB) |
| Violation types covered | 3/5 | 4/5 |
| Largest remaining gap | Damaged Structures, Missing Numbers | Missing Address Numbers + Vehicle on Unimproved Surface (deferred to real data) |

### Class Imbalance Warning

After merging, `trash_debris` dominates at 11,964 images (41.8%) while `structural_damage` has only 1,204 (4.2%) — a **9.9:1 imbalance ratio**. This was only 2.1:1 before adding Garbage Object Detection.

**Mitigation plan**: Downsample `trash_debris` to ~3,000 images before training (random sample from TACO + Garbage Object Detection), bringing the ratio down to ~2.5:1. Additionally use weighted cross-entropy loss during training.

See `data-cleaning-report.md` for the full audit results.
