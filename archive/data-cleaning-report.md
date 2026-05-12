# Data Cleaning Report

> Automated audit of 28,620 images across 7 proxy datasets.
> Generated 2026-04-07

---

## TL;DR — How to Reproduce

### 1. Download the datasets

**HuggingFace** (no account needed):

```bash
pip install huggingface_hub

# Garbage Object Detection (187 MB, 10,464 images)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='keremberke/garbage-object-detection', repo_type='dataset', local_dir='FinalProject/data/raw/garbage-object-detection')"
# Then extract: cd FinalProject/data/raw/garbage-object-detection/data && unzip train.zip -d ../train && unzip valid.zip -d ../valid && unzip test.zip -d ../test

# Building Surface Defect Detection (388 MB, 7,353 images)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='xueaidezhouzhou/buildingsurfacedefectdetection', repo_type='dataset', local_dir='FinalProject/data/raw/building-surface-defect')"
```

**Roboflow** (free account required for API key):

```bash
pip install roboflow

# Grass-Weeds (115 MB, 2,486 images)
# Aerial Dumping (291 MB, 1,555 images)
# Broken Fence Detection (99 MB, 1,297 images)
# → Download from Roboflow Universe in COCO format, place under FinalProject/data/raw/<dataset-name>/
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
rf.workspace('uji-thesis').project('broken-fence-detection').version(1).download('coco', location='FinalProject/data/raw/broken-fence')
"
```

**Other**:

- **BD3** (108 MB, 3,965 images) — Clone from https://github.com/Praveenkottari/BD3-Dataset
- **TACO** (2.6 GB, 1,500 images) — Download from http://tacodataset.org/ or `python FinalProject/data/raw/TACO/download.py`

### 2. Run the cleaning audit

```bash
python FinalProject/scripts/data_cleaning.py
```

Outputs:
- `FinalProject/data-cleaning-report.md` — this report
- `FinalProject/data/cleaning_results.json` — raw per-image results (for further analysis)

### 3. Expected directory structure

```
FinalProject/data/raw/
├── BD3/BD3_original_dataset/train/{algae,major_crack,...}/   (3,965 imgs)
├── TACO/data/{batch_1,...}/                                   (1,500 imgs)
├── grass-weeds/{train,valid,test}/                            (2,486 imgs)
├── aerial-dumping/{train,valid,test}/                         (1,555 imgs)
├── broken-fence/{train,valid,test}/                           (1,297 imgs)
├── garbage-object-detection/{train,valid,test}/               (10,464 imgs)
└── building-surface-defect/images/{train,val,test}/           (7,353 imgs)
Total: ~28,620 images, ~4.3 GB
```

---

## 1. Summary

| Metric | Count | % |
|--------|-------|---|
| Total images scanned | 28620 | 100% |
| Clean (no issues) | 28601 | 99.9% |
| Corrupt / unreadable | 0 | 0.0% |
| Tiny (< 32x32) | 0 | 0.0% |
| Extreme aspect ratio (> 5:1) | 0 | 0.0% |
| Low variance (blank/solid) | 19 | 0.1% |
| Exact duplicates (MD5) | 74 dupes in 39 groups | — |
| Cross-source perceptual dupes | 100 groups | — |

## 2. Per-Dataset Breakdown

| Dataset | Images | Issues | Resolution (median) | File Size (median) |
|---------|--------|--------|--------------------|--------------------|
| BD3 | 3965 | 12 | 512x512 | 20 KB |
| TACO | 1500 | 0 | 2921x3264 | 1694 KB |
| aerial-dumping | 1555 | 0 | 1024x1024 | 195 KB |
| broken-fence | 1297 | 0 | 640x640 | 74 KB |
| building-surface-defect | 7353 | 2 | 640x640 | 54 KB |
| garbage-object-detection | 10464 | 5 | 416x416 | 16 KB |
| grass-weeds | 2486 | 0 | 416x416 | 44 KB |

## 3. Resolution Distribution

### BD3
- Width:  min=512, median=512, max=512
- Height: min=512, median=512, max=512
- Unique resolutions: 1
- Top resolutions: 512x512 (3965)

### TACO
- Width:  min=842, median=2921, max=6000
- Height: min=474, median=3264, max=4618
- Unique resolutions: 53
- Top resolutions: 2448x3264 (490), 3264x2448 (280), 1824x4000 (105), 4032x3024 (101), 3120x4160 (84)

### aerial-dumping
- Width:  min=1024, median=1024, max=1024
- Height: min=1024, median=1024, max=1024
- Unique resolutions: 1
- Top resolutions: 1024x1024 (1555)

### broken-fence
- Width:  min=640, median=640, max=640
- Height: min=640, median=640, max=640
- Unique resolutions: 1
- Top resolutions: 640x640 (1297)

### building-surface-defect
- Width:  min=640, median=640, max=640
- Height: min=640, median=640, max=640
- Unique resolutions: 1
- Top resolutions: 640x640 (7353)

### garbage-object-detection
- Width:  min=416, median=416, max=416
- Height: min=416, median=416, max=416
- Unique resolutions: 1
- Top resolutions: 416x416 (10464)

### grass-weeds
- Width:  min=416, median=416, max=416
- Height: min=416, median=416, max=416
- Unique resolutions: 1
- Top resolutions: 416x416 (2486)

## 4. Label Distribution

| Label | Count | % |
|-------|-------|---|
| trash_debris | 11964 | 41.8% |
| exterior_deterioration | 9514 | 33.2% |
| overgrown_vegetation | 2486 | 8.7% |
| illegal_dumping | 1555 | 5.4% |
| damaged_structures | 1297 | 4.5% |
| structural_damage | 1204 | 4.2% |
| __excluded__ | 600 | 2.1% |

**Imbalance ratio**: 9.9:1 (largest class / smallest class)

> **Action plan**: Downsample `trash_debris` from 11,964 to ~2,500–3,000 images before training. This brings the imbalance ratio to ~2.5:1, consistent with our original 4-dataset setup. We will randomly sample from TACO and Garbage Object Detection proportionally. Additionally, use weighted cross-entropy loss during training to further compensate.
>
> After downsampling, expected distribution:
>
> | Label | Before | After (est.) |
> |-------|--------|-------------|
> | trash_debris | 11,964 | ~3,000 |
> | exterior_deterioration | 9,514 | 9,514 |
> | overgrown_vegetation | 2,486 | 2,486 |
> | illegal_dumping | 1,555 | 1,555 |
> | damaged_structures | 1,297 | 1,297 |
> | structural_damage | 1,204 | 1,204 |
> | **Total** | **28,020** | **~19,056** |

## 5. Flagged Images

### Low Variance (19)

- `cls01_531.jpg` (BD3): ['low_variance: std=2.6']
- `cls02_357.jpg` (BD3): ['low_variance: std=2.1']
- `cls02_397.jpg` (BD3): ['low_variance: std=2.8']
- `cls02_473.jpg` (BD3): ['low_variance: std=1.7']
- `cls02_594.jpg` (BD3): ['low_variance: std=2.7']
- `cls03_385.jpg` (BD3): ['low_variance: std=2.8']
- `cls04_215.jpg` (BD3): ['low_variance: std=2.5']
- `cls04_373.jpg` (BD3): ['low_variance: std=2.7']
- `cls04_383.jpg` (BD3): ['low_variance: std=2.7']
- `cls04_428.jpg` (BD3): ['low_variance: std=2.9']
- `cls06_049.jpg` (BD3): ['low_variance: std=2.8']
- `cls06_472.jpg` (BD3): ['low_variance: std=2.8']
- `cardboard1395_jpg.rf.667183ae96295c26c3fe3ac5b87cdf9c.jpg` (garbage-object-detection): ['low_variance: std=2.8']
- `cardboard1499_jpg.rf.f1f596893f7862fa538abc0c10559957.jpg` (garbage-object-detection): ['low_variance: std=2.9']
- `cardboard1610_jpg.rf.5644b06156abb6f6b35b2d9f37a1d90f.jpg` (garbage-object-detection): ['low_variance: std=1.8']
- `paper1557_jpeg.rf.928aca22684b12e9adc9120ee513d319.jpg` (garbage-object-detection): ['low_variance: std=2.5']
- `cardboard847_jpg.rf.6dc31dc2203134005538fff0c506b477.jpg` (garbage-object-detection): ['low_variance: std=2.5']
- `images_6276.jpg` (building-surface-defect): ['low_variance: std=2.8']
- `images_6753.jpg` (building-surface-defect): ['low_variance: std=2.8']

### Exact Duplicates (39 groups)

- Group 1 (3 files): cls00_096.jpg, cls00_133.jpg, cls00_492.jpg
- Group 2 (2 files): cls01_013.jpg, cls02_063.jpg
- Group 3 (2 files): cls01_015.jpg, cls01_567.jpg
- Group 4 (7 files): cls01_048.jpg, cls02_034.jpg, cls02_271.jpg, cls02_342.jpg, cls02_366.jpg
- Group 5 (2 files): cls01_477.jpg, cls02_596.jpg
- Group 6 (5 files): cls02_005.jpg, cls02_073.jpg, cls02_565.jpg, cls02_598.jpg, cls02_601.jpg
- Group 7 (2 files): cls02_019.jpg, cls02_053.jpg
- Group 8 (5 files): cls02_022.jpg, cls02_080.jpg, cls02_211.jpg, cls02_338.jpg, cls02_343.jpg
- Group 9 (7 files): cls02_033.jpg, cls02_039.jpg, cls02_186.jpg, cls02_237.jpg, cls02_281.jpg
- Group 10 (4 files): cls02_040.jpg, cls02_500.jpg, cls02_607.jpg, cls02_614.jpg
- ... and 29 more groups

### Cross-Source Perceptual Duplicates (100 groups)

- Group 1: cls00_330.jpg (BD3), cls06_388.jpg (BD3), glass576_jpg.rf.eabfd98da23cdc0f33ca2c5d7983f141.jpg (garbage-object-detection), glass91_jpg.rf.fcedca58135402a41b7dd5d75b1c78ea.jpg (garbage-object-detection), paper977_jpg.rf.9e73fd72a5d50f67f428ab712daa0a64.jpg (garbage-object-detection)
- Group 2: cls00_422.jpg (BD3), images_3450.jpg (building-surface-defect)
- Group 3: cls00_493.jpg (BD3), images_1945.jpg (building-surface-defect)
- Group 4: cls00_532.jpg (BD3), images_486.jpg (building-surface-defect)
- Group 5: cls01_030.jpg (BD3), cls02_578.jpg (BD3), cls03_331.jpg (BD3), cls05_132.jpg (BD3), images_1431.jpg (building-surface-defect)
- Group 6: cls01_049.jpg (BD3), images_4023.jpg (building-surface-defect)
- Group 7: cls01_101.jpg (BD3), IMG_4876.JPG (TACO), images_6946.jpg (building-surface-defect)
- Group 8: cls01_240.jpg (BD3), cls01_321.jpg (BD3), cls01_462.jpg (BD3), cls02_421.jpg (BD3), cls04_037.jpg (BD3)
- Group 9: cls01_250.jpg (BD3), cls02_047.jpg (BD3), images_3739.jpg (building-surface-defect), images_3740.jpg (building-surface-defect)
- Group 10: cls01_286.jpg (BD3), cls01_546.jpg (BD3), cls03_327.jpg (BD3), Capture-du-2022-03-04-11-31-04_0_jpg.rf.c6b884f91359a6332d226527a64375ad.jpg (broken-fence), Capture-du-2022-03-04-11-31-04_1_jpg.rf.7358432dd99304e20fdce144b9e0d4a6.jpg (broken-fence)
- ... and 90 more groups

## 6. Recommendations

- **Review 19 low-variance images** — likely blank/solid, add noise or remove
- **Deduplicate 74 exact duplicates** — keeping one copy per group
- **Review 100 cross-source perceptual duplicates** — same image may appear in multiple datasets
- **Address class imbalance (critical)** — `trash_debris` has 11,964 imgs vs `structural_damage` with 1,204 (9.9:1 ratio). **Plan: downsample `trash_debris` to ~3,000** by random sampling from TACO + Garbage Object Detection, plus use weighted cross-entropy loss during training
- **Standardize resolution** — datasets range from 416×416 to 3,264×2,921; `RandomResizedCrop(224)` in the augmentation pipeline normalizes this at training time
