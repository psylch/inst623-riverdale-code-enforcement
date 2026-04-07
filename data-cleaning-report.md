# Data Cleaning Report

> Automated audit of 9506 images across 4 proxy datasets.
> Generated 2026-04-07 12:44

---

## 1. Summary

| Metric | Count | % |
|--------|-------|---|
| Total images scanned | 9506 | 100% |
| Clean (no issues) | 9494 | 99.9% |
| Corrupt / unreadable | 0 | 0.0% |
| Tiny (< 32x32) | 0 | 0.0% |
| Extreme aspect ratio (> 5:1) | 0 | 0.0% |
| Low variance (blank/solid) | 12 | 0.1% |
| Exact duplicates (MD5) | 70 dupes in 35 groups | — |
| Cross-source perceptual dupes | 12 groups | — |

## 2. Per-Dataset Breakdown

| Dataset | Images | Issues | Resolution (median) | File Size (median) |
|---------|--------|--------|--------------------|--------------------|
| BD3 | 3965 | 12 | 512x512 | 20 KB |
| TACO | 1500 | 0 | 2921x3264 | 1694 KB |
| aerial-dumping | 1555 | 0 | 1024x1024 | 195 KB |
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

### grass-weeds
- Width:  min=416, median=416, max=416
- Height: min=416, median=416, max=416
- Unique resolutions: 1
- Top resolutions: 416x416 (2486)

## 4. Label Distribution

| Label | Count | % |
|-------|-------|---|
| overgrown_vegetation | 2486 | 26.2% |
| exterior_deterioration | 2161 | 22.7% |
| illegal_dumping | 1555 | 16.4% |
| trash_debris | 1500 | 15.8% |
| structural_damage | 1204 | 12.7% |
| __excluded__ | 600 | 6.3% |

**Imbalance ratio**: 2.1:1 (largest class / smallest class)

## 5. Flagged Images

### Low Variance (12)

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

### Exact Duplicates (35 groups)

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
- ... and 25 more groups

### Cross-Source Perceptual Duplicates (12 groups)

- Group 1: cls01_101.jpg (BD3), IMG_4876.JPG (TACO)
- Group 2: cls01_240.jpg (BD3), cls01_321.jpg (BD3), cls01_462.jpg (BD3), cls02_421.jpg (BD3), cls04_037.jpg (BD3)
- Group 3: cls01_457.jpg (BD3), cls01_503.jpg (BD3), 000013.jpg (TACO)
- Group 4: cls01_504.jpg (BD3), cls05_439.jpg (BD3), ridderzuring_2303_jpg.rf.896c614647362e96f56e7af404292b5e.jpg (grass-weeds)
- Group 5: cls01_558.jpg (BD3), 000075.jpg (TACO)
- Group 6: cls01_565.jpg (BD3), cls02_402.jpg (BD3), ridderzuring_1984_jpg.rf.dead13e5619a9e94c4d6eab4aaea4aa4.jpg (grass-weeds)
- Group 7: cls02_120.jpg (BD3), 000040.JPG (TACO)
- Group 8: cls04_081.jpg (BD3), ridderzuring_0216_jpg.rf.c878986dee0d984315a8aeebd4ac8bdc.jpg (grass-weeds)
- Group 9: cls04_157.jpg (BD3), 180_jpg.rf.56d70cd4da879344024a722a0d32328c.jpg (aerial-dumping)
- Group 10: cls06_035.jpg (BD3), 000087.jpg (TACO)
- ... and 2 more groups

## 6. Recommendations

- **Review 12 low-variance images** — likely blank/solid, add noise or remove
- **Deduplicate 70 exact duplicates** — keeping one copy per group
- **Review 12 cross-source perceptual duplicates** — same image may appear in multiple datasets
- **Standardize resolution** — datasets have mixed resolutions; `RandomResizedCrop(224)` in augmentation pipeline handles this at training time
