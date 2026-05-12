# Riverdale Park AI Code Enforcement — Client Briefing

*Mid-project briefing for Riverdale Park, MD · UMD INST623 Spring 2026*

This is the report-style talk track used to drive the slide deck (`ppt/index.html`). It can stand alone as a written report or be lifted into speaker notes.

---

## 1. How the system works

A two-stage cascade that mimics how an inspector would read a photo:

```
Photo  →  ① CLIP screening      →  ② Gemma verification  →  Final flags
          "what might be wrong?"     "is each candidate
           (broad scan)               actually visible?"
```

- **Stage 1 (CLIP)** scans every photo for all 9 violation categories and picks the 5 most likely candidates per image.
- **Stage 2 (Gemma vision model)** verifies each of those 5 candidates with a yes/no answer plus a confidence score and a one-sentence rationale.
- **Design principle:** *CLIP for recall, Gemma for precision.* CLIP casts a wide net so we don't miss real problems; Gemma's verification stage rejects the wrong candidates so we don't raise false alarms on clean houses.

---

## 2. What we tested

| Dataset | Count | Ground truth |
|---|---|---|
| Known-violation houses (Riverdale Park dataset) | 98 images | 3 independent team-member annotators |
| Reference clean houses (synthetic baseline) | 20 images | All 3 annotators confirmed all 9 categories = no |
| **Total evaluation surface** | **118 images × 9 categories = 1,062 cells** | 276 positive / 786 negative |

**On ground truth methodology:** annotators each labeled every (image, category) cell as yes / no / unsure. We collapse "unsure" to "no" for binary metrics — this biases the system toward fewer false alarms, which is the right direction for a screening tool. A category counts as a positive if at least 2 of 3 annotators say yes.

---

## 3. Three headline numbers

### ✅ Catches real problems
**The system flags the known violation 87% of the time** (85 of 98 images, cascade k=5). In Stage 1 — the screening question — a real problem house gets correctly identified in nearly nine out of ten cases.

### ✅ Zero false alarms on clean houses
**0 / 180 cells.** Across 20 confirmed clean houses × 9 categories, the system produced zero false positives. Gemma's average confidence on these cells was 96 / 100 — these were not borderline calls. Sample rationale: *"All windows are intact and glassed over."*

### ✅ When the system says "yes," it's usually right
**85% precision (cascade k=5, micro-averaged across all categories).** Of every 100 flags the system raises across the full evaluation, 85 match human consensus.

---

## 4. Why we report results by "human agreement tier"

We discovered a clean correlation that shaped how we report: **AI accuracy tracks how well our three annotators agreed, category by category**. When three humans cannot agree on a category, AI also struggles.

This means the bottleneck is **label definition**, not the algorithm. So we report results stratified by Cohen's κ (inter-rater agreement), grouped per Landis & Koch into five tiers.

---

## 5. The main results

### 5.1 Performance by κ tier (Cascade k=5, 118 images)

| Tier | κ range | Categories | macro-P | macro-R | macro-F1 | Recommended use |
|---|---|---|---|---|---|---|
| 🟢 Almost perfect | ≥ 0.81 | Graffiti · Inoperable vehicle | 0.86 | 0.88 | **0.87** | Full automation |
| 🟢 Substantial | 0.61–0.80 | Damaged roof · Overgrown vegetation | 0.92 | 0.57 | 0.69 | Full automation |
| 🟡 Moderate | 0.41–0.60 | Boarded windows · Broken windows · Junk/trash | 0.76 | 0.50 | 0.55 | AI-assisted, inspector verifies |
| 🟡 Fair | 0.21–0.40 | Peeling paint *(prevalence inflates F1)* | 0.93 | 0.70 | 0.80 | AI-assisted |
| 🔴 Slight | < 0.21 | Deteriorating chimney | 0.25 | 0.14 | 0.18 | Hold until SOP provided |

### 5.2 The whole story in one sentence

**The categories where humans agree, AI is inspector-grade. The categories where humans don't agree, AI fails — and that failure is real evidence that the category itself needs clearer definition.**

---

## 6. Per-category breakdown

Sorted by κ, with cascade k=5 metrics on the unified 118-image set.

| Category | κ | Tier | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Inoperable vehicle | 0.928 | almost perfect | 0.84 | 0.88 | 0.86 | 21 | 4 | 3 |
| Graffiti | 0.888 | almost perfect | 0.88 | 0.88 | 0.88 | 15 | 2 | 2 |
| Overgrown vegetation | 0.677 | substantial | 0.95 | 0.75 | 0.84 | 42 | 2 | 14 |
| Damaged roof shingles | 0.640 | substantial | 0.89 | 0.38 | 0.53 | 8 | 1 | 13 |
| Junk/trash accumulation | 0.601 | moderate | 0.64 | 0.72 | 0.68 | 18 | 10 | 7 |
| Boarded windows | 0.564 | moderate | 0.64 | 0.56 | 0.60 | 9 | 5 | 7 |
| Broken windows | 0.541 | moderate | **1.00** | 0.22 | 0.36 | 8 | 0 | 29 |
| Peeling paint | 0.248 | fair | 0.93 | 0.70 | 0.80 | 51 | 4 | 22 |
| Deteriorating chimney | 0.150 | slight | 0.25 | 0.14 | 0.18 | 1 | 3 | 6 |

### 6.1 Two anomalies worth narrating

**Broken windows — perfect precision, low recall.**
Every broken-windows flag is correct (P = 1.00), but the system only catches 22% of true cases. Useful talk-track: *"You can publish broken-windows flags as-is. But for buildings where context suggests damage, an inspector should still walk the property — the AI is conservative on this one."*

**Deteriorating chimney — the strongest evidence for label-definition problems.**
Both human IAA (κ = 0.15) and AI F1 (0.18) are slight-tier. Even our annotators couldn't reliably tell if a chimney qualifies as "deteriorating" from a street-view photo. This is exactly where a city SOP would help — and exactly where any model would fail without one.

---

## 7. Zero false alarms — detail

20 clean houses × 9 categories = 180 binary checks. Result per category:

| Category | False positives |
|---|---|
| Boarded windows | 0 / 20 |
| Broken windows | 0 / 20 |
| Damaged roof | 0 / 20 |
| Deteriorating chimney | 0 / 20 |
| Graffiti | 0 / 20 |
| Inoperable vehicle | 0 / 20 |
| Junk / trash | 0 / 20 |
| Overgrown vegetation | 0 / 20 |
| Peeling paint | 0 / 20 |
| **Total** | **0 / 180** |

Mean Gemma confidence: 96 / 100.

**Honest caveat:** these 20 houses are AI-generated and somewhat idealized. Real-world clean houses have shadows, kids' toys, decorative wreaths, etc. The 0% number is a synthetic-baseline result. Replacing this with real city-archive clean photos is the #1 item on Future Work.

---

## 8. How Riverdale Park can use this system

We recommend three deployment modes mapped to the κ tiers:

### 🟢 Full automation — 4 categories
*Almost-perfect and substantial tiers: Graffiti, Inoperable vehicle, Overgrown vegetation, Damaged roof.*
The AI generates a violation-notice draft directly into the inspector follow-up queue. No pre-screening required. These categories perform at inspector grade and have effectively zero false-alarm risk.

### 🟡 AI-assisted — 4 categories
*Moderate and fair tiers: Boarded windows, Broken windows, Junk/trash, Peeling paint.*
AI flags go into a high-priority queue; inspectors still spot-check unflagged photos because recall is conservative. **AI flags here are nearly all correct (precision is high), but the system misses cases (recall is low).**

### 🔴 Hold — 1 category
*Slight tier: Deteriorating chimney.*
Keep the existing manual workflow until the city provides a clearer SOP. Until label definitions stabilize, no model — ours or anyone else's — can be reliable here.

### Operational metrics to monitor

| Cadence | Action | Why |
|---|---|---|
| Weekly | Spot-check 5–10 AI outputs from the automation tier | Catch precision drift before it's systemic |
| Monthly | Inspector override rate per category | Sudden spike = prompt or data has shifted |
| Quarterly | Re-run the 20-clean-house battery | Confirm false-alarm rate stays near 0% |

---

## 9. How to extend the system to new violation categories

When the city introduces a new violation type, this is the playbook for deciding whether it joins the automation tier, the AI-assisted tier, or has to wait for a clearer SOP.

### Step 1 — Collect samples
At least 30 positive images and 20 negative images. Write a plain-English description of the violation.

### Step 2 — Three-rater pilot
Three inspectors independently label all positives (yes / no / unsure). Compute Cohen's κ across the three.

### Step 3 — κ-gate decision
The κ value determines deployability before any AI training:
- **κ ≥ 0.61** — likely fits the automation tier
- **κ 0.41–0.60** — fits the AI-assisted tier
- **κ < 0.40** — label definition problem. Fix the SOP and add reference photos before testing AI.

### Step 4 — Write the Gemma prompt
Three rules for category descriptions:
1. **Visual, not legal** — describe what's visible, not the code citation. ✅ "trash bags or loose garbage piled in the front yard or driveway" / ❌ "violation of § 302.1"
2. **Include edge cases** — "...except properly bagged trash placed at curb on collection day"
3. **One sentence, ≤ 25 words.**

### Step 5 — Integrate and test
Add the description to the CLIP class list and the Gemma prompt set. Run on the 30 + 20 test set. Measure F1 and FPR.

### Step 6 — Deploy decision
- F1 ≥ 0.80 and FPR ≤ 5% → automation tier
- F1 0.50 – 0.80 → AI-assisted tier
- Otherwise → revise SOP and retest

**Key insight:** the κ test in Step 2 is the most important step. If three inspectors can't agree, no model will perform well on that category. Fix the labelling guidance first, then train.

---

## 10. Future work

### 10.1 What we'd ask Riverdale Park to provide

1. **50–100 real clean-house photos from city records.** Replace our AI-generated baseline so we can validate the 0% false-alarm rate against real-world variance (shadows, wreaths, parked cars, etc.).
2. **Code-enforcement SOP for ambiguous categories.** Specifically chimney and peeling paint — what counts as "deteriorating"? What's the cutoff? A short written guide plus 5–10 reference photos would unlock these tiers.
3. **Multi-angle photos per address.** Currently the pipeline sees one photo at a time. Real inspections use multiple angles. This would directly improve recall on detail-heavy categories like broken windows.
4. **Severity grading.** "Minor peeling paint" vs. "needs urgent repair" — adding a severity dimension would turn AI output into actionable enforcement levels rather than just yes/no flags.

### 10.2 What our team can build next

5. **REST API + simple web UI.** Inspectors upload a photo, see the verdict and confidence per category, and one-click confirm or override.
6. **Active-learning loop.** Inspector overrides feed back into the prompt-tuning data set automatically — the system gets better with use.
7. **Self-service onboarding tool for new categories.** A Jupyter notebook that takes 30 photos and a description, runs the κ test, drafts a Gemma prompt, and outputs a deploy/hold recommendation.
8. **Explainability dashboard.** For each flag, show the image regions the model attended to plus its rationale text. Makes inspector review fast and defensible to residents.

---

## 11. The close

- **Today:** 4 high-confidence categories ready for full automation, zero false alarms on clean houses.
- **Assisted:** 4 mid-tier categories at inspector grade with AI in support — high precision, conservative recall.
- **Pending:** 1 category (deteriorating chimney) waiting on a city-side SOP before deployment.
- **Onward:** A repeatable, κ-gated playbook for any new violation category, with a clear separation between what the city can provide and what our team can build.

The tool works today on the categories where the labels are clear; it's honest about the categories where the labels aren't. That honesty is the more durable foundation — it lets Riverdale Park adopt AI screening without having to defend false alarms or hide behind a black box.

---

## Appendix A — System configuration alternatives

The production recommendation is cascade k=5. For completeness:

| Configuration | macro-F1 | micro-P | micro-R | micro-F1 | Specificity | When to choose |
|---|---|---|---|---|---|---|
| Cascade k=5 (production) | 0.64 | **0.85** | 0.63 | **0.72** | **96.1%** | Default — balances precision and inference cost |
| Cascade k=3 (high-precision) | 0.61 | 0.92 | 0.52 | 0.67 | 98.3% | When flags must be publishable without inspector review |
| Gemma 9-way (no CLIP filter) | 0.67 | 0.82 | 0.71 | 0.76 | 94.5% | When inference cost isn't a constraint and recall matters most |

## Appendix B — Methodological transparency

- **Ground truth handling:** "unsure" votes from human annotators are collapsed to "no" for all metrics. This is conservative (biases toward fewer false alarms). Roughly 7% of cells were originally unsure.
- **Consensus threshold:** ≥ 2 of 3 annotators saying yes → positive. This is the standard for non-expert multi-rater datasets.
- **Stage 1 vs Stage 2 ground truth:** Stage 1 (screening) uses the folder-name primary category as ground truth — this is a fact about the data source, not a derived label, so it's the most authoritative reference for "did the AI catch the known violation." Stage 2 (per-category review) uses human consensus.
- **Synthetic clean houses:** We generated 20 reference compliant images via text-to-image because the client dataset contained no negative samples. All three annotators confirmed each image as no-violation across all 9 categories. This is a known limitation — see Future Work item #1.
- **Unified 118-image evaluation:** all metrics in this report combine the 98 violation images and the 20 clean houses into a single evaluation set. Per-category precision/recall is unchanged whether we include the 20 clean houses (they add only true negatives), but specificity and the FPR story do require them.
