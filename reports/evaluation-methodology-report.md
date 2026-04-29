# Evaluation Methodology — How We Built Ground Truth for the Cascade Pipeline

> Riverdale Park AI Code Enforcement · INST623 Final Project
> Chihao Li · 2026-04-29

This report documents the multi-week effort to build a reliable evaluation methodology for our zero-shot cascade pipeline (CLIP → Gemma binary → Sonnet judge). It covers two parallel evaluation tracks (AI multi-agent labeling and human inter-rater labeling), the cross-source comparison, a prompt-refinement iteration (v3), and what these findings mean for the final deliverable.

---

## 1. Background — Why we needed this

Our cascade pipeline assigns each inspector photo to one or more of 9 violation classes. The original "ground truth" we received from the client was a folder structure: 98 photos sorted into folders by their **primary** violation code, one folder per code.

Two problems with using folder labels as evaluation truth, both surfaced in Phase 2:

1. **Folders are single-label** — each photo is filed under one code, even when the photo clearly shows multiple violations (a derelict house can show graffiti, peeling paint, and boarded windows simultaneously).
2. **Folders are under-complete** — 11 of 98 photos appeared in multiple folders (verified by md5), and many photos contain visible secondary violations the folder didn't track.

Without a multi-label truth, we could not measure cascade performance honestly. Phase 2 numbers like "Sonnet 91% precision vs folder" were technically meaningful but vastly understated the model's actual accuracy on real (multi-label) violations.

The work documented in this report builds that multi-label truth using two complementary tracks.

---

## 2. The Two Evaluation Tracks

### Track A — AI multi-agent labeling

Approach: dispatch independent Claude Sonnet sub-agents to label all 98 photos using a structured prompt, then aggregate by majority vote across multiple runs.

**Schema iteration history** (each version is the same task with refined instructions):

| Version | Schema | Result |
|---|---|---|
| v1 | yes/no only | Single pass, 236 yes flags. Manual review found ~5% borderline cells where forcing yes/no created bias (vehicle windows, small vines, tarp-covered windows). |
| v2 | yes/no/unsure + 3 class-boundary rules ("can't see → unsure"; "broken_windows = buildings only"; "overgrown = property-scale only") | Single pass, 195 yes / 47 unsure. 88.9% of v1 cells unchanged; the 11.1% that changed were the borderlines. |
| **v2 multi-rater** | Same schema, **run 3 independent times** to measure self-consistency | 89.1% of cells unanimous across 3 runs, 10.7% majority, only 0.2% (2 cells) truly split. Two errors I had spot-checked manually in v1 were auto-corrected by the multi-run consensus. |
| v3 | v2 schema + refined definitions for 3 hard classes (paint/chimney/junk) | Single pass tested against v2. Results in §5. |

**Final v2 truth** (after 3 runs, majority vote): 201 yes flags, 639 no, 42 unsure, 786 unanimous + 94 majority + 2 disputed. This is the AI ground truth used for cross-comparison.

### Track B — Human inter-rater labeling

Approach: three team members independently labeled all 98 photos using the same v2 schema. We compute pairwise inter-rater agreement (Cohen's κ) and build a human consensus by majority vote.

**Completion**:

| Labeler | Rows labeled | Yes-rate (on labeled cells) |
|---|---:|---:|
| Fechi | 56/98 (stopped at img_055) | 40% |
| Eileen | 98/98 | 33% |
| Jake | 98/98 | 24% |

The three labelers had very different styles. Jake was the most decisive (only 5 unsures total). Eileen used unsure 85 times. Fechi used unsure 127 times. Yes-rates spanned a 16-point range, reflecting different thresholds for borderline cases.

**Inter-rater agreement** (3-way Cohen's κ on the first 56 photos all three labeled):

| Class | κ | Interpretation |
|---|---:|---|
| graffiti | 0.65 | Substantial |
| boarded_windows | 0.60 | Substantial |
| inoperable_vehicle | 0.52 | Moderate |
| overgrown_vegetation | 0.51 | Moderate |
| broken_windows | 0.44 | Moderate |
| damaged_roof_shingles | 0.35 | Fair |
| junk_trash_accumulation | 0.23 | Fair |
| **peeling_paint** | **0.15** | Slight |
| **deteriorating_chimney** | **0.15** | Slight |

**Two classes have κ < 0.20** — humans themselves don't agree on what counts as `peeling_paint` or `deteriorating_chimney`. This is not a model deficiency. It is task definition fuzziness in the underlying labels.

**Human consensus** (majority of 3 on rows 0-55, agreement of 2 on rows 56-97):
- 882 cells total
- 247 unanimous (3-way) + 247 majority (2/3) + 324 unanimous-2 (Eileen+Jake agree) + 54 split-2way + 10 split-3way
- 250 yes flags total

---

## 3. Cross-source comparison

We compare four sources of labels on the same 98 photos × 9 classes:

| Source | Yes total | What it represents |
|---|---:|---|
| Folder primary | 98 | Client's original single-label assignment |
| Sonnet 3-run truth | 201 | AI multi-rater majority |
| Gemma 4 binary | 238 | Phase 2 baseline AI judge |
| **Human consensus** | **250** | Three independent humans, majority vote |

Three structured comparisons:

### 3.1 Primary recall (does the method recover folder's primary?)

| Method | Hits primary | Notes |
|---|---:|---|
| Folder | 100% (baseline) | — |
| **Sonnet 3-run** | **99% (97/98)** | Best of all methods |
| Eileen | 97% | — |
| Jake | 97% | — |
| Human consensus | 96% | One overgrown miss |
| Fechi (within 56) | 95% | — |
| **Gemma 4** | **89%** | Misses concentrated on boarded (62%) and broken (50%) — same blind spots Phase 2 documented |

**Reading**: every method except Gemma clears the "minimum bar" of recovering the folder primary. Sonnet 3-run is the most reliable at this, ahead of any individual human and ahead of human consensus. **AI judge is not just sufficient for primary detection — it is more consistent than humans**.

### 3.2 Secondary detection (yes flags beyond the primary)

| Method | Total yes | Primary hits | Secondary |
|---|---:|---:|---:|
| Folder | 98 | 98 | 0 |
| Sonnet 3-run | 201 | 97 | +104 |
| Gemma 4 | 238 | 87 | +151 |
| Human consensus | 250 | 94 | **+156** |

Secondary additions cluster heavily on `peeling_paint` (humans +48, Sonnet +26, Gemma +45) and `overgrown_vegetation` (humans +36, Sonnet +22, Gemma +36) — confirming the Phase 2 "folder under-complete" hypothesis with three independent sources.

### 3.3 Sonnet 3-run vs human consensus (decisive cells only)

| Metric | Value |
|---|---:|
| Macro-average agreement | 85% |
| Macro-average recall vs human | 60% |
| Macro-average precision vs human | **91%** |
| Cells where Sonnet=YES, Human=NO | **7** (false positives) |
| Cells where Sonnet=NO, Human=YES | 60 (false negatives) |

**Sonnet is high-precision, conservative-recall**. When Sonnet says yes, humans almost always agree (91% precision). Sonnet's 60 missed yes cases concentrate on `peeling_paint` (27), `overgrown_vegetation` (15), and `junk_trash` (10) — which led to the v3 prompt iteration described next.

---

## 4. The Hard Three Classes

Across both tracks, three classes consistently underperformed:

| Class | Human κ | Sonnet 3-run agreement | Sonnet vs human disagree |
|---|---:|---:|---:|
| peeling_paint | 0.15 | 84% | 27 cells |
| deteriorating_chimney | 0.15 | 93% | 2 cells |
| junk_trash_accumulation | 0.23 | 81% | 10 cells |

These are not model failures — humans don't agree among themselves either. The root causes (from rationale analysis):

- **`peeling_paint`**: ambiguous whether bare brick/stucco walls count, whether paint must be visibly flaking or merely faded
- **`deteriorating_chimney`**: most photos show distant chimneys where structural damage cannot be visually confirmed
- **`junk_trash_accumulation`**: unclear threshold (how much constitutes a "violation" pile?), unclear whether auto parts around vehicles count

This led us to test whether **prompt refinement** could help.

---

## 5. v3 Prompt Iteration

**Hypothesis**: refined definitions with explicit thresholds and inclusion/exclusion examples will improve Sonnet's alignment with human Tier A (unanimous) labels on the hard classes.

**Method**: re-wrote definitions for `peeling_paint`, `deteriorating_chimney`, `junk_trash` (the other 6 classes unchanged from v2). Re-ran one pass of Sonnet on all 98 photos.

### Results

| Class | v2 yes | v3 yes | Δ | v2 unsure | v3 unsure | Δ | Tier A agreement v2 → v3 |
|---|---:|---:|---:|---:|---:|---:|---|
| peeling_paint | 40 | 38 | −2 | 3 | 2 | −1 | 91% → **88%** |
| deteriorating_chimney | 4 | 3 | −1 | 9 | 14 | **+5** | 100% → 100% |
| junk_trash | 26 | 32 | +6 | 0 | 0 | 0 | 89% → 87% |
| (overgrown — unchanged in v3) | 29 | 45 | **+16** | 0 | 0 | 0 | 74% → **89%** |

### What the iteration shows

**(a) Prompt refinement works for visually-objective constraints.**
The chimney distance rule ("if chimney is <3-5% of photo width, mark unsure not yes") triggered on 5 cells, exactly the cases human Tier A had labeled NO. Sonnet became more honest about visual ambiguity. This is the cleanest positive result.

**(b) Prompt refinement fails for visual-perception ambiguity.**
For `peeling_paint`, we explicitly extended the definition to include stucco/plaster/render finishes (not just literal paint). Yet Sonnet kept labeling 4 of 5 known-failure cases as NO with rationales like "bare brick exterior, no paint coating to peel." The model's visual judgment about whether a wall was ever painted is the bottleneck, not the prompt language. **No amount of prompt rewriting solves this**.

**(c) Threshold-based refinement changes behavior but doesn't necessarily improve alignment.**
The `junk_trash` rewrite added a "≥3 distinct items OR clear pile" threshold. Sonnet flipped 11 NO→YES and 5 YES→NO. Net yes count rose by 6. But Tier A agreement actually dropped from 89% to 87%. The threshold made the model more aggressive, but the cases it picked up weren't the cases humans picked up. **Threshold language alone doesn't align with human judgment**.

**(d) Side effects propagate.**
The v3 prompt did not modify `overgrown_vegetation`, but yes count jumped from 29 to 45 and Tier A agreement rose from 74% to 89% — a 15-point improvement. Tightening one part of the prompt context appears to have made the model more decisive elsewhere. This is unstable behavior and a reason to test prompts holistically, not class-by-class.

---

## 6. Conclusions

### 6.1 The AI multi-agent judge is reliable for primary detection

- Sonnet 3-run primary recall: **99%** (better than any single human at 95-97%)
- Sonnet 3-run self-consistency: 89.1% unanimous, 10.7% majority, 0.2% disputed
- 91% precision vs human consensus on decisive cells

For practical deployment, **a Sonnet-class model is sufficient** as the primary classifier. Three independent runs catch the small fraction of unstable cells.

### 6.2 Some classes have a definition-level ceiling, not a model-level ceiling

- `peeling_paint` and `chimney` have human κ ≈ 0.15 — humans only agree about half the time after subtracting chance
- No model can achieve higher accuracy than humans agree among themselves
- The fix is not a better model. The fix is sharper class definitions, ideally with photo examples reviewed and approved by the client

### 6.3 Folder labels are confirmed under-complete

- Three independent sources (humans, Sonnet, Gemma) all find ~2× more violations than the folder
- Secondary violations cluster on `peeling_paint` and `overgrown_vegetation`
- The final ground truth for evaluation is the multi-label consensus, not the folder

### 6.4 Prompt refinement is a tool, not a solution

- It works for **objective visual constraints** (e.g., distance thresholds for `chimney`)
- It fails for **perceptual ambiguity** (e.g., "is this wall painted or bare?")
- For the latter, **human-in-the-loop review** is the only reliable fallback

### 6.5 Three labelers were enough

- Cohen's κ varied widely by class — the three-rater design surfaced this naturally
- Two-out-of-three majority resolved 99.8% of cells; only 2 cells truly split 1/1/1
- Per-rater style differences (yes-rate from 24% to 40%) were a feature, not a bug

---

## 7. Implications for the final deliverable

This methodology is not a one-time analysis. It becomes a reusable component of the pipeline package:

1. **Multi-rater AI evaluation** ships with the package. Clients can re-run the 3-pass Sonnet evaluation on new photo batches to validate pipeline updates.
2. **Tiered ground truth** (unanimous / majority / disputed) becomes the standard evaluation framework. Models are reported separately on tier-A high-confidence cells and on majority cells, so confidence is always traceable.
3. **Per-class reliability table** (κ, yes-rate distribution, unsure rate) is the headline diagnostic for any new class added. Classes with κ < 0.4 are flagged as "needs HITL or definition refinement."
4. **HITL trigger conditions** are derived directly from this report:
   - Confidence < 70 on any class
   - Class is on the "low-κ" list (paint, chimney, junk for now)
   - Sonnet 3-run majority is split or contains an unsure
5. **Photo capture recommendations** to the client come out of the unsure analysis: 89% of unsures are "can't see" cases. Inspector workflow should include at least one upward roof shot, one chimney closeup, and one window detail per property.

---

## 8. Next steps

| Item | Why | When |
|---|---|---|
| Lock v2 as the production schema, not v3 | v3 didn't improve hard classes and had unstable side effects | Now |
| Document the three-rule schema in the package | These rules ARE the prompt, and they need to ship | Next week |
| Build the HITL trigger logic into the cascade | Per §6.4, paint/chimney/junk need it | Next week |
| Generate the reliability table for any new class added | Reusable diagnostic | Next week |
| Refresh class definitions on `peeling_paint` with the client | Definition-level fix beats any prompt change | Pending client meeting |

---

## Appendix — Data artifacts

All evaluation data is on disk and reproducible:

- `checkpoints/runs/run_a/`, `run_b/`, `run_c/` — three full Sonnet labeling passes (98 photos each)
- `checkpoints/runs/run_v3/` — v3 single-pass with refined hard-class prompts
- `checkpoints/sonnet_3run_truth.npy` — multi-rater majority truth
- `checkpoints/human_consensus.npy` — 3-human majority consensus
- `data/client-data/labeling/human/*.xlsx` — original human labels (Fechi, Eileen, Jake)
- `data/client-data/labeling/agent_instructions.md` — v2 production schema
- `data/client-data/labeling/agent_instructions_v3.md` — v3 experimental schema (archived; not used in production)
