# Example Prompts — Zero-Shot Cascade Pipeline

## Overview

The Riverdale Park violation detector is fully zero-shot — we do not fine-tune any model on the 98 inspector photos. All the work of distinguishing one violation from another is done by the text prompts we feed the models, so this document catalogs every prompt we use, with worked examples drawn from real run logs.

The pipeline is a two-stage cascade:

- **Stage 1 — CLIP retrieval.** CLIP ViT-B/32 (LAION-2B) scores each image against each violation class using averaged text embeddings. The top-*k* classes per image are forwarded as candidates.
- **Stage 2 — Gemma per-class verification.** Gemma 4 E4B-IT (4-bit MLX) is asked one independent yes/no question per candidate class — no cross-class competition.

A legacy Phase 1 path is included for comparison: a single Gemma call that returns a ranked top-3 across all 9 classes at once. Phase 2 retired this path because forcing the model to pick the single best class at rank 1 caused it to miss broader categories (e.g. `boarded_windows`) when more visually specific neighbors (e.g. `peeling_paint`) were also present.

---

## Stage 1 — CLIP Zero-Shot Prompts

**How it works.** For each violation class, we write three natural-language paraphrases (e.g. `"a photo of graffiti spray-painted on an exterior wall"`). All three are encoded by CLIP, averaged, and stored as that class's text embedding. At inference, each image is also encoded by CLIP, and we compute the cosine similarity between the image embedding and each class embedding. The result is a 98 × 9 score matrix where each row says how strongly each photo matches each class. For Stage 1 of the cascade we keep this matrix and take the top-*k* most similar classes per image. Code: `src/zeroshot.py:38-88`. Prompt source: `src/client_data.py:69-115`.

### The full CLIP prompt list (9 classes × 3 paraphrases)

```text
boarded_windows:
  "a photo of a house with windows covered by wooden boards"
  "a building exterior with plywood nailed over the windows"
  "boarded-up windows on a vacant or damaged property"

broken_windows:
  "a photo of a broken or shattered window on a building"
  "a house exterior with cracked or smashed glass windows"
  "a window with holes, cracks, or missing glass panes"

damaged_roof_shingles:
  "a photo of a roof with missing or damaged shingles"
  "a rooftop with peeling, lifted, or broken asphalt shingles"
  "visible damage and gaps on the shingles of a house roof"

deteriorating_chimney:
  "a photo of a deteriorating brick chimney on a house"
  "a crumbling chimney with cracked bricks and missing mortar"
  "a damaged residential chimney in poor structural condition"

graffiti:
  "a photo of graffiti spray-painted on an exterior wall"
  "spray-painted tags and markings on a building surface"
  "a wall or fence vandalized with painted graffiti"

inoperable_vehicle:
  "a photo of an abandoned inoperable car on a property"
  "a derelict vehicle with flat tires, rust, or no license plate"
  "an unused junk car parked on a residential lot"

junk_trash_accumulation:
  "a photo of junk, debris, and trash piled on a property"
  "outdoor accumulation of discarded items and household rubbish"
  "a yard filled with garbage bags, broken furniture, and debris"

overgrown_vegetation:
  "a photo of overgrown grass and weeds on a property"
  "a lawn with tall unmaintained grass and weeds"
  "an unkempt yard with excessive vegetation"

peeling_paint:
  "a photo of peeling and chipping paint on a building exterior"
  "flaking and deteriorated paint on wooden siding"
  "a house facade with cracked, peeling exterior paint"
```

### Worked example — image index 0 (`boarded_windows`)

Image: `data/client-data/ Boarded Windows (§ 304.13 - 108.2)/15015565588_38e53262df_o.jpg`. Raw cosine similarities from `checkpoints/client_clip_similarity.npz`:

| class | cosine sim |
|---|---|
| **boarded_windows** | **+0.3028** |
| broken_windows | +0.2353 |
| damaged_roof_shingles | +0.1914 |
| deteriorating_chimney | +0.2020 |
| graffiti | +0.1938 |
| inoperable_vehicle | +0.2365 |
| junk_trash_accumulation | +0.1971 |
| overgrown_vegetation | +0.2267 |
| peeling_paint | +0.2505 |

CLIP argmax = `boarded_windows` (correct). Top-3 forwarded to Stage 2 = `[boarded_windows, peeling_paint, inoperable_vehicle]`.

---

## Stage 2 — Gemma 4 Per-Class Binary Prompts

**Mechanism.** For each (image, candidate_class) pair, the description string is substituted into a single template and the model is asked to return a one-line JSON object with `answer` (yes/no), `confidence` (0–100), and `rationale` (<15 words). Each call is independent — there is no cross-class competition. Template lives at `src/binary_prompt.py:19-29`; the substitution helper strips any leading `class_id —` prefix from the description so the model only sees the visual phrase (`src/binary_prompt.py:32-44`). Parsing falls back from JSON regex to bare `yes`/`no` token matching (`src/binary_prompt.py:47-100`).

### `BINARY_PROMPT_TEMPLATE` (verbatim)

```text
Look at the image carefully.

Question: Does this image show {description}

Answer in this EXACT format on one line, nothing else:
{"answer": "yes", "confidence": 85, "rationale": "short sentence"}
or
{"answer": "no", "confidence": 90, "rationale": "short sentence"}

The confidence is an integer from 0 to 100. The rationale must be under 15 words.
```

### Class descriptions substituted as `{description}` (from `src/client_data.py:119-138`)

After the `class_id — ` prefix is stripped and a trailing `?` is appended, the substitution becomes:

| class | substituted phrase |
|---|---|
| boarded_windows | `windows covered with plywood or boards, typical of vacant or damaged properties?` |
| broken_windows | `visibly cracked, shattered, or missing window glass on a building?` |
| damaged_roof_shingles | `visible damage, gaps, peeling, or missing shingles on a roof?` |
| deteriorating_chimney | `a chimney with structural damage, cracking, or crumbling bricks?` |
| graffiti | `spray-painted tags, markings, or drawings on exterior surfaces?` |
| inoperable_vehicle | `abandoned cars, vehicles missing plates, flat tires, or in obvious disrepair?` |
| junk_trash_accumulation | `piles of garbage, junk, or discarded materials on a property?` |
| overgrown_vegetation | `grass or weeds that are excessively tall or unmaintained?` |
| peeling_paint | `exterior paint that is chipping, flaking, or peeling off?` |

### Worked example A — true positive (record `i=0, j=0` from `client_gemma4_binary_stream.jsonl`)

Image: `15015565588_38e53262df_o.jpg`. True class: `boarded_windows`. Query class: `boarded_windows`.

Rendered prompt:

```text
Look at the image carefully.

Question: Does this image show windows covered with plywood or boards, typical of vacant or damaged properties?

Answer in this EXACT format on one line, nothing else:
{"answer": "yes", "confidence": 85, "rationale": "short sentence"}
or
{"answer": "no", "confidence": 90, "rationale": "short sentence"}

The confidence is an integer from 0 to 100. The rationale must be under 15 words.
```

Raw model output:

```json
{"answer": "yes", "confidence": 95, "rationale": "The structure has boarded-up openings suggesting vacant condition."}
```

| field | value |
|---|---|
| `is_target_class` | `true` |
| parsed `answer` | `yes` |
| parsed `confidence` | `95` |
| parsed `rationale` | "The structure has boarded-up openings suggesting vacant condition." |
| score → `(N,K)` matrix | `0.95` (= confidence/100 because answer=yes) |

### Worked example B — true negative (record `i=0, j=5`, same image)

Image: `15015565588_38e53262df_o.jpg`. True class: `boarded_windows`. Query class: `inoperable_vehicle`.

Rendered prompt:

```text
Look at the image carefully.

Question: Does this image show abandoned cars, vehicles missing plates, flat tires, or in obvious disrepair?

Answer in this EXACT format on one line, nothing else:
{"answer": "yes", "confidence": 85, "rationale": "short sentence"}
or
{"answer": "no", "confidence": 90, "rationale": "short sentence"}

The confidence is an integer from 0 to 100. The rationale must be under 15 words.
```

Raw model output:

```json
{"answer": "no", "confidence": 95, "rationale": "The image displays a building, not vehicles."}
```

| field | value |
|---|---|
| `is_target_class` | `false` |
| parsed `answer` | `no` |
| parsed `confidence` | `95` |
| parsed `rationale` | "The image displays a building, not vehicles." |
| score → `(N,K)` matrix | `0.05` (= 1 − confidence/100 because answer=no) |

---

## Phase 1 — Gemma "Ranked Top-3" Prompt (Legacy)

This was the single-call variant used before Phase 2. Gemma sees the full 9-class catalogue once and must commit to a top-3 ordering. Template lives at `src/zeroshot.py:94-107`; the class-list block is built by `_build_gemma_prompt` (`src/zeroshot.py:110-112`). The reciprocal-rank-to-softmax conversion (`src/zeroshot.py:145-174`) lets us still compute top-k metrics from a list output.

### Rendered ranked-prompt (verbatim, with the full 9-class list expanded)

```text
You are a municipal code enforcement assistant.

Look at the image and decide which single violation category it best matches.
Choose from this list only:

- boarded_windows: Boarded Windows — windows covered with plywood or boards, typical of vacant or damaged properties.
- broken_windows: Broken Windows — visibly cracked, shattered, or missing window glass on a building.
- damaged_roof_shingles: Damaged/Missing Roof Shingles — visible damage, gaps, peeling, or missing shingles on a roof.
- deteriorating_chimney: Deteriorating Chimney — a chimney with structural damage, cracking, or crumbling bricks.
- graffiti: Graffiti — spray-painted tags, markings, or drawings on exterior surfaces.
- inoperable_vehicle: Inoperable/Unlicensed Vehicles — abandoned cars, vehicles missing plates, flat tires, or in obvious disrepair.
- junk_trash_accumulation: Junk/Debris/Trash Accumulation — piles of garbage, junk, or discarded materials on a property.
- overgrown_vegetation: Long Grass/Overgrown Vegetation — grass or weeds that are excessively tall or unmaintained.
- peeling_paint: Peeling/Deteriorating Exterior Paint — exterior paint that is chipping, flaking, or peeling off.

Return your answer as a single JSON object on one line, with keys:
  "ranked": an array of exactly 3 category ids from the list above, ordered
            from most likely to least likely.

Do not include any other text. Example:
{"ranked": ["graffiti", "peeling_paint", "broken_windows"]}
```

### Worked example — record `i=30` from `client_gemma4_stream.jsonl`

Image: `data/client-data/Graffiti (§ 302.9)/16397396584_bc36463347_o.jpg`. True class: `graffiti`.

Raw model output:

```json
{"ranked": ["graffiti", "peeling_paint", "broken_windows"]}
```

| field | value |
|---|---|
| true class | `graffiti` |
| parsed `ranked` | `["graffiti", "peeling_paint", "broken_windows"]` |
| top-1 `pred` | `graffiti` |
| `ok` | `true` |

The reciprocal-rank weights `[1, 1/2, 1/3]` are then placed at the corresponding class indices and softmaxed (temperature 0.3) to a probability vector for downstream metrics.

---

## Why Two Prompt Styles

The ranked top-3 prompt forces the model to perform an **internal 9-way argmax** before answering. Empirically this hurts broader classes that compete with more visually specific neighbors. In Phase 1, `boarded_windows` recall was **25%** — Gemma's rationales showed it *did* see the boards, but it kept choosing the more fine-grained `peeling_paint` to fill the rank-1 slot, because both are technically present and the prompt forces a single ordering. Switching to per-class binary verification drops the cross-class competition entirely: every question is "is this class present?" answered in isolation. Same model, same checkpoint, same images, just a different prompt — `boarded_windows` recall climbed to **60%** in binary mode, and `graffiti`, `deteriorating_chimney`, and `peeling_paint` all hit **100%** recall ([Phase 2 evaluation report](https://drive.google.com/file/d/12cJu7xq6N7_DwWnFzrUMTkUSVoohdx3a/view)). Independent yes/no questions are easier for the model than implicit cross-class ranking, which is why Stage 2 of the cascade is built on the binary template.
