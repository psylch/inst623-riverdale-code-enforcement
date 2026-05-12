# Photo Labeling Guide — Riverdale Park Project

**Prepared by:** Chihao Li · **Date:** 2026-04-27
**For:** Eileen, Fechi, Jake (and anyone else helping label)

---

## What this project is

We are building a tool for the Town of Riverdale Park's code enforcement team. When an inspector takes a photo of a property violation (peeling paint, broken windows, junk in the yard, etc.), the tool should look at the photo and tell them which violation code applies. Right now, inspectors do this match by hand against several hundred code sections — slow and inconsistent.

The client gave us **98 real inspector photos** sorted into folders by violation code (one folder per code). That folder name is currently the only label we have for each photo.

## Why we need to label these photos again

When we looked closely at the 98 photos, we found two problems with the folder labels:

1. **Some photos belong in more than one folder.** A burned-out house can show graffiti, peeling paint, *and* boarded windows all at once. The client put each photo in its main folder, but several photos clearly show multiple violations.
2. **The folder usually only marks the main violation.** A photo of overgrown grass often also has peeling paint visible — but the folder only says "overgrown vegetation." So some violations are present but unlabeled.

This matters because if we measure our AI tool against folder labels, the tool gets penalized for "wrong" answers that are actually right. **We need a better label for each photo: a yes/no answer for every one of the 9 violation types**, not just one folder per photo.

That is what this labeling task produces.

## What we are asking each labeler to do

Each of you will independently look at all 98 photos and decide, for each photo and each of the 9 violation types, whether that violation is visible.

We are doing this independently (without comparing notes) on purpose — we want to see how often three reasonable people agree, and where they disagree. That tells us how reliable our final labels are.

### Files you'll work with

Everything is in our shared [Google Drive folder](https://drive.google.com/drive/folders/16KfQB_TEzrdHKlSnxSl6scfGAdieAixA):

| File | What it is |
|---|---|
| [`preview/`](https://drive.google.com/drive/folders/1GOrbVDlnXCLx_KTdGM1l9Df1GptNeBOG) | A folder with 98 images named `img_000.jpg` … `img_097.jpg`. Open these to view the photos. The names hide the original folder so you label without bias. |
| [`labeler_1.csv`](https://drive.google.com/file/d/1zbOrvhNNl-yzCx3IFblabj0GoYZvKZUH/view) · [`labeler_2.csv`](https://drive.google.com/file/d/1yVNs0dWfhFHbZS_xnDUiOfkSfabXQ0XH/view) · [`labeler_3.csv`](https://drive.google.com/file/d/1rren-ihTz8OAF6BNO51UY2dzGq7DqOlX/view) | Pick one — your blank labeling sheet. One row per image, one column per violation type. |

Open the CSV in Google Sheets directly (right-click → Open with Sheets), or download and use Excel / Numbers. The columns are pre-filled with the 9 violation types, the cells are empty for you to fill in.

### How to fill in each cell

For every cell (one image × one violation type), put one of three values:

- **`y`** — the violation is clearly visible in the photo
- **`n`** — you can see this part of the photo and the violation is not present
- **`?`** — you genuinely can't decide because something blocks the call (occlusion, distance, image quality, or a borderline case)

A photo can be `y` for multiple violations — that's expected. `?` is fine whenever you genuinely can't tell — don't agonize, just mark `?` and move on.

### Notes column is optional

If you have a quick reason for any `?` (or anything else weird about a photo), jot it in the `notes` column. Don't worry about formatting — a few words is fine. **Skip it entirely if you're in a rush.**

### Three quick rules

1. **Can't see → `?`.** Tarp over a window, distant building, occluded roof — don't guess.
2. **`broken_windows` = buildings only.** Smashed car windows go under `inoperable_vehicle`, not here.
3. **`overgrown_vegetation` = property-scale.** A vine on a wall or trees at the edge don't count; vegetation has to be visibly taking over.

### What each violation type means

| Violation type | What you're looking for |
|---|---|
| `boarded_windows` | Windows covered with plywood or wooden boards |
| `broken_windows` | Cracked, shattered, or missing window glass |
| `damaged_roof_shingles` | Roof with missing, peeling, or broken shingles |
| `deteriorating_chimney` | Chimney with cracked or crumbling bricks |
| `graffiti` | Spray-painted markings or tags on a surface |
| `inoperable_vehicle` | Abandoned car — flat tires, no plates, obvious disrepair |
| `junk_trash_accumulation` | Piles of garbage, junk, debris in the yard |
| `overgrown_vegetation` | Tall unmaintained grass, weeds, untended yard |
| `peeling_paint` | Paint chipping or flaking on the building |

### Two important rules

1. **Don't compare notes with the other labelers while you're labeling.** Once you're all done we'll compile the results — disagreement is what we're measuring.
2. **Don't peek at the original folder names.** They give away the "main" answer and make this exercise pointless. Just use the [`preview/`](https://drive.google.com/drive/folders/1GOrbVDlnXCLx_KTdGM1l9Df1GptNeBOG) folder where every photo is renamed `img_NNN.jpg`.

### How long this takes

Aim for ~15 seconds per photo and you're done in **~25 minutes**. Don't overthink any single call — first instinct is fine. The point is that three people each go fast independently, not that any single labeler is meticulous.

---

## What happens with your labels

Once all three of you submit, I'll compare the answers in three ways:

1. **How often the three of you agree.** This tells us whether the task is well-defined or whether reasonable people see different things.
2. **Where all three of you marked `?` for the same cell.** Those photos are the ones that genuinely need extra information (a follow-up inspection, a second photo from a better angle). Surfacing them is one of the most useful outputs of this whole exercise.
3. **What "the team" thinks vs. what the AI thinks.** I'll take the majority vote across the three of you as the team answer, then compare it against two AI judges:
   - Claude Sonnet (a frontier AI model, already labeled all 98 photos using the same instructions you're using).
   - Gemma 4 (the open-source model we tested in March).
4. **Team labels vs. the original folder labels.** This will confirm or disprove our hunch that the folders miss secondary violations.

The output is a short report sharing what the agreement looked like and which set of labels we're using as the official ground truth for the rest of the project.

---

## Timeline

| Date | Milestone |
|---|---|
| **Today, Mon 2026-04-27** | This guide goes out; templates ready |
| **Tue 2026-04-28** | Each of you knocks out the labels (~25 min) |
| **Wed 2026-04-29 morning** | All three sheets in — hard deadline |
| **Wed afternoon / Thu** | I run the agreement analysis and share results |

---

## Questions to discuss as a team

1. Are three labelers enough, or do we want a fourth voice? (Two-vs-one ties otherwise force me to be the tie-breaker.)
2. Is `y` / `n` / `?` enough, or do we want something like "definitely / probably / maybe / no"?
3. Should we have one shared rule for borderline cases, or trust each labeler's judgment?

Quick reply on the team channel if anything's blocking — otherwise just dive in.

---

## Where to find things

Everything lives in the shared [INST623 — Final Project Deliverables](https://drive.google.com/drive/folders/1zWaZ3hITHExBKG8LHJTed2HNfMrjbT9h) folder.

| You need | Where |
|---|---|
| The 98 photos | [Labeling/preview/](https://drive.google.com/drive/folders/1GOrbVDlnXCLx_KTdGM1l9Df1GptNeBOG) |
| Your labeling sheet | [labeler_1](https://drive.google.com/file/d/1zbOrvhNNl-yzCx3IFblabj0GoYZvKZUH/view) · [labeler_2](https://drive.google.com/file/d/1yVNs0dWfhFHbZS_xnDUiOfkSfabXQ0XH/view) · [labeler_3](https://drive.google.com/file/d/1rren-ihTz8OAF6BNO51UY2dzGq7DqOlX/view) |
| Image filename ↔ photo manifest | [manifest.csv](https://drive.google.com/file/d/1OGCm0AeiDhfB3e2nC0SNjH-CiE0NQzNd/view) |
| This guide | [sop-labeling-2026-04-27.docx](https://drive.google.com/file/d/1fyu_bGNS44Adb3E-b6C2wBKzEfpJul5Y/view) |
| When done | Save the filled CSV in place (Drive will keep your edits) or email a copy to Chihao |
