# Speaker Script — Riverdale Park AI Code Enforcement

*12 slides · ~12–15 minutes · English · companion to `ppt/index.html`*

Read fluently, don't memorize. The script aims for conversational delivery — pause on the bolded phrases, walk past the bullet detail. Page numbers match the deck.

---

## Slide 1 — Cover *(hero)*

> Hi everyone — and thanks especially to Ryan for joining today.
>
> What I want to walk you through over the next twelve slides is a mid-project briefing on the AI code-enforcement work we've been doing for Riverdale Park.
>
> The short version is this: **we have a system that can look at a street-view photo of a house and tell you which of nine code-enforcement categories it might be violating** — and we've now tested that system on enough real and clean houses to be honest with you about where it works, where it doesn't, and what we'd need from your office to push it further.
>
> Let me get into it.

*(advance to slide 2)*

---

## Slide 2 — How the system works

> Architecturally, the system is **a two-stage cascade.**
>
> When a photo comes in, the first stage — a model called CLIP — does a broad scan and picks the five violation categories most likely to be present. It's deliberately generous; we'd rather it consider too many candidates than miss a real one.
>
> The second stage — a vision-language model called Gemma — then takes each of those five candidates and asks a focused yes/no question: *"does this image actually show a broken window?"* It returns a verdict, a confidence score, and a one-sentence rationale.
>
> The slogan we use internally is **"CLIP for recall, Gemma for precision"** — CLIP makes sure we don't miss problems, Gemma makes sure we don't raise false alarms.

*(advance to slide 3)*

---

## Slide 3 — What we tested

> To be transparent about what these numbers actually mean, here's what the evaluation looked like.
>
> We tested on **98 known-violation houses** from the Riverdale Park dataset — these have a confirmed primary violation type baked into the folder structure.
>
> We also generated **20 reference clean houses** synthetically — three of our team members independently confirmed that none of these images contain any of the nine violations.
>
> For ground truth, **three independent team members each labeled every image across all nine categories.** We collapse "unsure" responses to "no" — which biases the system slightly toward fewer false alarms, the right direction for a screening tool — and a category counts as a positive when at least two of three annotators agree.
>
> All in, that's **118 images, 1,062 individual judgments, three human raters** behind every cell.

*(advance to slide 4)*

---

## Slide 4 — Three headline numbers

> If you take three numbers away from today, take these three.
>
> First — **the system catches 87% of known violations.** When we feed it a photo where we know there's a real problem, it correctly flags that problem in nearly nine out of ten cases.
>
> Second — and this is the one I'm proudest of — **zero false alarms on clean houses.** Across 180 binary checks on 20 confirmed clean homes, the system never falsely raised a violation. The model's average confidence on those clean cells was 96 out of 100. These weren't borderline calls; the model was decisively right.
>
> Third — **when the system says "yes," it's right 85% of the time.** That's the precision number. When you tell a resident "we flagged your house for graffiti," 85 out of 100 of those statements would hold up to inspection.

*(pause briefly, then advance to slide 5)*

---

## Slide 5 — Why bucket by agreement *(hero)*

> Now, those three numbers are the headline — but the most important *insight* we discovered is on this slide.
>
> When we looked at performance category by category, we noticed something striking: **AI accuracy tracks human inter-rater agreement.** When our three annotators couldn't agree on whether a category was present, the AI also couldn't reliably decide.
>
> What that means is the bottleneck isn't the algorithm. It's **how well the category itself is defined.** And so for the rest of this briefing, instead of giving you one big average accuracy number, we report results stratified by how well our humans agreed — because that's where the real story is.

*(advance to slide 6)*

---

## Slide 6 — Performance by tier *(main results)*

> Here are the results sorted by that human-agreement metric — Cohen's kappa.
>
> The top two tiers — **almost-perfect and substantial agreement** — that's where the system shines. Categories like graffiti and inoperable vehicles, where humans almost always agree, the AI gets an F1 of 0.87 — that's basically inspector-grade. Damaged roof and overgrown vegetation, slightly below that but still strong with very high precision.
>
> The middle tier — **moderate agreement** — that's where it becomes a useful assistant rather than a standalone judge. Boarded windows, broken windows, junk and trash. AI flags here are mostly correct, but it's conservative — it'll miss some cases.
>
> Peeling paint sits in the **fair tier**. Its F1 looks high at 0.80, but I'd ask you to read that with a grain of salt — most of our images have some peeling paint, so a model that says "yes" often gets credit. The human agreement on this one is genuinely low.
>
> And then **deteriorating chimney**, in the slight tier. F1 of 0.18. That's the one category where the system fails — but as we'll see in a moment, it fails because *no one* could agree on what counts as a violation in the first place.

*(advance to slide 7)*

---

## Slide 7 — Per-category detail

> This is the same data, broken out by individual category. I won't read through all nine — but two rows are worth highlighting.
>
> **Broken windows** — look at that precision: a perfect 1.00. Every single time the system flagged broken windows, it was correct. But its recall is only 22%. So practically: if you see a broken-windows flag come out of this system, you can act on it directly. But for any property where you suspect window damage, an inspector should still walk it — the AI is genuinely conservative here.
>
> **Deteriorating chimney** — F1 of 0.18, the worst in the deck. But notice the kappa: **0.15.** Our three annotators couldn't agree on this category either. The model isn't broken; the *task* is underspecified. From a single street-view photo, even humans can't reliably tell what counts as deteriorating. This is the cleanest possible argument that **what we need here isn't a better model — it's a clearer city SOP.**

*(advance to slide 8)*

---

## Slide 8 — Zero false alarms

> Coming back to that zero-false-alarms number, I want to show you what's behind it.
>
> Twenty clean houses, nine categories each, 180 binary checks. Across all of them — zero false positives. Not a single false alarm.
>
> And these aren't lucky guesses. The model's average confidence was 96 out of 100. Here's an actual sample rationale Gemma produced on one of these houses: *"All windows are intact and glassed over. The roof appears to be in good condition with no visible damage."* It's not gambling — it's actually looking at the photo.
>
> One honest caveat: those twenty houses are AI-generated, and they're a little idealized. Real clean houses have shadows, decorative wreaths, kids' bikes in the yard. So I'd treat 0% as our **synthetic baseline** — and our top ask in the future-work section is to validate it against real city-archive photos.

*(advance to slide 9)*

---

## Slide 9 — How to use it *(deployment modes)*

> Given those tiered results, here's how we recommend you actually deploy the system.
>
> **Full automation** — green tier, four categories: graffiti, inoperable vehicles, overgrown vegetation, damaged roof. The AI generates a violation-notice draft directly into the inspector follow-up queue. No pre-screening needed. These are inspector-grade.
>
> **AI-assisted** — yellow tier, four categories: boarded and broken windows, junk and trash, peeling paint. AI flags go into a high-priority queue, but inspectors should still spot-check unflagged photos because the system is conservative here.
>
> **Hold** — red tier, just one category: deteriorating chimney. Keep your existing manual workflow until you can give us clearer guidance.
>
> And we've sketched three operational checks at the bottom — a weekly spot-check, a monthly override-rate review, and a quarterly clean-house re-test — to make sure performance doesn't drift over time.

*(advance to slide 10)*

---

## Slide 10 — How to scale to a new category

> A reasonable question is: *"What if we want to add a new violation type next year?"* Here's the playbook.
>
> Step 1 — collect samples: at least 30 positive and 20 negative photos, with a plain-English description of the violation.
>
> Step 2 — and **this is the most important step** — get three inspectors to independently label the positives, and compute the kappa. That single number tells you in advance whether AI can succeed on this category.
>
> Step 3 is the gate: kappa above 0.61 → likely automation tier; between 0.41 and 0.60 → AI-assisted; below 0.40 → fix the SOP first, don't even bother training.
>
> Steps 4 through 6 are mechanical: write the prompt — visual not legal, one sentence, edge cases included; integrate it; measure F1 and false-positive rate; deploy or revise.
>
> The whole point of this flow is that **the kappa test in step 2 saves you weeks of model engineering** if the answer is going to be "this category isn't well-defined enough."

*(advance to slide 11)*

---

## Slide 11 — Future work

> Two tracks for what comes next.
>
> **From the city's side**, in priority order:
>
> One — fifty to a hundred real clean-house photos from your archives. That replaces our synthetic baseline and validates the false-alarm number against reality.
>
> Two — a brief SOP for the ambiguous categories, especially chimney and peeling paint. Two pages plus a few reference photos would be enough to unlock these tiers.
>
> Three — multi-angle photos per address, which would directly improve recall on the detail-heavy categories like broken windows.
>
> Four — severity grading, so we can output "minor versus major" instead of just yes/no.
>
> **From our team's side:**
>
> Five — a REST API plus a simple web UI so inspectors can upload a photo and one-click confirm or override.
>
> Six — an active-learning loop where inspector overrides automatically feed back into prompt tuning.
>
> Seven — a self-service onboarding tool so the city can add new categories without us in the loop.
>
> Eight — an explainability dashboard with image regions and rationale text per flag.

*(advance to slide 12)*

---

## Slide 12 — Closing *(hero)*

> To wrap up:
>
> **Today**, four categories are ready for full automation, with zero false alarms on the clean houses we tested.
>
> **Four more** are at inspector grade with AI in support.
>
> **One** — chimney — is on hold pending an SOP from the city.
>
> And we have a clear, repeatable playbook for any new category Riverdale Park wants to add.
>
> The thing I want you to take away most is this: **the system is honest about where it works and where it doesn't.** That honesty is the durable foundation. It means when you adopt this tool, you can stand behind every flag it raises in front of a resident — and you'll know exactly which categories still need a human in the loop.
>
> Thank you. I'd love to take questions.

---

## Q&A — likely questions and short answers

**Q: How long does this take per photo?**
A: A few seconds. The CLIP stage is near-instant; the Gemma verification stage is around five seconds for the five candidates. Easily fast enough for batch processing or interactive use.

**Q: Does it work on photos taken at night, or in bad weather?**
A: Honestly, we haven't tested that. Our 98 photos are mostly daytime. That's a known gap and would need a small follow-up evaluation if you have night-time complaints to handle.

**Q: What if a resident disputes a flag?**
A: Two things help. First, every flag includes a one-sentence rationale from the model — so you can show the resident *why* it was flagged. Second, the explainability dashboard we proposed in future work would highlight which part of the image triggered the call.

**Q: Why not use a single model instead of two?**
A: We tested that — single Gemma actually has slightly higher F1. But the cascade is roughly half the inference cost and has a structural specificity advantage: the four non-top-five categories are forced to "no," which means cross-category mistakes are impossible. For a screening tool aimed at low false-alarm rate, the cascade is the right tradeoff.

**Q: What's the cost to run this in production?**
A: Both models run locally on a laptop-class GPU; no per-call API fees. The main cost is engineering time for the API and UI work in our future-work track.

**Q: How does this handle a photo that doesn't show a house at all?**
A: We haven't built explicit "is this even a house?" rejection. In practice the cascade will mostly say "no" to all nine categories, which is the right behavior, but a sanity-check filter would be a small addition.

**Q: Can this be biased against certain neighborhoods?**
A: Important question. Our 98 images are violation-only, so we haven't done a demographic-fairness analysis on the dataset. A real deployment audit would compare flag rates across neighborhoods to make sure the model isn't over-flagging certain areas. That's a fair-AI follow-up we'd recommend before full rollout.
