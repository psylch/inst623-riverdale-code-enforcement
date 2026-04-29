# Riverdale Park Property Maintenance Code 2022 — Class Citations

**Source**: [Town of Riverdale Park Property Maintenance Code 2022](https://cms7files.revize.com/riverdalemd/Chapter%2067%20PROPERTY%20MAINTENANCE%20COMBINED.pdf), adopted June 6, 2022, effective June 26, 2022. Derivative of 2018 International Property Maintenance Code (ICC). Local revisions are noted in original PDF with `**...**` markers.

PDF saved at: `research/riverdale/Chapter67_Property_Maintenance.pdf` · Plain text at: `research/riverdale/Chapter67_Property_Maintenance.txt`

---

## Per-class legal text + our SOP alignment

### 1. `peeling_paint` — § 304.2 Protective treatment

> Exterior surfaces, including but not limited to, doors, door and window frames, cornices, porches, trim, balconies, decks and fences, shall be maintained in good condition. Exterior wood surfaces, other than decay-resistant woods, shall be **protected from the elements and decay by painting or other protective covering or treatment**. **Peeling, flaking and chipped paint shall be eliminated and surfaces repainted.** Siding and masonry joints, as well as those between the building envelope and the perimeter of windows, doors and skylights, shall be maintained weather resistant and waterproof.

**Notes**:
- The law applies to surfaces **that are required to be coated** (wood, masonry joints, etc.) — not bare brick or stone that was never intended to be coated.
- This validates Sonnet's "bare brick → no" reflex on a legal basis. v3 attempt to extend to "stucco/plaster/render" is consistent with the law (these are protective coverings); Sonnet's failure to flip on these images is a perception issue.
- Implication for SOP: for a `yes`, the labeler must (a) determine surface intended to be coated, (b) confirm peeling/flaking. The **first** step is the source of disagreement.

### 2. `deteriorating_chimney` — § 304.11 Chimneys and towers

> Chimneys, cooling towers, smoke stacks, and similar appurtenances shall be **maintained structurally safe and sound, and in good repair**. Exposed surfaces of metal or wood shall be protected from the elements and against decay or rust by periodic application of weathercoating materials, such as paint or similar surface treatment.

**Notes**:
- Two separate maintenance requirements: (a) structural safety + good repair, (b) weather coating on metal/wood components.
- Our SOP only captures (a). For "deteriorating" we focused on cracked/leaning/missing bricks. (b) — chimney paint or coating peeling — is technically also a violation but we never asked labelers to consider it.
- The "structural soundness" requirement is unobservable from typical street-level photos at distance, which is the data-coverage issue we identified.

### 3. `junk_trash_accumulation` — § 302.1 Sanitation + § 308.1

**§ 302.1 Sanitation** (local revision in **bold**):
> Exterior property and premises shall be maintained in a clean, safe and sanitary condition. The occupant shall keep that part of the exterior property that such occupant occupies or controls in a clean and sanitary condition. **Except as otherwise specifically authorized by law, the open storage on residential property of any household appliance, motor vehicle parts, building materials, furniture, weeds, dead trees, garbage, rubbish, or similar items or materials, or residue [is prohibited]...**

**§ 308.1 Accumulation of rubbish or garbage**:
> Exterior property and premises, and the interior of every structure, shall be **free from any accumulation of rubbish or garbage**.

**Notes**:
- The local revision in § 302.1 explicitly enumerates **motor vehicle parts** and **building materials** as prohibited open storage on residential property. This means **auto parts on the ground around vehicles count as junk_trash** — which our v3 prompt got right.
- "Accumulation" is not quantitatively defined. The law says "any accumulation" → in principle even a small pile triggers a violation, but the statutory phrasing is permissive. Our v3 "≥3 distinct items OR clearly accumulated debris" threshold is one operationalization but is not directly anchored in the law.
- The threshold ambiguity (Jake 24% yes vs Eileen 33%) is consistent with "any accumulation" being read with different practical thresholds by different inspectors.

### 4. `overgrown_vegetation` — § 302.4 Weeds (has a quantitative anchor!)

> Premises and exterior property shall be maintained free from weeds or plant growth in excess of **ten (10) inches**. Noxious weeds shall be prohibited. Weeds shall be defined as all grasses, annual plants and vegetation, other than trees or shrubs provided; however, this term shall not include cultivated flowers and gardens.

**Notes**:
- **The law has a hard quantitative threshold: 10 inches.** Our SOP did not reference this. Labelers used subjective "property-scale unmaintained" judgment.
- Eileen yes-rate 57/98 vs Jake 43/98 vs Fechi (on her 56 photos) reflects different visual estimates of "excessive height", not different interpretations of the law.
- **This is the cleanest fix available**: amend the SOP to say "yes if visible vegetation appears to exceed ~knee height (≈ 18-20 inches, generously above the 10-inch legal threshold to allow visual estimation error)."
- Trees and shrubs are statutorily excluded — our v3 rule "trees at the edge don't count" matches this.
- Cultivated flowers/gardens are excluded — we never told labelers to distinguish.

### 5. `inoperable_vehicle` — § 302.8 Motor vehicles

> Except as provided for in other regulations, **inoperative or unlicensed motor vehicles shall not be parked, kept or stored** on any premises, and vehicles shall not at any time be in a state of **major disassembly, disrepair, or in the process of being stripped or dismantled**. Painting of vehicles is prohibited unless conducted inside an approved spray booth.

**Notes**:
- "Inoperative" is not defined. Common visual proxies: flat tires, no plates, broken windows, rust/heavy disrepair.
- The law explicitly includes "major disassembly" or "stripped or dismantled" — this is broader than our SOP, which focused on "abandoned/disrepair" cues.
- This class had the highest inter-rater κ (0.94 between Eileen and Jake on the 2-way segment). The visual cues are unambiguous.

### 6. `graffiti` — § 302.9 Defacement of property

> A person shall not willfully or wantonly damage, mutilate or deface any exterior surface of any structure or building on any private or public property by placing thereon any **marking, carving or graffiti**.

**Notes**:
- Statute frames graffiti as **willful defacement** — i.e., vandalism. This excludes legitimate signage, painted business names, etc.
- This validates our v1 → v2 → v3 evolution: img_065 (Arabic shop signage on white wall) was correctly identified by v1 as NOT graffiti, and the 3-run majority confirmed.
- High human κ (0.65) and high model agreement reflect the relatively clear legal definition: vandalism vs sanctioned marking.

### 7. `boarded_windows` — § 304.13 + § 108.2 (local cite)

**§ 304.13 Window, skylight and door frames**:
> Every window, skylight, door and frame shall be kept in **sound condition, good repair and weather tight**.

**Notes**:
- The law does not directly prohibit boarded windows; rather, it requires windows to be in sound/repaired/weather-tight condition. Boards over windows are typically a sign that the underlying glass/frame has failed.
- § 108.2 (referenced in folder name) is in the IPMC Chapter 1 *Administration* section, dealing with "Closing of vacant structures" — boarding is the remedy for unsafe vacant buildings, not the violation per se.
- Implication: a boarded window is **evidence** of an underlying violation (broken glass, unsafe structure) rather than the violation itself. Our SOP treated "windows covered with plywood" as the class — this is a useful operational definition but doesn't fully match the legal framing.

### 8. `broken_windows` — § 304.13.1 Glazing

> Glazing materials shall be maintained **free from cracks and holes**.

**Notes**:
- Crisp definition: cracks **or** holes. Both qualify.
- Our SOP wording ("visibly cracked, shattered, or missing window glass on a building") matches this exactly with one extension (missing glass = a hole).
- v2's "buildings only" rule does not appear in the legal text but is a reasonable operational restriction (vehicle windows are covered under § 302.8).

### 9. `damaged_roof_shingles` — § 304.7 Roofs and drainage

> The roof and flashing shall be **sound, tight and not have defects that admit rain**. Roof drainage shall be adequate to prevent dampness or deterioration in the walls or interior portion of the structure. Roof drains, gutters and downspouts shall be maintained in good repair and free from obstructions.

**Notes**:
- The law focuses on **roof functionality** (water tightness), not aesthetic of shingles per se. A roof with missing shingles but no rain admittance is technically not a violation; visible damage is the proxy.
- Our SOP focused on visible shingle damage. This is the right operational proxy because rain admittance is invisible from a single photo.
- Coverage gap matches our findings: 16 unsures out of 98 because roofs aren't visible from street angle.

---

## Summary table — SOP vs Law alignment

| Class | Has quantitative legal threshold? | SOP fully grounded? | Gap |
|---|---|---|---|
| `peeling_paint` | No (qualitative: "peeling, flaking, chipped") | Mostly | Need to clarify "applicable surfaces only" — not bare brick |
| `deteriorating_chimney` | No | Partial | Missed weather-coating clause; structural assessment unobservable |
| `junk_trash` | No (statute: "any accumulation") | Mostly | Threshold operationalization is judgment, not law |
| `overgrown_vegetation` | **Yes — 10 inches** | **No — SOP missed this** | **Add 10-inch (or ≥knee-height) anchor to SOP** |
| `inoperable_vehicle` | No (statute also covers "major disassembly") | Mostly | Could broaden to include disassembled vehicles even if technically operable |
| `graffiti` | No (qualitative: willful defacement) | Yes | Excludes legitimate signage — already aligned |
| `boarded_windows` | No | Operational only | Boards are evidence, not the violation; legal framing differs |
| `broken_windows` | No (qualitative: "cracks and holes") | Yes | Direct match |
| `damaged_roof_shingles` | No (functional: rain tightness) | Yes (visual proxy) | Aesthetic vs functional — proxy is appropriate |

---

## Implications

1. **Adding the 10-inch anchor to `overgrown_vegetation`** is the single highest-leverage SOP change. Eileen-Jake inter-rater κ on this class is currently 0.67 (after the 3-rule SOP). Anchoring to a measurable threshold could push it higher and align all annotators to the same legal standard.

2. **`peeling_paint` uncertainty is partially law-grounded**: the law itself only applies to "surfaces required to be coated", and "required to be coated" is a context-dependent judgment (was this brick wall ever painted?). Our model can't infer history. **HITL is the legally correct fallback here**, not better prompts.

3. **`junk_trash`'s "any accumulation" statutory framing** explicitly includes auto parts, building materials, furniture, weeds, dead trees, garbage, rubbish, or "similar items or materials". Our v3 prompt's threshold ("≥3 items") is *narrower* than the statute. We could relax the threshold to match law.

4. **`boarded_windows` framing**: the law treats boarding as remedy, not violation. For client communication, we should frame our class as "windows boarded over (suggesting underlying broken/missing glass or unsafe vacancy)" rather than implying boarding itself is the offense.

5. **For the final deliverable, we can now anchor every class to a Riverdale Park ordinance citation**. This dramatically increases the credibility of our SOP and the cascade pipeline's outputs.
