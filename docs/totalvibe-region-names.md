# Naming the eleven regions of `totalvibe:body_regions`

**A deferred piece of work, with the evidence already gathered.** Dataset 278 segments a body
into eleven regions and names them `"1"` … `"11"`. Upstream publishes no name mapping at all.
This note records what was measured on 2026-09-06, why the names are worth supplying, where
they belong, and how to derive them without guessing — so the work can be picked up cold.

## What the model is

From upstream's table (`robert-graf/VIBESegmentator`, README):

| ID  | Name                          | Modality       | Resolution |
|-----|-------------------------------|----------------|------------|
| 278 | Splits the body in 11 regions | **NAKO VIBE-only** | iso 4 mm |

The narrowest entry in that table — the flagship model 100 is listed as "MRI/CT". The repo's
own title carries an asterisk, "Full Body MRI\* Segmentation … \*works also on CT!".

The checkpoint agrees about the scale it expects. From its `dataset_fingerprint.json`:

- 1924 training cases, **every one at exactly 4 × 4 × 4 mm**
- median field of view **236 × 324 × 416 mm**, reaching 1088 mm in one axis
- patch 96³ at 4 mm — the network sees a **384 mm cube** at a time
- `channel_names: {"0": "any"}`, `transpose_forward: [0, 1, 2]`

The regions are **axis-aligned blocks by design**, not anatomical contours: a left/right ×
body-level grid. Upstream's `imgs/roi.jpg` is the only documentation of the scheme — neck at
top, both arms, then four cranio-caudal bands each split left and right. Eleven blocks.

## Measured behaviour (2026-09-06)

Three inputs through `totalvibe:body_regions` on MPS. "Outside the body" is against a filled
envelope (closing + hole fill), because a plain HU threshold calls lung parenchyma air — an
error that first produced a spurious 32 %.

| input | extent | regions | labels outside body | midline offset | boundary wander |
|---|---|---|---|---|---|
| ABIDE brain T1 (1 mm) | 170 × 217 × 151 mm | 7/11 | 7.8 % | 11.5 mm | 38 mm |
| AMOS abdominal MR | 380 × 344 × 180 mm | 6/11 | 0.6 % | — | — |
| **NLST chest CT (IDC)** | 360 × 360 × 323 mm | **9/11** | **0.3 %** | **2.1 mm** | **3.5 mm** |

Read: **field of view matters far more than modality.** The chest CT sits near the median
training extent and the split lands within ~2 mm of the body's own midline. The brain T1 is
below the minimum training extent in two axes and gets a rough bisection displaced by more
than a centimetre. Modality is nearly irrelevant — the CT result is the best of the three,
which supports upstream's asterisk holding for 278 as well.

Sampling caveat for anyone re-measuring: pick slices where both members of a pair genuinely
divide the body. A slice inside the transition band where one pair gives way to the next has
few rows containing both, and reports a misleadingly large offset (17.6 mm at z=40 against
2.1 mm at z=20 on the same series).

## Why supply names, and where they belong

`"1"` … `"11"` is honest but unusable: a caller cannot select a region, and
`--structures thorax` cannot work. The regions are stable and well-defined, so names are real
information, not decoration.

**This does not violate "the checkpoint is the spec".** That rule (CLAUDE.md) governs the
*weights manifests* — `*_weights.json` hold url, folder, tag and digest, never labels, so a
manifest cannot rot away from the checkpoint it describes. The *task catalog* is a different
file with different rules: `ts_tasks.json` declares an explicit `label_map` for **all 51**
TotalSegmentator tasks, because TS's names live in TS's Python source rather than in its
checkpoints. `TaskSpec.label_map` already accepts a declared map and `TaskCatalog._load`
already parses it (`tasks.py:160`).

The rule exists to stop a catalog restating what the checkpoint already says, which then goes
stale. Here the checkpoint says nothing. Supplying absent information is a different act from
shadowing present information.

## How to derive them — do not read them off the figure

Reading regions off `roi.jpg` by eye is how a confident wrong mapping ships. It happened twice
while gathering the evidence above. Derive instead:

1. Run `ts:total` (117 **named** structures) on the same NLST chest CT that gave the 9/11
   result. Its weights are already installed locally.
2. For each of the eleven regions, tabulate which named structures fall inside it, by voxel
   count and by fraction of each structure contained.
3. Names follow from the table: lungs and heart → thorax; liver, spleen, kidneys → abdomen;
   and critically **`kidney_left` vs `kidney_right` settles the left/right assignment by
   evidence** — a question left open twice during the session, because bilateral symmetry
   cannot distinguish a correct assignment from a mirrored one.
4. A single CT covers neck to abdomen only. The caudal regions (thighs, and whatever region 11
   is — it appeared as 1,527 voxels at the very edge of the CT) need a larger field of view,
   or stay unnamed. **Leave a region unnamed rather than guess it.**

Ship as `tools/gen_totalvibe_regions.py`, matching the house rule that catalogs are generated
and never hand-written, so it is re-runnable when upstream ships a new release.

## Requirements on the result

- **Marked as ours.** The names are derived, not upstream's. Provenance must say so, so nobody
  mistakes them for something the checkpoint stated.
- **Evidence retained.** Keep the organ-overlap table the names came from, so the next reader
  can check the derivation instead of trusting it.
- **Partial is acceptable.** Eight named regions and three honest `"9"`, `"10"`, `"11"` beats
  eleven plausible names, three of which are wrong.

## Also worth fixing while in here

The manifest entry describes 278 as `"body-region splitter (head, thorax, abdomen, ...)"` with
modality `MR`. Given upstream's "NAKO VIBE-only" and the measured behaviour, something closer
to *whole-body VIBE; needs a large field of view, works on CT* would warn callers off the
misuse performed twice in this session.
