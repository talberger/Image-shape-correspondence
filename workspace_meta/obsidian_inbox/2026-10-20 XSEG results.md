---
date: 2026-10-20
tags: [experiment, xseg, horse56069]
---
# XSEG experiment — horse lvis56069 (2026-10-20)

**Question:** the nose (GT pt 3) collapses in two-stage training because cross-segment pairs use the 2D geodesic as teacher, which is wrong where parts are crowded in the photo (nose and belly are close in the image, far on the mesh).

**Arms** (one stage, 300 steps, current LVIS recipe):
- **perseg** (exp 0): only in-segment pairs (16 in + 0 out), no cross-segment pairs.
- **table / wonly / Fonly** (exp 1): 8 in + 8 out. In-segment pairs use `rkd_spatial_dist`; cross-segment pairs use feature RKD instead (teacher = cos of frozen photo features). The three arms differ in the gradient path: frozen phase-1 table / w only / F only. Weights calibrated to equal gradient norms.

**Results** (soft-snap geodesic error, lower = better; 3 GT points, 1 instance, 1 seed):

| arm | pt1 | pt2 | pt3 nose | mean |
|---|---|---|---|---|
| ZS | 6.3 | 24.6 | 0.0 | 10.3 |
| two-stage (baseline) | 7.4 | 15.3 | 95.1 | 39.3 |
| perseg | 18.1 | 25.8 | 10.1 | 18.0 |
| **table** | **4.1** | 27.7 | **6.0** | **12.6** |
| wonly | 26.2 | 22.3 | 20.9 | 23.1 |
| Fonly | 28.6 | 23.5 | 20.6 | 24.3 |
| SHIC | 2.3 | 18.3 | 0.0 | 6.9 |

**Takeaways:**
- Removing cross-segment pairs fixes the nose collapse (95.1 → 10.1) but hurts the body points.
- `table` keeps the nose and recovers pt1: best of our arms, still behind ZS and SHIC on mean.
- The circular arms (wonly, Fonly) are worse than perseg.
- Not reviewed by me yet.

**Problem found: segmentation.** The training map has only 4 segments; one is 68% of the horse and joins the head with the front body. Fine segments come from SAM 1 (automatic mask generator) + 1.5% merge. I'd prefer SAM 3 for the parts too. SAM 3 has no automatic mode, but a point-grid wrapper could do it (generic); text prompts per part would not be generic.

**Next:** review the figures, fix the segmentation (SAM 3?), bear 159218 as second example.

Figures (pod): `demo/shic/xseg/horse_lvis56069/<arm>/summary/SUMMARY_lvis56069.png`
