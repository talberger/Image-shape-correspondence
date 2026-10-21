# VLPart part map + XSEG (2026-10-21, pod 51)

## Goal
- Find a segmentation that splits the horse's head from its chest (SAM 1 / SAM 3 / UnSAMv2 cannot), then re-run XSEG on it.

## Semantic-SAM
- Dropped before install: the released weights are SA-1B only (no part data), the same data as SAM.

## VLPart (part detector, fixed part vocabulary)
- First model that separates the head from the chest/body.
![[lvis56069_vlpart_parts.jpg]]
- Map options vs SAM 1 (merge 1.5% + fillseg). Tal chose pascal_part score>=0.5: 5 segments, body 52 / muzzle 23 / head 21 / 2 ears.
![[lvis56069_vlpart_map_options.jpg]]

## XSEG on the VLPart map (soft-snap, lower = better)
| arm | pt1 | pt2 | pt3 nose | mean | mean with SAM 1 map |
|---|---|---|---|---|---|
| ZS | 6.3 | 24.6 | 0.0 | 10.3 | |
| perseg (exp 0) | 12.8 | 15.8 | 3.3 | 10.6 | 18.0 |
| table | 11.8 | 21.5 | 2.3 | 11.9 | 12.6 |
| wonly | 9.7 | 20.1 | 19.9 | 16.6 | 23.1 |
| Fonly | 11.0 | 14.6 | 3.2 | 9.6 | 24.3 |
| SHIC | 2.3 | 18.3 | 0.0 | 6.9 | |

- perseg:
![[lvis56069_xseg_vlpart5_perseg.jpg]]
- table:
![[lvis56069_xseg_vlpart5_table.jpg]]
- wonly:
![[lvis56069_xseg_vlpart5_wonly.jpg]]
- Fonly:
![[lvis56069_xseg_vlpart5_Fonly.jpg]]

## Conclusions
- A better part map improves all 4 arms.
- Fonly is the first arm below ZS (9.6 vs 10.3), within cross-pod noise (~2); 1 instance, 1 seed.
- pt1 is worse than ZS in every arm.

## Next
- Tal reviews the figures.
- Second example (bear 159218), a second seed, and VLPart as an option in build_fine_segments.py.
