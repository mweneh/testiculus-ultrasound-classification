# Image split and similarity audit

## Split summary

- Seed: `42`
- Protocol: provisional, stratified image-level 70/15/15 split.
- Patient/examination and centre identifiers are unavailable; patient leakage and centre overlap cannot be assessed.
- Labelled images included: 860.
- Exact duplicate groups: 0 (0 images).
- Exact pixel duplicates are kept in one partition.

| Partition | Total | Homogeneous (0) | Inhomogeneous (1) |
|---|---:|---:|---:|
| train | 602 | 488 | 114 |
| validation | 129 | 104 | 25 |
| test | 129 | 104 | 25 |

## Perceptual similarity screen

The screen uses a 64-bit grayscale difference hash (dHash); candidates have Hamming distance ≤ 4. This is a triage heuristic, not proof of duplicate acquisition or patient identity. Candidates need visual human review. No patient linkage is inferred.

Candidate pairs flagged: 22.
- `homogeneous/109.png` ↔ `homogeneous/127.png` (dHash distance 3; labels 0/0; partitions validation/train).
- `homogeneous/111.png` ↔ `homogeneous/393.png` (dHash distance 4; labels 0/0; partitions test/train).
- `homogeneous/118.png` ↔ `homogeneous/227.png` (dHash distance 3; labels 0/0; partitions train/test).
- `homogeneous/118.png` ↔ `homogeneous/417.png` (dHash distance 3; labels 0/0; partitions train/train).
- `homogeneous/121.png` ↔ `homogeneous/411.png` (dHash distance 4; labels 0/0; partitions train/train).
- `homogeneous/13.png` ↔ `inhomogeneous/147.png` (dHash distance 4; labels 0/1; partitions validation/train).
- `homogeneous/187.png` ↔ `homogeneous/323.png` (dHash distance 4; labels 0/0; partitions train/validation).
- `homogeneous/208.png` ↔ `homogeneous/288.png` (dHash distance 4; labels 0/0; partitions train/test).
- `homogeneous/312.png` ↔ `inhomogeneous/66.png` (dHash distance 4; labels 0/1; partitions train/train).
- `homogeneous/393.png` ↔ `homogeneous/58.png` (dHash distance 3; labels 0/0; partitions train/train).
- `homogeneous/393.png` ↔ `inhomogeneous/818.png` (dHash distance 4; labels 0/1; partitions train/validation).
- `homogeneous/419.png` ↔ `homogeneous/467.png` (dHash distance 4; labels 0/0; partitions train/train).
- `homogeneous/476.png` ↔ `homogeneous/478.png` (dHash distance 1; labels 0/0; partitions train/train).
- `homogeneous/477.png` ↔ `homogeneous/479.png` (dHash distance 0; labels 0/0; partitions train/train).
- `homogeneous/507.png` ↔ `homogeneous/605.png` (dHash distance 3; labels 0/0; partitions train/train).
- `homogeneous/561.png` ↔ `homogeneous/577.png` (dHash distance 4; labels 0/0; partitions validation/test).
- `homogeneous/571.png` ↔ `inhomogeneous/589.png` (dHash distance 4; labels 0/1; partitions test/train).
- `homogeneous/575.png` ↔ `homogeneous/595.png` (dHash distance 4; labels 0/0; partitions train/validation).
- `homogeneous/660.png` ↔ `homogeneous/706.png` (dHash distance 4; labels 0/0; partitions train/test).
- `homogeneous/687.png` ↔ `homogeneous/835.png` (dHash distance 4; labels 0/0; partitions train/train).
- `homogeneous/791.png` ↔ `homogeneous/811.png` (dHash distance 4; labels 0/0; partitions train/train).
- `homogeneous/856.png` ↔ `inhomogeneous/334.png` (dHash distance 4; labels 0/1; partitions test/train).

Near-hash candidates were not automatically grouped because similarity does not establish same-patient identity. Human review is pending; review these candidates and update the grouping policy before training if the review identifies duplicates.
