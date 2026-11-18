# TesticulUS dataset documentation

## Dataset attribution

Credit: Morelli et al. and the TesticulUS dataset contributors.

- Dataset: [TesticulUS](https://ditto.ing.unimore.it/testiculus/).
- Licence: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Required publications: Morelli et al., *Enhancing Testicular Ultrasound Image
  Classification Through Synthetic Data and Pretraining Strategies*, ICIAP 2025;
  and Morelli et al., *A New Multicenter Testicular US Dataset and a Lightweight
  Cond-UNet for Generalization in US Segmentation*, BMVC 2026.

The provider's complete BibTeX entries are in [references.bib](references.bib),
under `2025ICIAP` and `2026BMVC`, as supplied by the official dataset page.
Cite both publications in reports, presentations and shared dataset adaptations.

## Sharing images and adaptations

CC BY-NC-SA 4.0 permits noncommercial copying, sharing and adaptation with credit
and ShareAlike. Original images remain under that licence. Grad-CAM
visualizations derived from these images, overlays, crops, annotations and other
modified ultrasound images are adaptations. When distributing them, retain
supplied attribution notices, link the source and licence, indicate changes,
cite both publications and share the adaptations under CC BY-NC-SA 4.0 or a
compatible licence allowed by its ShareAlike terms. Do not imply creator
endorsement.

Example credit for an adapted visualization (replace the change description
with the transformations actually applied):

> Adapted from TesticulUS by Morelli et al. and the dataset contributors:
> https://ditto.ing.unimore.it/testiculus/. Original and adaptation: CC BY-NC-SA
> 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/. Changes: resized and
> overlaid with a Grad-CAM attribution map by this project's authors. References:
> Morelli et al., ICIAP 2025 (`2025ICIAP`) and BMVC 2026 (`2026BMVC`).

Keep the full references with a shared figure or in its accompanying document.
These terms apply to dataset content and its adaptations. Independently
developed software is governed by its own applicable software licence and
third-party notices. This checkout currently declares no repository-wide
software licence; the dataset notice does not select one or grant a software
licence. Dependencies and pretrained weights retain their own applicable terms.

## Local copy and frozen experiments

The local labelled copy contains 696 homogeneous images (class 0) and 164
inhomogeneous images (positive class 1), plus 200 unlabelled images. The tracked
split manifests record 602 training, 129 validation and 129 test images.
Unlabelled images and segmentation masks are not classification inputs.

The official page describes 1,053 real images and a separate synthetic release;
the local copy contains 1,060 images across the image folders. No archive
checksum or release identifier establishes a byte-for-byte version match.
The official licence is documented above; local release provenance remains a
separate reproducibility limitation. Verified patient and centre mappings are
unavailable, so the local image-level split does not establish patient or centre
independence.

Raw images, generated contact sheets and Grad-CAM outputs remain outside Git
under the repository's source-control policy. This policy does not override the
licence's permission to share attributed noncommercial adaptations. No further
images are removed and no Git history is rewritten for this documentation
update. Trained checkpoints, selected thresholds, manifests and experimental
results are preserved; no retraining or held-out image evaluation is needed.
