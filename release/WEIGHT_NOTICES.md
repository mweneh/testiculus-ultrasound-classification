# Model-weight provenance and redistribution review

Prepared for noncommercial research use; reviewed sources: 18 November 2026.
This notice records provenance and uncertainty. It does not invent a blanket
licence for the trained checkpoints or the independently developed software.
Publication remains subject to owner review and explicit authorization.

## Dataset material

Credit: Morelli et al. and the TesticulUS dataset contributors.

- Dataset: https://ditto.ing.unimore.it/testiculus/.
- Licence: CC BY-NC-SA 4.0,
  https://creativecommons.org/licenses/by-nc-sa/4.0/.
- Required citations: Morelli et al., ICIAP 2025, *Enhancing Testicular Ultrasound
  Image Classification Through Synthetic Data and Pretraining Strategies*;
  Morelli et al., BMVC 2026, *A New Multicenter Testicular US Dataset and a
  Lightweight Cond-UNet for Generalization in US Segmentation*.
  The supplied complete entries accompany the release as `DATASET_REFERENCES.bib`.

The dataset licence permits noncommercial sharing/adaptation with attribution,
change notices and applicable ShareAlike terms. It covers original dataset
images and their adaptations, including ultrasound Grad-CAM overlays and other
modified images. This package contains no dataset images or visualizations.
The dataset notice does not select a licence for independently developed code.
Whether, and to what extent, these trained parameter files constitute adapted
material subject to the dataset's ShareAlike terms is not established by the
available release documents. Do not infer either unrestricted checkpoint rights
or an exemption from applicable dataset terms. The owner should confirm the
appropriate checkpoint licence and distribution notices before publication.

## Initialization and software notices

| Model | Initialization / retained provenance |
|---|---|
| Custom CNN | Random initialization; trained by the project on TesticulUS labels. No ImageNet initialization. |
| ResNet18 | Colab record confirms ImageNet-pretrained initialization, followed by full fine-tuning. Current builder uses `ResNet18_Weights.DEFAULT`, resolving to `IMAGENET1K_V1` in the verified runtime. Original enum/runtime/download hash was not retained. Current official source: https://download.pytorch.org/models/resnet18-f37072fd.pth. |
| MobileNetV3-Small | `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth; frozen pretrained features and BatchNorm statistics, locally trained classifier. Initialization checksum retained in `reports/mobilenet_exploratory_results.json` in the source repository. |

Torchvision's software is BSD-3-Clause. The complete notice from the installed
0.29.1 distribution is included as `TORCHVISION_LICENSE.txt`; official source:
https://github.com/pytorch/vision/blob/v0.29.1/LICENSE. Retain its copyright,
conditions and disclaimer where applicable; do not imply contributor endorsement.
This software licence must not be assumed to settle pretrained-weight rights.

Torchvision explicitly says pretrained models may have their own licences or
terms derived from training datasets, and users must determine permission for
their use: https://docs.pytorch.org/vision/stable/models.html.
The two transfer-learning checkpoints include ImageNet-initialized parameters;
MobileNet retains the frozen feature extractor. ImageNet dataset access and
source-image rights are separate from Torchvision software licensing. ImageNet
terms pages (https://www.image-net.org/download.php and
https://image-net.org/download-images.php) were not retrievable during this
review. No checkpoint-specific redistribution grant or complete account/download
agreement for the original ImageNet weights was established. Noncommercial
intent alone does not resolve this uncertainty.

## Owner decisions still required

Confirm authority to distribute the project's trained checkpoints, applicable
TesticulUS/ShareAlike treatment of parameter files, ImageNet-derived weight
redistribution conditions and any original download/account terms. Select and
state an appropriate checkpoint licence after that review. The repository also
currently has no repository-wide software licence; third-party notices do not
license its independently developed code. This package records intended
noncommercial research use without granting unverified rights.

The hashes identify supplied bytes and detect corruption; they are not a
licence grant, digital signature or independent authentication of the original
Colab execution. Public release, upload, tag pushing and source pushing require
separate explicit authorization. No such action is performed by the packaging
or download utilities.
