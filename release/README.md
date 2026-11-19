# Frozen checkpoint release package

Tag: **`models-v1.0.0`**. Repository:
[`mweneh/testiculus-ultrasound-classification`](https://github.com/mweneh/testiculus-ultrasound-classification).
Status: **prepared locally; not uploaded or published**. Intended use:
noncommercial research. [Redistribution review](WEIGHT_NOTICES.md) still needs
owner resolution; preparation does not grant a checkpoint licence.

## Prepared assets

Local directory: `artifacts/releases/models-v1.0.0/` (ignored).

| Asset | Bytes | Purpose |
|---|---:|---|
| `testiculus-custom-cnn-v1.pth` | 1,571,781 | Final scratch-trained Custom CNN |
| `testiculus-resnet18-v1.pth` | 44,787,339 | Final ImageNet-initialized ResNet18 |
| `testiculus-mobilenet-v3-small-v1.pth` | 6,204,077 | Final exploratory MobileNetV3-Small |
| `validation_thresholds.json` | 56,212 | Complete original two-model validation record |
| `mobilenet_v3_small_thresholds.json` | 28,398 | Complete separate exploratory validation record |
| `release_manifest.json` | See manifest | Pinned names, sizes, SHA-256 and model-to-threshold mapping |
| `SHA256SUMS` | See local file | Checksums of every other asset, including manifest |
| `RELEASE_NOTES.md` | See manifest | Models, dataset, frozen metrics, limits and intended use |
| `WEIGHT_NOTICES.md` | See manifest | Provenance, redistribution conditions and uncertainty |
| `DATASET_REFERENCES.bib` | 851 | Both required provider-supplied BibTeX entries |
| `TORCHVISION_LICENSE.txt` | See manifest | Complete BSD-3-Clause software notice |

The [source manifest](manifest.json) is the download trust reference and records
exact sizes for every payload asset. [SHA256SUMS](SHA256SUMS) is identical to the
packaged checksum file. [verification.json](verification.json) records CPU load
checks and runtime. Neither that verification record nor this maintainer guide
is a release asset. The allowlist excludes images, contact sheets, prediction
CSVs, histories, credentials, environments and unrelated artifacts.

All checkpoint and threshold files are copied without reserialization, tensor
changes, threshold edits or removal of their original metadata. Original
weights remain in `training_runs/`. Both threshold records preserve prediction
hashes, selection criteria, metrics and complete validation sweeps. The scripts
never open dataset partitions or run model inference/training.

## Reproduce locally

From the repository root, with original checkpoints present and the development
or serving environment installed:

```bash
OMP_NUM_THREADS=1 python -m src.release_assets prepare
OMP_NUM_THREADS=1 python -m src.release_assets verify \
  --package artifacts/releases/models-v1.0.0
(cd artifacts/releases/models-v1.0.0 && sha256sum -c SHA256SUMS)
OMP_NUM_THREADS=1 python -m pytest -q tests/test_release_assets.py
```

`prepare` refuses an existing output directory. To compare a second assembly
without replacing anything, supply `--output artifacts/releases/comparison`.
All payload bytes, including the manifest and checksum file, are reproducible;
filesystem modification times are not part of the release identity. Before
loading, source files must match the manifest's recorded hashes and sizes.
Model loading uses `weights_only=True`, strict CPU state restoration and
existing preprocessing/threshold validation. No forward pass is needed.

The 18 release tests pass, including the original checkpoint integration check
when those files are locally present. Source-only checkouts skip that one test.
Other tests use small synthetic byte fixtures and mocked HTTPS, not models or
medical images. The historical 150-pass application run and four report tests
are separate records; no full-suite rerun or new scientific evaluation is
claimed by this release work.

## Installation after publication

The optional public downloader requires only the standard library:

```bash
python3 -m src.release_assets download
```

It pins the repository, tag, names, sizes and checksums in the trusted checked-out
manifest. Downloaded control files are also compared to this source copy.
All assets are staged and checked before installing a new directory under
`artifacts/checkpoints/models-v1.0.0/`; failures leave no partial installation,
and existing destinations are never overwritten. `--verify-load` also checks
CPU loading when ML dependencies are installed. Otherwise run `verify` after
setting up the serving environment, as documented in the main README.

This public-URL script does not read or transmit credentials. Private/draft
assets require authorized GitHub CLI access and subsequent local verification.
The release has not been published, so a live download was not attempted;
unavailable-release handling and successful downloads are mocked in tests.
Use a trusted source checkout: hashes detect byte changes, not an independently
signed publisher identity. A different weight file requires a new reviewed
manifest/release rather than overwriting this frozen version.

## Commands for an authorized maintainer

**Do not execute these remote commands until explicit authorization to push,
create a release and publish has been given, and the redistribution questions
in `WEIGHT_NOTICES.md` are resolved.** This work has performed none of them.
The commands use the complete local asset allowlist, not the repository's
large data directories. Authenticate using your normal GitHub CLI workflow;
never put tokens in files or command examples.

First verify the local package, then push the reviewed source commit if
separately authorized:

```bash
OMP_NUM_THREADS=1 python -m src.release_assets verify \
  --package artifacts/releases/models-v1.0.0
git push origin main:main
```

Run from the clean, reviewed `main` checkout; `RELEASE_COMMIT` captures its full
commit SHA after all submission-readiness changes. The commit must be available
in the remote repository. The manifest's `experiment_source_revision` identifies
the earlier frozen experiment implementation, not the release-tag target.
`gh release create` creates a tag at that commit if it does not exist. Do not
retarget an existing tag or use `--clobber` to replace assets.

```bash
RELEASE_COMMIT=$(git rev-parse main)
test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$RELEASE_COMMIT"
ASSET_DIR=artifacts/releases/models-v1.0.0
gh release create models-v1.0.0 \
  --repo mweneh/testiculus-ultrasound-classification \
  --target "$RELEASE_COMMIT" \
  --draft --title 'Frozen TesticulUS research checkpoints v1.0.0' \
  --notes-file release/RELEASE_NOTES.md \
  "$ASSET_DIR/testiculus-custom-cnn-v1.pth" \
  "$ASSET_DIR/testiculus-resnet18-v1.pth" \
  "$ASSET_DIR/testiculus-mobilenet-v3-small-v1.pth" \
  "$ASSET_DIR/validation_thresholds.json" \
  "$ASSET_DIR/mobilenet_v3_small_thresholds.json" \
  "$ASSET_DIR/release_manifest.json" \
  "$ASSET_DIR/SHA256SUMS" \
  "$ASSET_DIR/RELEASE_NOTES.md" \
  "$ASSET_DIR/WEIGHT_NOTICES.md" \
  "$ASSET_DIR/DATASET_REFERENCES.bib" \
  "$ASSET_DIR/TORCHVISION_LICENSE.txt"
```

After checking the draft's uploaded files/checksums and receiving authorization
to publish it:

```bash
gh release edit models-v1.0.0 \
  --repo mweneh/testiculus-ultrasound-classification --draft=false
```

After publication, verify the tag resolves to `RELEASE_COMMIT`, inspect the
release metadata and check public downloads against the trusted source manifest:

```bash
git fetch origin tag models-v1.0.0
test "$(git rev-parse 'models-v1.0.0^{commit}')" = "$RELEASE_COMMIT"
gh release view models-v1.0.0 \
  --repo mweneh/testiculus-ultrasound-classification \
  --json tagName,isDraft,publishedAt,url,assets
python3 -m src.release_assets download --output artifacts/checkpoints/models-v1.0.0
OMP_NUM_THREADS=1 python -m src.release_assets verify \
  --package artifacts/checkpoints/models-v1.0.0
```

Use a new empty destination for the download check. The standard-library
downloader uses public URLs without credentials. Only after these checks succeed,
record the verified versioned URL and source revision in the README/report,
remove obsolete pending-status text and rebuild the PDF. Review remote source
paths for the report, API, Dockerfile, Grad-CAM, MobileNet, documentation and
tests after the push; local existence alone does not verify remote delivery.

If publication remains pending, the owner-confirmed alternative is Strathmore
University Virtual Campus — DSA 8401, **Capstone Project (23 Nov)** assignment
submission. Supply the matching source checkout and this complete 11-file
package, then verify examiner access. **Upload and access verification are
pending.** GitHub Releases remains preferred, with the repository URL submitted
through Virtual Campus. This names a fallback channel, not a completed delivery.

An immutable release setting, if available/enabled for the repository, can help
prevent later asset replacement. Check the repository settings rather than
assuming it is enabled. The command semantics above follow the official
[GitHub release guide](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
and [GitHub CLI reference](https://cli.github.com/manual/gh_release_create).

## Attribution and unresolved rights

Dataset: [TesticulUS](https://ditto.ing.unimore.it/testiculus/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), credited
to Morelli et al. and the contributors. Cite both Morelli et al. ICIAP 2025
(classification/synthetic pretraining) and BMVC 2026 (dataset/segmentation).
Provider BibTeX entries are preserved in the package; release notes identify the
ICIAP version of record (2026, pp. 26–39, DOI `10.1007/978-3-032-10192-1_3`).
Original dataset images
and adaptations, including modified ultrasound/Grad-CAM images, remain subject
to attribution, change notices, noncommercial use and applicable ShareAlike.

The dataset licence and Torchvision's BSD software notice do not by themselves
establish an unrestricted redistribution grant for trained parameter files.
The applicable checkpoint licence, possible dataset-derived obligations,
ImageNet-related terms and the developer's authority to release the weights
need owner review. The original ResNet initialization download cannot be
independently authenticated. ImageNet terms pages were unavailable during this
review. See [WEIGHT_NOTICES.md](WEIGHT_NOTICES.md) for primary sources and all
qualifications. Independently developed software has no repository-wide licence
selected in this checkout. No additional licence is silently assigned here.
