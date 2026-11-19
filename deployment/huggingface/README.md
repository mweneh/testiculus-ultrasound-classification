# Hugging Face Docker deployment

**Status: prepared locally; no public application URL exists yet.** Account,
names, plan/spending approval and hosting-rights clearance are pending. The local
CLI reported `Not logged in`. No source push, private checkpoint transfer or
Space creation has been performed for this task. Do not present the URL templates
below as working services.

## Architecture and resources

One Docker Space serves the compiled React interface and FastAPI on
`0.0.0.0:7860`; the platform supplies public HTTPS. The same image still serves
local Docker/Compose on port 8000. `SPACE_README.md` supplies Docker SDK metadata,
`app_port: 7860` and a 10-minute startup allowance. UID/GID 1000 follows the Docker
Spaces permissions guidance. No GPU, analytics, external fonts or client-side
Hub credentials are used.

At startup, `src.serve` checks the private Hub repository's visibility and full
commit SHA, downloads only the 11 pinned package assets, checks byte sizes and
SHA-256 against `release/manifest.json`, and installs them atomically under the
application user's private directory (0700; files 0600). Existing caches are
reverified; failures stop startup. Redirects to HTTPS asset CDNs lose the Hub
Authorization header. The token is not written to disk and is removed from the
serving process environment after downloading. Socket reads time out after 30
seconds; a 180-second download budget is checked between chunks. A blocked read
can extend that budget by its socket timeout. Retrying requires a restart.

The selected models load once, using the **downloaded, verified model-specific
threshold configurations**. All three frozen checkpoints remain unchanged.
Checkpoint storage is separate from `/app/frontend/dist`; there is no file or
checkpoint download route. Source code publication, hosted inference and private
weight storage are three separate arrangements. The unpublished GitHub model
release is not required by this profile.

The official platform documentation currently lists CPU Basic as **2 vCPUs,
16 GB RAM, 50 GB ephemeral disk**, with no hourly compute charge. It also says
creating Docker/Gradio Spaces requires a paid plan; the current personal PRO
price is **US$9/month**, before applicable taxes. CPU Upgrade is listed at
US$0.03/hour and is **not** selected or authorized here. CPU Basic can sleep after
48 hours of inactivity; there is no always-on/SLA claim. Check live account
eligibility, storage allowance and pricing before approving any spend. The
checkpoint package is about 50 MiB; Python/PyTorch and the container image require
substantially more disk/RAM. Local measurements are recorded separately below;
the actual Space's capacity, cold starts and load behaviour remain unverified.

Official sources checked for preparation:
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[overview/resources](https://huggingface.co/docs/hub/spaces-overview),
[hardware/sleep](https://huggingface.co/docs/hub/spaces-gpus),
[pricing](https://huggingface.co/pricing),
[fine-grained tokens](https://huggingface.co/docs/hub/security-tokens),
[CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).

## Hosting-rights review before any checkpoint transfer

Private storage avoids offering weights for public download; it does not by
itself establish permission to upload them to a service provider. Hugging Face's
[terms, section “Your Content”](https://huggingface.co/terms-of-service) require
authority to grant service rights to uploaded content, including private content.
They also preserve applicable open-source terms. This review has not established
the necessary authority for these exact pretrained-derived files; this is an
unresolved condition, not a finding that research hosting is prohibited.

Before transferring files, obtain a documented owner/institutional determination
covering private third-party hosting and public noncommercial inference for the
actual checkpoints, the applicable TesticulUS ShareAlike treatment, ImageNet-derived
parameters and any original account/download conditions. The existing
[`WEIGHT_NOTICES.md`](../../release/WEIGHT_NOTICES.md) records those uncertainties.
Torchvision's BSD software licence does not settle weight permissions; its
[official model documentation](https://docs.pytorch.org/vision/stable/models.html)
explicitly requires checking permission for the particular use. No blanket licence
for project software or checkpoint contributions is invented here. Keep all
supplied attribution, references and notices with the private package. Do not
make its repository public under this deployment authorization.

Dataset credit: Morelli et al. and the TesticulUS contributors,
[TesticulUS](https://ditto.ing.unimore.it/testiculus/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Both mandatory ICIAP 2025 and BMVC 2026 citations accompany the package as
`DATASET_REFERENCES.bib` and remain in `data/references.bib`. The ICIAP version of
record is published in 2026, pp. 26–39,
[DOI](https://doi.org/10.1007/978-3-032-10192-1_3).
Original dataset images and adapted ultrasound visualizations, including
Grad-CAM overlays, require attribution, change identification and the applicable
noncommercial/ShareAlike terms when shared. This dataset licence is separate from
independent software and pretrained-component terms.

## Configuration

Set these in the Docker Space's Settings **after approval**:

| Name | Kind | Value / purpose |
|---|---|---|
| `DEPLOYMENT_MODE` | Variable | `space`; enables verified bootstrap and public API limits |
| `PORT` | Variable | `7860`, matching the Space README |
| `MODEL_REPO_ID` | Secret | Approved `namespace/private-model-repository`; avoids publishing its name as a public variable |
| `MODEL_REVISION` | Variable | Full 40-character commit SHA of the approved private model upload; never `main` or `latest` |
| `HF_TOKEN` | Secret | Fine-grained read access to **only** that private model repository; no write access |
| `MODEL_TYPES` | Variable | Default `custom_cnn,resnet18,mobilenet_v3_small`; reduced profile `resnet18,mobilenet_v3_small` |
| `MODEL_CACHE_DIR` | Variable, optional | Default `/home/inference/.local/share/testiculus`; writable, private, outside the static directory |
| `MODEL_PACKAGE_DIR` | Local verification only | Existing complete verified package mounted outside static files; bypasses network download |

`DEVICE=cpu` and `PUBLIC_MODE=1` are enforced by Space bootstrap. Downloaded
per-model paths are configured internally; do not set conflicting checkpoint,
`THRESHOLD_CONFIG` or per-model threshold variables. `FRONTEND_DIST` is normally
unset, using the image's compiled build. Do not set `VITE_*` secrets. Do not use
Docker ARG/ENV values, command-line tokens, browser storage or Git files for
credentials. Operator upload credentials must be separate from the runtime
read-only token. Revoke/rotate credentials through account settings when needed.

## Approved operator publication sequence (not executed)

1. Resolve the hosting-rights review above; approve the account, repository names,
   region/terms, plan and spending limit. Authenticate locally using `hf auth login`
   (interactive; never paste tokens in chat or command history). CLI help was
   checked with `hf` 1.16.1. Confirm the identity with `hf auth whoami`.
2. Verify the existing private package without evaluating models:

   ```bash
   OMP_NUM_THREADS=1 .venv/bin/python -m src.release_assets verify \
     --package artifacts/releases/models-v1.0.0
   ```

3. Create a **new private** model repository using the approved namespace and name:

   ```bash
   hf repos create "$MODEL_REPO_ID" --repo-type model --private
   ```

   Confirm its private visibility in the Hub **before transferring any file**.
   If it already exists, stop and inspect its visibility; `--private` does not
   change an existing repository's visibility. Confirm anonymous access is denied.
   Upload **only** the verified package, including notices and checksum manifest:

   ```bash
   hf upload "$MODEL_REPO_ID" artifacts/releases/models-v1.0.0 . \
     --repo-type model --private --commit-message "Store verified frozen capstone checkpoints privately"
   ```

   Record the returned full model repository commit SHA as `MODEL_REVISION`.
   Recheck private visibility. Do not upload raw images, histories or prediction CSVs.
4. Prepare the reviewed source from a clean committed deployment branch:

   ```bash
   .venv/bin/python -m deployment.prepare_space --output artifacts/huggingface-space
   ```

   This only creates a narrow source bundle, without datasets, checkpoints,
   credentials, local dependencies or test files. It adds the Space README and
   `SOURCE_REVISION.txt`. It refuses an existing output directory; use a new path
   for a later revision. Inspect the file list before upload.
5. Create a Docker Space **privately first**, selecting approved CPU Basic hardware:

   ```bash
   hf repos create "$SPACE_REPO_ID" --repo-type space --space-sdk docker \
     --flavor cpu-basic --private
   ```

   In its Settings, enter the variables and secrets above. Confirm the billing
   plan before proceeding. Then upload only the prepared source bundle:

   ```bash
   hf upload "$SPACE_REPO_ID" artifacts/huggingface-space . --repo-type space \
     --commit-message "Deploy reviewed CPU research application"
   ```

6. Inspect build/startup logs and resource metrics without printing secrets.
   Confirm readiness and model count. After private checks and approved public
   hosting, set **the Space only** to public in Settings. Keep the model repository
   private. Record both its exact public Space page and its HTTPS application URL.
7. Run the examiner/browser checks below anonymously. Only after they pass, add
   the actual public URLs and hosted measurements to the root README. Push source
   changes only after the required account/hosting arrangements are approved.

No publishing command above has been run by the preparation scripts.

## Local Docker reproduction

The prepared package includes all three checkpoints and both threshold files.
No Hugging Face account is needed for this **local** compatibility check:

```bash
docker build -t testiculus-inference:huggingface-review .
docker run --rm --cpus 2 --memory 2g --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m --cap-drop ALL \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:7860:7860 \
  --mount "type=bind,src=$PWD/artifacts/releases/models-v1.0.0,dst=/private/models,readonly" \
  -e DEPLOYMENT_MODE=space -e MODEL_PACKAGE_DIR=/private/models \
  testiculus-inference:huggingface-review
```

Open <http://127.0.0.1:7860>. To load fewer models add
`-e MODEL_TYPES=resnet18,mobilenet_v3_small` before the image name; the UI disables
Custom CNN. All 11 package assets are still verified, even with fewer models.
For a genuinely partial local checkpoint collection, use the existing ordinary
Docker profile in the root README and set only the available checkpoint variables.
`docker compose up --build` remains the original local all-model workflow on 8000.

## Verification and operational limits

The local acceptance record is `deployment/huggingface/verification.json`.
It must not be presented as a hosted deployment record. The pending live Hub
download, public HTTPS, anonymous browser and hosted memory checks remain open.

Observed locally with all **three** frozen models, a two-CPU quota and **2 GiB**
memory cap: startup to `/health` ready **20.38 s**; peak cgroup memory **482.39 MiB**;
process RSS approximately **378 MiB** at readiness and **370 MiB** after subsequent
requests. Cgroup current memory decreased from 471.68 to 383.77 MiB during this
short check; no OOM kills occurred. This supports keeping all models loaded for
the initial deployment; it does not establish long-term memory stability or
performance on Hugging Face's actual hardware. The two-model profile also loaded
successfully (23.99 s locally; peak 381.80 MiB). Startup measurements exclude the
live Hub download and image build.

| Model | Prediction HTTP latency | Explanation HTTP latency |
|---|---:|---:|
| Custom CNN | 0.756 s | 0.376 s |
| ResNet18 | 0.737 s | 0.815 s |
| MobileNetV3-Small | 0.539 s | 0.191 s |

These are single-request smoke timings, including HTTP, decode and preprocessing;
explanation also includes prediction. Cold/warm effects and other host activity
are uncontrolled, so they must not be compared as a model-speed benchmark. All
three probabilities matched local CPU inference exactly on the same licensed
training image. Frozen thresholds, class decisions and 224×224 Grad-CAM responses
matched. Separate headless Chrome checks with synthetic pixels passed for all
three models on desktop (1440×900) and mobile (390×844): no pre-action uploads,
console errors, third-party requests or horizontal overflow. Oversize-body
rejection, rate limiting, hidden documentation/checkpoint paths and closure of
a >1 MiB spooled synthetic upload passed. `/tmp/torchinductor_inference` is library
runtime state, not a retained upload. Temporary test containers were removed.

The regression suite passed **201 backend tests**, with **two synthetic optimizer
tests deliberately excluded**, plus **34 frontend** and **4 report** tests. The
existing Starlette TestClient deprecation warning remains. Production frontend
build, Compose configuration and the local Docker build passed. Python serving
dependencies were freshly downloaded in this build; base images and the frontend
dependency layer were cached. A fully uncached build and live Space build are not
claimed. The scientific report/PDF, original checkpoints, thresholds and frozen
results were preserved.

After public deployment, use an anonymized, appropriately licensed non-test
demonstration image and compare genuine API responses with the same local frozen
checkpoints. This helper saves no image/heatmap and sends no Hub token to the app:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m deployment.verify_app \
  --url "$PUBLIC_APP_URL" --image /path/to/authorized-non-test-demo.png \
  --package artifacts/releases/models-v1.0.0
```

For the local check add `--allow-local` and use `http://127.0.0.1:7860`.
The checked TesticulUS training image `inhomogeneous/128.png` is one possible
local demo, **not** a held-out example. Its pixels were manually reviewed for
visible identifiers. Credit/licence remain as above. Do not add its pixels or
generated overlay to source Git; uploading it to a public host also requires the
approved data-handling arrangement. The helper relies on the operator to confirm
non-test provenance and licensing; it does not discover/open any partition.

In an anonymous desktop/mobile browser, confirm `/`, `/health`, loaded-model
selection, image preview, model probability and frozen threshold, genuine Grad-CAM,
and recoverable API errors. Check console/CSP errors and network requests. Verify
`/docs`, `/redoc`, `/openapi.json` and guessed checkpoint paths return 404. Compare
hosted results with local inference (probability tolerance 1e-5, exact frozen
threshold and class). Observe actual Space memory before/after repeated requests
and cold startup, and record hardware, model count and the deployed source/model
commit SHAs. The historical CPU benchmark in the scientific report is unchanged;
these application timings are a separate operational measurement.

Public mode has a 5 MiB file limit, 20 MP image limit, PNG/JPEG decoding checks,
**6 MiB total body cap before parsing**, one file/no extra fields, and a 15-second
body-receive deadline. One inference/upload runs at a time; excess concurrent
work returns 503 with `Retry-After`. Twelve admitted requests per sliding minute
are shared globally; excess requests return 429. IP/forwarded headers are not
trusted or retained. Uvicorn limits active connections/tasks to 16, uses one
worker and a five-second keep-alive, and disables access logs in Space mode.
Model execution errors return generic messages, and model responses use `no-store`.
Multipart uploads may spool temporarily under `/tmp`; form/image contexts close
them after each request. The application does not persist uploads or heatmaps.
The host/proxy may keep infrastructure logs under its own privacy policy.

Native PyTorch execution is **not hard-preempted** by an asyncio timeout: its
admission slot is held until real computation completes, avoiding overlapping
abandoned CPU jobs. Upload duration, dimensions, model size and concurrency are
bounded; unexpected stuck native execution still needs a process/container
restart. The 30-second graceful-shutdown deadline is not an inference timeout.
These in-process controls are for a single-replica academic demo, not distributed
DDoS protection or a service availability guarantee. Global limits permit one
visitor to temporarily exhaust capacity. Multiple workers/replicas would require
shared controls and a new memory review. The actual hosting proxy's timeout and
abuse controls must be checked during live deployment.
