"""Select a small, documented set of held-out test images for interface demos.

Selection uses only the saved test-partition predictions of the frozen models;
no model is trained, tuned or re-thresholded. Images flagged by the similarity
screen are skipped so a demo image has no known near-duplicate elsewhere.
The selected files are copied under neutral names (the web interface shows the
uploaded filename) into a git-ignored folder with a README answer key.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

MODELS = ("custom_cnn", "resnet18", "mobilenet_v3_small")
MODEL_NAMES = {"custom_cnn": "Custom CNN", "resnet18": "ResNet18", "mobilenet_v3_small": "MobileNetV3-Small"}
CLASS_NAMES = {0: "homogeneous", 1: "inhomogeneous"}


@dataclass(frozen=True)
class Case:
    """A demo scenario: a true label and the predictions expected from each model."""

    key: str
    label: int
    predictions: tuple[int, int, int]  # ordered as MODELS
    count: int
    description: str


CASES = (
    Case("agree-homogeneous", 0, (0, 0, 0), 2, "Homogeneous; all three models correct"),
    Case("agree-inhomogeneous", 1, (1, 1, 1), 2, "Inhomogeneous; all three models detect it"),
    Case("resnet-only-detects", 1, (0, 1, 0), 1, "Inhomogeneous; only ResNet18 detects it"),
    Case("mobilenet-only-correct", 0, (1, 1, 0), 1, "Homogeneous; Custom CNN and ResNet18 raise a false positive"),
    Case("all-miss", 1, (0, 0, 0), 1, "Inhomogeneous; missed by all three models"),
)


def load_predictions(results_dir: Path) -> dict[str, dict]:
    """Join the saved per-image test predictions of every model by image path."""
    joined: dict[str, dict] = {}
    for model in MODELS:
        with (results_dir / f"{model}_test" / "predictions.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                entry = joined.setdefault(row["image_path"], {"label": int(row["label"]), "models": {}})
                if entry["label"] != int(row["label"]):
                    raise ValueError(f"Inconsistent label for {row['image_path']}")
                entry["models"][model] = {
                    "score": float(row["probability_class_1"]),
                    "threshold": float(row["threshold"]),
                    "prediction": int(row["prediction"]),
                }
    incomplete = [path for path, entry in joined.items() if set(entry["models"]) != set(MODELS)]
    if incomplete:
        raise ValueError(f"Images missing a model prediction: {incomplete[:5]}")
    return joined


def load_flagged_images(audit_path: Path) -> set[str]:
    """Return every image named in a similarity-screen candidate pair."""
    pairs = re.findall(r"`([a-z]+/\d+\.png)` ↔ `([a-z]+/\d+\.png)`", audit_path.read_text())
    return {path for pair in pairs for path in pair}


def clarity(entry: dict) -> float:
    """Smallest distance between any model's score and its threshold (larger is clearer)."""
    return min(abs(result["score"] - result["threshold"]) for result in entry["models"].values())


def select_cases(predictions: dict[str, dict], test_paths: set[str], excluded: set[str]) -> list[tuple[Case, str]]:
    """Pick the clearest examples of each case from the test partition, deterministically."""
    unknown = set(predictions) - test_paths
    if unknown:
        raise ValueError(f"Predictions include non-test images: {sorted(unknown)[:5]}")
    selected = []
    for case in CASES:
        candidates = [
            path
            for path, entry in predictions.items()
            if path not in excluded
            and entry["label"] == case.label
            and tuple(entry["models"][model]["prediction"] for model in MODELS) == case.predictions
        ]
        candidates.sort(key=lambda path: (-clarity(predictions[path]), path))
        if len(candidates) < case.count:
            raise ValueError(f"Only {len(candidates)} candidates for case {case.key}")
        selected.extend((case, path) for path in candidates[: case.count])
    return selected


def write_readme(out_dir: Path, rows: list[dict]) -> None:
    header = "| File | Source | True label | Case | " + " | ".join(MODEL_NAMES[m] for m in MODELS) + " |"
    lines = [
        "# Demo images (local only, not tracked)",
        "",
        "Held-out **test-partition** images from TesticulUS, selected by `python -m src.demo_images` from the saved",
        "predictions of the frozen models. Files are renamed so the interface does not reveal the label.",
        "Each model cell shows the saved score for the inhomogeneous class, the model's frozen threshold, and whether",
        "the saved prediction was correct. These are illustrative examples chosen for contrast, not a performance estimate.",
        "",
        header,
        "|" + "---|" * (4 + len(MODELS)),
    ]
    for row in rows:
        cells = [
            f"{result['score']:.3f} / {result['threshold']:.3f} {'✓' if result['prediction'] == row['label'] else '✗'}"
            for result in (row["models"][model] for model in MODELS)
        ]
        lines.append(
            f"| `{row['file']}` | `{row['source']}` | {CLASS_NAMES[row['label']]} | {row['description']} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "Images flagged by the similarity screen (`reports/similarity_audit.md`) were not eligible.",
        "",
        "Credit: TesticulUS by Morelli et al. and the dataset contributors, https://ditto.ing.unimore.it/testiculus/,",
        "licensed CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/). Images are unmodified copies;",
        "see `data/README.md` for the required citations. Research and educational use only; not for clinical diagnosis.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def verify_with_checkpoints(rows: list[dict], out_dir: Path, checkpoints: dict[str, Path], tolerance: float) -> None:
    """Re-score the copied files with the release checkpoints and compare with the saved scores."""
    from PIL import Image

    from src.inference import CheckpointPredictor

    for model, checkpoint in checkpoints.items():
        predictor = CheckpointPredictor(checkpoint, device="cpu")
        if predictor.model_type != model:
            raise ValueError(f"{checkpoint} holds {predictor.model_type}, expected {model}")
        for row in rows:
            with Image.open(out_dir / row["file"]) as image:
                live = predictor.predict(image)
            saved = row["models"][model]
            if abs(live["probability_class_1"] - saved["score"]) > tolerance or live["prediction"] != saved["prediction"]:
                raise ValueError(f"{model} on {row['file']}: live {live['probability_class_1']:.6f}, saved {saved['score']:.6f}")
        print(f"verified {MODEL_NAMES[model]}: {len(rows)} images match the saved predictions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--test-split", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--similarity-audit", type=Path, default=Path("reports/similarity_audit.md"))
    parser.add_argument("--out-dir", type=Path, default=Path("demo_images"))
    parser.add_argument("--checkpoint-dir", type=Path, help="Release checkpoint folder; re-scores the copies to confirm the saved predictions")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    with args.test_split.open(newline="") as handle:
        test_paths = {row["image_path"] for row in csv.DictReader(handle)}
    predictions = load_predictions(args.results_dir)
    selected = select_cases(predictions, test_paths, load_flagged_images(args.similarity_audit))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (case, source) in enumerate(selected, start=1):
        name = f"demo-{index:02d}.png"
        shutil.copyfile(args.data_dir / source, args.out_dir / name)
        rows.append({"file": name, "source": source, "case": case.key, "description": case.description, **predictions[source]})
    write_readme(args.out_dir, rows)
    (args.out_dir / "selection.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} images and an answer key to {args.out_dir}/")

    if args.checkpoint_dir:
        checkpoints = {
            "custom_cnn": args.checkpoint_dir / "testiculus-custom-cnn-v1.pth",
            "resnet18": args.checkpoint_dir / "testiculus-resnet18-v1.pth",
            "mobilenet_v3_small": args.checkpoint_dir / "testiculus-mobilenet-v3-small-v1.pth",
        }
        verify_with_checkpoints(rows, args.out_dir, checkpoints, args.tolerance)


if __name__ == "__main__":
    main()
