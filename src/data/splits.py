"""Create reproducible, image-level classification splits and similarity audits.

The split is provisional because patient and acquisition-centre identifiers are
not available. Exact pixel duplicates are assigned together; perceptual-hash
matches are reported for human review and are not interpreted as patient links.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_TO_LABEL = {"homogeneous": 0, "inhomogeneous": 1}
DEFAULT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def _discover_images(data_dir: Path) -> list[dict]:
    records = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.glob("*.png")):
            relative_path = image_path.relative_to(data_dir).as_posix()
            try:
                with Image.open(image_path) as image:
                    pixels = np.asarray(image.convert("L"))
            except Exception as exc:
                raise ValueError(f"Could not read {image_path}: {exc}") from exc

            pixel_digest = hashlib.sha256(
                f"L:{pixels.shape[1]}x{pixels.shape[0]}:".encode()
                + pixels.tobytes()
            ).hexdigest()
            records.append({
                "image_path": relative_path,
                "label": label,
                "pixel_digest": pixel_digest,
                "dhash": _difference_hash(pixels),
            })
    if not records:
        raise ValueError(f"No labelled PNG images found under {data_dir}")
    return records


def _difference_hash(pixels: np.ndarray) -> int:
    """Return a 64-bit difference hash of a grayscale image."""
    image = Image.fromarray(pixels).resize((9, 8), Image.Resampling.LANCZOS)
    small = np.asarray(image, dtype=np.int16)
    bits = small[:, 1:] > small[:, :-1]
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def audit_similarities(data_dir: Path, near_threshold: int = 4) -> dict:
    """Audit exact pixel duplicates and close 64-bit difference hashes."""
    if not 0 <= near_threshold <= 64:
        raise ValueError("near_threshold must be between 0 and 64")
    records = _discover_images(Path(data_dir))
    exact_groups_by_digest = defaultdict(list)
    for record in records:
        exact_groups_by_digest[record["pixel_digest"]].append(record)
    exact_groups = [
        group for group in exact_groups_by_digest.values() if len(group) > 1
    ]

    near_pairs = []
    for index, first in enumerate(records):
        for second in records[index + 1:]:
            if first["pixel_digest"] == second["pixel_digest"]:
                continue
            distance = (first["dhash"] ^ second["dhash"]).bit_count()
            if distance <= near_threshold:
                near_pairs.append({
                    "image_path_a": first["image_path"],
                    "label_a": first["label"],
                    "image_path_b": second["image_path"],
                    "label_b": second["label"],
                    "dhash_distance": distance,
                })
    return {
        "records": records,
        "exact_groups": exact_groups,
        "near_pairs": near_pairs,
        "near_threshold": near_threshold,
    }


def _assign_groups(groups: list[list[dict]], seed: int) -> dict[str, list[dict]]:
    """Greedily balance indivisible exact-image groups across three splits."""
    split_names = tuple(DEFAULT_RATIOS)
    total = sum(len(group) for group in groups)
    targets = {name: total * DEFAULT_RATIOS[name] for name in split_names}
    rng = random.Random(seed)
    ordered_groups = list(groups)
    rng.shuffle(ordered_groups)
    ordered_groups.sort(key=len, reverse=True)

    assignments = {name: [] for name in split_names}
    counts = {name: 0 for name in split_names}
    for group in ordered_groups:
        candidates = list(split_names)
        rng.shuffle(candidates)
        # Select the split with the smallest normalized squared size error.
        destination = min(
            candidates,
            key=lambda name: (
                ((counts[name] + len(group) - targets[name])
                 / max(targets[name], 1)) ** 2
                + sum(
                    ((counts[other] - targets[other])
                     / max(targets[other], 1)) ** 2
                    for other in split_names if other != name
                )
            ),
        )
        assignments[destination].extend(group)
        counts[destination] += len(group)
    return assignments


def make_splits(data_dir: Path, output_dir: Path, seed: int = 42,
                near_threshold: int = 4,
                report_path: Path | None = None) -> tuple[dict, dict]:
    """Create stratified 70/15/15 manifests, keeping exact duplicates together."""
    audit = audit_similarities(data_dir, near_threshold=near_threshold)
    groups_by_label = {label: [] for label in CLASS_TO_LABEL.values()}
    for group in audit["exact_groups"]:
        labels = {record["label"] for record in group}
        if len(labels) != 1:
            paths = ", ".join(record["image_path"] for record in group)
            raise ValueError(
                "Exact pixel duplicates have conflicting labels; resolve the "
                f"label conflict before splitting: {paths}"
            )
        groups_by_label[next(iter(labels))].append(group)

    exact_members = {
        record["image_path"]
        for group in audit["exact_groups"]
        for record in group
    }
    for record in audit["records"]:
        if record["image_path"] not in exact_members:
            groups_by_label[record["label"]].append([record])

    # Split each class separately to retain stratification while keeping each
    # exact duplicate group indivisible.
    assigned = {name: [] for name in DEFAULT_RATIOS}
    for label in sorted(groups_by_label):
        class_assignment = _assign_groups(
            groups_by_label[label], seed=seed + int(label)
        )
        for split_name, records in class_assignment.items():
            assigned[split_name].extend(records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in assigned.items():
        records = sorted(records, key=lambda record: record["image_path"])
        path = output_dir / f"{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("image_path", "label"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {key: record[key] for key in ("image_path", "label")}
                for record in records
            )
        assigned[split_name] = records

    if report_path is not None:
        _write_audit_report(audit, assigned, Path(report_path), seed)
    return assigned, audit


def _write_audit_report(audit: dict, splits: dict, report_path: Path,
                        seed: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partition_for_path = {
        record["image_path"]: split_name
        for split_name, records in splits.items()
        for record in records
    }
    lines = [
        "# Image split and similarity audit",
        "",
        "## Split summary",
        "",
        f"- Seed: `{seed}`",
        "- Protocol: provisional, stratified image-level 70/15/15 split.",
        "- Patient/examination and centre identifiers are unavailable; patient leakage "
        "and centre overlap cannot be assessed.",
        f"- Labelled images included: {len(audit['records'])}.",
        f"- Exact duplicate groups: {len(audit['exact_groups'])} "
        f"({sum(len(group) for group in audit['exact_groups'])} images).",
        "- Exact pixel duplicates are kept in one partition.",
        "",
        "| Partition | Total | Homogeneous (0) | Inhomogeneous (1) |",
        "|---|---:|---:|---:|",
    ]
    for name, records in splits.items():
        counts = {label: sum(record["label"] == label for record in records)
                  for label in (0, 1)}
        lines.append(f"| {name} | {len(records)} | {counts[0]} | {counts[1]} |")
    lines.extend([
        "",
        "## Perceptual similarity screen",
        "",
        "The screen uses a 64-bit grayscale difference hash (dHash); candidates "
        f"have Hamming distance ≤ {audit['near_threshold']}. This is a triage "
        "heuristic, not proof of duplicate acquisition or patient identity. "
        "Candidates need visual human review. No patient linkage is inferred.",
        f"",
        f"Candidate pairs flagged: {len(audit['near_pairs'])}.",
    ])
    for pair in audit["near_pairs"]:
        lines.append(
            f"- `{pair['image_path_a']}` ↔ `{pair['image_path_b']}` "
            f"(dHash distance {pair['dhash_distance']}; labels "
            f"{pair['label_a']}/{pair['label_b']}; partitions "
            f"{partition_for_path[pair['image_path_a']]}/"
            f"{partition_for_path[pair['image_path_b']]})."
        )
    lines.extend([
        "",
        "Near-hash candidates were not automatically grouped because similarity "
        "does not establish same-patient identity. Human review is pending; "
        "review these candidates and update the grouping policy before training "
        "if the review identifies duplicates.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--report-path", type=Path,
        default=Path("reports/similarity_audit.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=int, default=4)
    args = parser.parse_args()
    splits, audit = make_splits(
        args.data_dir, args.output_dir, args.seed, args.near_threshold,
        args.report_path,
    )
    for split_name, records in splits.items():
        counts = {label: sum(row["label"] == label for row in records)
                  for label in (0, 1)}
        print(f"{split_name}: {len(records)} images; class counts {counts}")
    print(f"Exact duplicate groups: {len(audit['exact_groups'])}")
    print(f"Perceptual candidate pairs: {len(audit['near_pairs'])}")


if __name__ == "__main__":
    main()
