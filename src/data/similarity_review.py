"""Verify split manifests and create a visual sheet for dHash review pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.data.splits import CLASS_TO_LABEL, audit_similarities


PARTITIONS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
PAIR_PATTERN = re.compile(
    r"- `([^`]+)` ↔ `([^`]+)` \(dHash distance (\d+); labels ([01])/([01]); "
    r"partitions (train|validation|test)/(train|validation|test)\)\."
)


def read_manifests(split_dir: Path) -> dict[str, list[dict]]:
    """Load manifest rows without changing any split artifacts."""
    manifests = {}
    for partition in PARTITIONS:
        manifest_path = Path(split_dir) / f"{partition}.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing {partition} manifest: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            manifests[partition] = list(csv.DictReader(stream))
    return manifests


def parse_audit_report(report_path: Path) -> dict:
    """Parse split summaries and review-pair entries from the Markdown audit."""
    text = Path(report_path).read_text(encoding="utf-8")
    total_match = re.search(r"- Labelled images included: (\d+)\.", text)
    group_match = re.search(
        r"- Exact duplicate groups: (\d+) \((\d+) images\)\.", text
    )
    pair_count_match = re.search(r"Candidate pairs flagged: (\d+)\.", text)
    threshold_match = re.search(r"Hamming distance ≤ (\d+)", text)
    if not all((total_match, group_match, pair_count_match, threshold_match)):
        raise ValueError(f"Audit report is missing required summary fields: {report_path}")

    split_summary = {}
    for row in re.finditer(
        r"^\| (train|validation|test) \| (\d+) \| (\d+) \| (\d+) \|$",
        text,
        re.MULTILINE,
    ):
        split_summary[row.group(1)] = {
            "total": int(row.group(2)),
            0: int(row.group(3)),
            1: int(row.group(4)),
        }
    pairs = []
    for match in PAIR_PATTERN.finditer(text):
        pairs.append({
            "image_path_a": match.group(1),
            "image_path_b": match.group(2),
            "dhash_distance": int(match.group(3)),
            "label_a": int(match.group(4)),
            "label_b": int(match.group(5)),
            "partition_a": match.group(6),
            "partition_b": match.group(7),
        })
    return {
        "total": int(total_match.group(1)),
        "exact_duplicate_groups": int(group_match.group(1)),
        "exact_duplicate_images": int(group_match.group(2)),
        "candidate_count": int(pair_count_match.group(1)),
        "near_threshold": int(threshold_match.group(1)),
        "split_summary": split_summary,
        "pairs": pairs,
    }


def _labelled_images(data_dir: Path) -> dict[str, int]:
    expected = {}
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.glob("*.png")):
            relative_path = image_path.relative_to(data_dir).as_posix()
            expected[relative_path] = label
    return expected


def verify_split_integrity(
    data_dir: Path,
    split_dir: Path,
    audit_path: Path,
    *,
    expected_total: int | None = 860,
    expected_class_counts: dict[int, int] | None = None,
    distribution_tolerance: float = 0.015,
) -> dict:
    """Check full coverage, uniqueness, labels, stratification, and audit agreement."""
    if expected_class_counts is None and expected_total == 860:
        expected_class_counts = {0: 696, 1: 164}

    expected_paths = _labelled_images(data_dir)
    manifests = read_manifests(split_dir)
    partition_for_path = {}
    labels_for_path = {}
    occurrences = Counter()
    counts = {
        partition: {"total": 0, 0: 0, 1: 0}
        for partition in PARTITIONS
    }
    errors = []
    all_paths_resolve = True

    for partition, rows in manifests.items():
        for row_number, row in enumerate(rows, start=2):
            relative_path = row.get("image_path", "")
            if not relative_path:
                errors.append(f"{partition}.csv:{row_number} has an empty image_path")
                continue
            path = Path(relative_path)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                errors.append(f"Invalid relative image path: {relative_path}")
                all_paths_resolve = False
                continue
            expected_label = expected_paths.get(path.as_posix())
            if expected_label is None:
                errors.append(f"Image path is not in a labelled class folder: {relative_path}")
                all_paths_resolve = False
                continue
            if not (Path(data_dir) / path).is_file():
                errors.append(f"Image path does not resolve: {relative_path}")
                all_paths_resolve = False
            try:
                label = int(row.get("label", ""))
            except (TypeError, ValueError):
                errors.append(f"Invalid label in {partition}.csv:{row_number}")
                continue
            if label not in (0, 1):
                errors.append(f"Invalid binary label {label} for {relative_path}")
                continue
            if label != expected_label:
                errors.append(
                    f"Label mismatch for {relative_path}: manifest {label}, "
                    f"class folder {expected_label}"
                )
            occurrences[path.as_posix()] += 1
            partition_for_path.setdefault(path.as_posix(), partition)
            labels_for_path[path.as_posix()] = label
            counts[partition]["total"] += 1
            counts[partition][label] += 1

    duplicate_paths = sorted(path for path, count in occurrences.items() if count > 1)
    missing_paths = sorted(set(expected_paths) - set(occurrences))
    unexpected_paths = sorted(set(occurrences) - set(expected_paths))
    if duplicate_paths:
        errors.append(f"{len(duplicate_paths)} image paths occur more than once")
    if missing_paths:
        errors.append(f"{len(missing_paths)} labelled image paths are missing from manifests")
    if unexpected_paths:
        errors.append(f"{len(unexpected_paths)} manifest paths are not in the labelled dataset")

    observed_class_counts = {
        label: sum(split_counts[label] for split_counts in counts.values())
        for label in (0, 1)
    }
    manifest_total = sum(split_counts["total"] for split_counts in counts.values())
    if expected_total is not None and len(expected_paths) != expected_total:
        errors.append(f"Dataset has {len(expected_paths)} labelled images; expected {expected_total}")
    if expected_class_counts is not None and observed_class_counts != expected_class_counts:
        errors.append(
            f"Manifest class counts {observed_class_counts} do not match "
            f"expected {expected_class_counts}"
        )

    distribution = {}
    distributions_reasonable = True
    for label in (0, 1):
        class_total = observed_class_counts[label]
        distribution[label] = {}
        for partition in PARTITIONS:
            observed = counts[partition][label] / class_total if class_total else 0.0
            target = TARGET_RATIOS[partition]
            distribution[label][partition] = {
                "observed": observed,
                "target": target,
                "absolute_difference": abs(observed - target),
            }
            if abs(observed - target) > distribution_tolerance:
                distributions_reasonable = False
    if not distributions_reasonable:
        errors.append(
            "At least one per-class partition ratio is outside the "
            f"{distribution_tolerance:.3f} tolerance"
        )

    report = parse_audit_report(audit_path)
    report_summary_matches = report["total"] == manifest_total == len(expected_paths)
    report_summary_matches = report_summary_matches and all(
        report["split_summary"].get(partition) == counts[partition]
        for partition in PARTITIONS
    )
    if not report_summary_matches:
        errors.append("Split manifest totals or class counts disagree with the audit report")

    similarity = audit_similarities(
        Path(data_dir), near_threshold=report["near_threshold"]
    )
    report_exact_matches = (
        report["exact_duplicate_groups"] == len(similarity["exact_groups"])
        and report["exact_duplicate_images"]
        == sum(len(group) for group in similarity["exact_groups"])
    )
    if not report_exact_matches:
        errors.append("Exact-duplicate findings disagree with the similarity report")

    reported_pairs = {
        (
            pair["image_path_a"], pair["image_path_b"], pair["label_a"],
            pair["label_b"], pair["dhash_distance"],
        )
        for pair in report["pairs"]
    }
    recomputed_pairs = {
        (
            pair["image_path_a"], pair["image_path_b"], pair["label_a"],
            pair["label_b"], pair["dhash_distance"],
        )
        for pair in similarity["near_pairs"]
    }
    pair_hashes_match = (
        report["candidate_count"] == len(report["pairs"])
        and len(reported_pairs) == len(report["pairs"])
        and reported_pairs == recomputed_pairs
    )
    pair_partitions_match = all(
        pair["partition_a"] == partition_for_path.get(pair["image_path_a"])
        and pair["partition_b"] == partition_for_path.get(pair["image_path_b"])
        and pair["label_a"] == labels_for_path.get(pair["image_path_a"])
        and pair["label_b"] == labels_for_path.get(pair["image_path_b"])
        for pair in report["pairs"]
    )
    if not pair_hashes_match:
        errors.append("Reported perceptual-hash pairs do not match a fresh audit")
    if not pair_partitions_match:
        errors.append("Perceptual-pair labels or partitions disagree with manifests")

    return {
        "ok": not errors,
        "errors": errors,
        "dataset_total": len(expected_paths),
        "manifest_total": manifest_total,
        "class_counts": observed_class_counts,
        "partition_counts": counts,
        "per_class_partition_proportions": distribution,
        "unique_paths_exactly_once": not duplicate_paths and not missing_paths
        and not unexpected_paths,
        "all_paths_resolve": all_paths_resolve,
        "class_distributions_reasonable": distributions_reasonable,
        "audit_summary_matches": report_summary_matches,
        "audit_exact_hashes_match": report_exact_matches,
        "audit_candidate_pairs_match": pair_hashes_match,
        "audit_pair_partitions_match": pair_partitions_match,
        "candidate_pair_count": len(report["pairs"]),
        "candidate_pairs_crossing_partitions": sum(
            pair["partition_a"] != pair["partition_b"] for pair in report["pairs"]
        ),
    }


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_contact_sheet(data_dir: Path, split_dir: Path,
                         report_path: Path, output_path: Path) -> Path:
    """Render all review candidates two pairs per row with IDs and split info."""
    report = parse_audit_report(report_path)
    read_manifests(split_dir)

    margin = 24
    pair_width = 485
    tile_size = 210
    pair_height = 315
    pairs_per_row = 2
    rows = math.ceil(len(report["pairs"]) / pairs_per_row)
    title_height = 84
    canvas = Image.new(
        "RGB",
        (margin * 2 + pair_width * pairs_per_row,
         title_height + rows * pair_height + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(23)
    pair_font = _load_font(18)
    info_font = _load_font(14)
    draw.text((margin, 14), "Perceptual similarity candidates", fill="#172b4d", font=title_font)
    draw.text(
        (margin, 48),
        "Image IDs are filename stems, not patient IDs. dHash threshold: "
        f"{report['near_threshold']} bits. Review signal only.",
        fill="#4b5563",
        font=info_font,
    )

    for index, pair in enumerate(report["pairs"]):
        column = index % pairs_per_row
        row = index // pairs_per_row
        x = margin + column * pair_width
        y = title_height + row * pair_height
        draw.rounded_rectangle(
            (x, y, x + pair_width - 10, y + pair_height - 12),
            radius=8,
            outline="#cbd5e1",
            width=2,
            fill="#f8fafc",
        )
        draw.text(
            (x + 12, y + 8),
            f"Pair {index + 1:02d}  |  dHash distance {pair['dhash_distance']}",
            fill="#172b4d",
            font=pair_font,
        )
        for side, relative_path, label, partition in (
            (0, pair["image_path_a"], pair["label_a"], pair["partition_a"]),
            (1, pair["image_path_b"], pair["label_b"], pair["partition_b"]),
        ):
            tile_x = x + 12 + side * 228
            class_name = "homogeneous" if label == 0 else "inhomogeneous"
            image_id = Path(relative_path).stem
            draw.text(
                (tile_x, y + 36), relative_path,
                fill="#111827", font=info_font,
            )
            draw.text(
                (tile_x, y + 54),
                f"ID {image_id} | class {label}",
                fill="#374151",
                font=info_font,
            )
            draw.text(
                (tile_x, y + 71),
                f"{class_name} | {partition}",
                fill="#374151",
                font=info_font,
            )
            with Image.open(Path(data_dir) / relative_path) as image:
                thumbnail = ImageOps.fit(
                    image.convert("L"),
                    (tile_size, tile_size),
                    method=Image.Resampling.LANCZOS,
                ).convert("RGB")
            canvas.paste(thumbnail, (tile_x, y + 90))
            draw.rectangle(
                (tile_x, y + 90, tile_x + tile_size - 1, y + 90 + tile_size - 1),
                outline="#94a3b8",
                width=1,
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--audit", type=Path, default=Path("reports/similarity_audit.md")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/similarity_contact_sheet.png"),
    )
    parser.add_argument("--expected-total", type=int, default=860)
    args = parser.parse_args()

    result = verify_split_integrity(
        args.data_dir,
        args.split_dir,
        args.audit,
        expected_total=args.expected_total,
    )
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
    output_path = create_contact_sheet(
        args.data_dir, args.split_dir, args.audit, args.output
    )
    print(f"Contact sheet: {output_path}")


if __name__ == "__main__":
    main()
