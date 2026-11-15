import csv

import numpy as np
from PIL import Image

from src.data.similarity_review import (
    create_contact_sheet,
    verify_split_integrity,
)
from src.data.splits import make_splits


def _make_random_dataset(root, per_class=20):
    rng = np.random.default_rng(42)
    for class_name in ("homogeneous", "inhomogeneous"):
        folder = root / class_name
        folder.mkdir(parents=True)
        for index in range(per_class):
            image = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
            Image.fromarray(image).save(folder / f"{index:03d}.png")


def test_integrity_checks_coverage_stratification_paths_and_report(tmp_path):
    data_dir = tmp_path / "raw"
    split_dir = tmp_path / "splits"
    audit_path = tmp_path / "audit.md"
    _make_random_dataset(data_dir)
    make_splits(data_dir, split_dir, seed=42, report_path=audit_path)

    result = verify_split_integrity(
        data_dir,
        split_dir,
        audit_path,
        expected_total=40,
        expected_class_counts={0: 20, 1: 20},
    )

    assert result["ok"]
    assert result["dataset_total"] == 40
    assert result["manifest_total"] == 40
    assert result["unique_paths_exactly_once"]
    assert result["all_paths_resolve"]
    assert result["class_distributions_reasonable"]
    assert result["audit_summary_matches"]
    assert result["audit_candidate_pairs_match"]


def test_integrity_check_detects_duplicate_manifest_path(tmp_path):
    data_dir = tmp_path / "raw"
    split_dir = tmp_path / "splits"
    audit_path = tmp_path / "audit.md"
    _make_random_dataset(data_dir)
    make_splits(data_dir, split_dir, seed=42, report_path=audit_path)
    with (split_dir / "test.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["homogeneous/000.png", 0])

    result = verify_split_integrity(
        data_dir,
        split_dir,
        audit_path,
        expected_total=40,
        expected_class_counts={0: 20, 1: 20},
    )

    assert not result["ok"]
    assert not result["unique_paths_exactly_once"]
    assert any("occur more than once" in error for error in result["errors"])


def test_contact_sheet_renders_pair_metadata_and_images(tmp_path):
    data_dir = tmp_path / "raw"
    (data_dir / "homogeneous").mkdir(parents=True)
    (data_dir / "inhomogeneous").mkdir()
    Image.fromarray(np.full((30, 30), 90, dtype=np.uint8)).save(
        data_dir / "homogeneous" / "12.png"
    )
    Image.fromarray(np.full((30, 30), 100, dtype=np.uint8)).save(
        data_dir / "inhomogeneous" / "12.png"
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text(
        "image_path,label\nhomogeneous/12.png,0\n", encoding="utf-8"
    )
    (split_dir / "validation.csv").write_text(
        "image_path,label\ninhomogeneous/12.png,1\n", encoding="utf-8"
    )
    (split_dir / "test.csv").write_text("image_path,label\n", encoding="utf-8")
    report = tmp_path / "audit.md"
    report.write_text(
        """# Audit

- Labelled images included: 2.
- Exact duplicate groups: 0 (0 images).
The dHash candidates have Hamming distance ≤ 4.
| train | 1 | 1 | 0 |
| validation | 1 | 0 | 1 |
| test | 0 | 0 | 0 |
Candidate pairs flagged: 1.
- `homogeneous/12.png` ↔ `inhomogeneous/12.png` (dHash distance 0; labels 0/1; partitions train/validation).
""",
        encoding="utf-8",
    )

    output = create_contact_sheet(
        data_dir, split_dir, report, tmp_path / "review.png"
    )
    with Image.open(output) as contact_sheet:
        assert contact_sheet.format == "PNG"
        assert contact_sheet.width > 400
        assert contact_sheet.height > 250
