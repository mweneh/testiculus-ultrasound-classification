import csv
import shutil

import numpy as np
from PIL import Image

from src.data.dataset import TesticulUSDataset
from src.data.splits import audit_similarities, make_splits


def _make_dataset(root, per_class=12):
    for class_name, offset in (("homogeneous", 0), ("inhomogeneous", 1)):
        folder = root / class_name
        folder.mkdir(parents=True)
        for index in range(per_class):
            pixels = np.zeros((16, 16), dtype=np.uint8)
            pixels[index % 16, :] = 40 + offset * 100 + index
            Image.fromarray(pixels).save(folder / f"{index}.png")


def _read_manifest(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_splits_are_reproducible_stratified_and_nonoverlapping(tmp_path):
    data_dir = tmp_path / "raw"
    _make_dataset(data_dir)
    first, _ = make_splits(data_dir, tmp_path / "first", seed=42)
    second, _ = make_splits(data_dir, tmp_path / "second", seed=42)

    assert {
        name: [(row["image_path"], row["label"]) for row in rows]
        for name, rows in first.items()
    } == {
        name: [(row["image_path"], row["label"]) for row in rows]
        for name, rows in second.items()
    }
    path_sets = [
        {row["image_path"] for row in _read_manifest(tmp_path / "first" / f"{name}.csv")}
        for name in ("train", "validation", "test")
    ]
    assert not (path_sets[0] & path_sets[1])
    assert not (path_sets[0] & path_sets[2])
    assert not (path_sets[1] & path_sets[2])
    assert set.union(*path_sets) == {
        f"{class_name}/{index}.png"
        for class_name in ("homogeneous", "inhomogeneous")
        for index in range(12)
    }
    assert all("/" in path for paths in path_sets for path in paths)
    assert [len(path_set) for path_set in path_sets] == [16, 4, 4]


def test_exact_duplicates_stay_in_one_partition(tmp_path):
    data_dir = tmp_path / "raw"
    _make_dataset(data_dir)
    shutil.copyfile(
        data_dir / "homogeneous" / "0.png",
        data_dir / "homogeneous" / "duplicate.png",
    )

    splits, audit = make_splits(data_dir, tmp_path / "splits", seed=42)
    assert len(audit["exact_groups"]) == 1
    locations = {
        split_name
        for split_name, rows in splits.items()
        if {row["image_path"] for row in rows}
        & {"homogeneous/0.png", "homogeneous/duplicate.png"}
    }
    assert len(locations) == 1


def test_perceptual_similarity_candidates_are_reported(tmp_path):
    data_dir = tmp_path / "raw"
    _make_dataset(data_dir, per_class=2)
    # Changing all pixel values preserves this image's dHash while making it
    # an inexact pixel duplicate, so it should be a review candidate.
    image = np.full((16, 16), 180, dtype=np.uint8)
    Image.fromarray(image).save(data_dir / "homogeneous" / "flat_a.png")
    Image.fromarray(image + 1).save(data_dir / "homogeneous" / "flat_b.png")

    audit = audit_similarities(data_dir, near_threshold=0)
    assert any(
        {pair["image_path_a"], pair["image_path_b"]}
        == {"homogeneous/flat_a.png", "homogeneous/flat_b.png"}
        for pair in audit["near_pairs"]
    )


def test_dataset_consumes_class_qualified_manifest_and_checks_label(tmp_path):
    data_dir = tmp_path / "raw"
    _make_dataset(data_dir, per_class=1)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_path,label\nhomogeneous/0.png,0\ninhomogeneous/0.png,1\n",
        encoding="utf-8",
    )

    dataset = TesticulUSDataset(data_dir, manifest_path=manifest)
    assert [sample["label"] for sample in dataset.samples] == [0, 1]
    assert dataset[1]["image_path"] == "inhomogeneous/0.png"

    manifest.write_text("image_path,label\nhomogeneous/0.png,1\n", encoding="utf-8")
    try:
        TesticulUSDataset(data_dir, manifest_path=manifest)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched manifest label was accepted")
