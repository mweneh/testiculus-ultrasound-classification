import pytest

from src.demo_images import CASES, MODELS, load_flagged_images, select_cases


def entry(label, predictions, margin=0.2):
    return {
        "label": label,
        "models": {
            model: {"score": 0.5 + (margin if prediction else -margin), "threshold": 0.5, "prediction": prediction}
            for model, prediction in zip(MODELS, predictions)
        },
    }


def full_pool():
    """Enough candidates for every case, each at a distinct clarity."""
    pool = {}
    for case in CASES:
        for index in range(case.count + 1):
            pool[f"{case.key}/{index}.png"] = entry(case.label, case.predictions, margin=0.1 + index / 100)
    return pool


def test_selects_clearest_candidates_per_case_and_skips_flagged():
    pool = full_pool()
    clearest = f"{CASES[0].key}/{CASES[0].count}.png"
    selected = select_cases(pool, set(pool), excluded={clearest})
    chosen = [path for _, path in selected]
    assert clearest not in chosen
    assert len(chosen) == sum(case.count for case in CASES) == len(set(chosen))
    for case, path in selected:
        assert pool[path]["label"] == case.label
        assert tuple(pool[path]["models"][m]["prediction"] for m in MODELS) == case.predictions


def test_rejects_predictions_outside_the_test_partition():
    pool = full_pool()
    with pytest.raises(ValueError, match="non-test"):
        select_cases(pool, set(list(pool)[1:]), excluded=set())


def test_fails_when_a_case_has_too_few_candidates():
    pool = {path: value for path, value in full_pool().items() if not path.startswith("all-miss/")}
    with pytest.raises(ValueError, match="all-miss"):
        select_cases(pool, set(pool), excluded=set())


def test_reads_both_sides_of_similarity_pairs(tmp_path):
    audit = tmp_path / "audit.md"
    audit.write_text("- `homogeneous/1.png` ↔ `inhomogeneous/22.png` (dHash distance 3)\n")
    assert load_flagged_images(audit) == {"homogeneous/1.png", "inhomogeneous/22.png"}
