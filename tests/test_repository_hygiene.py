"""Keep saved notebook displays from committing medical image pixels."""

import json
from pathlib import Path


def test_repository_notebooks_have_no_saved_outputs_or_attachments():
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    assert notebooks, "Expected the source exploration notebook"
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            location = f"{path.name}, cell {index}"
            assert not cell.get("attachments"), location
            assert not cell.get("outputs"), location
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, location
        assert "data:image/" not in path.read_text(encoding="utf-8"), path.name
