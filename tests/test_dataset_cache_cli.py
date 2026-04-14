import os

import pytest
from typer.testing import CliRunner

from nl_code.datasets.cache import read_manifest
from nl_code.datasets.cache_cli import app
from nl_code.datasets.humaneval_dataset import HumanEvalDataset

from conftest import fail_on_hf, make_humaneval_row, mock_hf_dataset

runner = CliRunner()
pytestmark = pytest.mark.docker


@pytest.mark.usefixtures("dataset_cache_dir")
def test_cli_rebuild_status_and_clear(monkeypatch) -> None:
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setattr(
        "nl_code.datasets.dataset.load_dataset",
        lambda *_args, **_kwargs: mock_hf_dataset([make_humaneval_row()]),
    )

    rebuild_result = runner.invoke(app, ["rebuild", "humaneval-plus"])

    assert rebuild_result.exit_code == 0
    assert os.environ["MPLBACKEND"] == "Agg"
    manifest = read_manifest(HumanEvalDataset().dataset_id, HumanEvalDataset().split)
    assert manifest is not None
    assert manifest.task_count == 1

    monkeypatch.setattr("nl_code.datasets.dataset.load_dataset", fail_on_hf)

    status_result = runner.invoke(app, ["status", "humaneval-plus"])
    assert status_result.exit_code == 0
    assert "cached 1 tasks" in status_result.output

    clear_result = runner.invoke(app, ["clear", "humaneval-plus"])
    assert clear_result.exit_code == 0
    assert "cleared" in clear_result.output

    missing_result = runner.invoke(app, ["status", "humaneval-plus"])
    assert missing_result.exit_code == 0
    assert "missing" in missing_result.output
