import pytest

from nl_code.code_execution.models import CodeExecutionInfrastructureError
from nl_code.datasets.dataset import FlawedSample
from nl_code.datasets.humaneval_dataset import HumanEvalDataset

from conftest import make_humaneval_row, prime_dataset_cache

pytestmark = pytest.mark.docker


@pytest.mark.usefixtures("dataset_cache_dir")
class TestHumanEvalDataset:
    def test_load_valid_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_humaneval_row(task_id="HumanEval/0")]
        ds = prime_dataset_cache(HumanEvalDataset(), rows, monkeypatch)

        assert len(ds.raw_samples) == 1
        assert "HumanEval/0" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "HumanEval/0" in ds.tasks

    def test_flawed_rows_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_row = make_humaneval_row(
            task_id="HumanEval/bad",
            canonical_solution="    return a - b\n",
        )
        good_row = make_humaneval_row(task_id="HumanEval/0")
        ds = prime_dataset_cache(HumanEvalDataset(), [bad_row, good_row], monkeypatch)

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "HumanEval/bad" in ds.flawed_raw_samples
        flawed = ds.flawed_raw_samples["HumanEval/bad"]
        assert isinstance(flawed, FlawedSample)
        assert flawed.error.startswith("dataset_failure:")

    def test_derived_tasks_use_stripped_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = prime_dataset_cache(
            HumanEvalDataset(), [make_humaneval_row()], monkeypatch
        )
        task = ds.tasks["HumanEval/0"]
        assert '"""' not in task.gt_solution

    def test_derived_tasks_have_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = prime_dataset_cache(
            HumanEvalDataset(), [make_humaneval_row()], monkeypatch
        )
        task = ds.tasks["HumanEval/0"]
        assert task.description == "Add two integers and return the result."

    def test_docker_failures_are_tracked_separately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "nl_code.datasets.humaneval_dataset.batch_run_assertion_tests",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                CodeExecutionInfrastructureError(
                    stage="docker_unavailable",
                    execution_mode="docker_worker",
                    detail="docker unavailable",
                )
            ),
        )
        ds = prime_dataset_cache(
            HumanEvalDataset(), [make_humaneval_row(task_id="HumanEval/0")], monkeypatch
        )
        assert "HumanEval/0" in ds.flawed_raw_samples
        assert ds.flawed_raw_samples["HumanEval/0"].error.startswith("docker_failure:")
