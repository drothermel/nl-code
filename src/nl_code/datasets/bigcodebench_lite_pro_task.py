from functools import cached_property

from nl_code.datasets.pro_task import ProAssertionTestSuite, RawProTask


class RawBigCodeBenchLiteProTask(RawProTask):
    @cached_property
    def test_suite(self) -> ProAssertionTestSuite:
        return ProAssertionTestSuite(
            source=self.source.test_code,
            docker_env={"MPLBACKEND": "Agg"},
        )
