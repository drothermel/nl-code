import nl_code.datasets as datasets_pkg


def test_dataset_package_exports_are_importable() -> None:
    for export_name in datasets_pkg.__all__:
        assert getattr(datasets_pkg, export_name) is not None
