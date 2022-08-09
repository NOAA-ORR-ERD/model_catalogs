"""
Test process functions.
"""

import warnings

import model_catalogs as mc


# @pytest.mark.slow
def test_process():
    """Test all models for transform."""

    main_cat = mc.setup()

    for model in list(main_cat):
        for timing in list(main_cat[model]):

            source = main_cat[model][timing]

            # use local version of model output
            # Some of the test files still don't exist.
            if not mc.TEST_PATH_FILE(model, timing).exists():
                warnings.warn(
                    f"Test file for model {model} with timing {timing} does not exist yet. This test is being skipped",  # noqa: E501
                    RuntimeWarning,
                )
                continue

            source.__dict__["_captured_init_kwargs"]["transform_kwargs"]["urlpath"] = [
                mc.TEST_PATH_FILE(model, timing)
            ]

            try:
                ds = source.to_dask()
            except OSError:
                warnings.warn(f'MODEL {model} timing {timing} would not run', RuntimeWarning)
                continue

            # check axis attributes have been assigned
            checks = [
                ds[var_name].attrs["axis"] == axis
                for axis, var_names in source.metadata["axis"].items()
                for var_name in mc.astype(var_names, list)
                if var_name in ds.dims
            ]
            assert all(checks)

            # check standard_names attributes have been assigned
            checks = [
                ds[var_name].attrs["standard_name"] == st_name
                for st_name, var_names in source.metadata["standard_names"].items()
                for var_name in mc.astype(var_names, list)
                if var_name in ds.data_vars
            ]
            assert all(checks)

            # check cf-xarray
            # AXIS X and Y won't be defined for unstructured model
            assert sorted(list(ds.cf.axes.keys())) == ["T", "X", "Y", "Z"]
            assert sorted(list(ds.cf.coordinates.keys())) == [
                "latitude",
                "longitude",
                "time",
                "vertical",
            ]

            ds.close()
