"""
Make sure catalog creation is working.

Some tests won't consistently run, I think because of connection issues with
NOAA servers. I can't make progress on these so I will save them at the bottom
of this script, commented out, and I will run abbreviated sample versions of
them instead.
"""

import warnings
import yaml

import pandas as pd
import pytest

import model_catalogs as mc


def test_setup():
    """Make sure main catalog is created correctly."""

    main_cat = mc.setup()

    # check that all compiled catalog files exist
    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        fname = mc.FILE_PATH_COMPILED(cat_loc.name)
        assert fname.exists()
        assert mc.is_fresh(fname)

    # Check that timings are correct for one test case
    assert sorted(list(main_cat["CBOFS"])) == ["forecast", "hindcast", "nowcast"]
    assert main_cat["CBOFS"].metadata["geospatial_bounds"]


@pytest.mark.slow
def test_find_availability():
    """Test find_availability.

    Using warnings instead of exception since sometimes servers aren't reliable.
    """

    # test models with fast static links and that require aggregations
    test_models = {"RTOFS-ALASKA": "forecast",
                   "HYCOM": "forecast",
                   "DBOFS": "nowcast",
                   "SFBOFS": "hindcast",
                   }

    main_cat = mc.setup()
    for model, timing in test_models.items():
        cat = mc.find_availability(main_cat[model], timings=timing)
        if "start_datetime" not in cat[timing].metadata:
            warnings.warn(f"Running model {model} with timing {timing} in `find_availability()` did not result in `start_datetime` in the catalog metadata.",
            RuntimeWarning,
        )
        fname = mc.FILE_PATH_START(model, timing)
        if not mc.is_fresh(fname):
            warnings.warn(f"Filename {fname} is not found as fresh.", RuntimeWarning)


@pytest.mark.slow
def test_boundaries():
    """Test one faster model and compare with existing file."""

    model = "HYCOM"

    # Calculate
    boundaries = mc.calculate_boundaries(file_locs=mc.FILE_PATH_ORIG(model), save_files=False, return_boundaries=True)

    # Read in saved
    with open(mc.FILE_PATH_BOUNDARIES(model), "r") as stream:
        boundaries_read = yaml.safe_load(stream)

    assert boundaries[model]["bbox"] == boundaries_read["bbox"]
    assert boundaries[model]["wkt"] == boundaries_read["wkt"]


@pytest.mark.slow
def test_select_date_range():
    """Test functionality in `select_date_range()`.

    Should filter date range for static and nonstatic links.
    """

    test_models = {"HYCOM": "forecast",
                   "CIOFS": "nowcast"}

    today = pd.Timestamp.today()
    tom = today + pd.Timedelta('1 day')

    main_cat = mc.setup()
    for model, timing in test_models.items():
        source = mc.select_date_range(main_cat[model], today, tom, timing=timing)

        try:
            ds = source.to_dask()
            assert pd.Timestamp(ds.cf['T'].cf.isel(T=0).values).date() == today.date()
            assert pd.Timestamp(ds.cf['T'].cf.isel(T=-1).values).date() == tom.date()

        except OSError:
            warnings.warn(f"Running model {model} with timing {timing} in `select_date_range()` did not return the correct date range.",
            RuntimeWarning,
        )


@pytest.mark.slow
def test_process():
    """Test that dataset is processed."""

    main_cat = mc.setup()

    # if this dataset hasn't been processed, lon and lat won't be in coords
    assert "lon" in main_cat["LOOFS"]["nowcast"].to_dask().coords


@pytest.mark.slow
def test_forecast():
    """Test all known models for running in forecast mode.

    Fails gracefully. Does not require running `select_date_range()` because forecasts always have some known files included or a static link.
    """

    main_cat = mc.setup()
    timing = "forecast"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()
        try:
            ds = main_cat[model][timing].to_dask()
            ds.close()
        except OSError:
            warnings.warn(f"Model {model} with timing {timing} is not working right now.", RuntimeWarning)


@pytest.mark.slow
def test_nowcast():
    """Test all known models for running in nowcast mode.

    Fails gracefully. Does not require running `select_date_range()` because they always have some known files included or a static link.
    """

    main_cat = mc.setup()
    timing = "nowcast"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()

        if timing in list(main_cat[model]):
            try:
                ds = main_cat[model][timing].to_dask()
                ds.close()
            except OSError:
                warnings.warn(f"Model {model} with timing {timing} is not working right now.", RuntimeWarning)


@pytest.mark.slow
def test_hindcast():
    """Test all known models for running in hindcast mode."""

    main_cat = mc.setup()
    timing = "hindcast"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()

        if timing in list(main_cat[model]):
            try:
                ds = main_cat[model][timing].to_dask()
                ds.close()
            except OSError:
                warnings.warn(f"Model {model} with timing {timing} is not working right now.", RuntimeWarning)


@pytest.mark.slow
def test_hindcast_forecast_aggregation():
    """Test all known models for running in hindcast aggregation mode."""

    main_cat = mc.setup()
    timing = "hindcast-forecast-aggregation"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()

        if timing in list(main_cat[model]):
            try:
                ds = main_cat[model][timing].to_dask()
                ds.close()
            except OSError:
                warnings.warn(f"Model {model} with timing {timing} is not working right now.", RuntimeWarning)
