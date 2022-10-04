"""
Make sure catalog creation is working.

Some tests won't consistently run, I think because of connection issues with
NOAA servers. I can't make progress on these so I will save them at the bottom
of this script, commented out, and I will run abbreviated sample versions of
them instead.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from intake.catalog import Catalog
from intake_xarray.opendap import OpenDapSource
from pandas import Timestamp

import model_catalogs as mc

from model_catalogs import process
from model_catalogs.process import DatasetTransform


def test_setup():
    """Make sure main catalog is created correctly."""

    main_cat = mc.setup(override=True)

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
    test_models = {
        "HYCOM": "forecast",
        "DBOFS": "nowcast",
        "SFBOFS": "hindcast",
    }

    main_cat = mc.setup()
    for model, timing in test_models.items():
        cat = mc.find_availability(main_cat[model], timing=timing, override=True)
        if "start_datetime" not in cat[timing].metadata:
            warnings.warn(
                f"Running model {model} with timing {timing} in `find_availability()` did not result in `start_datetime` in the catalog metadata.",  # noqa: E501
                RuntimeWarning,
            )
        fname = mc.FILE_PATH_START(model, timing)
        if not mc.is_fresh(fname):
            warnings.warn(f"Filename {fname} is not found as fresh.", RuntimeWarning)

        # make sure catalog output since catalog was input
        assert isinstance(cat, Catalog)

        # also compare with requesting source directly
        source = mc.find_availability(main_cat[model][timing])

        # make sure source output since source was input
        assert isinstance(source, (OpenDapSource, DatasetTransform))

        assert (
            cat[timing].metadata["start_datetime"] == source.metadata["start_datetime"]
        )
        assert cat[timing].metadata["end_datetime"] == source.metadata["end_datetime"]

        # test that if server status is False, start_datetime, end_datetime are None
        in_source = main_cat[model][timing]
        in_source._status = False
        with pytest.warns(RuntimeWarning):
            out_source = mc.find_availability(in_source)
        assert out_source.metadata["start_datetime"] is None
        assert out_source.metadata["end_datetime"] is None


@pytest.mark.slow
def test_boundaries():
    """Test one faster model and compare with existing file."""

    model = "hycom"

    # Calculate
    boundaries = mc.calculate_boundaries(
        file_locs=mc.FILE_PATH_ORIG(model), save_files=False, return_boundaries=True
    )

    # Read in saved
    with open(mc.FILE_PATH_BOUNDARIES(model), "r") as stream:
        boundaries_read = yaml.safe_load(stream)

    assert boundaries[model]["bbox"] == boundaries_read["bbox"]
    assert boundaries[model]["wkt"] == boundaries_read["wkt"]


@pytest.mark.slow
def test_select_date_range():
    """Test functionality in `select_date_range()`.

    Should filter date range for static and nonstatic links. Should return all model output for today
    and tomorrow.
    """

    test_models = {"HYCOM": "forecast", "CIOFS": "nowcast"}

    today = pd.Timestamp.today() - pd.Timedelta("1 day")
    # today = pd.Timestamp.today(tz="UTC") - pd.Timedelta("1 day")
    tom = today + pd.Timedelta("1 day")

    main_cat = mc.setup()
    for model, timing in test_models.items():
        source = mc.select_date_range(
            main_cat[model][timing], today.date(), tom.date(), override=True
        )

        if source.status:
            ds = source.to_dask()

            assert ds.cf["T"][0] == today.normalize()

            # calculate delta times
            dts = ds.cf["T"].diff(dim=ds.cf.axes["T"][0])

            # make sure all dts are the same (some are 42 seconds off for some reason but that is ok)
            assert all(
                [
                    pd.Timedelta(f"{float(dt)} {np.datetime_data(dt)[0]}") < pd.Timedelta("1 minute")
                    for dt in dts - dts[0]
                ]
            )

            end_of_day = tom.normalize() + pd.Timedelta("1 day")
            assert bool(
                end_of_day - pd.Timedelta(f"{dts[0]}") <= ds.cf["T"][-1] < end_of_day
            )

        else:
            warnings.warn(
                f"Source {model}, {timing} server status is False.",  # noqa: E501
                RuntimeWarning,
            )

    # also make sure an incorrect requested datetime range returns a warning
    # check this for static link models
    test_models = {"HYCOM": "forecast", "CIOFS": "forecast"}

    main_cat = mc.setup()
    for model, timing in test_models.items():
        source = mc.select_date_range(main_cat[model][timing], "1980-1-1", "1980-1-2")

        with pytest.warns(RuntimeWarning):
            ds = source.to_dask()


def test_select_date_range_dates():
    """Detailed tests for resulting start/end datetimes.

    Doesn't run `to_dask()`, and only checks unaggregated models.
    All the tested models have hourly output.
    """

    yes = (
        pd.Timestamp.today().normalize()
        - pd.Timedelta("1 day")
        + pd.Timedelta("6:00:00")
    )
    yes_date = str(yes.date())
    yes_st = yes.normalize()
    yes_end = yes_st + pd.Timedelta("23:00:00")
    tod = pd.Timestamp.today().normalize() + pd.Timedelta("6:00:00")

    test_models = {"CBOFS": "nowcast", "NYOFS": "nowcast"}
    test_conditions = [
        {"sday": yes, "eday": yes, "tst_known": yes_st, "tend_known": yes_end},
        {
            "sday": yes_date,
            "eday": yes_date,
            "tst_known": yes_st,
            "tend_known": yes_end,
        },
        {"sday": yes, "eday": None, "tst_known": yes, "tend_known": None},
        {"sday": tod, "eday": None, "tst_known": tod, "tend_known": None},
    ]

    main_cat = mc.setup()
    for model, timing in test_models.items():
        for tc in test_conditions:
            if tc["tend_known"] is None:
                cat = mc.find_availability(
                    main_cat[model], timing=timing, override=True
                )
            else:
                cat = main_cat[model]
            source = mc.select_date_range(
                cat[timing],
                start_date=tc["sday"],
                end_date=tc["eday"],
                override=True,
            )

            assert source.dates[0] == tc["tst_known"]
            if tc["tend_known"] is None:
                assert source.dates[-1] == pd.Timestamp(source.metadata["end_datetime"])
            else:
                assert source.dates[-1] == tc["tend_known"]

            # check that dates are all consistent
            ddf = pd.Series(source.dates).diff()
            assert (ddf[1:] - ddf.median() < pd.Timedelta("1 min")).all()

    # check hindcast, which stops 4 days ago and does not have forecast files
    fivedays = (
        pd.Timestamp.today().normalize()
        - pd.Timedelta("6 days")
        + pd.Timedelta("6:00:00")
    )
    fivedays_date = str(fivedays.date())
    fivedays_st = fivedays.normalize()
    fivedays_end = fivedays_st + pd.Timedelta("23:00:00")

    test_models = {"SFBOFS": "hindcast"}
    # check t-1_known of None with find_availability output
    test_conditions = [
        {
            "sday": fivedays,
            "eday": fivedays,
            "tst_known": fivedays_st,
            "tend_known": fivedays_end,
        },
        {
            "sday": fivedays_date,
            "eday": fivedays_date,
            "tst_known": fivedays_st,
            "tend_known": fivedays_end,
        },
    ]

    main_cat = mc.setup()
    for model, timing in test_models.items():
        for tc in test_conditions:
            if tc["tend_known"] is None:
                cat = mc.find_availability(
                    main_cat[model], timing=timing, override=True
                )
            else:
                cat = main_cat[model]
            source = mc.select_date_range(
                cat[timing],
                start_date=tc["sday"],
                end_date=tc["eday"],
                override=True,
            )

            assert source.dates[0] == tc["tst_known"]
            if tc["tend_known"] is None:
                assert source.dates[-1] == pd.Timestamp(source.metadata["end_datetime"])
            else:
                assert source.dates[-1] == tc["tend_known"]

            # check that dates are all consistent
            ddf = pd.Series(source.dates).diff()
            assert (ddf[1:] - ddf.median() < pd.Timedelta("1 min")).all()


@pytest.mark.slow
def test_process():
    """Test that dataset is processed."""

    main_cat = mc.setup()

    # if this dataset hasn't been processed, lon and lat won't be in coords
    assert "lon" in main_cat["LOOFS"]["nowcast"].to_dask().coords


def check_source(source):
    """Check attributes of source for other tests."""

    try:
        if source.status:
            ds = source.to_dask()
        else:
            warnings.warn(
                f"Source {source.cat.name}, {source.name} server status is False.",
                RuntimeWarning,
            )
            return
    except Exception as e:
        warnings.warn(
            f"Source {source.cat.name}, {source.name} could not be read in by `xarray`, with uncaught exception: {e}."
        )
        return
    # except OSError:
    #     warnings.warn(
    #         f"Model {source.cat.name} with timing {source.name} is not working right now.",
    #         RuntimeWarning,
    #     )
    #     return

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
    # AXIS X and Y won't be defined for unstructured model unless interpolated
    if "SELFE" in source.cat.description or "FVCOM" in source.cat.description:
        if "REGULARGRID" in source.cat.name:
            assert sorted(list(ds.cf.axes.keys())) == ["T", "X", "Y", "Z"]
        elif "2DS" in source.cat.name:
            assert sorted(list(ds.cf.axes.keys())) == ["T"]
        else:
            assert sorted(list(ds.cf.axes.keys())) == ["T", "Z"]
    else:
        assert sorted(list(ds.cf.axes.keys())) == ["T", "X", "Y", "Z"]

    # the 2D cases are weird
    if "NGOFS2-2DS" in source.cat.name:
        assert sorted(list(ds.cf.coordinates.keys())) == [
            "latitude",
            "longitude",
            "time",
        ]
    else:
        assert sorted(list(ds.cf.coordinates.keys())) == [
            "latitude",
            "longitude",
            "time",
            "vertical",
        ]
    ds.close()


@pytest.mark.slow
def test_forecast():
    """Test all known models for running in forecast mode.

    Fails gracefully. Does not require running `select_date_range()` because forecasts always have some
    known files included or a static link.
    """

    main_cat = mc.setup()
    timing = "forecast"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()
        source = main_cat[model][timing]
        try:
            check_source(source)
        except AssertionError:
            warnings.warn(
                f"Model {model} with timing {timing} does not have proper attributes.",
                RuntimeWarning,
            )


@pytest.mark.slow
def test_nowcast():
    """Test all known models for running in nowcast mode.

    Fails gracefully. Does not require running `select_date_range()` because they always have some known
    files included or a static link.
    """

    main_cat = mc.setup()
    timing = "nowcast"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()
        if timing in list(main_cat[model]):
            source = main_cat[model][timing]
            try:
                check_source(source)
            except AssertionError:
                warnings.warn(
                    f"Model {model} with timing {timing} does not have proper attributes.",
                    RuntimeWarning,
                )


@pytest.mark.slow
def test_hindcast():
    """Test all known models for running in hindcast mode."""

    main_cat = mc.setup()
    timing = "hindcast"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()
        if timing in list(main_cat[model]):
            source = main_cat[model][timing]
            try:
                check_source(source)
            except AssertionError:
                warnings.warn(
                    f"Model {model} with timing {timing} does not have proper attributes.",
                    RuntimeWarning,
                )


@pytest.mark.slow
def test_hindcast_forecast_aggregation():
    """Test all known models for running in hindcast aggregation mode."""

    main_cat = mc.setup()
    timing = "hindcast-forecast-aggregation"

    for cat_loc in mc.CAT_PATH_ORIG.glob("*.yaml"):
        model = cat_loc.stem.upper()
        if timing in list(main_cat[model]):
            source = main_cat[model][timing]
            try:
                check_source(source)
            except AssertionError:
                warnings.warn(
                    f"Model {model} with timing {timing} does not have proper attributes.",
                    RuntimeWarning,
                )


def test_urlpath():
    """urlpath is exposed to transform from target now."""

    main_cat = mc.setup()

    # shows 2 default files
    assert len(main_cat["NGOFS2"]["forecast"].urlpath) == 2


def test_file2dt():
    """Test a few sample files."""

    fname = "nos.cbofs.fields.n001.20220913.t00z.nc"
    date = Timestamp("20220912T19:00")
    assert mc.file2dt(fname) == date

    fname = "nos.cbofs.fields.f001.20220914.t12z.nc"
    date = Timestamp("20220914T13:00")
    assert mc.file2dt(fname) == date

    fname = "glofs.loofs.fields.nowcast.20220919.t00z.nc"
    date = [
        Timestamp("2022-09-18 19:00:00"),
        Timestamp("2022-09-18 20:00:00"),
        Timestamp("2022-09-18 21:00:00"),
        Timestamp("2022-09-18 22:00:00"),
        Timestamp("2022-09-18 23:00:00"),
        Timestamp("2022-09-19 00:00:00"),
    ]
    assert mc.file2dt(fname) == date

    fname = "nos.nyofs.fields.forecast.20220920.t17z.nc"
    date = [
        Timestamp("2022-09-20 18:00:00"),
        Timestamp("2022-09-20 19:00:00"),
        Timestamp("2022-09-20 20:00:00"),
        Timestamp("2022-09-20 21:00:00"),
        Timestamp("2022-09-20 22:00:00"),
        Timestamp("2022-09-20 23:00:00"),
        Timestamp("2022-09-21 00:00:00"),
        Timestamp("2022-09-21 01:00:00"),
        Timestamp("2022-09-21 02:00:00"),
        Timestamp("2022-09-21 03:00:00"),
        Timestamp("2022-09-21 04:00:00"),
        Timestamp("2022-09-21 05:00:00"),
        Timestamp("2022-09-21 06:00:00"),
        Timestamp("2022-09-21 07:00:00"),
        Timestamp("2022-09-21 08:00:00"),
        Timestamp("2022-09-21 09:00:00"),
        Timestamp("2022-09-21 10:00:00"),
        Timestamp("2022-09-21 11:00:00"),
        Timestamp("2022-09-21 12:00:00"),
        Timestamp("2022-09-21 13:00:00"),
        Timestamp("2022-09-21 14:00:00"),
        Timestamp("2022-09-21 15:00:00"),
        Timestamp("2022-09-21 16:00:00"),
        Timestamp("2022-09-21 17:00:00"),
        Timestamp("2022-09-21 18:00:00"),
        Timestamp("2022-09-21 19:00:00"),
        Timestamp("2022-09-21 20:00:00"),
        Timestamp("2022-09-21 21:00:00"),
        Timestamp("2022-09-21 22:00:00"),
        Timestamp("2022-09-21 23:00:00"),
        Timestamp("2022-09-22 00:00:00"),
        Timestamp("2022-09-22 01:00:00"),
        Timestamp("2022-09-22 02:00:00"),
        Timestamp("2022-09-22 03:00:00"),
        Timestamp("2022-09-22 04:00:00"),
        Timestamp("2022-09-22 05:00:00"),
        Timestamp("2022-09-22 06:00:00"),
        Timestamp("2022-09-22 07:00:00"),
        Timestamp("2022-09-22 08:00:00"),
        Timestamp("2022-09-22 09:00:00"),
        Timestamp("2022-09-22 10:00:00"),
        Timestamp("2022-09-22 11:00:00"),
        Timestamp("2022-09-22 12:00:00"),
        Timestamp("2022-09-22 13:00:00"),
        Timestamp("2022-09-22 14:00:00"),
        Timestamp("2022-09-22 15:00:00"),
        Timestamp("2022-09-22 16:00:00"),
        Timestamp("2022-09-22 17:00:00"),
        Timestamp("2022-09-22 18:00:00"),
        Timestamp("2022-09-22 19:00:00"),
        Timestamp("2022-09-22 20:00:00"),
        Timestamp("2022-09-22 21:00:00"),
        Timestamp("2022-09-22 22:00:00"),
        Timestamp("2022-09-22 23:00:00"),
    ]
    assert mc.file2dt(fname) == date


def test_filedates2df():
    """Checking sorting and deduplicating indices."""

    fnames = [
        "nos.creofs.fields.f000.20220920.t15z.nc",
        "nos.creofs.fields.f001.20220920.t15z.nc",
        "nos.creofs.fields.f002.20220920.t15z.nc",
        "nos.creofs.fields.f003.20220920.t15z.nc",
        "nos.creofs.fields.n000.20220920.t15z.nc",
        "nos.creofs.fields.n001.20220920.t15z.nc",
        "nos.creofs.fields.n002.20220920.t15z.nc",
        "nos.creofs.fields.n003.20220920.t15z.nc",
        "nos.creofs.fields.n004.20220920.t15z.nc",
        "nos.creofs.fields.n005.20220920.t15z.nc",
        "nos.creofs.fields.n006.20220920.t15z.nc",
    ]

    df = mc.filedates2df(fnames)

    # also no duplicate in ordered_fnames
    ordered_fnames = [
        "nos.creofs.fields.n000.20220920.t15z.nc",
        "nos.creofs.fields.n001.20220920.t15z.nc",
        "nos.creofs.fields.n002.20220920.t15z.nc",
        "nos.creofs.fields.n003.20220920.t15z.nc",
        "nos.creofs.fields.n004.20220920.t15z.nc",
        "nos.creofs.fields.n005.20220920.t15z.nc",
        "nos.creofs.fields.f000.20220920.t15z.nc",
        "nos.creofs.fields.f001.20220920.t15z.nc",
        "nos.creofs.fields.f002.20220920.t15z.nc",
        "nos.creofs.fields.f003.20220920.t15z.nc",
    ]

    assert ordered_fnames == list(df["filenames"].values)


@pytest.mark.slow
def test_urlpath_after_select():
    """urlpath is replaced after select_date_range"""

    main_cat = mc.setup()

    day = "2022-1-1"
    source = mc.select_date_range(
        main_cat["CREOFS"]["hindcast"],
        start_date=day,
        end_date=day,
        override=True,
    )

    # compare datetimes instead of filenames since filenames are not unique
    known_dates = [
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
        pd.Timestamp("2022-01-01 02:00:00"),
        pd.Timestamp("2022-01-01 03:00:00"),
        pd.Timestamp("2022-01-01 04:00:00"),
        pd.Timestamp("2022-01-01 05:00:00"),
        pd.Timestamp("2022-01-01 06:00:00"),
        pd.Timestamp("2022-01-01 07:00:00"),
        pd.Timestamp("2022-01-01 08:00:00"),
        pd.Timestamp("2022-01-01 09:00:00"),
        pd.Timestamp("2022-01-01 10:00:00"),
        pd.Timestamp("2022-01-01 11:00:00"),
        pd.Timestamp("2022-01-01 12:00:00"),
        pd.Timestamp("2022-01-01 13:00:00"),
        pd.Timestamp("2022-01-01 14:00:00"),
        pd.Timestamp("2022-01-01 15:00:00"),
        pd.Timestamp("2022-01-01 16:00:00"),
        pd.Timestamp("2022-01-01 17:00:00"),
        pd.Timestamp("2022-01-01 18:00:00"),
        pd.Timestamp("2022-01-01 19:00:00"),
        pd.Timestamp("2022-01-01 20:00:00"),
        pd.Timestamp("2022-01-01 21:00:00"),
        pd.Timestamp("2022-01-01 22:00:00"),
        pd.Timestamp("2022-01-01 23:00:00"),
    ]

    assert source.dates == known_dates


def test_setting_std_name():
    """Test updating standard name metadata attributes."""
    lon = xr.DataArray(
        data=np.arange(360), dims=("lon",), attrs={"units": "degrees_east"}
    )
    lat = xr.DataArray(
        data=np.arange(-90, 90), dims=("lat",), attrs={"units": "degrees_north"}
    )
    temp = xr.DataArray(
        data=np.arange(360 * 180).reshape(180, 360),
        dims=("lat", "lon"),
        attrs={"units": "degrees_C"},
    )
    ds = xr.Dataset({"lon": lon, "lat": lat, "temp": temp})
    metadata = {
        "standard_names": {
            "longitude": ["lon"],
            "latitude": ["lat"],
        }
    }
    ds = process.add_attributes(ds, metadata=metadata)
    assert ds["lon"].attrs["standard_name"] == "longitude"
    assert ds["lat"].attrs["standard_name"] == "latitude"
