"""
Make sure catalogs work correctly. If this doesn't run once, try again since the servers can be finnicky.
"""

import warnings

from unittest import mock

import intake
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from intake.catalog import Catalog
from intake_xarray.opendap import OpenDapSource
from pandas import Timestamp

import model_catalogs as mc

from model_catalogs import process
from model_catalogs.process import DatasetTransform


def test_setup():
    """Make sure main catalog is created correctly."""

    main_cat = mc.setup(override=True, boundaries=False)

    # check that all compiled catalog files exist
    should_exist = [
        cat.lstrip("mc_") for cat in list(intake.cat) if cat.startswith("mc_")
    ]
    assert sorted(list(main_cat)) == sorted(should_exist)

    # Check that model_sources are correct for one test case
    assert sorted(list(main_cat["CBOFS"])) == [
        "coops-forecast-agg",
        "coops-forecast-noagg",
        "ncei-archive-noagg",
    ]
    # assert main_cat["CBOFS"].metadata["geospatial_bounds"]


# @pytest.mark.slow
def test_find_availability():
    """Test find_availability.

    Using warnings instead of exception since sometimes servers aren't reliable.
    """

    # test models with fast static links and that require aggregations
    test_models = {
        "GOFS": "hycom-forecast-agg",
        "DBOFS": "coops-forecast-noagg",
        "SFBOFS": "ncei-archive-noagg",
    }

    main_cat = mc.setup(boundaries=False)
    for model, model_source in test_models.items():
        cat = mc.find_availability(
            main_cat[model], model_source=model_source, override=True
        )
        if "start_datetime" not in cat[model_source].metadata:
            warnings.warn(
                f"Running model {model} with model_source {model_source} in `find_availability()` did not result in `start_datetime` in the catalog metadata.",  # noqa: E501
                RuntimeWarning,
            )
        fname = mc.FILE_PATH_START(model, model_source)
        if not mc.is_fresh(fname, cat[model_source]):
            warnings.warn(f"Filename {fname} is not found as fresh.", RuntimeWarning)

        # make sure catalog output since catalog was input
        assert isinstance(cat, Catalog)

        # also compare with requesting source directly
        source = mc.find_availability(main_cat[model][model_source])

        # make sure source output since source was input
        assert isinstance(source, (OpenDapSource, DatasetTransform))

        #         assert (
        #             cat[model_source].metadata["start_datetime"]
        #             == source.metadata["start_datetime"]
        #         )
        #         assert (
        #             cat[model_source].metadata["end_datetime"]
        #             == source.metadata["end_datetime"]
        #         )

        # test that if server status is False, start_datetime, end_datetime are None
        in_source = main_cat[model][model_source]
        in_source._status = False
        with pytest.warns(RuntimeWarning):
            out_source = mc.find_availability(in_source)
        assert out_source.metadata["start_datetime"] is None
        assert out_source.metadata["end_datetime"] is None


# @pytest.mark.slow
def test_boundaries():
    """Test one faster model and make sure variables come through."""

    model = "mc_GOFS"
    cat = intake.cat[model]

    # Calculate
    boundaries = mc.calculate_boundaries(
        cats=cat, save_files=False, return_boundaries=True
    )

    assert "bbox" in boundaries[model]
    assert "wkt" in boundaries[model]


# @pytest.mark.slow
# this test continues to be too brittle with CO-OPS servers breaking constantly but in different ways each time
# def test_select_date_range():
#     """Test functionality in `select_date_range()`.

#     Should filter date range for static and nonstatic links. Should return all model output for today
#     and tomorrow.
#     """

#     test_models = {"GOFS": "hycom-forecast-agg", "CIOFS": "coops-forecast-noagg"}

#     today = pd.Timestamp.today() - pd.Timedelta("1 day")
#     # today = pd.Timestamp.today(tz="UTC") - pd.Timedelta("1 day")
#     tom = today + pd.Timedelta("1 day")

#     main_cat = mc.setup()
#     for model, model_source in test_models.items():
#         try:
#             source = mc.select_date_range(
#                 main_cat[model][model_source], today.date(), tom.date(), override=True
#             )
#         except RuntimeError as e:
#             warnings.warn(
#                 f"Source {model}, {model_source} had runtime issue with exception {e}.",  # noqa: E501
#                 RuntimeWarning,
#             )


#         # if source.status:
#         try:
#             # import pdb; pdb.set_trace()
#             ds = source.to_dask()

#             assert ds.cf["T"][0] == today.normalize()

#             # calculate delta times
#             dts = ds.cf["T"].diff(dim=ds.cf.axes["T"][0])

#             # make sure all dts are the same (some are 42 seconds off for some reason but that is ok)
#             assert all(
#                 [
#                     pd.Timedelta(f"{float(dt)} {np.datetime_data(dt)[0]}")
#                     < pd.Timedelta("1 minute")
#                     for dt in dts - dts[0]
#                 ]
#             )

#             end_of_day = tom.normalize() + pd.Timedelta("1 day")
#             assert bool(
#                 end_of_day
#                 - pd.Timedelta(f"{float(dts[0])} {np.datetime_data(dts[0])[0]}")
#                 <= ds.cf["T"][-1]
#                 < end_of_day
#             )

#         except OSError as e:
#             warnings.warn(
#                 f"Some aource {model}, {model_source} file is unavailable.",  # noqa: E501
#                 RuntimeWarning,
#             )


#         # else:
#         #     warnings.warn(
#         #         f"Source {model}, {model_source} server status is False.",  # noqa: E501
#         #         RuntimeWarning,
#         #     )

#     # also make sure an incorrect requested datetime range returns a warning
#     # check this for static link models
#     test_models = {"GOFS": "hycom-forecast-agg", "CIOFS": "coops-forecast-agg"}

#     main_cat = mc.setup()
#     for model, model_source in test_models.items():
#         source = mc.select_date_range(
#             main_cat[model][model_source], "1980-1-1", "1980-1-2"
#         )

#         with pytest.raises(RuntimeError):
#             ds = source.to_dask()


# this test continues to be too brittle with CO-OPS servers breaking constantly but in different ways each time
# def test_select_date_range_dates():
#     """Detailed tests for resulting start/end datetimes.

#     Doesn't run `to_dask()`, and only checks unaggregated models.
#     All the tested models have hourly output.
#     """

#     yes = (
#         pd.Timestamp.today().normalize()
#         - pd.Timedelta("1 day")
#         + pd.Timedelta("6:00:00")
#     )
#     yes_date = str(yes.date())
#     yes_st = yes.normalize()
#     yes_end = yes_st + pd.Timedelta("23:00:00")
#     tod = pd.Timestamp.today().normalize() + pd.Timedelta("6:00:00")

#     test_models = {"CBOFS": "coops-forecast-noagg", "NYOFS": "coops-forecast-noagg"}
#     test_conditions = [
#         {"sday": yes, "eday": yes, "tst_known": yes_st, "tend_known": yes_end},
#         {
#             "sday": yes_date,
#             "eday": yes_date,
#             "tst_known": yes_st,
#             "tend_known": yes_end,
#         },
#         {"sday": yes, "eday": None, "tst_known": yes, "tend_known": None},
#         {"sday": tod, "eday": None, "tst_known": tod, "tend_known": None},
#     ]

#     main_cat = mc.setup()
#     for model, model_source in test_models.items():
#         for tc in test_conditions:
#             if tc["tend_known"] is None:
#                 cat = mc.find_availability(
#                     main_cat[model], model_source=model_source, override=True
#                 )
#             else:
#                 cat = main_cat[model]

#             try:
#                 source = mc.select_date_range(
#                     cat[model_source],
#                     start_date=tc["sday"],
#                     end_date=tc["eday"],
#                     override=True,
#                 )
#             except RuntimeError:
#                 print(f"server isn't working correctly right now for {cat.name}, {model_source}")

#             assert source.dates[0] == tc["tst_known"]
#             if tc["tend_known"] is None:
#                 assert source.dates[-1] == pd.Timestamp(source.metadata["end_datetime"])
#             else:
#                 assert source.dates[-1] == tc["tend_known"]

#             # check that dates are all consistent
#             ddf = pd.Series(source.dates).diff()
#             assert (ddf[1:] - ddf.median() < pd.Timedelta("1 min")).all()

#     # check archive, which stopped 10/22/22 and does not have forecast files
#     date = pd.Timestamp("2022-10-20T06:00")
#     date_date = str(date.date())
#     date_st = date.normalize()
#     date_end = date_st + pd.Timedelta("23:00:00")

#     test_models = {"SFBOFS": "ncei-archive-noagg"}
#     # check t-1_known of None with find_availability output
#     test_conditions = [
#         {
#             "sday": date,
#             "eday": date,
#             "tst_known": date_st,
#             "tend_known": date_end,
#         },
#         {
#             "sday": date_date,
#             "eday": date_date,
#             "tst_known": date_st,
#             "tend_known": date_end,
#         },
#     ]

#     main_cat = mc.setup()
#     for model, model_source in test_models.items():
#         for tc in test_conditions:
#             if tc["tend_known"] is None:
#                 cat = mc.find_availability(
#                     main_cat[model], model_source=model_source, override=True
#                 )
#             else:
#                 cat = main_cat[model]
#             source = mc.select_date_range(
#                 cat[model_source],
#                 start_date=tc["sday"],
#                 end_date=tc["eday"],
#                 override=True,
#             )

#             assert source.dates[0] == tc["tst_known"]
#             if tc["tend_known"] is None:
#                 assert source.dates[-1] == pd.Timestamp(source.metadata["end_datetime"])
#             else:
#                 assert source.dates[-1] == tc["tend_known"]

#             # check that dates are all consistent
#             ddf = pd.Series(source.dates).diff()
#             assert (ddf[1:] - ddf.median() < pd.Timedelta("1 min")).all()


# @pytest.mark.slow
def test_process():
    """Test that dataset is processed."""

    main_cat = mc.setup(boundaries=False)

    # if this dataset hasn't been processed, lon and lat won't be in coords
    assert "lon" in list(
        main_cat["LOOFS_FVCOM"]["coops-forecast-noagg"].to_dask().coords
    )


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
    #         f"Model {source.cat.name} with model_source {source.name} is not working right now.",
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
    # AXIS Z won't be defined for unstructured 2D model output
    if (
        "SELFE" in source.target.description or "FVCOM" in source.target.description
    ) and "2DS" in source.cat.name:
        assert sorted(list(ds.cf.axes.keys())) == ["T", "X", "Y"]
    else:
        assert sorted(list(ds.cf.axes.keys())) == ["T", "X", "Y", "Z"]

    # the 2D cases are weird
    if "NGOFS2_2DS" in source.cat.name:
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
def test_sources():
    """Test all known model sources.

    Fails gracefully. Does not require running `select_date_range()` because sources always have some
    known files included or a static link.
    """

    main_cat = mc.setup(boundaries=False)

    for model in list(main_cat):
        for model_source in list(main_cat[model]):
            source = main_cat[model][model_source]
            try:
                check_source(source)
            except AssertionError:
                warnings.warn(
                    f"Model {model} with model_source {model_source} does not have proper attributes.",
                    RuntimeWarning,
                )


def test_urlpath():
    """urlpath is exposed to transform from target now."""

    main_cat = mc.setup()

    # shows 2 default files
    assert len(main_cat["NGOFS2"]["coops-forecast-noagg"].urlpath) == 2


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


# @pytest.mark.slow
def test_urlpath_after_select():
    """urlpath is replaced after select_date_range"""

    main_cat = mc.setup(boundaries=False)

    day = "2022-1-1"
    source = mc.select_date_range(
        main_cat["CREOFS"]["ncei-archive-noagg"],
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


@mock.patch("xarray.open_dataset")
@mock.patch("xarray.open_mfdataset")
def test_wrong_time_range(mock_open_dataset, mock_open_mfdataset):
    ds = xr.Dataset()
    dates = pd.date_range("2000-1-1", "2000-2-1", freq="1D")
    ds["time"] = ("time", dates, {"axis": "T"})
    mock_open_dataset.return_value = ds
    mock_open_mfdataset.return_value = ds

    # have to use a real cat/source pair to get this to work, but it isn't actually called in to_dask
    main_cat = mc.setup(boundaries=False)
    cat0 = main_cat[list(main_cat)[0]]
    source0 = cat0[list(cat0)[0]]
    source0.metadata = {}
    source0 = mc.select_date_range(
        source0, start_date="2001-10-1", end_date="2001-10-2"
    )
    with pytest.raises(RuntimeError):
        source0.to_dask()
