"""
Make sure catalog creation is working.
"""

import os

import pandas as pd
import pytest

import model_catalogs as mc


def test_make_source_catalog():
    """Make sure source_catalog is created correctly."""

    # make source catalog
    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs",
        source_catalog_name="source_catalog_test.yaml",
        make_source_catalog=True,
    )

    # source_catalog.yaml in main dir?
    assert os.path.exists(
        "model_catalogs/tests/catalogs/source_catalogs/source_catalog_test.yaml"
    )

    # has dir name (date) encoded in metadata at top of catalog?
    assert cats.source_cat.metadata["source_catalog_dir"] == cats.source_catalog_dir

    # has specific date/source catalog files encoded in the catalog to trace which
    # files are being used? Just check one.
    assert cats.source_cat["CBOFS"].path == f"{cats.source_catalog_dir}/cbofs.yaml"

    assert sorted(list(cats.source_cat["CBOFS"])) == ["forecast", "hindcast", "nowcast"]


def test_user_cat_3ways():
    """Make sure the 3 ways of setting up user catalog return same results."""

    # make sets of catalogs
    cats1 = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs",
        source_catalog_name="source_catalog_test.yaml",
        make_source_catalog=True,
    )
    cats1.setup_cat(
        [
            dict(
                model="DBOFS",
                timing="forecast",
                start_date=None,
                end_date=None,
                treat_last_day_as_forecast=False,
            )
        ]
    )

    cats2 = mc.Management(catalog_path=f"{mc.__path__[0]}/tests/catalogs")
    cats2.setup_cat(
        dict(
            model="DBOFS",
            timing="forecast",
            start_date=None,
            end_date=None,
            treat_last_day_as_forecast=False,
        )
    )

    cats3 = mc.Management(catalog_path=f"{mc.__path__[0]}/tests/catalogs")
    cats3.setup_cat(
        model="DBOFS",
        timing="forecast",
        start_date=None,
        end_date=None,
        treat_last_day_as_forecast=False,
    )

    assert (
        cats1.user_cat["DBOFS-forecast"]
        == cats2.user_cat["DBOFS-forecast"]
        == cats3.user_cat["DBOFS-forecast"]
    )


def setup_user_catalog_for_test():
    """setup for two other tests."""

    # make source catalog
    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs",
        source_catalog_name="source_catalog_test.yaml",
        make_source_catalog=True,
    )

    # make user catalog with several model orientations
    today = pd.Timestamp.today()
    twoweeksago = today - pd.Timedelta("14 days")
    twoweeksplus1day = twoweeksago + pd.Timedelta("1 day")
    yesterday = today - pd.Timedelta("1 day")
    hindstart = pd.Timestamp("2016-1-1")
    hindend = hindstart + pd.Timedelta("1 day")
    cats.setup_cat(
        [
            dict(
                model="DBOFS",
                timing="forecast",
                start_date=None,
                end_date=None,
                treat_last_day_as_forecast=False,
            ),
            dict(
                model="DBOFS",
                timing="nowcast",
                start_date=twoweeksago,
                end_date=twoweeksplus1day,
                treat_last_day_as_forecast=False,
            ),
            dict(
                model="DBOFS_REGULARGRID",
                timing="nowcast",
                start_date=yesterday,
                end_date=today,
                treat_last_day_as_forecast=False,
            ),
            dict(
                model="DBOFS",
                timing="hindcast",
                start_date=hindstart,
                end_date=hindend,
                treat_last_day_as_forecast=False,
            ),
        ]
    )
    return cats


def test_make_user_catalog():
    """Make sure user_catalog is created correctly."""

    cats = setup_user_catalog_for_test()

    # check user catalog name/location
    assert os.path.exists(cats.user_catalog_name)

    entries = [
        "DBOFS-forecast",
        "DBOFS-hindcast",
        "DBOFS-nowcast",
        "DBOFS_REGULARGRID-nowcast",
    ]

    # has all catalog entries
    assert set(entries).issubset(set(list(cats.user_cat)))

    # has source catalog at top of this file: name and associated source directory?
    assert cats.user_cat.metadata["source_catalog_dir"] == cats.source_catalog_dir
    assert cats.user_cat.metadata["source_catalog_name"] == cats.source_catalog_name

    # # has dir name (date) encoded in metadata at top of catalog?
    # assert cats.user_cat.metadata["updated_catalog_dir"] == cats.updated_catalog_dir


@pytest.mark.slow
def test_make_user_catalog_dask():
    """Make sure user_catalog model output can be read in."""

    cats = setup_user_catalog_for_test()

    entries = [
        "DBOFS-forecast",
        "DBOFS-hindcast",
        "DBOFS-nowcast",
        "DBOFS_REGULARGRID-nowcast",
    ]

    # check that can read in model output
    for entry in entries:
        cats.user_cat[entry].to_dask()
        assert cats.user_cat[entry]._ds
        cats.user_cat[entry]._ds.close()


@pytest.mark.slow
def test_treat_last_day_as_forecast():
    """Make sure more files found for forecast."""

    # make source catalog
    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs",
        source_catalog_name="source_catalog_test.yaml",
        make_source_catalog=True,
    )

    today = pd.Timestamp.today()
    cats.setup_cat(
        [
            dict(
                model="GOMOFS",
                timing="nowcast",
                start_date=today,
                end_date=today,
                treat_last_day_as_forecast=False,
            ),
            dict(
                model="GOMOFS",
                timing="nowcast",
                start_date=today,
                end_date=today,
                treat_last_day_as_forecast=True,
            ),
        ]
    )

    entries = ["GOMOFS-nowcast", "GOMOFS-nowcast-with_forecast"]

    # has all catalog entries
    assert set(entries).issubset(set(list(cats.user_cat)))

    # make sure with_forecast has more files
    assert len(cats.user_cat["GOMOFS-nowcast-with_forecast"].metadata["urlpath"]) > len(
        cats.user_cat["GOMOFS-nowcast"].metadata["urlpath"]
    )

    # make sure with_forecast can be read in
    assert cats.user_cat["GOMOFS-nowcast-with_forecast"].to_dask()


@pytest.mark.slow
def test_forecast():
    """Test all known models for running in forecast mode."""

    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs", make_source_catalog=True
    )
    today = pd.Timestamp.today()

    # not every forecast need start_date and end_date, but some do, and all can have extra inputs.
    cats_to_make = [
        dict(
            model=model,
            timing="forecast",
            start_date=today,
            end_date=today,
            treat_last_day_as_forecast=False,
        )
        for model in list(cats.source_cat)
        if "forecast" in list(cats.source_cat[model])
    ]

    cats.setup_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        print(source)
        ds = cats.user_cat[source].to_dask()
        ds.close()


@pytest.mark.slow
def test_nowcast():
    """Test all known models for running in nowcast mode."""

    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs", make_source_catalog=True
    )
    today = pd.Timestamp.today()

    # not every forecast need start_date and end_date, but some do, and all can have extra inputs.
    cats_to_make = [
        dict(
            model=model,
            timing="nowcast",
            start_date=today,
            end_date=today,
            treat_last_day_as_forecast=False,
        )
        for model in list(cats.source_cat)
        if "nowcast" in list(cats.source_cat[model])
    ]

    cats.setup_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()


@pytest.mark.slow
def test_hindcast():
    """Test all known models for running in hindcast mode."""

    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs", make_source_catalog=True
    )
    day = pd.Timestamp.today() - pd.Timedelta("150 days")
    nextday = day + pd.Timedelta("1 day")

    cats_to_make = [
        dict(
            model=model,
            timing="hindcast",
            start_date=day,
            end_date=nextday,
            treat_last_day_as_forecast=False,
        )
        for model in list(cats.source_cat)
        if "hindcast" in list(cats.source_cat[model])
    ]

    cats.setup_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()


@pytest.mark.slow
def test_hindcast_forecast_aggregation():
    """Test all known models for running in hindcast mode."""

    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs", make_source_catalog=True
    )
    day = pd.Timestamp.today() - pd.Timedelta("365 days")
    nextday = day + pd.Timedelta("1 day")

    cats_to_make = [
        dict(
            model=model,
            timing="hindcast-forecast-aggregation",
            start_date=day,
            end_date=nextday,
            treat_last_day_as_forecast=False,
        )
        for model in list(cats.source_cat)
        if "hindcast-forecast-aggregation" in list(cats.source_cat[model])
    ]

    cats.setup_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()


@pytest.mark.slow
def test_derived():
    """Test known hindcast model that will break without
    derived dataset."""

    cats = mc.Management(
        catalog_path=f"{mc.__path__[0]}/tests/catalogs", make_source_catalog=True
    )
    day = pd.Timestamp("2021-02-20")

    cats_to_make = [
        dict(
            model="CBOFS",
            timing="hindcast",
            start_date=day,
            end_date=day,
            treat_last_day_as_forecast=False,
        )
    ]

    cats.setup_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()
