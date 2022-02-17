"""
Make sure catalog creation is working.
"""

import model_catalogs as mc
import os
import pandas as pd
import pytest


def test_make_source_catalog():
    """Make sure source_catalog is created correctly."""

    # make source catalog
    cats = mc.Management(catalog_path='tests/catalogs',
                         source_catalog_name='source_catalog_test.yaml',
                         make_source_catalog=True)

    # source_catalog.yaml in main dir?
    assert os.path.exists('tests/catalogs/source_catalogs/source_catalog_test.yaml')

    # has dir name (date) encoded in metadata at top of catalog?
    assert cats.source_cat.metadata['source_catalog_dir'] == cats.source_catalog_dir

    # has specific date/source catalog files encoded in the catalog to trace which
    # files are being used? Just check one.
    assert cats.source_cat['CBOFS'].path == f'{cats.source_catalog_dir}/ref_cbofs.yaml'

    assert sorted(list(cats.source_cat['CBOFS'])) == ['forecast', 'hindcast', 'nowcast']


def test_make_updated_catalog():
    """Make sure updated_catalog is created correctly."""

    # make source catalog
    cats = mc.Management(catalog_path='tests/catalogs',
                         source_catalog_name='source_catalog_test.yaml',
                         make_source_catalog=True)

    # make updated catalog
    cats.run_updated_cat(model_names=['TBOFS'],
                         updated_catalog_name='updated_catalog_test.yaml',
                         make_updated_catalog=True)

    # check updated_catalog.yaml location
    assert os.path.exists(cats.updated_catalog_name)

    # check for only hycom in catalog
    assert list(cats.updated_cat) == ['TBOFS']
    assert sorted(list(cats.updated_cat['TBOFS'])) == ['forecast', 'hindcast',
                                                       'hindcast-forecast-aggregation', 'nowcast']

    # has dir name (date) encoded in metadata at top of catalog?
    assert cats.updated_cat.metadata['updated_catalog_dir'] == cats.updated_catalog_dir

    # has source catalog at top of this file: name and associated source directory?
    assert cats.updated_cat.metadata['source_catalog_dir'] == cats.source_catalog_dir
    assert cats.updated_cat.metadata['source_catalog_name'] == cats.source_catalog_name


def setup_user_catalog_for_test():
    """setup for two other tests."""

    # make source catalog
    cats = mc.Management(catalog_path='tests/catalogs',
                         source_catalog_name='source_catalog_test.yaml',
                         make_source_catalog=True)

    # make updated catalog
    cats.run_updated_cat(model_names=['DBOFS'],
                         updated_catalog_name='updated_catalog_test.yaml',
                         make_updated_catalog=True)

    # make user catalog with several model orientations
    today = pd.Timestamp.today()
    twoweeksago = today - pd.Timedelta('14 days')
    twoweeksplus1day = twoweeksago + pd.Timedelta('1 day')
    yesterday = today - pd.Timedelta('1 day')
    hindstart = pd.Timestamp('2016-1-1')
    hindend = hindstart + pd.Timedelta('1 day')
    cats.setup_user_cat([dict(model='DBOFS', timing='forecast',
                              start_date=None, end_date=None,
                              filetype='fields', treat_last_day_as_forecast=False),
                         dict(model='DBOFS', timing='nowcast', start_date=twoweeksago, end_date=twoweeksplus1day,
                              filetype='fields', treat_last_day_as_forecast=False),
                         dict(model='DBOFS', timing='nowcast', start_date=yesterday, end_date=today,
                              filetype='regulargrid', treat_last_day_as_forecast=False),
                         dict(model='DBOFS', timing='hindcast', start_date=hindstart, end_date=hindend,
                              filetype='fields', treat_last_day_as_forecast=False),
                        ])
    return cats


def test_make_user_catalog():
    """Make sure user_catalog is created correctly."""

    cats = setup_user_catalog_for_test()

    # check user catalog name/location
    assert os.path.exists(cats.user_catalog_name)

    entries = ['DBOFS-forecast', 'DBOFS-hindcast-fields',
               'DBOFS-nowcast-fields', 'DBOFS-nowcast-regulargrid']

    # has all catalog entries
    assert sorted(list(cats.user_cat)) == entries

    # has source catalog at top of this file: name and associated source directory?
    assert cats.user_cat.metadata['source_catalog_dir'] == cats.source_catalog_dir
    assert cats.user_cat.metadata['source_catalog_name'] == cats.source_catalog_name

    # has dir name (date) encoded in metadata at top of catalog?
    assert cats.user_cat.metadata['updated_catalog_dir'] == cats.updated_catalog_dir


@pytest.mark.slow
def test_make_user_catalog_dask():
    """Make sure user_catalog model output can be read in."""

    cats = setup_user_catalog_for_test()

    entries = ['DBOFS-forecast', 'DBOFS-hindcast-fields',
               'DBOFS-nowcast-fields', 'DBOFS-nowcast-regulargrid']

    # check that can read in model output
    for entry in entries:
        cats.user_cat[entry].to_dask()
        assert cats.user_cat[entry]._ds
        cats.user_cat[entry]._ds.close()


@pytest.mark.slow
def test_treat_last_day_as_forecast():
    """Make sure more files found for forecast."""

    # make source catalog
    cats = mc.Management(catalog_path='tests/catalogs',
                         source_catalog_name='source_catalog_test.yaml',
                         make_source_catalog=True)

    today = pd.Timestamp.today()
    cats.setup_user_cat([dict(model='GOMOFS', timing='nowcast', start_date=today, end_date=today,
                              filetype='fields', treat_last_day_as_forecast=False),
                         dict(model='GOMOFS', timing='nowcast', start_date=today, end_date=today,
                              filetype='fields', treat_last_day_as_forecast=True),
                        ])

    entries = ['GOMOFS-nowcast-fields', 'GOMOFS-nowcast-fields-with_forecast']

    # has all catalog entries
    assert sorted(list(cats.user_cat)) == entries

    # make sure with_forecast has more files
    assert len(cats.user_cat['GOMOFS-nowcast-fields-with_forecast'].urlpath) > \
           len(cats.user_cat['GOMOFS-nowcast-fields'].urlpath)

    # make sure with_forecast can be read in
    assert cats.user_cat['GOMOFS-nowcast-fields-with_forecast'].to_dask()


@pytest.mark.slow
def test_forecast():
    """Test all known models for running in forecast mode."""

    cats = mc.Management(make_source_catalog=True)
    today = pd.Timestamp.today()

    # not every forecast need start_date and end_date, but some do, and all can have extra inputs.
    cats_to_make = {model: dict(timing='forecast', start_date=today,
                                end_date=today, treat_last_day_as_forecast=False)
                    for model in list(cats.updated_cat)}

    cats.setup_user_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()


@pytest.mark.slow
def test_nowcast():
    """Test all known models for running in nowcast mode."""

    cats = mc.Management(make_source_catalog=True)
    today = pd.Timestamp.today()

    # not every forecast need start_date and end_date, but some do, and all can have extra inputs.
    cats_to_make = [dict(model=model, timing='nowcast', start_date=today,
                        end_date=today, treat_last_day_as_forecast=False)
                    for model in list(cats.updated_cat) if 'nowcast' in list(cats.updated_cat[model])]

    cats.setup_user_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()


@pytest.mark.slow
def test_hindcast():
    """Test all known models for running in hindcast mode."""

    cats = mc.Management(make_source_catalog=True)
    day = pd.Timestamp.today() - pd.Timedelta('60 days')
    nextday = day + pd.Timedelta('1 day')

    cats_to_make = [dict(model=model, timing='hindcast', start_date=day,
                         end_date=nextday, treat_last_day_as_forecast=False)
                    for model in list(cats.updated_cat) if 'hindcast' in list(cats.updated_cat[model])]

    cats.setup_user_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()


@pytest.mark.slow
def test_hindcast_forecast_aggregation():
    """Test all known models for running in hindcast mode."""

    cats = mc.Management(make_source_catalog=True)
    day = pd.Timestamp.today() - pd.Timedelta('365 days')
    nextday = day + pd.Timedelta('1 day')

    cats_to_make = [dict(model=model, timing='hindcast-forecast-aggregation',
                         start_date=day, end_date=nextday, treat_last_day_as_forecast=False)
                    for model in list(cats.updated_cat) if 'hindcast-forecast-aggregation' in list(cats.updated_cat[model])]

    cats.setup_user_cat(cats_to_make)

    # make sure can load them all in too
    for source in list(cats.user_cat):
        ds = cats.user_cat[source].to_dask()
        ds.close()
