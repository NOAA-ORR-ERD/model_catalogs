"""Test different ways to input catalogs."""

from pathlib import PurePath

import intake
import numpy as np
import xarray as xr

import model_catalogs as mc


def test_default_catalogs_intake():
    """Make sure some default 'mc_' catalogs are present since are required."""

    assert "mc_CBOFS" in list(intake.cat)
    assert "mc_CIOFS" in list(intake.cat)


def test_setup_default():
    """Check that default `mc_` catalogs come through setup."""

    main_cat = mc.setup(boundaries=False)
    assert "CBOFS" in list(main_cat)
    assert "CIOFS" in list(main_cat)


def test_open_catalog(tmpdir):
    dsfname = PurePath(tmpdir) / "example_dataset.nc"
    fname = PurePath(tmpdir) / "example_catalog.yaml"

    ds = xr.Dataset()
    ds["time"] = (
        "Time",
        np.arange(10),
        # {"standard_name": "time"},
    )
    ds["Lat"] = (
        "y",
        np.arange(10),
        # {"units": "degrees_north", "standard_name": "latitude"},
    )
    ds["Lon"] = (
        "x",
        np.arange(10),
        # {"units": "degrees_east", "standard_name": "longitude"},
    )
    ds.to_netcdf(dsfname)

    catalog_text = f"""
    name: example
    metadata:
        alpha_shape: [4,50]  # dd, alpha
    sources:
        example_source:
            args:
                urlpath: {dsfname}
            description: ''
            driver: netcdf
            metadata:
                axis:
                    T: Time
                    X: x
                    Y: y
                standard_names:
                    time: time
                    latitude: Lat
                    longitude: Lon
                coords:
                    - Lon
                    - Lat
                    - Time
    """

    with open(fname, "w") as fp:
        fp.write(catalog_text)
    cat = mc.open_catalog(fname)
    ds = cat["example_source"].to_dask()
    assert ds.cf.axes["X"] == ["x"]
    assert ds.cf.axes["Y"] == ["y"]
    assert ds.cf.axes["T"] == ["Time"]
    assert ds.cf.coordinates["latitude"] == ["Lat"]
    assert ds.cf.coordinates["longitude"] == ["Lon"]
    assert ds.cf.coordinates["time"] == ["Time"]
