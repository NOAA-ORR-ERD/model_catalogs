"""
This file contains all information for transforming the Datasets.
"""

import cf_xarray  # noqa
import numpy as np
import xarray as xr

from intake.source.derived import GenericTransform


class DatasetTransform(GenericTransform):
    """Transform where the input and output are both Dask-compatible Datasets

    This derives from GenericTransform, and you must supply ``transform`` and
    any ``transform_kwargs``.
    """

    input_container = "xarray"
    container = "xarray"
    optional_params = {}
    _ds = None

    def to_dask(self):
        """Makes it so can read in model output."""
        if self._ds is None:
            self._pick()
            self._ds = self._transform(
                self._source.to_dask(), **self._params["transform_kwargs"]
            )
        return self._ds

    def read(self):
        """Same here."""
        return self.to_dask()


def add_attributes(ds, axis, standard_names):
    """Update Dataset metadata.

    Using supplied axis variable names and variable name mapping to associated
    standard names (from CF conventions), update the Dataset metadata.
    """

    # set standard_names for all variables
    for stan_name, var_names in standard_names.items():
        if not isinstance(var_names, list):
            var_names = [var_names]
        for var_name in var_names:
            ds[var_name].attrs["standard_name"] = stan_name

    # Run code to find vertical coordinates
    try:
        ds.cf.decode_vertical_coords()
    except Exception:
        pass

    # set axis attributes (time, lon, lat, z potentially)
    for ax_name, var_names in axis.items():
        if not isinstance(var_names, list):
            var_names = [var_names]
        for var_name in var_names:
            # var_name needs to be a coord to have attributes
            if var_name not in ds.coords:
                ds[var_name] = (
                    var_name,
                    np.arange(ds.sizes[var_name]),
                    {"axis": ax_name},
                )
            else:
                ds[var_name].attrs["axis"] = ax_name

    # this won't run for e.g. GFS which has multiple time variables
    # but also doesn't need to have the calendar updated
    try:
        attrs = ds[ds.cf["T"].name].attrs
        if ("calendar" in attrs) and (attrs["calendar"] == "gregorian_proleptic"):
            attrs["calendar"] = "proleptic_gregorian"
            ds[ds.cf["T"].name].attrs = attrs
    except KeyError:
        pass

    return xr.decode_cf(ds)
