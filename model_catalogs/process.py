"""
This file contains all information for transforming the Datasets.
"""
from typing import Optional

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
            kwargs = self._params["transform_kwargs"]

            # check for "yesterday" in kwargs and use it if present
            # this is for some RTOFS models
            if "yesterday" in kwargs:
                self._source = self._source(yesterday=kwargs["yesterday"])

            # check for 'urlpath' update being sent in, if so use it to update
            if "urlpath" in kwargs:
                self._source.urlpath = kwargs["urlpath"]

            # Checks to catch potential user pitfalls
            # Make sure that user has filled in urlpath if needed: OFS nowcast
            # and some OFS forecast. These are unaggregated and come with 2
            # sample files but user should be warned if they haven't been
            # replaced since it might be a mistake
            if "sample_locs" in self._source.metadata:
                if self._source.urlpath == self._source.metadata["sample_locs"]:
                    # CHANGE TO LOGGER WARNING and also print warning
                    print(
                        "Note that you are using the original example files in your input source. You may want to instead first run `mc.select_date_range()` to search for and add the correct model output files for your desired date range, then run `to_dask()`."  # noqa: E501
                    )

            # Make sure that user has filled in urlpath if needed: OFS hindcast
            # check for if the urlpath is null and if so `select_date_range()`
            # needs to be run to fill it in
            elif self._source.urlpath is None:
                raise KeyError(
                    "The input source `urlpath` does not have a value. You probably want to run `mc.select_date_range()` before running `to_dask()`."  # noqa: E501
                )

            # This sends the metadata to `add_attributes()`
            self._ds = self._transform(
                self._source.to_dask(),
                metadata=self.metadata,
            )

            # check for 'urlpath' update being sent in, if so use it to
            # subselect ds in time
            if "start_date" in kwargs and "end_date" in kwargs:
                self._ds = self._ds.cf.sel(
                    T=slice(kwargs["start_date"], kwargs["end_date"])
                )

        return self._ds

    def read(self):
        """Same here."""
        return self.to_dask()


def add_attributes(ds, metadata: Optional[dict] = None):
    """Update Dataset metadata.

    Update the Dataset metadata with metadata passed in from catalog files.

    Parameters
    ----------
    ds : Dataset
        xarray Dataset containing model output.
    metadata : dict, optional
        Metadata that has processing information to apply

    Returns
    -------
    Improved Dataset.
    """
    # set standard_names for all variables
    if metadata is not None and "standard_names" in metadata:
        for stan_name, var_names in metadata["standard_names"].items():
            if not isinstance(var_names, list):
                var_names = [var_names]
            for var_name in var_names:
                if var_name in ds.data_vars:
                    ds[var_name].attrs["standard_name"] = stan_name

    # # Run code to find vertical coordinates
    # try:
    #     # create name mapping
    #     snames = ['ocean_s_coordinate_g1', 'ocean_s_coordinate_g2', 'ocean_sigma_coordinate']
    #     s_vars = [standard_names[sname] for sname in snames if sname in standard_names][0]
    #     z_vars = axis['Z']
    #     outnames = {s_var: z_var for s_var, z_var in zip(s_vars, z_vars)}
    #     ds.cf.decode_vertical_coords(outnames=outnames)
    # except Exception:
    #     pass

    if metadata is not None and "coords" in metadata:
        ds = ds.assign_coords({k: ds[k] for k in metadata["coords"]})

    # set axis attributes (time, lon, lat, z potentially)
    if metadata is not None and "axis" in metadata:
        for ax_name, var_names in metadata["axis"].items():
            if not isinstance(var_names, list):
                var_names = [var_names]
            for var_name in var_names:
                # var_name needs to exist
                # if ax_name == 'X':
                #     import pdb; pdb.set_trace()

                if var_name in ds.dims:
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

    # check for a dataset having both `missing_value` and `_FillValue` attributes defined
    # delete `missing_value` in this case (arbitrary decision of which to delete)
    for data_var in ds.data_vars:
        if "missing_value" in ds[data_var].attrs and "_FillValue" in ds[data_var].attrs:
            del ds[data_var].attrs["missing_value"]

    # decode times if times are floats.
    # Some datasets like GFS have multiple time coordinates for different phenomena like
    # precipitation accumulation vs winds vs surface albedo average.
    if (
        "T" in metadata["axis"]
        and isinstance(metadata["axis"]["T"], list)
        and len(metadata["axis"]["T"]) > 1
    ):
        for time_var in metadata["axis"]["T"]:
            if ds[time_var].dtype == "float64":
                ds = xr.decode_cf(ds, decode_times=True)
                break
    elif ds.cf["T"].dtype == "float64":
        ds = xr.decode_cf(ds, decode_times=True)

    if metadata is not None and "formula_terms" in metadata:
        for varname in metadata["formula_terms"]:
            ds[varname].attrs["formula_terms"] = metadata["formula_terms"][varname]

    # This is an internal attribute used by netCDF which xarray doesn't know or care about, but can
    # be returned from THREDDS.
    if "_NCProperties" in ds.attrs:
        del ds.attrs["_NCProperties"]

    return ds
