"""
This file contains all information for transforming the Datasets.
"""
import warnings

from typing import Optional

import cf_xarray  # noqa
import numpy as np
import pandas as pd
import xarray as xr

from intake.source.derived import GenericTransform

import model_catalogs as mc


# extract_model might be necessary for reading in model output, if using FVCOM model
try:
    import extract_model  # noqa: F401

    EM_AVAILABLE = True
except ImportError:  # pragma: no cover
    EM_AVAILABLE = False  # pragma: no cover


yesterday = pd.Timestamp.today() - pd.Timedelta("1 day")


class DatasetTransform(GenericTransform):
    """Transform where the input and output are both Dask-compatible Datasets

    This derives from GenericTransform, and you must supply ``transform`` and
    any ``transform_kwargs``.
    """

    input_container = "xarray"
    container = "xarray"
    optional_params = {}
    _ds = None

    @property
    def urlpath(self):
        """Data location for target

        Can be overwritten by `update_urlpath`

        Returns
        -------
        list
            Location(s) for where data can be found
        """

        if not hasattr(self, "_urlpath"):
            self.target
        return self._urlpath

    @property
    def status(self):
        """Status of server for source.

        Returns
        -------
        bool
            If True, server was reachable.
        """

        if not hasattr(self, "_status"):

            if self.target.describe()["driver"][0] == "opendap":
                suffix = ".das"
            else:
                suffix = ""
            self._status = mc.status(mc.astype(self.urlpath, list)[0], suffix=suffix)
        return self._status

    @property
    def dates(self):
        """Dates associated with urlpath files

        ...if there is more than one. Doesn't work for static links or RTOFS models. So, this is really for NOAA OFS models.

        Returns
        -------
        list
            Ordered dates to match `urlpath` locations.
        """

        if "catloc" in self.metadata:
            dates = []
            for url in self.urlpath:
                dates.extend(mc.astype(mc.file2dt(url), list))
            self._dates = dates
        else:
            self._dates = None

        return self._dates

    @property
    def target(self):
        """Connect target into Transform

        This way can expose some information to query. This will only run once per object.

        Returns
        -------
        Intake Source
            Source that is the target of the object Transform
        """

        # all functions here call follow_target to start
        # need to pick the source only once
        if self._source is None:
            self._pick()

            # if "yesterday" is in user_parameters for original source
            # run with that sent in
            if any(
                [
                    "yesterday" in d.values()
                    for d in self._source.describe()["user_parameters"]
                ]
            ):
                self._source = self._source(yesterday=yesterday)

            # self is of type DatasetTransform instead of OpenDapSource
            # since the OpenDapSource is the target of the Transform
            # but make it easy to check urlpath
            self._urlpath = self._source.urlpath

        return self._source

    def update_urlpath(self):
        """Update urlpath for transform.

        Run this in `select_date_range` for aggregated sources. This can be run more than once.
        """

        if not hasattr(self, "target"):
            self.target

        kwargs = self._params["transform_kwargs"]

        # this is the one that is used when .to_dask() is run
        self._source.urlpath = kwargs["urlpath"]

        # but this one is printed when looking at the source, so better change it too
        self._source._captured_init_kwargs["urlpath"] = kwargs["urlpath"]

        # self is of type DatasetTransform instead of OpenDapSource
        # since the OpenDapSource is the target of the Transform
        # but make it easy to check urlpath
        self._urlpath = kwargs["urlpath"]

    def to_dask(self):
        """Makes it so can read in model output.

        Returns
        -------
        Dataset
            xarray Dataset that has been read in
        """

        if self._ds is None:

            # This is a special flag to be able to just quickly connect to the model output
            # without all the extra checks and processing (other than to transform the original
            # target). Used to find_availability without modifying the Dataset itself.
            if (
                "skip_dask_processing" in self._kwargs
                and self._kwargs["skip_dask_processing"]
            ):
                return self._transform(
                    self._source.to_dask(),
                    metadata=self.metadata,
                )

            if not hasattr(self, "target"):
                self.target

            kwargs = self._params["transform_kwargs"]

            # Checks to catch potential user pitfalls
            # Make sure that user has filled in urlpath if needed if model_source is coops-forecast-noagg.
            # These are unaggregated and come with 2 sample files but user should be warned if they haven't been
            # replaced since it might be a mistake
            if "sample_locs" in self._source.metadata:
                if self._source.urlpath == self._source.metadata["sample_locs"]:
                    warnings.warn(
                        "Note that you are using the original example files in your input source. You may want to instead first run `mc.select_date_range()` to search for and add the correct model output files for your desired date range, then run `to_dask()`.",  # noqa: E501
                        UserWarning,
                    )

            # Make sure that source has urlpath
            # check for if the urlpath is null and if so `select_date_range()`
            # needs to be run to fill it in
            elif self.target.urlpath is None:
                raise KeyError(
                    "The input source `urlpath` does not have a value. You probably want to run `mc.select_date_range()` before running `to_dask()`."  # noqa: E501
                )

            # Alert if triangularmesh engine is required (from FVCOM) but not present
            if (
                self.target.describe()["driver"][0] == "opendap"
                and self.target.engine == "triangularmesh_netcdf"
                and not EM_AVAILABLE
            ):
                raise ModuleNotFoundError(  # pragma: no cover
                    "`extract_model` is not available but contains the 'triangularmesh_netcdf' engine that is required for a model."
                )

            # if "yesterday" is in user_parameters for original source, check that yesterday is still
            # yesterday. Otherwise, info in source is old.
            if any(
                [
                    "yesterday" in d.values()
                    for d in self._source.describe()["user_parameters"]
                ]
            ):
                check_yesterday = pd.Timestamp.today() - pd.Timedelta("1 day")
                if yesterday.date() != check_yesterday.date():
                    warnings.warn(
                        f"You may be running with an out of date source, and you may consider restarting the kernel to update. Yesterday from code: {yesterday.date()}, yesterday right now: {check_yesterday.date()}.",  # noqa: E501
                        UserWarning,
                    )

            # if "today" is in user_parameters for original source, check that today is still
            # today. Otherwise, info in source is old.
            if any(
                [
                    "tod" in d.values()
                    for d in self._source.describe()["user_parameters"]
                ]
            ):
                check_today = pd.Timestamp.today()
                today = [
                    d["default"]
                    for d in self._source.describe()["user_parameters"]
                    if "tod" in d.values()
                ][0]
                if today.date() != check_today.date():
                    warnings.warn(
                        f"You may be running with an out of date source, and you may consider restarting the kernel to update. Today from code: {today.date()}, today right now: {check_today.date()}.",  # noqa: E501
                        RuntimeWarning,
                    )

            # This sends the metadata to `add_attributes()`
            self._ds = self._transform(
                self._source.to_dask(),
                metadata=self.metadata,
            )

            # drop any time duplicates that may be present (RTOFS can have)
            self._ds = self._ds.drop_duplicates(dim=self._ds.cf.axes["T"])

            # check for 'urlpath' update being sent in, if so use it to
            # subselect ds in time
            if "start_date" in kwargs and "end_date" in kwargs:

                try:
                    ds_temp = self._ds.cf.sel(
                        T=slice(kwargs["start_date"], kwargs["end_date"])
                    )

                    if len(ds_temp.cf["T"]) == 0:
                        self._ds = None
                        raise RuntimeError(
                            f"The time slice requested for source {self.name}, {self.cat.name}, {kwargs['start_date']} to {kwargs['end_date']}, results in no times in the Dataset."
                        )
                    else:
                        self._ds = ds_temp

                except KeyError:
                    # self._ds = self._ds
                    warnings.warn(
                        f"The time slice requested for source {self.name}, {self.cat.name}, {kwargs['start_date']} to {kwargs['end_date']}, did not result in a valid Dataset, and so was not used.",
                        RuntimeWarning,
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
    Dataset
        Improved Dataset.
    """
    # set standard_names for all variables
    if metadata is not None and "standard_names" in metadata:
        for stan_name, var_names in metadata["standard_names"].items():
            if not isinstance(var_names, list):
                var_names = [var_names]
            for var_name in var_names:
                if var_name in ds.variables:
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

    # set axis attributes (T, X, Y, Z potentially)
    if metadata is not None and "axis" in metadata:
        for ax_name, var_names in metadata["axis"].items():
            var_names = mc.astype(var_names, list)
            for var_name in var_names:

                # Check dims, coords, and data_vars:
                if (
                    var_name in ds.dims
                    or var_name in ds.data_vars.keys()
                    or var_name in ds.coords
                ):
                    # var_name needs to be a coord to have attributes
                    if var_name not in ds.coords:
                        ds = ds.assign_coords(
                            {
                                var_name: (
                                    var_name,
                                    np.arange(ds[var_name].size),
                                    {"axis": ax_name},
                                )
                            }
                        )
                    else:
                        ds[var_name].attrs["axis"] = ax_name

                else:
                    warnings.warn(
                        f"The variable {var_name} input in a catalog file is not present in the Dataset.",
                        UserWarning,
                    )

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
        metadata is not None
        and "axis" in metadata
        and "T" in metadata["axis"]
        and isinstance(metadata["axis"]["T"], list)
        and len(metadata["axis"]["T"]) > 1
    ):
        for time_var in metadata["axis"]["T"]:
            if ds[time_var].dtype == "float64":
                ds = xr.decode_cf(ds, decode_times=True)
                break
    elif "T" in ds.cf and ds.cf["T"].dtype == "float64":
        ds = xr.decode_cf(ds, decode_times=True)

    if metadata is not None and "formula_terms" in metadata:
        for varname in metadata["formula_terms"]:
            ds[varname].attrs["formula_terms"] = metadata["formula_terms"][varname]

    # This is an internal attribute used by netCDF which xarray doesn't know or care about, but can
    # be returned from THREDDS.
    if "_NCProperties" in ds.attrs:
        del ds.attrs["_NCProperties"]

    return ds
