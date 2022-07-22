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
            kwargs = self._params['transform_kwargs']
            kwargs['metadata'] = self.metadata
            self._ds = self._transform(
                self._source.to_dask(),
                **kwargs,
            )

        return self._ds

    def read(self):
        """Same here."""
        return self.to_dask()


def add_attributes(ds, axis, standard_names, metadata: Optional[dict] = None):
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

    if metadata is not None and 'coords' in metadata:
        ds = ds.assign_coords({ k: ds[k] for k in metadata['coords'] })

    # set axis attributes (time, lon, lat, z potentially)
    for ax_name, var_names in axis.items():
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

    # decode times if times are floats
    if ds.cf["T"].dtype == "float64":
        ds = xr.decode_cf(ds, decode_times=True)

    return ds
