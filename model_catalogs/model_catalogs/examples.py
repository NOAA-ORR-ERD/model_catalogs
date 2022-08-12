#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module for common components for scripting examples."""
import time

from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
import xarray as xr

from extract_model import utils as em_utils

import model_catalogs as mc


class Timer:
    """A class to aid with measuring timing."""

    def __init__(self, msg=None):
        """Initializes the current time."""
        self.t0 = time.time()
        self.msg = msg

    def tick(self):
        """Update the timer start."""
        self.t0 = time.time()

    def tock(self) -> float:
        """Return elapsed time in ms."""
        return (time.time() - self.t0) * 1000.0

    def format(self):
        time_in_ms = self.tock()
        if time_in_ms > 60000:
            return f"{time_in_ms / 60000:.1f} min"
        if time_in_ms > 2000:
            return f"{time_in_ms / 1000:.1f} s"
        return f"{time_in_ms:.1f} ms"

    def __enter__(self):
        """With context."""
        self.tick()

    def __exit__(self, type, value, traceback):
        """With exit."""
        if self.msg is not None:
            print(self.msg.format(self.format()))


STANDARD_NAMES = [
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
    "eastward_wind",
    "northward_wind",
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "sea_floor_depth",
]


@dataclass
class FetchConfig:
    """Configuration data class for fetching."""

    model_name: str
    output_pth: Path
    start: pd.Timestamp
    end: pd.Timestamp
    bbox: Tuple[float, float, float, float]
    timing: str
    standard_names: List[str] = field(default_factory=lambda: STANDARD_NAMES)
    surface_only: bool = False


def parse_bbox(val: str) -> Tuple[float, float, float, float]:
    """Return the bounding box parsed from the comma-delimited string.

    Parameters
    ----------
    val : str
        Comma-delimited sequence of 4 float values for (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
        tuple of 4 floats
    """
    values = val.split(",")
    if len(values) != 4:
        raise ValueError(
            "bbox should include four numbers: lon_min,lat_min,lon_max,lat_max"
        )
    return tuple(float(i) for i in values)


def get_surface(ds: xr.Dataset) -> xr.Dataset:
    """Return a dataset that is reduced to only the surface layer."""
    model_guess = em_utils.guess_model_type(ds)
    if all([ds[zaxis].ndim < 2 for zaxis in ds.cf.axes["Z"]]):
        return ds.cf.sel(Z=0, method="nearest")
    elif model_guess == "FVCOM":
        vertical_dims = set()
        for varname in ds.cf.axes["Z"]:
            vertical_dim = ds[varname].dims[0]
            vertical_dims.add(vertical_dim)
        sel_kwargs = {vdim: 0 for vdim in vertical_dims}
        return ds.isel(**sel_kwargs)
    elif model_guess == "SELFE":
        return ds.isel(nv=-1)
    raise ValueError("Can't decode vertical coordinates.")


def fetch(fetch_config: FetchConfig):
    """Downloads and subsets the model data.

    Parameters
    ----------
    fetch_config : FetchConfig
        The configuration object which contains the model name, timing, start/end dates of the
        request, etc.
    """
    print("Setting up source catalog")
    with Timer("\tSource catalog generated in {}"):
        main_cat = mc.setup()

    print(
        f"Generating catalog specific for {fetch_config.model_name} {fetch_config.timing}"
    )
    with Timer("\tSpecific catalog generated in {}"):
        source = mc.select_date_range(
            main_cat[fetch_config.model_name],
            start_date=fetch_config.start,
            end_date=fetch_config.end,
            timing=fetch_config.timing,
        )
    print("Getting xarray dataset for model data")
    with Timer("\tCreated dask-based xarray dataset in {}"):
        ds = source.to_dask()

    if fetch_config.surface_only:
        print("Selecting only surface data.")
        with Timer("\tIndexed surface data in {}"):
            ds = get_surface(ds)
    print("Subsetting data")
    with Timer("\tSubsetted dataset in {}"):
        ds_ss = (
            ds.em.filter(fetch_config.standard_names)
            # .cf.sel(T=slice(fetch_config.start, fetch_config.end))
            .em.sub_grid(bbox=fetch_config.bbox)
        )
    print(
        f"Writing netCDF data to {fetch_config.output_pth}. This may take a long time..."
    )
    with Timer("\tWrote output to disk in {}"):
        ds_ss.to_netcdf(fetch_config.output_pth)
    print("Complete")


def parse_config(
    main: Callable,
    model_name: str,
    default_bbox: Tuple[float, float, float, float],
    output_dir: Path,
    default_timing: str = "hindcast",
    standard_names: List[str] = None,
    default_start: datetime = None,
    default_end: datetime = None,
) -> FetchConfig:
    """Parse command line arguments into a FetchConfig object.

    Parameters
    ----------
    main : function
        A reference to the main function of the script. This is used to fill in the help output
        while parsing arguments.
    model_name : str
        Name of the model (as it appears in the catalog files).
    default_bbox : tuple of floats
        The default bounding box to use for subsetting if the user does not specify one in the
        command line arguments.
    output_dir : Path
        The output path for where to write resulting netCDF files to.
    default_timing : str
        The default model run-type to use if not specified by CLI arguments. One of "forecast",
        "nowcast", or "hindcast".
    standard_names : list of strings
        The default list of standard names to use to filter on if not specified by CLI arguments.
    default_start : datetime
        The default start time of the query to use if not specified by CLI arguments.
    default_end : datetime
        The default end time of the query to use if not specified by CLI arguments.

    Returns
    -------
    FetchConfig
        An object which contains all of the information needed by the `fetch` function for
        requesting, subsetting, and filtering a dataset.

    """
    if standard_names is None:
        standard_names = STANDARD_NAMES
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "-t",
        "--timing",
        choices=["hindcast", "nowcast", "forecast"],
        default=default_timing,
        help="Model Timing Choice.",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=pd.Timestamp,
        default=default_start,
        help="Request start time",
    )
    parser.add_argument(
        "-e", "--end", type=pd.Timestamp, default=default_end, help="Request end time"
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--bbox", type=parse_bbox, default=default_bbox, help="Specify the bounding box"
    )
    parser.add_argument(
        "--surface", action="store_true", default=False, help="Fetch only surface data."
    )
    args = parser.parse_args()

    # Sanity check on start/end time
    if args.start is None and args.end is None:
        start = pd.Timestamp("2022-06-20")
        end = pd.Timestamp("2022-06-21")
    elif args.start is None or args.end is None:
        raise ValueError("start time or end time not specified")
    else:
        start = args.start
        end = args.end
    if start >= end:
        raise ValueError("end time must be greater than start time")

    output_filename = f"{model_name}_{args.timing}_{start:%Y%m%d}-{end:%Y%m%d}.nc"
    output_pth = output_dir / output_filename
    if output_pth.exists():
        if args.force:
            output_pth.unlink()
        else:
            raise FileExistsError(f"{output_pth} already exists")

    return FetchConfig(
        model_name=model_name,
        output_pth=output_pth,
        start=start,
        end=end,
        bbox=args.bbox,
        timing=args.timing,
        standard_names=standard_names,
        surface_only=args.surface,
    )
