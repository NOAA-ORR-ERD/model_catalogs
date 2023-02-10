"""
Set up for using package.
"""

import importlib

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from appdirs import AppDirs

from .model_catalogs import (  # noqa
    find_availability,
    make_catalog,
    open_catalog,
    select_date_range,
    setup,
    transform_source,
)
from .utils import (  # noqa
    agg_for_date,
    astype,
    calculate_boundaries,
    file2dt,
    filedates2df,
    find_bbox,
    find_catrefs,
    find_filelocs,
    get_fresh_parameter,
    is_fresh,
    status,
)


try:
    __version__ = version("model_catalogs")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

# this forces single threading which avoids an issue described here:
# https://github.com/pydata/xarray/issues/7079
# https://github.com/Unidata/netcdf4-python/issues/1192
import dask


dask.config.set(scheduler="single-threaded")

# set up known locations for catalogs.
# this is where the original model catalog files and previously-calculated
# model boundaries can be found, which are hard-wired in the repo
# version change for this behavior
try:  # >= Python 3.9
    CAT_PATH = importlib.resources.files("model_catalogs") / "support_files"
except AttributeError:  # < Python3.9
    with importlib.resources.path("model_catalogs", "support_files") as pth:
        CAT_PATH = pth
# CAT_PATH_ORIG = CAT_PATH / "orig"
# CAT_PATH_BOUNDARIES = CAT_PATH / "boundaries"
CAT_PATH_TRANSFORM = CAT_PATH / "transform.yaml"

# test files
try:  # >= Python 3.9
    TEST_PATH = importlib.resources.files("model_catalogs") / "tests"
except AttributeError:  # < Python3.9
    with importlib.resources.path("model_catalogs", "tests") as pth:
        TEST_PATH = pth
TEST_PATH_FILES = TEST_PATH / "test_files"

# set up cache directories for package to use
# user application cache directory, appropriate to each OS
dirs = AppDirs("model_catalogs", "NOAA-ORR-ERD")
cache_dir = Path(dirs.user_cache_dir)


# This is where files are saved to refer back to for saving time
CACHE_PATH = cache_dir
CACHE_PATH_COMPILED = CACHE_PATH / "compiled"
CACHE_PATH_AVAILABILITY = CACHE_PATH / "availability"
CACHE_PATH_FILE_LOCS = CACHE_PATH / "file_locs"

# whenever a package of catalogs is installed, have it copy any available boundaries to here
CAT_PATH_BOUNDARIES = CACHE_PATH / "boundaries"

# make directories
CACHE_PATH_COMPILED.mkdir(parents=True, exist_ok=True)
CACHE_PATH_AVAILABILITY.mkdir(parents=True, exist_ok=True)
CACHE_PATH_FILE_LOCS.mkdir(parents=True, exist_ok=True)
CAT_PATH_BOUNDARIES.mkdir(parents=True, exist_ok=True)


def TEST_PATH_FILE(model, model_source):
    """Return file path to test file."""
    model = model.lower().replace("-", "_")
    return (TEST_PATH_FILES / f"{model}_{model_source}").with_suffix(".nc")


def FILE_PATH_COMPILED(model):
    """Return filename for model boundaries information."""
    return (CACHE_PATH_COMPILED / model).with_suffix(".yaml")


# availability file names
def FILE_PATH_START(model, model_source):
    """Return filename for model/model_source start time."""
    return CACHE_PATH_AVAILABILITY / f"{model}_{model_source}_start_datetime.yaml"


def FILE_PATH_END(model, model_source):
    """Return filename for model/model_source end time."""
    return CACHE_PATH_AVAILABILITY / f"{model}_{model_source}_end_datetime.yaml"


def FILE_PATH_BOUNDARIES(model):
    """Return filename for model boundaries information."""
    return (CAT_PATH_BOUNDARIES / model).with_suffix(".yaml")


def FILE_PATH_CATREFS(model, model_source):
    """Return filename for model/model_source start time."""
    return CACHE_PATH_AVAILABILITY / f"{model}_{model_source}_catrefs.yaml"


def FILE_PATH_AGG_FILE_LOCS(model, model_source, date, is_fore):
    """Return filename for aggregated file locations.

    Date included to day."""
    date = astype(date, pd.Timestamp)
    name = f"{model}_{model_source}_{date.isoformat()[:10]}_is-forecast_{is_fore}.yaml"
    return CACHE_PATH_FILE_LOCS / name


# Fresh parameters: how long until model output avialability will be refreshed
# for `find_availabililty()` if requested
FRESH = {
    "start": "1 day",
    "end": "4 hours",
    "catrefs": "6 hours",
    "file_locs": "4 hours",
    "compiled": "6 hours",  # want to be on the same calendar day as when they were compiled; this approximates that.  # noqa: E501
}
