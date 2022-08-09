"""
Set up for using package.
"""

import importlib
import shutil
import pandas as pd

from pathlib import Path
from appdirs import AppDirs

from pkg_resources import DistributionNotFound, get_distribution

from .model_catalogs import (  # noqa
    find_availability,
    make_catalog,
    select_date_range,
    setup,
    transform_source,
)
from .utils import (  # noqa
    agg_for_date,
    astype,
    calculate_boundaries,
    find_bbox,
    find_catrefs,
    find_filelocs,
    get_dates_from_ofs,
    get_fresh_parameter,
    is_fresh
)


try:
    __version__ = get_distribution("model_catalogs").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"

# set up known locations for catalogs. Can be overwritten HOW
# this is where the original model catalog files and previously-calculated
# model boundaries can be found, which are hard-wired in the repo
# CAT_PATH = importlib.resources.path('model_catalogs', 'catalogs')
# CAT_PATH = Path(__file__).parent / "catalogs"
with importlib.resources.path('model_catalogs', 'catalogs') as pth:
    CAT_PATH = pth
CAT_PATH_ORIG = CAT_PATH / "orig"
CAT_PATH_BOUNDARIES = CAT_PATH / "boundaries"
CAT_PATH_TRANSFORM = CAT_PATH / "transform.yaml"

# set up cache directories for package to use
# user application cache directory, appropriate to each OS
dirs = AppDirs("model_catalogs", "NOAA-ORR-ERD")
cache_dir = Path(dirs.user_cache_dir)

# This is where files are saved to refer back to for saving time
CACHE_PATH = cache_dir / "catalogs"
# SOURCE_CATALOG_NAME = "source_catalog.yaml"
CACHE_PATH_COMPILED = CACHE_PATH / "compiled"
CACHE_PATH_AVAILABILITY = CACHE_PATH / "availability"
CACHE_PATH_FILE_LOCS = CACHE_PATH / "file_locs"

# make directories
# CAT_PATH.mkdir(parents=True, exist_ok=True)
# CAT_PATH_ORIG.mkdir(parents=True, exist_ok=True)
# CAT_PATH_BOUNDARIES.mkdir(parents=True, exist_ok=True)
CACHE_PATH_COMPILED.mkdir(parents=True, exist_ok=True)
CACHE_PATH_AVAILABILITY.mkdir(parents=True, exist_ok=True)
CACHE_PATH_FILE_LOCS.mkdir(parents=True, exist_ok=True)

# # Move "orig" catalog files to catalog dir
# PKG_CAT_PATH_ORIG = Path(__path__[0]) / "catalogs" / "orig"
# [
#     shutil.copy(fname, CAT_PATH_ORIG)
#     for fname in PKG_CAT_PATH_ORIG.glob("*.yaml")
# ]
# PKG_CAT_PATH_BOUNDARY = Path(__path__[0]) / "catalogs" / "boundaries"
# [
#     shutil.copy(fname, CAT_PATH_BOUNDARIES)
#     for fname in PKG_CAT_PATH_BOUNDARY.glob("*.yaml")
# ]

# # Move "transform.yaml" to catalog dir
# SOURCE_TRANSFORM_REPO = Path(__path__[0]) / "catalogs" / "transform.yaml"
# shutil.copy(SOURCE_TRANSFORM_REPO, SOURCE_TRANSFORM)


def FILE_PATH_COMPILED(modelyaml):
    """Return filename for model boundaries information."""
    return CACHE_PATH_COMPILED / modelyaml


# availability file names
def FILE_PATH_START(model, timing):
    """Return filename for model/timing start time."""
    return CACHE_PATH_AVAILABILITY / f"{model}_{timing}_start_datetime.yaml"


def FILE_PATH_END(model, timing):
    """Return filename for model/timing end time."""
    return CACHE_PATH_AVAILABILITY / f"{model}_{timing}_end_datetime.yaml"


def FILE_PATH_BOUNDARIES(modelyaml):
    """Return filename for model boundaries information."""
    return CAT_PATH_BOUNDARIES / modelyaml


def FILE_PATH_CATREFS(model, timing):
    """Return filename for model/timing start time."""
    return CACHE_PATH_AVAILABILITY / f"{model}_{timing}_catrefs.yaml"


def FILE_PATH_AGG_FILE_LOCS(model, timing, date, is_fore):
    """Return filename for aggregated file locations.

    Date included to day."""
    date = astype(date, pd.Timestamp)
    name = f"{model}_{timing}_{date.isoformat()[:10]}_is-forecast_{is_fore}.yaml"
    return CACHE_PATH_FILE_LOCS / name


# Fresh parameters: how long until model output avialability will be refreshed
# for `find_availabililty()` if requested
FRESH = {'forecast': {'start': '1 day', 'end': '4 hours', 'catrefs': '6 hours',
                        'file_locs': '4 hours'},
         'nowcast': {'start': '3 days', 'end': '4 hours', 'catrefs': '6 hours',
                    'file_locs': '4 hours'},
         'hindcast': {'start': '7 days', 'end': '1 day', 'catrefs': '1 day',
                    'file_locs': '1 day'},
         'hindcast-forecast-aggregation': {'start': '7 days', 'end': '1 day'},
         'compiled': '6 hours',  # want to be on the same calendar day as when they were compiled; this approximates that.
        }
