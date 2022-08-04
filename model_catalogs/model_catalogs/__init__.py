"""
Set up for using package.
"""

import shutil

from pathlib import Path
from appdirs import AppDirs

from pkg_resources import DistributionNotFound, get_distribution

from .model_catalogs import (  # noqa
    add_url_path,
    calculate_boundaries,
    find_availability,
    make_catalog,
    setup,
    transform_source,
)
from .utils import (  # noqa
    agg_for_date,
    astype,
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

# set up cache directories for package to use
# user application cache directory, appropriate to each OS
dirs = AppDirs("model_catalogs", "NOAA-ORR-ERD")
cache_dir = Path(dirs.user_cache_dir)

# set up known locations for catalogs. Can be overwritten HOW
# By default, put catalog directory in home directory
# CATALOG_PATH = cache_dir / "catalogs"
CATALOG_PATH = Path.home() / "catalogs"
# UPDATE THESE
SOURCE_CATALOG_NAME = "source_catalog.yaml"
CATALOG_PATH_DIR_ORIG = CATALOG_PATH / "orig"
CATALOG_PATH_DIR_BOUNDARY = CATALOG_PATH / "boundaries"
CATALOG_PATH_DIR_COMPILED = CATALOG_PATH / "compiled"
CATALOG_PATH_DIR_AVAILABILITY = CATALOG_PATH / "availability"
CATALOG_PATH_DIR = CATALOG_PATH / "complete"
CATALOG_PATH_UPDATED = CATALOG_PATH / "updated"
CATALOG_PATH_TMP = CATALOG_PATH / "tmp"
SOURCE_TRANSFORM = CATALOG_PATH / "transform.yaml"

# make directories
# CATALOG_PATH.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_DIR_ORIG.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_UPDATED.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_TMP.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_DIR_BOUNDARY.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_DIR_AVAILABILITY.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_DIR_COMPILED.mkdir(parents=True, exist_ok=True)

# Move "orig" catalog files to catalog dir
PKG_CATALOG_PATH_DIR_ORIG = Path(__path__[0]) / "catalogs" / "orig"
[
    shutil.copy(fname, CATALOG_PATH_DIR_ORIG)
    for fname in PKG_CATALOG_PATH_DIR_ORIG.glob("*.yaml")
]
PKG_CATALOG_PATH_DIR_BOUNDARY = Path(__path__[0]) / "catalogs" / "boundaries"
[
    shutil.copy(fname, CATALOG_PATH_DIR_BOUNDARY)
    for fname in PKG_CATALOG_PATH_DIR_BOUNDARY.glob("*.yaml")
]

# Move "transform.yaml" to catalog dir
SOURCE_TRANSFORM_REPO = Path(__path__[0]) / "catalogs" / "transform.yaml"
shutil.copy(SOURCE_TRANSFORM_REPO, SOURCE_TRANSFORM)


# availability file names
def start_filename(model, timing):
    """Return filename for model/timing start time."""
    return CATALOG_PATH_DIR_AVAILABILITY / f"{model}_{timing}_start_datetime.yaml"


def end_filename(model, timing):
    """Return filename for model/timing end time."""
    return CATALOG_PATH_DIR_AVAILABILITY / f"{model}_{timing}_end_datetime.yaml"


# Fresh parameters: how long until model output will be refreshed if requested
FRESH = {'forecast': {'start': '1 day', 'end': '4 hours'},
         'nowcast': {'start': '3 days', 'end': '1 day'},
         'hindcast': {'start': '7 days', 'end': '7 days'},
         'hindcast-forecast-aggregation': {'start': '7 days', 'end': '7 days'},
         'compiled': '1 day',
        }
