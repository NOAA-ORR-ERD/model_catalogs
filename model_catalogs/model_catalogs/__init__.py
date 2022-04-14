"""
Set up for using package.
"""

import shutil

from pathlib import Path

from pkg_resources import DistributionNotFound, get_distribution

from .model_catalogs import (  # noqa
    add_url_path,
    complete_source_catalog,
    find_availability,
    make_catalog,
    setup_source_catalog,
)
from .utils import (  # noqa
    agg_for_date,
    find_bbox,
    find_catrefs,
    find_filelocs,
    get_dates_from_ofs,
)


try:
    __version__ = get_distribution("model_catalogs").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"


# set up known locations for catalogs. Can be overwritten HOW
# By default, put catalog directory in home directory
CATALOG_PATH = Path.home() / "catalogs"
SOURCE_CATALOG_NAME = "source_catalog.yaml"
CATALOG_PATH_DIR_ORIG = CATALOG_PATH / "orig"
CATALOG_PATH_DIR = CATALOG_PATH / "complete"
CATALOG_PATH_UPDATED = CATALOG_PATH / "updated"
CATALOG_PATH_TMP = CATALOG_PATH / "tmp"
SOURCE_TRANSFORM = CATALOG_PATH / "transform.yaml"

# make directories
# CATALOG_PATH.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_DIR_ORIG.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_UPDATED.mkdir(parents=True, exist_ok=True)
CATALOG_PATH_TMP.mkdir(parents=True, exist_ok=True)

# Move "orig" catalog files to catalog dir
PKG_CATALOG_PATH_DIR_ORIG = Path(__path__[0]) / "catalogs" / "orig"
[
    shutil.copy(fname, CATALOG_PATH_DIR_ORIG)
    for fname in PKG_CATALOG_PATH_DIR_ORIG.glob("*.yaml")
]

# Move "transform.yaml" to catalog dir
SOURCE_TRANSFORM_REPO = Path(__path__[0]) / "catalogs" / "transform.yaml"
shutil.copy(SOURCE_TRANSFORM_REPO, SOURCE_TRANSFORM)
