"""
Set up for using package.
"""

import os
import shutil

from glob import glob

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
CATALOG_PATH = f"{__path__[0]}/catalogs"
SOURCE_CATALOG_NAME = "source_catalog.yaml"

CATALOG_PATH_DIR_ORIG = f"{CATALOG_PATH}/orig"
CATALOG_PATH_DIR = f"{CATALOG_PATH}/complete"
CATALOG_PATH_UPDATED = f"{CATALOG_PATH}/updated"
CATALOG_PATH_TMP = f"{CATALOG_PATH}/tmp"

# make directories
os.makedirs(".catalogs", exist_ok=True)
os.makedirs(f"{__path__[0]}/catalogs/updated", exist_ok=True)
os.makedirs(f"{__path__[0]}/catalogs/tmp", exist_ok=True)

# set up testing
os.makedirs(f"{__path__[0]}/tests/catalogs", exist_ok=True)
# # copy source catalogs to tests
# os.makedirs(f"{__path__[0]}/tests/catalogs/source_catalogs/", exist_ok=True)
os.makedirs(f"{__path__[0]}/tests/catalogs/orig", exist_ok=True)
[
    shutil.copy(fname, f"{__path__[0]}/tests/catalogs/orig/")
    for fname in glob(f"{__path__[0]}/catalogs/orig/*")
]
# also copy transform template
shutil.copy(
    f"{__path__[0]}/catalogs/transform.yaml",
    f"{__path__[0]}/tests/catalogs/",
)
