"""
Set up for using package.
"""

import os
import shutil

from glob import glob

from pkg_resources import DistributionNotFound, get_distribution

from .model_catalogs import (setup_source_catalog, agg_for_date, make_catalog,
                             complete_source_catalog, update_catalog, add_url_path)  # noqa


try:
    __version__ = get_distribution("model_catalogs").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"


# set up known locations for catalogs. Can be overwritten HOW
CATALOG_PATH = f"{__path__[0]}/catalogs"
SOURCE_CATALOG_NAME = "source_catalog.yaml"
CAT_SOURCE_BASE = f"{CATALOG_PATH}/source_catalogs"
CAT_USER_BASE = f"{CATALOG_PATH}/user_catalogs"

SOURCE_CATALOG_DIR_ORIG = f"{CAT_SOURCE_BASE}/orig"
SOURCE_CATALOG_DIR = f"{CAT_SOURCE_BASE}/complete"

# make directories
# os.makedirs('.catalogs', exist_ok=True)
# os.makedirs('.catalogs/source_catalogs', exist_ok=True)
# os.makedirs(f"{__path__[0]}/catalogs/updated_catalogs", exist_ok=True)
os.makedirs(f"{__path__[0]}/catalogs/user_catalogs", exist_ok=True)
#
# set up testing
os.makedirs(f"{__path__[0]}/tests/catalogs", exist_ok=True)
# copy source catalogs to tests
os.makedirs(f"{__path__[0]}/tests/catalogs/source_catalogs/", exist_ok=True)
os.makedirs(f"{__path__[0]}/tests/catalogs/source_catalogs/orig", exist_ok=True)
[
    shutil.copy(fname, f"{__path__[0]}/tests/catalogs/source_catalogs/orig/")
    for fname in glob(f"{__path__[0]}/catalogs/source_catalogs/orig/*")
]
# also copy transform template
shutil.copy(
    f"{__path__[0]}/catalogs/source_catalogs/transform.yaml",
    f"{__path__[0]}/tests/catalogs/source_catalogs/",
)
