import os
import shutil

from glob import glob
from .model_catalogs import Management, agg_for_date, make_catalog


# make directories
# os.makedirs('.catalogs', exist_ok=True)
# os.makedirs('.catalogs/source_catalogs', exist_ok=True)
os.makedirs('../catalogs/updated_catalogs', exist_ok=True)
os.makedirs('../catalogs/user_catalogs', exist_ok=True)
#
# set up testing
os.makedirs('../tests/catalogs', exist_ok=True)
# copy source catalogs to tests
os.makedirs('../tests/catalogs/source_catalogs/', exist_ok=True)
os.makedirs('../tests/catalogs/source_catalogs/orig', exist_ok=True)
[shutil.copy(fname, '../tests/catalogs/source_catalogs/orig/') for fname in glob('../catalogs/source_catalogs/orig/*')]
