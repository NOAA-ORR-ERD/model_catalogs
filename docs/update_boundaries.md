# How to Update Boundaries

Boundaries files will be searched for automatically when `mc.setup()` is run. If the command has previously been run with the requested catalog files, then the boundaries files should already exist. If new catalog files are being used, then boundaries files will be calculated as each catalog file is handled.

Boundaries files are saved to `mc.FILE_PATH_BOUNDARIES(catalog_name)` where the `catalog_name` is determined at the top of the catalog file itself under "name".

If you want to calculate the boundaries separately from the call to `mc.setup()`, you can do so with

```
import model_catalogs as mc
boundaries = mc.calculate_boundaries([CATALOGS], save_files=False, return_boundaries=True)
```

You can also calculate them by opening a single catalog and requesting that the boundaries be calculated with:

```
import model_catalogs as mc
cat = mc.open_catalog(CATALOG_PATH, boundaries=True, save_boundaries=False)
```

You can save the boundaries to the cache with the relevant keyword argument for either approach.
