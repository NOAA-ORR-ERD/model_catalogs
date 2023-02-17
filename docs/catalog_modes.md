---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3.8.13 ('model_catalogs')
  language: python
  name: python3
---

```{code-cell} ipython3
import model_catalogs as mc
import intake
from tempfile import TemporaryDirectory
from pathlib import PurePath
import xarray as xr
import numpy as np
```

# Different Modes for Catalog Use

## Installed Catalogs

`model_catalogs`, as of v0.6.0, can use catalog files flexibly. This page demonstrates some of the uses.

You can install your catalog files or use local versions. Catalogs that are meant to be used long-term should be installed as Python packages. Once they are installed, they are available through the default `intake` catalog. See information on how to install your catalog files in the {doc}`docs page <add_model>`.

See all available installed catalogs with:

```{code-cell} ipython3
list(intake.cat)
```

Installed catalogs to be used with `model_catalogs` should have a specified prefix ending with an underscore so they can be easily selected from the default catalog. `model_catalogs` required the installation of [`mc-goods`](https://github.com/axiom-data-science/mc-goods), a package of model catalogs, which have the prefix "mc_".

+++

### With `mc.setup()`

The default behavior is to read in the `mc-goods` catalogs, as follows

```{code-cell} ipython3
main_cat = mc.setup()  # default `locs="mc_"`
list(main_cat)
```

If you have other catalogs available in the default catalog, you can instead input that prefix so the relevant catalogs can be found. For example, if the catalogs are installed with the prefix "other_", you can set up your main cat with them using

```
main_cat = mc.setup(locs="other_")
```

If you want to use those catalogs and the default catalogs, you can do so with

```
main_cat = mc.setup(locs=["mc_", "other_"])
```

You may instead set up with a local catalog file in which you input a string or Path of its location, for example,

```
main_cat = mc.setup(locs="path_to_catalog_file")
```

Or, you can input a combination of these.

## Local Catalog File

### With or without `mc.setup()`

Let's write a catalog file to demonstrate using a local catalog file:

```{code-cell} ipython3
with TemporaryDirectory() as tmpdirname:

    tmpdir = PurePath(tmpdirname)

    # Create a Dataset and save it
    dsfname = PurePath(tmpdir) / "example.nc"
    fname = PurePath(tmpdir) / 'example.yaml'

    ds = xr.Dataset()
    ds["time"] = ("Time", np.arange(10),)
    ds["Lat"] = ("y", np.arange(10),)
    ds["Lon"] = ("x", np.arange(10),)
    ds.to_netcdf(dsfname)

    catalog_text = f"""
    name: example
    metadata:
        alpha_shape: [4,50]  # dd, alpha
    sources:
        example_source:
            args:
                urlpath: {dsfname}
            description: ''
            driver: netcdf
            metadata:
                axis:
                    T: Time
                    X: x
                    Y: y
                standard_names:
                    time: time
                    latitude: Lat
                    longitude: Lon
                coords:
                    - Lon
                    - Lat
                    - Time
    """
    fp = open(fname, 'w')
    fp.write(catalog_text)
    fp.close()

    main_cat = mc.setup(str(fname), override=True)
    print(list(main_cat["example"]))
```

Alternatively you can read in and work with a single catalog file with having the structure of nested catalogs in `mc.setup()`. For that, you can simply read in a catalog file with the following, and still have access to all the associated functions.

```
mc.open_catalog(catalog_path)
```
