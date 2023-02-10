How to Include a New Model
==========================

Also:

* what to include in catalog file
* freshness parameters
* catalog packages


Scenario: you want to use a new model with ``model_catalogs``. How should you go about doing this?


Make a new catalog for it
-------------------------

One Intake catalog file should represent a single model domain whose sources all provide access to model output from running on the same grid. Take a look at the top of an existing catalog file to see what catalog-level metadata is set to know what should be consistent between sources in a catalog file. If the horizontal grid is different (e.g., a subset of a model domain), that should be a different catalog file. If the vertical grid is different (e.g., the output is only at the surface of a 3D model), that should be a different catalog file.

What should your catalog file include?
**************************************

A basic template for your ``.yaml`` file is::

  name: MODEL_NAME
  metadata:
    alpha_shape: [2,5]  # dd, alpha
  sources:
    local:  # name of source
      driver: netcdf  # this is from ``intake-xarray``. Could be opendap or zarr, etc. Intake driver for source.
      args:
        [any kwargs to pass to ``xr.open_mfdataset``]
        urlpath: [path to local file]
      # if you want to be able to use ``cf-xarray`` you can fill in metadata here
      # also if you want to be able to refer to variables with the same standard_names
      metadata:
        axis:
          T: time  # T is the cf-xarray axis name for time. time is the variable name for the example dataset.
          # also X, Y, and Z dimensions if you want
        standard_names:
          sea_surface_elevation: zeta  # sea_surface_elevation is the standard name to be matched to variable named zeta
          # etc
        coords:
          latitude: Lat  # latitude is the cf-xarray name for the coordinate. Lat is the example variable dataset name.
          # can assign coordinates by listing names here. coords must already exist and be named
        freshness:  # any or none of these can be included to override the package defaults in __init__. See details below.
          start: "7 days"
          end: "4 hours"
          catrefs: "3 days"
          file_locs: "14 days"

Your catalog file needs to have the ``.yaml`` suffix.


Chunks
******

Selections for chunks can be input in your catalog file under args as kwargs to pass to xarray. Chunks are difficult to choose in general. Start with an existing model's options if there is something similar to the new model. In general, reasonable chunks options are: None, -1, 'auto', or {'[time axis name': 1}. You'll need to test the resulting behavior with each option. What is best depends on a lot of things. You can get more information by searching online with something like "xarray best chunks to use".

Freshness
*********

The "freshness" parameters, which determine how much time can pass before different actions must be rerun, now have defaults (set in the `__init__` file) for each of the five actions that have freshness parameters associated with them. Possible parameters are:

* start
* end
* catrefs
* file_locs
* compiled

The "compiled" freshness parameter cannot be overridden by metadata in a catalog source. However, the others can be overridden for any model source with the "freshness" parameter in the source metadata. The value of any of these should be a pandas Timestamp-interpretable string.

In a catalog file, they would look like this::

  name: catalog_or_model_names
  sources:
    source_name:
      ...
      metadata:
        freshness:
          start: "7 days"
          end: "4 hours"



For calculating boundaries
**************************

The parameters necessary for calculating the polygon describing the boundary of a numerical model are `dd` and `alpha`. `dd` is how much to decimate the model output: 1 is no decimation, 2 is taking every other point, etc. `alpha` is the parameter used in `alphashape <https://github.com/bellockk/alphashape>`_ that determines how tight of a polygon to make around the points. An `alpha` of 0 gives the qhull of the numerical domain back. 1 to 5 are medium values, and higher is 20 or 50. This should be provided in the catalog file at the catalog level::

  name: catalog_name
  metadata:
    alpha_shape: [4,50]  # dd, alpha

You will probably need to experiment with these values to get a good representation of your model boundary.


How to use your new catalog file
--------------------------------

Make a new package
******************

Catalog files of models that are to be reused should be formally put together in a Python package and made installable by PyPI and conda-forge. This is possible through ``intake``. Useful and relevant ``intake`` docs pages are `here <https://intake.readthedocs.io/en/latest/data-packages.html>`_ and `there <https://github.com/intake/intake-examples/tree/master/data_package>`_

If you want to make a new package of catalog files, it might be easiest to start from existing packages:

* mc-goods: https://github.com/axiom-data-science/mc-goods
* mc-nwgoa: https://github.com/axiom-data-science/mc-nwgoa

For use with ``model_catalogs``, prefix all of your catalogs from a package with a string like ``"mc_"`` or ``"othermc_"`` so they are easily used in the setup function.


Use local catalog file
**********************

You can also use your catalog file in ``model_catalogs`` as a local file to which you provide the path. You can use this path as an input to ``mc.setup()`` or ``mc.open_catalog()``.
