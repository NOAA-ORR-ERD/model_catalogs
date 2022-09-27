How to Add a New Model
======================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation


Scenario: you are given a netCDF file containing model output, so it is available only locally on your computer. How should you use this file? Some approaches:

Make a new catalog for it
-------------------------

Make your new catalog entry at ``mc.CAT_PATH_ORIG``. The benefit of doing this is you have a chance to add the model output's specific metadata in the file, and you can share this for other people to use.

A basic template for your ``.yaml`` file is::

  name: Model name
  sources:
    local:
      driver: netcdf  # this is from ``intake-xarray``
      args:
        [any kwargs to pass to ``xr.open_mfdataset``]
        urlpath: [path to local file]
      # if you want to be able to use ``cf-xarray`` you can fill in metadata here
      # also if you want to be able to refer to variables with the same standard_names
      metadata:
        axis:
          T: time
          # also X, Y, and Z dimensions if you want
        standard_names:
          sea_surface_elevation: zeta
          # etc
        coords:
          # can assign coordinates by listing names here

Your catalog file needs to have the ``.yaml`` suffix and be present in ``mc.CAT_PATH_ORIG`` to be found.

Chunks are difficult to choose in general. Start with an existing model's options if there is something similar to the new model. In general, reasonable chunks options are: None, -1, 'auto', or {'[time axis name': 1}. You'll need to test the resulting behavior with each option. What is best depends on a lot of things.


Skip straight to ``extract_model``
----------------------------------

If you will be using `extract_model <https://github.com/axiom-data-science/extract_model>`_ with your ``xarray Dataset``, try using the file directly with ``extract_model`` (skipping ``model_catalogs``) with::

  import extract_model as em
  import xarray as xr

  ds = xr.open_mfdataset([file location], preprocess=em.preprocess)

``extract_model`` has some logic to improve metadata built into ``em.preprocess`` which may be adequate for you. If, however, as you go through your workflow you find that metadata is missing, you may want to try the other approach and write up a catalog file to represent the model output.
