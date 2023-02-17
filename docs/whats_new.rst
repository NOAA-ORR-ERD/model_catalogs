:mod:`What's New`
----------------------------

v0.6.0 (February 17, 2023)
==========================
* Updated docs.
* The "freshness" parameter, which determines how much time can pass before different actions must be rerun, now has a default for each of the five actions that have freshness parameters associated with them. These parameters are set in the `__init__` file but can be overridden for any model source with the "freshness" parameter in the source metadata. More details are available :doc:`here <add_model>`.
* The "known" GOODS model catalog yaml files are no longer distributed with ``model_catalogs`` itself in order to enforce more separation between the catalog files themselves and this code. However, the package of catalogs is currently a requirement of ``model_catalogs`` and can be found at `mc-goods <https://github.com/axiom-data-science/mc-goods>`_. Note that catalog names that had names like `CBOFS-RGRID` are now called `CBOFS_RGRID` with underscores instead of hyphens. This was a necessary change for setting up the models in their own packages with entry points.
* Enforcing single threading in ``model_catalogs`` to avoid issue when using ``xr.open_mfdataset`` (which is used with `noagg` sources) in which the first time you read something in you hit an error but the second time it works. For more information check this `xarray issue <https://github.com/pydata/xarray/issues/7079>`_ or this `netcdf issue <https://github.com/Unidata/netcdf4-python/issues/1192>`_.
* User can work with a local catalog file now! See :doc:`here <catalog_modes>` for details.

  * boundaries are optionally calculated when using `mc.open_catalog()`.
  * boundaries are calculated the first time a catalog file is worked with through `mc.setup()`

* Removed requirement for `filetype` to be in catalog if sources in catalog do not need to be aggregated.
* LSOFS and LOOFS have new FVCOM versions. So, there are new versions of the model files:

  * `lsofs.yaml` and `loofs.yaml` are still the legacy POM version of the models but no longer have source `coops-forecast-noagg`, and their metadata have been updated to reflect the end dates of the model sources.
  * new catalog files `lsofs-fvcom.yaml` and `loofs-fvcom.yaml` have source `coops-forecast-noagg` that points to the new FVCOM version of the models.

* If user requests time range that is not available for a source, it will now error instead of warn.
* Bug fixed in `find_availability` so that when a source that does not have a catloc entry is checked, the Dataset is read in without extra processing and checks (including limiting the time range which otherwise would impact checking the time availability).

v0.5.0 (October 7, 2022)
========================

* Added `long_name` and `html_desc` to the catalog files
* Revised overall_start/end_datetime attributes to be more consistent
* Renamed Intake Source names from "timing"s to "model_source"s. Also some of the specific model names changed. All source names changed to reflect where the model output is coming from and whether the files are unaggregated or not.


v0.4.1 (September 27, 2022)
===========================

* Updates to ``select_date_range()``:

  - can input catalog or source
  - if ``find_availability`` needs to have been run, it will be run for you

* added note about choosing chunks to `How to Add a New Model <https://model-catalogs.readthedocs.io/en/latest/add_model.html#>`_


v0.4.0 (September 26, 2022)
===========================

* Files found for unaggregated NOAA OFS models are now sorted by correct time order.
* Repeated files (between nowcast and forecast files, and different nowcast timing cycles) are not included anymore.
* Improved behavior in aggregated NOAA OFS file selection for seamless time series between days with no forecast included and the final day of the selection that includes a forecast.
* Updated many details in `Model Aggregations <https://model-catalogs.readthedocs.io/en/latest/aggregations.html#>`_ page of docs.
* Package now available on conda-forge!
* ``source.follow_target()`` was renamed ``source.target``. It exposes the original catalog information, which is the target of the source transform. It is a property now instead of a method.
* Fixed bug when applying `standard_name` attributes to coordinate variables.
* ``model_catalogs`` can now understand the datetimes associated with NOAA OFS models. See `Model Aggregations <https://model-catalogs.readthedocs.io/en/latest/aggregations.html#>`_ for more details.
* Improved date and datetime behavior for `select_date_range()`.
* Updates chunking parameter for several models for performance reasons.
* Updates CF coordinate standard names for several catalogs.
* Input to ``mc.find_availability()`` can now be an Intake Catalog or Intake Source.
* More robust calls to ``mc.find_availability()`` in case time range cannot be found.
* Better server checks, using ``requests`` package to see if server is working. A source's server status can be checked with `source.status`.


v0.3.0 (September 9, 2022)
==========================

* Can now query the transform catalogs for the target ``urlpath`` (see `docs <https://model-catalogs.readthedocs.io/en/latest/demo.html#urlpath:-model-output-source>`_)
* Can explicitly follow the transform catalog to the target catalog with `cat.follow_target()`.
* ``select_date_range()`` now explicitly replaces the ``urlpath`` of the target when the function is run rather than later when ``to_dask()`` is run.


v0.2.0 (August 24, 2022)
========================

* Lots of updates to docs and installation instructions
