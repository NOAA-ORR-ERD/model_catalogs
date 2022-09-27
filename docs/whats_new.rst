:mod:`What's New`
----------------------------

v0.4.1 (unreleased)
===================

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
