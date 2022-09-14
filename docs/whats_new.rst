:mod:`What's New`
----------------------------

v0.4.0 (unreleased)
===================

* Files found for unaggregated NOAA OFS models are now sorted by correct time order.
* Repeated files (between nowcast and forecast files, and different nowcast timing cycles) are not included anymore.
* Improved behavior in aggregated NOAA OFS file selection for seamless time series between days with no forecast included and the final day of the selection that includes a forecast.
* Updated many details in `Model Aggregations <https://model-catalogs.readthedocs.io/en/latest/aggregations.html#>`_ page of docs.


v0.3.0 (September 9, 2022)
==========================

* Can now query the transform catalogs for the target `urlpath` (see `docs <https://model-catalogs.readthedocs.io/en/latest/demo.html#urlpath:-model-output-source>`_)
* Can explicitly follow the transform catalog to the target catalog with `cat.follow_target()`.
* `select_date_range()` now explicitly replaces the `urlpath` of the target when the function is run rather than later when `to_dask()` is run.


v0.2.0 (August 24, 2022)
========================

* Lots of updates to docs and installation instructions
