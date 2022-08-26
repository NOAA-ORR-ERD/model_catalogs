:mod:`What's New`
----------------------------

v0.3.0 (unreleased)
===================

* Can now query even the transform catalogs for the target `urlpath` (see future update in docs)
* Can explicitly follow the transform catalog to the target catalog with `cat.follow_target()`.
* `select_date_range()` now explicitly replaces the `urlpath` of the target when the function is run rather than later when `to_dask()` is run.


v0.2.0 (August 24, 2022)
========================

* Lots of updates to docs and installation instructions
