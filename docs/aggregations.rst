Model Aggregations
==================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation


NOAA OFS Aggregations
---------------------

Overview
********

All of the NOAA OFS models available through ``model_catalogs`` have model output available that is unaggregated. They are unaggregated in that there are nowcast and forecast model output files available, but a subset of the files need to be aggregated in a particular way to get a coherent time series from the files.

Building an aggregation requires understanding their naming conventions:

- 2-D surface field output: nos.wcofs.2ds.[n|f]HHH.YYYYMMDD.tCCz.nc
- 3-D field output: nos.wcofs.fields.[n|f]NNN.YYYYMMDD.tCCz.nc

Where [nowcast/forecast] or [n/f] denotes either the nowcast or forecast results; YYYYMMDD is the date of the model run, tCCz is the cycle of the day; HHH is the nowcast or forecast hour.

How to aggregate
****************

Nowcast and forecast files are created four times a day. Output is hourly in individual files. So each update generates 6 nowcast files and 48 forecast files. The update cycle time will be the last model output timestep in the nowcast files and the first timestep in the forecast files.

Example filenames from one update cycle (20141027.t15z):

Nowcast:

- nos.ngofs.fields.n000.20141027.t15z.nc
- nos.ngofs.fields.n001.20141027.t15z.nc
- ...
- nos.ngofs.fields.n006.20141027.t15z.nc

Forecast:

- nos.ngofs.fields.f000.20141027.t15z.nc
- nos.ngofs.fields.f002.20141027.t15z.nc
- ...
- nos.ngofs.fields.f048.20141027.t15z.nc

So to make a time series, use subsequent nowcasts updates strung together sequentially
by update date/time then by n0001-n005 (leave off the last one as it overlaps with
the next set of files)

Similarly append the forecast that is the same update cycle as the most recent nowcast.


Aggregation in ``model_catalogs``
*********************************

Aggregation occurs in ``model_catalogs`` when a user calls ``mc.select_date_range()``. For NOAA OFS models, there are functions called there that:

* learn the structure of the thredds catalogs (``mc.find_catrefs()``),
* find the URLs of the relevant model output files on the thredds server (``mc.find_filelocs()``),
* determine exactly which files from the file locations should be used to create the aggregation (``mc.agg_for_date()``).
* the file locations are now available in a list (``filelocs_urlpath``) which are inserted into the model source ``urlpath`` variable so that when ``source.to_dask()`` is run, those file locations are read in using `xarray` `open_mfdataset()`.


How to Extend
-------------

Aggregation of full files
*************************

Add another conditional statement in ``mc.select_date_range()`` for the new model aggregation case (this may need to be an indicator of some kind in the new model source metadata that can be checked for). For the new case, put in the necessary logic to pull out the file locations that should be aggregated together, and save them into variable ``filelocs_urlpath``.


Aggregation of partial files
****************************

Aggregating part of a set of files requires an additional step. You would need to first find the set of files to aggregate as in the previous listing. However, to select which times from the files you want to keep you would need to run preprocessing code on each file as it is being read in with ``xarray``'s ``open_mfdataset()``. A good approach to set this up would be:

- in the new catalog file, have an argument that will go to the xarray read in step called `preprocess` that indicates preprocessing is necessary, for example part of the catalog file would look like::

    name: CBOFS
    sources:
      nowcast:
    driver: opendap
    description: Unaggregated 3D Chesapeake Bay model in ROMS from 30 days ago with 48 hour forecast.
    args:
      chunks:
        ocean_time: 1
      parallel: True
      engine: netcdf4
      preprocess: True

- in ``model_catalogs`` ``process.py``, a conditional statement can look for the ``preprocess: True`` flag and if present, run preprocessing code for this case that will pull out the first N timesteps of each model output file.
