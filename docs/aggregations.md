# Model Aggregations

## NOAA OFS Aggregations

### Overview

All of the NOAA OFS models available through ``model_catalogs`` have model output available that is unaggregated. They are unaggregated in that there are nowcast and forecast model output files available, but which files to use and the order of the files is not apparent from the thredds server. A subset of the files need to be aggregated in a particular way to get a coherent time series from the files.

#### General

Building an aggregation requires understanding their naming conventions. Most NOAA OFS models follow this convention:

- 2-D surface field output: `nos.MODELNAME.2ds.[n|f]HHH.YYYYMMDD.tCCz.nc`
- 3-D field output: `nos.MODELNAME.fields.[n|f]HHH.YYYYMMDD.tCCz.nc`

where `MODELNAME` is the short model name (e.g. `WCOFS`), `[n|f]` denotes either the nowcast or forecast results, `YYYYMMDD` is the date of the model run, `tCCz` is the cycle of the day, `HHH` is the nowcast or forecast hour, and there is one model output per file.

#### LSOFS, LOOFS, NYOFS

However, LSOFS, LOOFS, and NYOFS follow a different convention. For 3-D field outputs:

- LSOFS: `glofs.lsofs.fields.[nowcast|forecast].YYYYMMDD.tCCz.nc`
- LOOFS: `glofs.loofs.fields.[nowcast|forecast].YYYYMMDD.tCCz.nc`
- NYOFS: `nos.nyofs.fields.[nowcast|forecast].YYYYMMDD.tCCz.nc`

where `[nowcast|forecast]` denotes either the nowcast or forecast results, `tCCz` is the cycle of the day and there are 6 model outputs per file.

**Note**
As of October 2022, developmental versions of LSOFS and LOOFS have started to replace the old POM models. So, whether the files follow the general aggregation rules or the specific older rules depends on what source you retrieve the output from.

### File selection and order

#### Forecast

##### General

Usually, nowcast and forecast files are created four times a day, and output is hourly in individual files. So, each update generates 6 nowcast files and 48 forecast files (for a 48 hour forecast; the forecast length varies by model). The update cycle time will be the last model output timestep in the nowcast files and the first timestep in the forecast files.

Example filenames from one update cycle (`20141027.t15z`):

Nowcast:

- `nos.ngofs.fields.n000.20141027.t15z.nc`
- `nos.ngofs.fields.n001.20141027.t15z.nc`
- ...
- `nos.ngofs.fields.n006.20141027.t15z.nc`

Forecast:

- `nos.ngofs.fields.f000.20141027.t15z.nc`
- `nos.ngofs.fields.f002.20141027.t15z.nc`
- ...
- `nos.ngofs.fields.f048.20141027.t15z.nc`

So to make a time series, use subsequent nowcasts updates strung together sequentially
by update date/time then by ``n0001``-``n006``. If a file with ``n000`` is present, leave it off because it is a duplicate of the previous nowcast cycle.

Similarly append the forecast that is the same update cycle as the most recent nowcast. If a file with ``f000`` is present, leave off as it overlaps with the nowcast ``n006`` file.

##### LSOFS, LOOFS, NYOFS

LSOFS and LOOFS before October 2022, and NYOFS are different. They have 6 model outputs per file, and a sequence of files to make a time series forward in time with a forecast looks like:

 - `glofs.lsofs.fields.nowcast.20220916.t00z.nc`
 - `glofs.lsofs.fields.nowcast.20220916.t06z.nc`
 - `glofs.lsofs.fields.nowcast.20220916.t12z.nc`
 - `glofs.lsofs.fields.forecast.20220916.t12z.nc`

#### Nowcast cycles

##### General

To create a time series for a day if you don't want the forecast, you use only nowcast files. The pattern you should use is:

 - `nos.creofs.fields.n001.20220912.t03z.nc`
 - `nos.creofs.fields.n002.20220912.t03z.nc`
 - ...
 - `nos.creofs.fields.n006.20220912.t03z.nc`
 - `nos.creofs.fields.n001.20220912.t09z.nc`
 - ...
 - `nos.creofs.fields.n006.20220912.t09z.nc`
 - `nos.creofs.fields.n001.20220912.t15z.nc`
 - ...
 - `nos.creofs.fields.n006.20220912.t15z.nc`
 - `nos.creofs.fields.n001.20220912.t21z.nc`
 - ...
 - `nos.creofs.fields.n006.20220912.t21z.nc`

where ``n000`` files have been left off the list since they are duplicates.

##### LSOFS, LOOFS, NYOFS

For LSOFS and LOOFS before October 2022, and NYOFS, a day of model output with no forecast looks like:

 - `glofs.lsofs.fields.nowcast.20220915.t00z.nc`
 - `glofs.lsofs.fields.nowcast.20220915.t06z.nc`
 - `glofs.lsofs.fields.nowcast.20220915.t12z.nc`
 - `glofs.lsofs.fields.nowcast.20220915.t18z.nc`

### Datetimes associated with files by filename

The datetimes associated with a given NOAA OFS file is not obvious from the file name itself. Here we provide some translations. There is also a function in `model_catalogs` that will return the datetime for a NOAA OFS file: `mc.file2dt()`. Note that the times are assumed to be in UTC.

#### General

The ``n006`` file for timing cycle ``t00z`` is at midnight of the day listed in the filename. Files ``n000`` to ``n005`` for timing cycle ``t00z`` count backward in time from there. Forecast files do not have the 6 hour shift backward. The hour in the timing cycle should be added to this convention. Datetime translations are given in the following table for sample files.

The formula are:

- Nowcast files: time shift from midnight on date listed = CC + HHH - 6
- Forecast files: time shift from midnight on date listed = CC + HHH

<details>

<summary>Datetime Translation Examples</summary>

| File Name                              | Time Formula | Resulting Datetime |
| -------------------------------------- | -- | -- |
|                                        | N: CC + HHH - 6 = time shift from midnight |  |
|                                        | F: CC + HHH = time shift from midnight |  |
| nos.cbofs.fields.n001.20220913.t00z.nc | 0 + 1 - 6 = -5 | 20220912T19:00 |
| nos.cbofs.fields.n002.20220913.t00z.nc | 0 + 2 - 6 = -4 | 20220912T20:00 |
| nos.cbofs.fields.n003.20220913.t00z.nc | 0 + 3 - 6 = -3 | 20220912T21:00 |
| nos.cbofs.fields.n004.20220913.t00z.nc | 0 + 4 - 6 = -2 | 20220912T22:00 |
| nos.cbofs.fields.n005.20220913.t00z.nc | 0 + 5 - 6 = -1 | 20220912T23:00 |
| nos.cbofs.fields.n006.20220913.t00z.nc | 0 + 6 - 6 = 0 | 20220913T00:00 |
| nos.cbofs.fields.n001.20220913.t06z.nc | 6 + 1 - 6 = 1 | 20220913T01:00 |
| nos.cbofs.fields.n002.20220913.t06z.nc | 6 + 2 - 6 = 2 | 20220913T02:00 |
| nos.cbofs.fields.n003.20220913.t06z.nc | 6 + 3 - 6 = 3 | 20220913T03:00 |
| nos.cbofs.fields.n004.20220913.t06z.nc | 6 + 4 - 6 = 4 | 20220913T04:00 |
| nos.cbofs.fields.n005.20220913.t06z.nc | 6 + 5 - 6 = 5 | 20220913T05:00 |
| nos.cbofs.fields.n006.20220913.t06z.nc | 6 + 6 - 6 = 6 | 20220913T06:00 |
| nos.cbofs.fields.n001.20220913.t12z.nc | 12 + 1 - 6 = 7 | 20220913T07:00 |
| ... | ... | ... |
| nos.cbofs.fields.n006.20220913.t12z.nc | 12 + 6 - 6 = 12 | 20220913T12:00 |
| nos.cbofs.fields.n001.20220913.t18z.nc | 18 + 1 - 6 = 13 | 20220913T13:00 |
| ... | ... | ... |
| nos.cbofs.fields.n006.20220913.t18z.nc | 18 + 6 - 6 = 18 | 20220913T18:00 |
| nos.cbofs.fields.n001.20220914.t00z.nc | 0 + 1 - 6 = -5 | 20220913T19:00 |
| nos.cbofs.fields.n002.20220914.t00z.nc | 0 + 2 - 6 = -4 | 20220913T20:00 |
| nos.cbofs.fields.n003.20220914.t00z.nc | 0 + 3 - 6 = -3 | 20220913T21:00 |
| nos.cbofs.fields.n004.20220914.t00z.nc | 0 + 4 - 6 = -2 | 20220913T22:00 |
| nos.cbofs.fields.n005.20220914.t00z.nc | 0 + 5 - 6 = -1 | 20220913T23:00 |
| nos.cbofs.fields.n006.20220914.t00z.nc | 0 + 6 - 6 = 0 | 20220914T00:00 |
| nos.cbofs.fields.n001.20220914.t06z.nc | 6 + 1 - 6 = 1 | 20220914T01:00 |
| nos.cbofs.fields.n002.20220914.t06z.nc | 6 + 2 - 6 = 2 | 20220914T02:00 |
| nos.cbofs.fields.n003.20220914.t06z.nc | 6 + 3 - 6 = 3 | 20220914T03:00 |
| nos.cbofs.fields.n004.20220914.t06z.nc | 6 + 4 - 6 = 4 | 20220914T04:00 |
| nos.cbofs.fields.n005.20220914.t06z.nc | 6 + 5 - 6 = 5 | 20220914T05:00 |
| nos.cbofs.fields.n006.20220914.t06z.nc | 6 + 6 - 6 = 6 | 20220914T06:00 |
| nos.cbofs.fields.n001.20220914.t12z.nc | 12 + 1 - 6 = 7 | 20220914T07:00 |
| ... | ... | ... |
| nos.cbofs.fields.n006.20220914.t12z.nc | 12 + 6 - 6 = 12 | 20220914T12:00 |
| nos.cbofs.fields.f001.20220914.t12z.nc | 12 + 1 = 13 | 20220914T13:00 |
| nos.cbofs.fields.f002.20220914.t12z.nc | 12 + 2 = 14 | 20220914T14:00 |
| nos.cbofs.fields.f003.20220914.t12z.nc | 12 + 3 = 15 | 20220914T15:00 |
| ... | ... | ... |
| nos.cbofs.fields.f012.20220914.t12z.nc | 12 + 12 = 24 | 20220915T00:00 |
| ... | ... | ... |
| nos.cbofs.fields.f036.20220914.t12z.nc | 12 + 36 = 48 | 20220916T00:00 |
| ... | ... | ... |
| nos.cbofs.fields.f047.20220914.t12z.nc | 12 + 47 = 59 | 20220916T11:00 |
| nos.cbofs.fields.f048.20220914.t12z.nc | 12 + 48 = 60 | 20220916T12:00 |

 </details>


#### LSOFS, LOOFS, NYOFS

**Note**
LSOFS and LOOFS models changed in October 2022 to be FVCOM instead of POM and model output going forward from then do not follow these legacy rules.

Each nowcast model output file for these three models contains 6 model time steps. The last model time step in a given file corresponds to the datetime information in the file name, and the other file times are each an hour previous.

A forecast file contains all time steps for the forecast, the first time of which is an hour after the datetime represented in the file name.

The formula are:
- Nowcast files: time shifts from midnight on date listed in file = [CC - 5, CC - 4, CC - 3, CC - 2, CC - 1, CC]
- Forecast files: time shifts from midnight on date listed in file = [CC + 1, CC + 2, ..., CC + N], where N is 60 for LSOFS and LOOFS and N is 54 for NYOFS.

<details>

<summary>Datetime Translation Examples</summary>

| File Name                              | Time Formula | Resulting Datetime |
| -------------------------------------- | -- | -- |
| glofs.lsofs.fields.nowcast.20220915.t00z.nc | -5, -4, -3, -2, -1, 0 | [2022-09-14 19:00:00, 2022-09-14 20:00:00, 2022-09-14 21:00:00,2022-09-14 22:00:00,2022-09-14 23:00:00,2022-09-15 00:00:00] |
| glofs.lsofs.fields.nowcast.20220915.t06z.nc | 1, 2, 3, 4, 5, 6 | [2022-09-15 01:00:00, 2022-09-15 02:00:00, 2022-09-15 03:00:00, 2022-09-15 04:00:00, 2022-09-15 05:00:00, 2022-09-15 06:00:00] |
| glofs.lsofs.fields.forecast.20220916.t12z.nc | 13, 14, ..., 13+60 | [2022-09-16 13:00:00, 2022-09-16 14:00:00, ..., 2022-09-19 00:00:00] |

</details>

## Aggregation in ``model_catalogs``

Aggregation occurs in ``model_catalogs`` when a user calls ``mc.select_date_range()``. For NOAA OFS models, there are functions called there that:

* learn the structure of the thredds catalogs (``mc.find_catrefs()``),
* find the URLs of the relevant model output files on the thredds server (``mc.find_filelocs()``),
* determine exactly which files from the file locations should be used to create the aggregation (``mc.agg_for_date()``).
* the file locations are now available in a list (``filelocs_urlpath``) which are inserted into the model source ``urlpath`` variable so that when ``source.to_dask()`` is run, those file locations are read in using `xarray` `open_mfdataset()`.

Note that the base thredds catalog location is saved for each relevant NOAA OFS model source in the metadata and can be accessed with

```
import model_catalogs as mc
main_cat = mc.setup()
main_cat['CBOFS']['coops-forecast-noagg'].metadata['catloc']
```

which would return

```
'https://opendap.co-ops.nos.noaa.gov/thredds/catalog/NOAA/CBOFS/MODELS/catalog.xml'
```

### Translate NOAA OFS filenames to datetimes

There is a function in `model_catalogs` that interprets the known NOAA OFS model file names and returns the datetime(s) in the file. Here is how to use that:

```
import model_catalogs as mc
main_cat = mc.setup()
[mc.file2dt(url) for url in main_cat['NGOFS2']['coops-forecast-noagg'].urlpath]
```

returns, for example:

```
[Timestamp('2022-09-14 21:00:00'), Timestamp('2022-09-15 00:00:00')]
```

Once `mc.select_date_range()` has been run, which overwrites the example files in `source.urlpath` with the file locations for the date range entered, there would be more/different urlpath file locations and dates, accordingly. The dates associated with the `urlpath` can also be checked with `source.dates`.

### How to Extend

#### Aggregation of full files

Add another conditional statement in ``mc.select_date_range()`` for the new model aggregation case (this may need to be an indicator of some kind in the new model source metadata that can be checked for). For the new case, put in the necessary logic to pull out the file locations that should be aggregated together, and save them into variable ``filelocs_urlpath``.


#### Aggregation of partial files

Aggregating part of a set of files requires an additional step. You would need to first find the set of files to aggregate as in the previous listing. However, to select which times from the files you want to keep you would need to run preprocessing code on each file as it is being read in with ``xarray``'s ``open_mfdataset()``. A good approach to set this up would be:

- in the new catalog file, have an argument that will go to the xarray read in step called `preprocess` that indicates preprocessing is necessary, for example part of the catalog file would look like:

```
    name: CBOFS
    sources:
      coops-forecast-noagg:
      driver: opendap
      description: Unaggregated 3D Chesapeake Bay model in ROMS from 30 days ago with 48 hour forecast.
      args:
        chunks:
          ocean_time: 1
        parallel: True
        engine: netcdf4
        preprocess: True
```

- in ``model_catalogs`` ``process.py``, a conditional statement can look for the ``preprocess: True`` flag and if present, run preprocessing code for this case that will pull out the first N timesteps of each model output file.
