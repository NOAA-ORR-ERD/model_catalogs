"""
Utilities to help with catalogs.
"""

import fnmatch
import intake
import model_catalogs as mc
import numpy as np
import os
import pandas as pd
import re
import shapely.geometry
from siphon.catalog import TDSCatalog


def find_bbox(ds, dd=None, alpha=None):
    """Determine bounds and boundary of model.

    Parameters
    ----------
    ds: Dataset
        xarray Dataset containing model output.
    dd: int, optional
        Number to decimate model output lon/lat by, as a stride.
    alpha: float, optional
        Number for alphashape to determine what counts as the convex hull.
        Larger number is more detailed, 1 is a good starting point.

    Returns
    -------
    List containing the name of the longitude and latitude variables for ds,
    geographic bounding box of model output: [min_lon, min_lat, max_lon,
    max_lat], low res and high res wkt representation of model boundary.
    """

    hasmask = False

    try:
        lon = ds.cf["longitude"].values
        lat = ds.cf["latitude"].values
        lonkey = ds.cf["longitude"].name
        latkey = ds.cf["latitude"].name

    except KeyError:
        if "lon_rho" in ds:
            lonkey = "lon_rho"
            latkey = "lat_rho"
        else:
            lonkey = list(ds.cf[["longitude"]].coords.keys())[0]
            # need to make sure latkey matches lonkey grid
            latkey = f"lat{lonkey[3:]}"
        # In case there are multiple grids, just take first one;
        # they are close enough
        lon = ds[lonkey].values
        lat = ds[latkey].values

    # check for corresponding mask (rectilinear and curvilinear grids)
    if any([var for var in ds.data_vars if "mask" in var]):
        if ("mask_rho" in ds) and (lonkey == "lon_rho"):
            maskkey = lonkey.replace("lon", "mask")
        elif "mask" in ds:
            maskkey = "mask"
        else:
            maskkey = None
        if maskkey in ds:
            lon = ds[lonkey].where(ds[maskkey] == 1).values
            lon = lon[~np.isnan(lon)].flatten()
            lat = ds[latkey].where(ds[maskkey] == 1).values
            lat = lat[~np.isnan(lat)].flatten()
            hasmask = True

    # This is structured, rectilinear
    # GFS
    if (lon.ndim == 1) and ("nele" not in ds.dims) and not hasmask:

        nlon, nlat = ds["lon"].size, ds["lat"].size
        lonb = np.concatenate(([lon[0]] * nlat, lon[:], [lon[-1]] * nlat, lon[::-1]))
        latb = np.concatenate((lat[:], [lat[-1]] * nlon, lat[::-1], [lat[0]] * nlon))
        # boundary = np.vstack((lonb, latb)).T
        p = shapely.geometry.Polygon(zip(lonb, latb))
        p0 = p.simplify(1)
        p1 = p

    elif hasmask or ("nele" in ds.dims):  # unstructured

        assertion = (
            "dd and alpha need to be defined in the source_catalog for this model."
        )
        assert dd is not None and alpha is not None, assertion

        # need to calculate concave hull or alphashape of grid
        import alphashape

        # low res, same as convex hull
        p0 = alphashape.alphashape(list(zip(lon, lat)), 0.0)
        # downsample a bit to save time, still should clearly see shape of domain
        pts = shapely.geometry.MultiPoint(list(zip(lon[::dd], lat[::dd])))
        p1 = alphashape.alphashape(pts, alpha)

    # useful things to look at: p.wkt  #shapely.geometry.mapping(p)
    return lonkey, latkey, list(p0.bounds), p1.wkt
    # return lonkey, latkey, list(p0.bounds), p0.wkt, p1.wkt


def agg_for_date(date, strings, filetype, is_forecast=False, pattern=None):
    """Aggregate NOAA OFS-style nowcast/forecast files.

    Parameters
    ----------
    date: str of datetime, pd.Timestamp
        Date of day to find model output files for. Doesn't pay attention to hours/minutes/seconds.
    strings: list
        List of strings to be filtered. Expected to be file locations from a
        thredds catalog.
    filetype: str
        Which filetype to use. Every NOAA OFS model has "fields" available, but some have "regulargrid"
        or "2ds" also. This availability information is in the source catalog for the model under
        `filetypes` metadata.
    is_forecast: bool, optional
        If True, then date is the last day of the time period being sought and the forecast files
        should be brought in along with the nowcast files, to get the model output the length of the
        forecast out in time. The forecast files brought in will have the latest timing cycle of the
        day that is available. If False, all nowcast files (for all timing cycles) are brought in.
    pattern: str, optional
        If a model file pattern doesn't match that assumed in this code, input one that will work.
        Currently only NYOFS doesn't match but the pattern is built into the catalog file.

    Returns
    -------
    List of URLs for where to find all of the model output files that match the keyword arguments.
    """

    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    # # brings in nowcast and forecast for any date in the catalog
    # pattern0 = f'*{filetype}*.t??z.*'
    # # brings in nowcast and forecast for only the day specified
    # pattern0 = date.strftime(f'*{filetype}*.%Y%m%d.t??z.*')

    if pattern is None:
        pattern = date.strftime(f"*{filetype}*.n*.%Y%m%d.t??z.*")
        # pattern = date.strftime(f'*{filetype}*.n???.%Y%m%d.t??z.*')
    else:
        pattern = eval(f"f'{pattern}'")
    # pattern = date.strftime(f'*{filetype}*.n*.%Y%m%d.t??z.*')

    # '*{filetype}.n???.{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}.t??z.*'

    # pattern = eval(f"f'{pattern}'")
    # import pdb; pdb.set_trace()
    fnames = fnmatch.filter(strings, pattern)

    if is_forecast:

        import re

        regex = re.compile(".t[0-9]{2}z.")
        # substrings
        subs = [substr[2:4] for substr in regex.findall("".join(strings))]
        cycle = sorted(list(set(subs)))[-1]  # noqa: F841

        # cycle = sorted(list(set([fname[ fname.find(start:='.t') + len(start):fname.find('z.')] for fname in fnames])))[-1]  # noqa: E501
        # import pdb; pdb.set_trace()
        # pattern1 = f'*{filetype}*.t{cycle}z.*'

        # replace '.t??z.' in pattern with '.t{cycle}z.' and replace '.n*.' with '.*.'
        pattern1 = pattern.replace(".t??z.", ".t{cycle}z.").replace(".n*.", ".*.")
        pattern1 = eval(f"f'{pattern1}'")
        fnames = fnmatch.filter(strings, pattern1)

    # filelocs = [cat.datasets.get(fname).access_urls["OPENDAP"] for fname in fnames]

    return fnames


def find_catrefs(catloc):
    """Find hierarchy of catalog references for thredds catalog.

    Parameters
    ----------
    catloc: str
        Search in thredds catalog structure from base catalog, catloc.

    Returns
    -------
    list of tuples containing the hierarchy of directories in the thredds catalog
    structure to get to where the datafiles start.
    """

    # 0th level catalog
    cat = TDSCatalog(catloc)
    catrefs_to_check = list(cat.catalog_refs)

    # 1st level catalog
    catrefs_to_check1 = [cat.catalog_refs[catref].follow().catalog_refs for catref in catrefs_to_check]

    # Combine the catalog references together
    catrefs = [(catref0, catref1) for catref0, catrefs1 in zip(catrefs_to_check, catrefs_to_check1) for catref1 in catrefs1]

    # Check first one to see if there are more catalog references or not
    cat_ref_test = cat.catalog_refs[catrefs[0][0]].follow().catalog_refs[catrefs[0][1]].follow().catalog_refs
    # If there are more catalog references, run another level of catalog and combine, ## 2
    if len(cat_ref_test) > 1:
        catrefs2 = [cat.catalog_refs[catref[0]].follow().catalog_refs[catref[1]].follow().catalog_refs for catref in catrefs]
        catrefs = [(catref[0], catref[1], catref22) for catref, catref2 in zip(catrefs, catrefs2) for catref22 in catref2]

    # OUTPUT
    return catrefs


def find_filelocs(catref, catloc, filetype='fields'):
    """Find thredds file locations.

    TEST with filetype still

    Parameters
    ----------
    catref: tuple
        2 or 3 labels describing the directories from catlog to get the data
        locations.
    catloc: str
        Base thredds catalog location.
    filetype FILL IN

    Returns
    -------
    Locations of files found from catloc to hierarchical location desribed by
    catref.
    """

    # ADD FILETYPE IN
    filelocs = []
    cat = TDSCatalog(catloc)
    if len(catref) == 2:
        catref1, catref2 = catref
        cat1 = cat.catalog_refs[catref1].follow()
        cat2 = cat1.catalog_refs[catref2].follow()
        last_cat = cat2
        datasets = cat2.datasets
    elif len(catref) == 3:
        catref1, catref2, catref3 = catref
        cat1 = cat.catalog_refs[catref1].follow()
        cat2 = cat1.catalog_refs[catref2].follow()
        cat3 = cat2.catalog_refs[catref3].follow()
        last_cat = cat3
        datasets = cat3.datasets
    # import pdb; pdb.set_trace()
    for dataset in datasets:
        if 'stations' not in dataset and 'vibrioprob' not in dataset and filetype in dataset:
            # print(dataset)
            url = last_cat.datasets[dataset].access_urls['OPENDAP']
            # print(url)
            filelocs.append(url)
    return filelocs


def get_dates_from_ofs(filelocs, filetype, norf, firstorlast):
    """Return either start or end datetime from list of filenames.

    MORE
    """

    pattern = f"*{filetype}*.{norf}*.????????.t??z.*"
    # pattern = date.strftime(f"*{filetype}*.n*.%Y%m%d.t??z.*")
    if firstorlast == 'first':
        ind = 0
    elif firstorlast == 'last':
        ind = -1
    fileloc = sorted(fnmatch.filter(filelocs, pattern))[ind]
    filename = fileloc.split('/')[-1]
    date = pd.Timestamp(filename.split('.')[4])
    regex = re.compile(".t[0-9]{2}z.")
    cycle = [substr[2:4] for substr in regex.findall("".join(filename))][0]
    regex = re.compile(f".{norf}" + "[0-9]{3}.")
    filetime = [substr[2:5] for substr in regex.findall("".join(filename))][0]
    # in UTC
    datetime = date + pd.Timedelta(f'{cycle} hours') + pd.Timedelta(f'{filetime} hours')

    return datetime


def find_availability(model, override=False):
    """Find availability for model for 'forecast' and 'hindcast'.

    Parameters
    ----------
    model: str
        Name of model, e.g., CBOFS
    """

    model = model.upper()

    ran_forecast, ran_hindcast = False, False

    # check to see if catalog file already exists and is not stale:
    catpath = f"{mc.CATALOG_PATH_UPDATED}/{model.lower()}.yaml"
    if os.path.exists(catpath):
        cat = intake.open_catalog(catpath)
    else:
        ref_cat = mc.setup_source_catalog()
        cat = ref_cat[model]
    # ref_cat = setup_source_catalog()
    # cat = ref_cat[model]

    # deal with RTOFS completely separately
    if 'RTOFS' in model:
        ds = cat['forecast'].to_dask()
        start_datetime = str(ds.time.values[0])
        end_datetime = str(ds.time.values[-1])
        cat['forecast'].metadata['start_datetime'] = start_datetime
        cat['forecast'].metadata['end_datetime'] = end_datetime
        cat_metadata = cat.metadata
        metadata = {
            "catalog_path": mc.CATALOG_PATH,
            # "source_catalog_name": mc.SOURCE_CATALOG_NAME,
            # "filetype": cat.metadata["filetype"]
        }
        cat_metadata.update(metadata)
        new_user_cat = mc.make_catalog(
            cat['forecast'],
            f"{model.upper()}",
            f"Model {model} with availability included.",
            cat_metadata,
            cat['forecast']._entry._driver,
            cat_path=mc.CATALOG_PATH_UPDATED,
        )
        return new_user_cat

    # determine filetype to send to `agg_for_date`
    if "regulargrid" in model.lower():
        filetype = "regulargrid"
    elif "2ds" in model.lower():
        filetype = "2ds"
    else:
        filetype = "fields"

    timings = ['forecast', 'hindcast']

    new_sources = []
    for timing in timings:

        catloc = cat[timing].metadata['catloc']
        catrefs = find_catrefs(catloc)

        # forecast: don't need to check for consecutive dates bc files are by day
        # just find first file from earliest catref and last file from last catref
        if 'stale' in cat[timing].metadata:
            stale = pd.Timedelta(cat[timing].metadata['stale'])
        else:
            stale = pd.Timedelta('1 second')
        if 'time_last_checked' in cat[timing].metadata:
            time_last_checked = cat[timing].metadata['time_last_checked']
        else:
            time_last_checked = pd.Timestamp.now()
        dt = pd.Timestamp.now() - pd.Timestamp(time_last_checked)

        if timing == 'forecast' and (dt > stale or override):

            # find start_datetime
            filelocs = find_filelocs(catrefs[-1], catloc, filetype=filetype)
            start_datetime = get_dates_from_ofs(filelocs, filetype, 'n', 'first')

            # find end_datetime
            filelocs = find_filelocs(catrefs[0], catloc, filetype=filetype)
            end_datetime = get_dates_from_ofs(filelocs, filetype, 'f', 'last')

            ran_forecast = True
            # time_last_checked = pd.Timestamp.now()

        elif timing == 'hindcast' and (dt > stale or override):

            # Find start_datetime by checking catrefs from the old end [-1]
            for catref in catrefs[::-1]:
                filelocs = find_filelocs(catref, catloc, filetype=filetype)
                if len(filelocs) == 0:
                    continue

                # import pdb; pdb.set_trace()
                # determine unique dates
                dates = sorted(list(set([pd.Timestamp(fileloc.split('/')[-1].split('.')[4]) for fileloc in filelocs])))

                # determine consecutive dates
                dates = [dates[i] for i in range(len(dates) - 2) if (dates[i] + pd.Timedelta('1 day') in dates) and (dates[i] + pd.Timedelta('2 days') in dates)]
                if len(dates) > 0:
                    # keep filelocs if their date matches one in dates
                    # filelocs that don't exceed date range found
                    filelocs = [fileloc for fileloc in filelocs if dates[0] <= pd.Timestamp(fileloc.split('/')[-1].split('.')[4]) <= dates[-1]]
                    start_datetime = get_dates_from_ofs(filelocs, filetype, 'n', 'first')
                    break

            # find end_datetime, no need to search through files on this end of time
            filelocs = find_filelocs(catrefs[0], catloc, filetype=filetype)
            end_datetime = get_dates_from_ofs(filelocs, filetype, 'n', 'last')

            ran_hindcast = True
            # time_last_checked = pd.Timestamp.now()
        else:
            start_datetime = cat[timing].metadata['start_datetime']
            end_datetime = cat[timing].metadata['end_datetime']

        # stale parameter: 4 hours for forecast, 1 day for hindcast
        if timing == 'forecast':
            stale = '4 hours'
        elif timing == 'hindcast':
            stale = '1 day'

        metadata = {
            "model": model,
            "timing": timing,
            "filetype": filetype,
            "time_last_checked": time_last_checked,
            "stale": stale,
            "start_datetime": str(start_datetime),
            "end_datetime": str(end_datetime),
        }
        cat[timing].metadata.update(metadata)
        new_sources.append(cat[timing])

    if not (ran_forecast or ran_hindcast):
        return cat
    else:

        cat_metadata = cat.metadata
        metadata = {
            "catalog_path": mc.CATALOG_PATH,
            # "source_catalog_name": mc.SOURCE_CATALOG_NAME,
            "filetype": new_sources[0].metadata["filetype"]
        }
        cat_metadata.update(metadata)

        new_user_cat = mc.make_catalog(
            new_sources,
            f"{model.upper()}",
            f"Model {model} with availability included.",
            cat_metadata,
            [source._entry._driver for source in new_sources],
            cat_path=mc.CATALOG_PATH_UPDATED,
        )
        return new_user_cat
