"""
Utilities to help with catalogs.
"""

import fnmatch
import pathlib
import re

import cf_xarray  # noqa
import numpy as np
import pandas as pd
import requests
import yaml

from intake.catalog import Catalog
from siphon.catalog import TDSCatalog

import model_catalogs as mc


def astype(value, type_):
    """Return `value` as type `type_`.

    Particularly made to work correctly for returning string, `PosixPath`, or `Timestamp` as list.
    """
    if not isinstance(value, type_):
        if type_ == list and isinstance(
            value, (str, pathlib.PurePath, pd.Timestamp, Catalog)
        ):
            return [value]
        return type_(value)
    return value


def status(urlpath, suffix=".das"):
    """Check status of server for urlpath.

    Parameters
    ----------
    urlpath : str
        Path to file location to check.

    Returns
    -------
    bool
        If True, server was reachable.
    """

    resp = requests.get(urlpath + suffix)
    if resp.status_code != 200:
        status = False
    else:
        status = True
    return status


def file2dt(filename):
    """Return Timestamp of NOAA OFS filename

    ...without reading in the filename to xarray. See `docs <https://model-catalogs.readthedocs.io/en/latest/aggregations.html>`_ for details on the formula. Most NOAA OFS models have 1 timestep per file, but NYOFS has 6.

    Parameters
    ----------
    filename : str
        Filename for which to decipher datetime. Can be full path or just file name.

    Returns
    -------
    Timestamp
        pandas Timestamp of the time(s) in the file.

    Examples
    --------
    >>> url = 'https://www.ncei.noaa.gov/thredds/dodsC/model-cbofs-files/2022/07/nos.cbofs.fields.n001.20220701.t00z.nc'
    >>> mc.filename2datetime(url)
    Timestamp('2022-06-30 19:00:00')
    """

    # strip off path if present since can mess up the matching
    filename = filename.split("/")[-1]

    # read in date from filename
    date = pd.to_datetime(filename, format="%Y%m%d", exact=False)  # has time 00:00

    # pull timing cycle from filename
    regex = re.compile(".t[0-9]{2}z.")
    cycle = int(regex.findall(filename)[0][2:4])

    # NYOFS: multiple times per file
    if fnmatch.fnmatch(filename, "*.nowcast.*"):

        date = [date + pd.Timedelta(f"{cycle - dt} hours") for dt in range(6)[::-1]]

    # NYOFS: multiple times per file
    elif fnmatch.fnmatch(filename, "*.forecast.*"):

        # models all have different forecast lengths! Though only nyofs is left in this category
        if "nyofs" in filename:
            nfiles = 54

        date = [date + pd.Timedelta(f"{cycle + 1 + dt} hours") for dt in range(nfiles)]

    # Main style of NOAA OFS files, 1 file per time step
    elif fnmatch.fnmatch(filename, "*.n???.*") or fnmatch.fnmatch(filename, "*.f???.*"):

        # pull hours from filename
        regex = re.compile(".[n,f][0-9]{3}.")
        hour = int(regex.findall(filename)[0][2:-1])

        # calculate hours
        dt = cycle + hour

        # if nowcast file, subtract 6 hours
        if fnmatch.fnmatch(filename, "*.n???.*"):
            dt -= 6

        # construct datetime. dt might be negative.
        date += pd.Timedelta(f"{dt} hours")

    return date


def get_fresh_parameter(filename, source):
    """Get freshness parameter, based on the filename.

    A freshness parameter is stored in ``__init__`` for required scenarios which is looked up using the logic in this function, based on the filename. The source is checked for most types of actions for an overriding freshness parameter value, otherwise the default is used.

    Parameters
    ----------
    filename : Path
        Filename to determine freshness.
    source : Intake Source
        Source from which to check for an overriding freshness parameter. Is not used for "compiled" catalog files.

    Returns
    -------
    str
        mu, a pandas Timedelta-interpretable string describing the amount of time that filename should be considered fresh before needing to be recalculated.
    """

    # a start or end datetime file
    if filename.parent == mc.CACHE_PATH_AVAILABILITY:

        if source is None:
            raise ValueError("source cannot be None for this freshness calculation.")

        # which type are we after
        if "start" in filename.name:
            parameter = "start"
        elif "end" in filename.name:
            parameter = "end"
        elif "catrefs" in filename.name:
            parameter = "catrefs"

        # check for overriding freshness parameter in source metadata
        if "freshness" in source.metadata and parameter in source.metadata["freshness"]:
            mu = source.metadata["freshness"][parameter]
        else:
            mu = mc.FRESH[parameter]
            # a file of file locs for aggregation
    elif filename.parent == mc.CACHE_PATH_FILE_LOCS:

        if source is None:
            raise ValueError("source cannot be None for this freshness calculation.")

        parameter = "file_locs"
        # check for overriding freshness parameter in source metadata
        if "freshness" in source.metadata and parameter in source.metadata["freshness"]:
            mu = source.metadata["freshness"][parameter]
        else:
            mu = mc.FRESH[parameter]

    # a compiled catalog file
    elif filename.parent == mc.CACHE_PATH_COMPILED:
        mu = mc.FRESH["compiled"]

    return mu


def is_fresh(filename, source=None):
    """Check if file called filename is fresh.

    If filename doesn't exist, return False.

    Parameters
    ----------
    filename : Path
        Filename to determine freshness
    source : Intake Source
        Source from which to check for an overriding freshness parameter. Is not used for "compiled" catalog files.

    Returns
    -------
    Boolean
        True if fresh and False if not or if filename is not found.
    """

    now = pd.Timestamp.today(tz="UTC")
    try:
        filetime = pd.Timestamp(filename.stat().st_mtime_ns).tz_localize("UTC")

        mu = get_fresh_parameter(filename, source=source)

        return now - filetime < pd.Timedelta(mu)

    except FileNotFoundError:
        return False


def find_bbox(ds, dd=None, alpha=None):
    """Determine bounds and boundary of model.

    Parameters
    ----------
    ds: Dataset
        xarray Dataset containing model output.
    dd: int, optional
        Number to decimate model output lon/lat, as a stride.
    alpha: float, optional
        Number for alphashape to determine what counts as the convex hull. Larger number is more detailed, 1 is a good starting point.

    Returns
    -------
    List
        Contains the name of the longitude and latitude variables for ds, geographic bounding box of model output (`[min_lon, min_lat, max_lon, max_lat]`), low res and high res wkt representation of model boundary.
    """

    import shapely.geometry

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
    # GFS, RTOFS, HYCOM
    if (lon.ndim == 1) and ("nele" not in ds.dims) and not hasmask:
        nlon, nlat = ds["lon"].size, ds["lat"].size
        lonb = np.concatenate(([lon[0]] * nlat, lon[:], [lon[-1]] * nlat, lon[::-1]))
        latb = np.concatenate((lat[:], [lat[-1]] * nlon, lat[::-1], [lat[0]] * nlon))
        # boundary = np.vstack((lonb, latb)).T
        p = shapely.geometry.Polygon(zip(lonb, latb))
        p0 = p.simplify(1)
        # Now using the more simplified version because all of these models are boxes
        p1 = p0

    elif hasmask or ("nele" in ds.dims):  # unstructured

        assertion = (
            "dd and alpha need to be defined in the catalog metadata for this model."
        )
        assert dd is not None and alpha is not None, assertion

        # need to calculate concave hull or alphashape of grid
        import alphashape

        # downsample a bit to save time, still should clearly see shape of domain
        lon, lat = lon[::dd], lat[::dd]
        pts = list(zip(lon, lat))

        # low res, same as convex hull
        p0 = alphashape.alphashape(pts, 0.0)
        p1 = alphashape.alphashape(pts, alpha)

    # useful things to look at: p.wkt  #shapely.geometry.mapping(p)
    return lonkey, latkey, list(p0.bounds), p1.wkt
    # return lonkey, latkey, list(p0.bounds), p0.wkt, p1.wkt


def filedates2df(filelocs):
    """Set up dataframe of datetimes to filenames.

    Parameters
    ----------
    filelocs : list of str
        File locations.

    Returns
    -------
    DataFrame
        Contains the index datetimes corresponding to file locations (column 'filenames').
    """

    # 1+ number of times possible from mc.file2dt, need the number of filenames to match
    filedates, filenames = [], []
    for fname in filelocs:
        filedate = mc.astype(mc.file2dt(fname), list)
        filenames.extend([fname] * len(filedate))
        filedates.extend(filedate)

    # Make dataframe
    df = pd.DataFrame(index=filedates, data={"filenames": filenames})

    # Sort resulting df by filenames and then by index which is the datetime of each file
    df = df.sort_values(axis="index", by="filenames").sort_index()

    # remove rows if index is duplicated, sorting makes it so nowcast files are kept
    df = df[~df.index.duplicated(keep="first")]

    return df


def agg_for_date(date, strings, filetype, is_forecast=False, pattern=None):
    """Select NOAA OFS-style nowcast/forecast files for aggregation.

    This function finds the files whose path includes the given date, regardless of times which might change the date forward or backward.

    Parameters
    ----------
    date: str of datetime, pd.Timestamp
        Date of day to find model output files for. Doesn't pay attention to hours/minutes seconds.
    strings: list
        List of strings to be filtered. Expected to be file locations from a thredds catalog.
    filetype: str
        Which filetype to use. Every NOAA OFS model has "fields" available, but some have "regulargrid" or "2ds" also. This availability information is in the catalog metadata for the model under `filetypes` metadata.
    is_forecast: bool, optional
        If True, then date is the last day of the time period being sought and the forecast files should be brought in along with the nowcast files, to get the model output the length of the forecast out in time. The forecast files brought in will have the latest timing cycle of the day that is available. If False, all nowcast files (for all timing cycles) are brought in.
    pattern: str, optional
        If a model file pattern doesn't match that assumed in this code, input one that will work. Currently only NYOFS doesn't match but the pattern is built into the catalog file.

    Returns
    -------
    List
        Contains URLs for where to find all of the model output files that match the keyword arguments. List is not sorted correctly for times (this happens later).
    """

    date = astype(date, pd.Timestamp)

    if pattern is None:
        pattern = date.strftime(f"*{filetype}*.n*.%Y%m%d.t??z.*")
    else:
        pattern = eval(f"f'{pattern}'")

    # if using forecast, find nowcast and forecast files for the latest full timing cycle
    if is_forecast:

        import re

        # Find the most recent, complete timing cycle for the forecast day
        regex = re.compile(".t[0-9]{2}z.")
        # substrings: list of repeated strings of hours, e.g. ['12', '06', '00', '12', ...]
        subs = [substr[2:4] for substr in regex.findall("".join(strings))]
        # unique str times, in increasing order, e.g. ['00', '06', '12']
        times = sorted(list(set(subs)))
        # choose the timing cycle that is latest but also has the most times available
        # sometimes the forecast files aren't available yet so don't want to use that time
        cycle = sorted(times, key=subs.count)[-1]  # noqa: F841

        # find all nowcast files with only timing "cycle"
        # replace '.t??z.' in pattern with '.t{cycle}z.' to get the latest timing cycle only
        pattern1 = pattern.replace(".t??z.", ".t{cycle}z.").replace(".n*.", ".[n,f]*.")
        pattern1 = eval(f"f'{pattern1}'")  # use `eval` to sub in value of `cycle`
        # sort nowcast files alone to get correct time order
        fnames_cycle = fnmatch.filter(strings, pattern1)

        # Include the nowcast files between the start of the day and when the time series
        # represented in fnames begins
        # fnames_now = find_nowcast_cycles(strings, pattern)
        fnames_nowcast = fnmatch.filter(strings, pattern)

        fnames = fnames_cycle + fnames_nowcast  # combine

        if len(fnames) == 0:
            raise ValueError(
                f"Error finding filenames. Filenames found so far: {fnames}. "
                "Maybe you have the wrong source for the days requested."
            )

    # if not using forecast, find all nowcast files matching pattern
    else:
        fnames = fnmatch.filter(strings, pattern)

    return fnames


def find_catrefs(catloc):
    """Find hierarchy of catalog references for thredds catalog.

    Parameters
    ----------
    catloc: str
        Search in thredds catalog structure from base catalog, catloc.

    Returns
    -------
    list
        Contains tuples containing the hierarchy of directories in the thredds catalog structure to get to where the datafiles start.
    """

    # 0th level catalog
    cat = TDSCatalog(catloc)
    catrefs_to_check = list(cat.catalog_refs)
    # only keep numerical directories
    catrefs_to_check = [catref for catref in catrefs_to_check if catref.isnumeric()]

    # 1st level catalog
    catrefs_to_check1 = [
        cat.catalog_refs[catref].follow().catalog_refs for catref in catrefs_to_check
    ]

    # Combine the catalog references together
    catrefs = [
        (catref0, catref1)
        for catref0, catrefs1 in zip(catrefs_to_check, catrefs_to_check1)
        for catref1 in catrefs1
    ]

    # Check first one to see if there are more catalog references or not
    cat_ref_test = (
        cat.catalog_refs[catrefs[0][0]]
        .follow()
        .catalog_refs[catrefs[0][1]]
        .follow()
        .catalog_refs
    )
    # If there are more catalog references, run another level of catalog and combine, ## 2
    if len(cat_ref_test) > 0:
        catrefs2 = [
            cat.catalog_refs[catref[0]]
            .follow()
            .catalog_refs[catref[1]]
            .follow()
            .catalog_refs
            for catref in catrefs
        ]
        catrefs = [
            (catref[0], catref[1], catref22)
            for catref, catref2 in zip(catrefs, catrefs2)
            for catref22 in catref2
        ]

    return catrefs


def find_filelocs(catref, catloc, filetype="fields"):
    """Find thredds file locations.

    Parameters
    ----------
    catref: tuple
        2 or 3 labels describing the directories from catlog to get the data
        locations.
    catloc: str
        Base thredds catalog location.
    filetype: str
        Which filetype to use. Every NOAA OFS model has "fields" available, but
        some have "regulargrid" or "2ds" also (listed in separate catalogs in the
        model name).

    Returns
    -------
    list
        Locations of files found from catloc to hierarchical location described by catref.
    """

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

    for dataset in datasets:
        if (
            "stations" not in dataset
            and "vibrioprob" not in dataset
            and filetype in dataset
        ):

            url = last_cat.datasets[dataset].access_urls["OPENDAP"]
            filelocs.append(url)
    return filelocs


def calculate_boundaries(cats, save_files=True, return_boundaries=False):
    """Calculate boundary information for all models.

    This loops over all input catalogs and will try with multiple model_source if necessary (in case servers aren't working) to access the example model output files and calculate the bounding box and numerical domain boundary. The numerical domain boundary is calculated using `alpha_shape` with previously-chosen parameters stored in the original model catalog files. The bounding box and boundary string representation (as WKT) are then saved to files.

    The files are saved the first time you run this function, so this function should only be rerun if you suspect that a model domain has changed or you have a new model catalog.

    Parameters
    ----------
    cats : Catalog, list of Catalogs
        The Catalog or Catalogs for which to find boundaries.
    save_files : boolean, optional
        Whether to save files or not. Defaults to True. Saves to ``mc.FILE_PATH_BOUNDARIES(cat_loc.name)``.
    return_boundaries : boolean, optional
        Whether to return boundaries information from this call. Defaults to False.

    Examples
    --------

    Calculate boundary information for CBOFS:

    >>> import model_catalogs as mc
    >>> main_cat = mc.setup()
    >>> mc.calculate_boundaries(main_cat["CBOFS"])
    """

    # loop over all orig catalogs
    boundaries = {}
    for cat in mc.astype(cats, list):

        # loop over available sources and use the first that works
        for model_source in list(cat):
            source_orig = cat[model_source]
            source_transform = mc.transform_source(source_orig)

            # need to make catalog to transfer information properly from
            # source_orig to source_transform
            cat_transform = mc.make_catalog(
                source_transform,
                full_cat_name=cat.name,  # model name
                full_cat_description=cat.description,
                full_cat_metadata=cat.metadata,
                cat_driver=mc.process.DatasetTransform,
                cat_path=None,
                save_catalog=False,
            )

            if cat_transform[model_source].status:

                # read in model output
                ds = cat_transform[model_source].to_dask()
                break

        # find boundary information for model
        if "alpha_shape" in cat.metadata:
            dd, alpha = cat.metadata["alpha_shape"]
        else:
            dd, alpha = None, None
        lonkey, latkey, bbox, wkt = mc.find_bbox(ds, dd=dd, alpha=alpha)

        ds.close()

        # save boundary info to file
        if save_files:
            with open(mc.FILE_PATH_BOUNDARIES(cat.name.lower()), "w") as outfile:
                yaml.dump({"bbox": bbox, "wkt": wkt}, outfile, default_flow_style=False)

        if return_boundaries:
            boundaries[cat.name] = {"bbox": bbox, "wkt": wkt}

    if return_boundaries:
        return boundaries
