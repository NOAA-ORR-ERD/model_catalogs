"""
Utilities to help with catalogs.
"""

import fnmatch
import pathlib
import re

import cf_xarray  # noqa
import intake
import numpy as np
import pandas as pd
import yaml

from siphon.catalog import TDSCatalog

import model_catalogs as mc


def astype(value, type_):
    """Return string or list as list"""
    if not isinstance(value, type_):
        if type_ == list and isinstance(value, (str, pathlib.PosixPath)):
            return [value]
        return type_(value)
    return value


def get_fresh_parameter(filename):
    """Get freshness parameter, based on the filename.

    A freshness parameter is stored in `__init__` for required scenarios which is looked up using the
    logic in this function, based on the filename.

    Parameters
    ----------
    filename : Path
        Filename to determine freshness

    Returns
    -------
    mu, a pandas Timedelta-interpretable string describing the amount of time that filename should be
    considered fresh before needing to be recalculated.
    """

    # a start or end datetime file
    if filename.parent == mc.CACHE_PATH_AVAILABILITY:
        timing = filename.name.split("_")[1]
        if "start" in filename.name:
            mu = mc.FRESH[timing]["start"]
        elif "end" in filename.name:
            mu = mc.FRESH[timing]["end"]
        elif "catrefs" in filename.name:
            mu = mc.FRESH[timing]["catrefs"]
    # a file of file locs for aggregation
    elif filename.parent == mc.CACHE_PATH_FILE_LOCS:
        timing = filename.name.split("_")[1]
        mu = mc.FRESH[timing]["file_locs"]
    # a compiled catalog file
    elif filename.parent == mc.CACHE_PATH_COMPILED:
        mu = mc.FRESH["compiled"]

    return mu


def is_fresh(filename):
    """Check if file called filename is fresh.

    If filename doesn't exist, return False.

    Parameters
    ----------
    filename : Path
        Filename to determine freshness

    Returns
    -------
    Boolean. True if fresh and False if not or if filename is not found.
    """

    now = pd.Timestamp.today(tz="UTC")
    try:
        filetime = pd.Timestamp(filename.stat().st_mtime_ns).tz_localize("UTC")

        mu = get_fresh_parameter(filename)

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
        Date of day to find model output files for. Doesn't pay attention to hours/minutes seconds.
    strings: list
        List of strings to be filtered. Expected to be file locations from a thredds catalog.
    filetype: str
        Which filetype to use. Every NOAA OFS model has "fields" available, but some have "regulargrid"
        or "2ds" also. This availability information is in the source catalog for the model under
        `filetypes` metadata.
    is_forecast: bool, optional
        If True, then date is the last day of the time period being sought and the forecast files should
        be brought in along with the nowcast files, to get the model output the length of the forecast
        out in time. The forecast files brought in will have the latest timing cycle of the day that is
        available. If False, all nowcast files (for all timing cycles) are brought in.
    pattern: str, optional
        If a model file pattern doesn't match that assumed in this code, input one that will work.
        Currently only NYOFS doesn't match but the pattern is built into the catalog file.

    Returns
    -------
    List of URLs for where to find all of the model output files that match the keyword arguments.
    """

    date = astype(date, pd.Timestamp)

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
    if len(cat_ref_test) > 1:
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

    # OUTPUT
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
        some have "regulargrid"or "2ds" also (listed in separate catalogs in the
        model name).

    Returns
    -------
    Locations of files found from catloc to hierarchical location described by
    catref.
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


def get_dates_from_ofs(filelocs, filetype, norf, firstorlast):
    """Return either start or end datetime from list of filenames.

    This looks at the actual nowcast and forecast file cycle times to understand the earliest and last
    model times, as opposed to just the date in the file name.

    Parameters
    ----------
    filelocs: list
        Locations of files found from catloc to hierarchical location described by
        catref.
    filetype: str
        Which filetype to use. Every NOAA OFS model has "fields" available, but
        some have "regulargrid"or "2ds" also (listed in separate catalogs in the
        model name).
    norf: str
        "n" or "f" for "nowcast" or "forecast", for OFS files.
    firstorlast: str
        Whether to get the "first" or "last" entry of the filelocs, which will
        translate to the index to use.
    """

    pattern = f"*{filetype}*.{norf}*.????????.t??z.*"
    # pattern = date.strftime(f"*{filetype}*.n*.%Y%m%d.t??z.*")
    if firstorlast == "first":
        ind = 0
    elif firstorlast == "last":
        ind = -1
    fileloc = sorted(fnmatch.filter(filelocs, pattern))[ind]
    filename = fileloc.split("/")[-1]
    date = pd.Timestamp(filename.split(".")[4])
    regex = re.compile(".t[0-9]{2}z.")
    cycle = [substr[2:4] for substr in regex.findall("".join(filename))][0]
    regex = re.compile(f".{norf}" + "[0-9]{3}.")
    try:
        filetime = [substr[2:5] for substr in regex.findall("".join(filename))][0]
    except IndexError:  # NYOFS uses this
        filetime = 0
    # in UTC
    datetime = date + pd.Timedelta(f"{cycle} hours") + pd.Timedelta(f"{filetime} hours")

    return datetime


def calculate_boundaries(file_locs=None, save_files=True, return_boundaries=False):
    """Calculate boundary information for all models.

    This loops over all catalog files available in mc.CAT_PATH_ORIG, tries first with forecast source and
    then with nowcast source if necessary to access the example model output files and calculate the
    bounding box and numerical domain boundary. The numerical domain boundary is calculated using
    `alpha_shape` with previously-chosen parameters stored in the original model catalog files. The
    bounding box and boundary string representation (as WKT) are then saved to files.

    The files that are saved by running this function have been previously saved into the repository, so
    this function should only be run if you suspect that a model domain has changed.

    Parameters
    ----------
    file_locs : Path, list of Paths, optional
        List of Path objects for model catalog files to read from. If not input, will use all catalog
        files available at mc.CAT_PATH_ORIG.glob("*.yaml").
    save_files : boolean, optional
        Whether to save files or not. Defaults to True. Saves to mc.CAT_PATH_BOUNDARY / cat_loc.name.
    return_boundaries : boolean, optional
        Whether to return boundaries information from this call. Defaults to False.

    Examples
    --------
    Calculate boundary information for all available models:
    >>> mc.calculate_boundaries()

    Calculate boundary information for CBOFS:
    >>> mc.calculate_boundaries([mc.CAT_PATH_ORIG / "cbofs.yaml"])
    """

    if file_locs is None:
        file_locs = mc.CAT_PATH_ORIG.glob("*.yaml")
    else:
        file_locs = mc.astype(file_locs, list)

    # loop over all orig catalogs
    boundaries = {}
    for cat_loc in file_locs:

        # open model catalog
        cat_orig = intake.open_catalog(cat_loc)

        # try with forecast but if it doesn't work, use nowcast
        # this avoids problematic NOAA OFS aggregations when they are broken
        try:
            timing = "forecast"
            source_orig = cat_orig[timing]
            source_transform = mc.transform_source(source_orig)

            # need to make catalog to transfer information properly from
            # source_orig to source_transform
            cat_transform = mc.make_catalog(
                source_transform,
                full_cat_name=cat_orig.name,  # model name
                full_cat_description=cat_orig.description,
                full_cat_metadata=cat_orig.metadata,
                cat_driver=mc.process.DatasetTransform,
                cat_path=None,
                save_catalog=False,
            )

            # read in model output
            ds = cat_transform[timing].to_dask()

        except OSError:
            timing = "nowcast"
            source_orig = cat_orig[timing]
            source_transform = mc.transform_source(source_orig)

            # need to make catalog to transfer information properly from
            # source_orig to source_transform
            cat_transform = mc.make_catalog(
                source_transform,
                full_cat_name=cat_orig.name,  # model name
                full_cat_description=cat_orig.description,
                full_cat_metadata=cat_orig.metadata,
                cat_driver=mc.process.DatasetTransform,
                cat_path=None,
                save_catalog=False,
            )

            # read in model output
            ds = cat_transform[timing].to_dask()

        # find boundary information for model
        if "alpha_shape" in cat_orig.metadata:
            dd, alpha = cat_orig.metadata["alpha_shape"]
        else:
            dd, alpha = None, None
        lonkey, latkey, bbox, wkt = mc.find_bbox(ds, dd=dd, alpha=alpha)

        ds.close()

        # save boundary info to file
        if save_files:
            with open(mc.FILE_PATH_BOUNDARIES(cat_loc.name), "w") as outfile:
                yaml.dump({"bbox": bbox, "wkt": wkt}, outfile, default_flow_style=False)

        if return_boundaries:
            boundaries[cat_loc.stem] = {"bbox": bbox, "wkt": wkt}

    if return_boundaries:
        return boundaries
