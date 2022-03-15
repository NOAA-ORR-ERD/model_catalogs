"""
Everything dealing with the catalogs.
"""

import fnmatch
import os

from glob import glob

import cf_xarray  # noqa
import intake
import intake.source.derived
import numpy as np
import pandas as pd
import shapely.geometry

from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
from siphon.catalog import TDSCatalog

import model_catalogs as mc
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()


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


def make_catalog(
    cats,
    full_cat_name,
    full_cat_description,
    full_cat_metadata,
    cat_driver,
    cat_path=None,
):
    """Construct single catalog from multiple catalogs or sources.

    Parameters
    ----------
    cats: list
       List of Intake catalog or source objects that will be combined into a single catalog.
    full_cat_name: str
       Name of overall catalog.
    full_cat_descrption: str
       Description of overall catalog.
    full_cat_metadata: dict
       Dictionary of metadata for overall catalog.
    cat_driver: str or Intake object or list
       Driver to apply to all catalog entries. For example:
       * intake.catalog.local.YAMLFileCatalog
       * 'opendap'
       If list, must be same length as cats and contains drivers that
       correspond to cats.
    cat_path: str, optional
       Path with catalog name to use for saving catalog. With or without yaml suffix. If not provided,
       will use `full_cat_name`.

    Returns
    -------

    Intake catalog.

    Examples
    --------

    Make catalog:

    >>> make_catalog([list of catalogs], 'catalog name', 'catalog desc', {}, 'opendap')
    """

    if cat_path is None:
        cat_path = full_cat_name
    if ("yaml" not in cat_path) and ("yml" not in cat_path):
        cat_path = f"{cat_path}.yaml"

    if not isinstance(cats, list):
        cats = [cats]
    if not isinstance(cat_driver, list):
        cat_driver = [cat_driver] * len(cats)
    assert len(cat_driver) == len(
        cats
    ), "Number of catalogs and catalog drivers must match"

    # create dictionary of catalog entries
    entries = {
        cat.name: LocalCatalogEntry(
            cat.name,
            description=cat.description,
            driver=catd,
            args=cat._yaml()["sources"][cat.name]["args"],
            metadata=cat._yaml()["sources"][cat.name]["metadata"],
        )
        for cat, catd in zip(cats, cat_driver)
    }

    # create catalog
    cat = Catalog.from_dict(
        entries,
        name=full_cat_name,
        description=full_cat_description,
        metadata=full_cat_metadata,
    )

    # save catalog
    cat.save(cat_path)

    return cat


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


def agg_for_date_old(
    date, catloc, filetype, treat_last_day_as_forecast=False, pattern=None
):
    """Aggregate NOAA OFS-style nowcast/forecast files.

    PROBABLY LEGACY

    Parameters
    ----------
    date: str of datetime, pd.Timestamp
        Date of day to find model output files for. Doesn't pay attention to hours/minutes/seconds.
    catloc: str
        URL of thredds base catalog. Should be a pattern like the following in which the date will be
        filled in as a f-string variable. Can be xml or html:
        https://opendap.co-ops.nos.noaa.gov/thredds/catalog/NOAA/CBOFS/MODELS/{date.year}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/catalog.xml
    filetype: str
        Which filetype to use. Every NOAA OFS model has "fields" available, but some have "regulargrid"
        or "2ds" also. This availability information is in the source catalog for the model under
        `filetypes` metadata.
    treat_last_day_as_forecast: bool, optional
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

    # f format catloc name with input date
    catloc = eval(f"f'{catloc}'")
    cat = TDSCatalog(catloc)

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
    fnames = fnmatch.filter(cat.datasets, pattern)

    if treat_last_day_as_forecast:

        import re

        regex = re.compile(".t[0-9]{2}z.")
        cycle = sorted(  # noqa: F841
            list(set([substr[2:4] for substr in regex.findall("".join(cat.datasets))]))
        )[
            -1
        ]  # noqa: E501

        # cycle = sorted(list(set([fname[ fname.find(start:='.t') + len(start):fname.find('z.')] for fname in fnames])))[-1]  # noqa: E501
        # import pdb; pdb.set_trace()
        # pattern1 = f'*{filetype}*.t{cycle}z.*'

        # replace '.t??z.' in pattern with '.t{cycle}z.' and replace '.n*.' with '.*.'
        pattern1 = pattern.replace(".t??z.", ".t{cycle}z.").replace(".n*.", ".*.")
        pattern1 = eval(f"f'{pattern1}'")
        fnames = fnmatch.filter(cat.datasets, pattern1)

    filelocs = [cat.datasets.get(fname).access_urls["OPENDAP"] for fname in fnames]

    return filelocs


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


def find_availability(cat, filetype='fields'):
    """Find availability for model.
    MORE
    """

    catloc = cat.metadata['catloc']
    # catloc = cats.source_cat[model]['forecast'].metadata['catloc']
    catrefs = find_catrefs(catloc)

    filelocs3 = Parallel(n_jobs=num_cores)(
        # print(catref)
        delayed(find_filelocs)(catref, catloc, filetype=filetype)
        for catref in catrefs
    )

    # expand into single list from nested list
    filelocs = []
    for fileloc in filelocs3:
        filelocs.extend(fileloc)

    filelocs = sorted(filelocs)

    # determine unique dates
    dates = sorted(list(set([pd.Timestamp(fileloc.split('/')[-1].split('.')[4]) for fileloc in filelocs])))

    # determine consecutive dates
    dates = [dates[i] for i in range(len(dates)-2) if (dates[i] + pd.Timedelta('1 day') in dates) and (dates[i] + pd.Timedelta('2 days') in dates)]

    # keep filelocs if their date matches one in dates
    # filelocs that don't exceed date range found
    filelocs = [fileloc for fileloc in filelocs if dates[0] <= pd.Timestamp(fileloc.split('/')[-1].split('.')[4]) <= dates[-1]]

    # to get range of available model output, can use dates:
    return dates[0], dates[-1], filelocs


def setup_source_catalog():
    """Setup source catalog for models.

    Returns
    -------
    Source catalog `source_cat`.
    """

    cat_source_description = "Source catalog for models."

    # find most recent set of source_catalogs
    if not os.path.exists(mc.SOURCE_CATALOG_DIR):
        print(
            '"complete" model source files are not yet available. Run `model_catalogs.complete_source_catalog()` to create this directory.'  # noqa: E501
        )
        cat_dir = mc.SOURCE_CATALOG_DIR_ORIG
    else:
        cat_dir = mc.SOURCE_CATALOG_DIR

    # open catalogs
    cat_locs = glob(f"{cat_dir}/*.yaml")
    cats = [intake.open_catalog(cat_loc) for cat_loc in cat_locs]

    metadata = {"source_catalog_dir": cat_dir}

    return make_catalog(
        cats,
        mc.SOURCE_CATALOG_NAME,
        cat_source_description,
        metadata,
        intake.catalog.local.YAMLFileCatalog,
    )


def complete_source_catalog(source_cat):
    """Update model source files in 'source_catalogs/orig'.

    This will add outer boundary files for the model domains and create
    "complete" directory of model files that mirror those in "orig".

    Returns
    -------
    Nothing, but resaves all model source catalogs into
    f"{self.cat_source_base}/complete" with domain boundaries added.

    Examples
    --------

    Add boundary calculations to source model files:
    >>> cats.update_source_files()
    """

    # if os.path.exists(f"{self.cat_source_base}/complete"):
    #     print('"complete" source model catalog files already exist.')
    #     return

    # # update default source_catalog model file location to complete
    # # instead of orig
    # self.source_catalog_dir_old = self.source_catalog_dir
    # self.source_catalog_dir = f"{self.cat_source_base}/complete"

    models = list(source_cat)
    timing = "forecast"

    for model in models:

        # read in model output
        ds = source_cat[model][timing].to_dask()

        # find metadata
        # select lon/lat for use. There may be more than one and we also want the name.
        if "alpha_shape" in source_cat[model].metadata:
            dd, alpha = source_cat[model].metadata["alpha_shape"]
        else:
            dd, alpha = None, None
        lonkey, latkey, bbox, wkt = find_bbox(ds, dd=dd, alpha=alpha)
        # lonkey, latkey, bbox, wkt_low, wkt_high = find_bbox(ds, dd=dd, alpha=alpha)

        # metadata for overall source_id0
        metadata0 = {
            "geospatial_bounds": wkt,
            "bounding_box": bbox,
        }

        # add Dataset metadata to specific source metadata
        # change metadata attributes to strings so catalog doesn't barf on them
        for attr in ds.attrs:
            source_cat[model][timing].metadata[attr] = str(ds.attrs[attr])

        # add 0th level metadata to 0th level model entry
        source_cat[model].metadata.update(metadata0)

        timings = list(source_cat[model])
        sources = [source_cat[model][timing] for timing in timings]
        make_catalog(
            sources,
            model,
            source_cat[model].description,
            source_cat[model].metadata,
            "opendap",
            f"{mc.SOURCE_CATALOG_DIR}/{model.lower()}.yaml",
        )

    return setup_source_catalog()


def _update_catalog(
    model,
    timing,
    # start_date=None,
    # end_date=None,
    # filetype="fields",
    # treat_last_day_as_forecast=False,
):
    """Make a set of user catalog sources.

    This is meant to be called by `setup_cat()`, not directly by user.

    Parameters
    ----------
    model: str
        Name of model, e.g., CBOFS
    timing: str
        Which timing to use. Normally "forecast", "nowcast", or "hindcast", if
        available for model, but could have different names and/or options.
        Find model options available with `list(self.source_cat[model])` or
        `list(self.updated_cat[model])`.
    start_date, end_date: datetime-interpretable str or pd.Timestamp, optional
        If model has an aggregated link for timing, start_date and end_date
        are not used. Otherwise they should be input. Only year-month-day
        will be used in date. end_date is inclusive.
    treat_last_day_as_forecast: bool, optional
        CHANGE BEHAVIOR
        If True, then date is the last day of the time period being sought and the forecast files
        should be brought in along with the nowcast files, to get the model output the length of the
        forecast out in time. The forecast files brought in will have the latest timing cycle of the
        day that is available. If False, all nowcast files (for all timing cycles) are brought in.

    Returns
    -------
    Source associated with the catalog entry.
    """

    model = model.upper()

    # # save source to already-made user catalog loc
    # self.user_catalog_dir = self.cat_user_base
    # self.user_catalog_name = (
    #     f"{self.user_catalog_dir}/{self.time_ref.isoformat().replace(':','_')}.yaml"
    # )

    # if not isinstance(start_date, pd.Timestamp):
    #     start_date = pd.Timestamp(start_date)
    # if not isinstance(end_date, pd.Timestamp):
    #     end_date = pd.Timestamp(end_date)
    #
    # # use updated_cat unless hasn't been run in which case use source_cat
    # if hasattr(self, "_updated_cat"):
    #     ref_cat = self.updated_cat
    # else:
    #     ref_cat = self.source_cat

    ref_cat = setup_source_catalog()
    # from copy import deepcopy
    # # source = deepcopy(ref_cat[model][timing])
    # source = ref_cat[model][timing]
    # import pdb; pdb.set_trace()

    # THIS WILL NEED TO BE REARRANGED
    # determine filetype to send to `agg_for_date`
    if "regulargrid" in model.lower():
        filetype = "regulargrid"
    elif "2ds" in model.lower():
        filetype = "2ds"
    else:
        filetype = "fields"

    start_datetime, end_datetime, filelocs = find_availability(ref_cat[model][timing], filetype=filetype)

    # # urlpath is None or a list of filler files if the filepaths need to be determined
    # if (
    #     ref_cat[model][timing].urlpath is None
    #     or isinstance(ref_cat[model][timing].urlpath, list)
    # ) and "catloc" in ref_cat[model][timing].metadata:
    #     if "pattern" in ref_cat[model][timing].metadata:
    #         pattern = ref_cat[model][timing].metadata["pattern"]
    #     else:
    #         pattern = None

    # # make sure necessary variables are present
    # assertion = f'You need to provide a `start_date` and `end_date` for finding the relevant model output locations.\nFor {model} and {timing}, the `overall_start_date` is: {ref_cat[model][timing].metadata["overall_start_datetime"]} `overall_end_date` is: {ref_cat[model][timing].metadata["overall_end_datetime"]}.'  # noqa
    # assert start_date is not None and end_date is not None, assertion

    # assertion = f'If timing is "hindcast", `treat_last_day_as_forecast` must be False because the forecast files are not available. `timing`=={timing}.'  # noqa
    # if timing == "hindcast":
    #     assert not treat_last_day_as_forecast, assertion

    # catloc = ref_cat[model][timing].metadata["catloc"]

    # # loop over dates
    # filelocs_urlpath = []
    # for date in pd.date_range(start=start_date, end=end_date, freq="1D"):
    #     if date == end_date and timing == 'forecast':
    #         is_forecast = True
    #     else:
    #         is_forecast = False
    #     filelocs_urlpath.extend(
    #         agg_for_date(
    #             date,
    #             filelocs,
    #             filetype,
    #             is_forecast=is_forecast,
    #             pattern=pattern,
    #         )
    #     )
    #
    # source_orig = ref_cat[model][timing](urlpath=filelocs_urlpath)  # [:2])

    # # urlpath is already available if the link is consistent in time
    # else:
    #     source_orig = ref_cat[model][timing]

    # # open the skeleton transform cat entry and then alter
    # # a few things so can use it with source_orig
    # source_transform_loc = f"{mc.CAT_SOURCE_BASE}/transform.yaml"
    # source_transform = intake.open_catalog(source_transform_loc)["name"]
    # from copy import deepcopy
    #
    # # change new source information
    # # update source's info with model name since user would probably prefer this over timing?
    # # also other metadata to bring into user catalog
    # source_transform.name = f"{model}-{timing}"
    # # if treat_last_day_as_forecast:
    # #     source_transform.name += "-with_forecast"
    #     # source_transform.description += "-with_forecast"
    # # rename source_orig to match source_transform
    # source_orig.name = source_transform.name + "_orig"
    # source_transform.description = (
    #     f"Catalog entry for transform of dataset {source_orig.name}"
    # )
    #
    # # copy over axis and standard_names to transform_kwargs and metadata
    # # also fill in target
    # axis = deepcopy(source_orig.metadata["axis"])
    # snames = deepcopy(source_orig.metadata["standard_names"])
    # source_transform.metadata["axis"] = axis
    # source_transform.metadata["standard_names"] = snames
    # source_transform.__dict__["_captured_init_kwargs"]["transform_kwargs"][
    #     "axis"
    # ] = axis
    # source_transform.__dict__["_captured_init_kwargs"]["transform_kwargs"][
    #     "standard_names"
    # ] = snames
    #
    # # make source_orig the target since will be made available in same catalog
    # target = f"{source_orig.name}"
    # source_transform.__dict__["_captured_init_kwargs"]["targets"] = [target]
    metadata = {
        "model": model,
        "timing": timing,
        "filetype": filetype,
        # "start_date": start_date.isoformat() if start_date is not None else None,
        # "end_date": end_date.isoformat() if end_date is not None else None,
        # # "treat_last_day_as_forecast": treat_last_day_as_forecast,
        # "urlpath": deepcopy(source_orig.urlpath),
        # "cat_source_base": self.cat_source_base,
        # "cat_user_base": self.cat_user_base,
        # "source_catalog_dir": self.source_catalog_dir,
        # "source_catalog_name": self.source_catalog_name,
        # "source_cat": self.source_cat,
        # "user_catalog_dir": self.user_catalog_dir,
        # "user_catalog_name": self.user_catalog_name,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "all_filepaths": filelocs,
    }
    ref_cat[model][timing].metadata.update(metadata)
    return ref_cat[model][timing]
    # source_transform.metadata.update(metadata)
    # source_transform.metadata.update(source_orig.metadata)
    # source_transform.metadata.update(ref_cat[model].metadata)

    # return [source_orig, source_transform]


def add_url_path(
    cat,
    # timing=None,
    start_date=None,
    end_date=None,
    # filetype="fields",
    # treat_last_day_as_forecast=False,
):
    """Make a set of user catalog sources.

    This is meant to be called by `setup_cat()`, not directly by user.

    Parameters
    ----------
    model: str
        Name of model, e.g., CBOFS
    timing: str
        Which timing to use. Normally "forecast", "nowcast", or "hindcast", if
        available for model, but could have different names and/or options.
        Find model options available with `list(self.source_cat[model])` or
        `list(self.updated_cat[model])`.
    start_date, end_date: datetime-interpretable str or pd.Timestamp, optional
        If model has an aggregated link for timing, start_date and end_date
        are not used. Otherwise they should be input. Only year-month-day
        will be used in date. end_date is inclusive.
    treat_last_day_as_forecast: bool, optional
        CHANGE BEHAVIOR
        If True, then date is the last day of the time period being sought and the forecast files
        should be brought in along with the nowcast files, to get the model output the length of the
        forecast out in time. The forecast files brought in will have the latest timing cycle of the
        day that is available. If False, all nowcast files (for all timing cycles) are brought in.

    Returns
    -------
    Source associated with the catalog entry.
    """

    # model = model.upper()

    # # save source to already-made user catalog loc
    # self.user_catalog_dir = self.cat_user_base
    # self.user_catalog_name = (
    #     f"{self.user_catalog_dir}/{self.time_ref.isoformat().replace(':','_')}.yaml"
    # )

    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)

    # # use updated_cat unless hasn't been run in which case use source_cat
    # if hasattr(self, "_updated_cat"):
    #     ref_cat = self.updated_cat
    # else:
    #     ref_cat = self.source_cat
    # ref_cat = setup_source_catalog()

    # which source to use from catalog for desired date range
    if start_date >= pd.Timestamp(cat['forecast'].metadata['start_datetime']) \
       and end_date <= pd.Timestamp(cat['forecast'].metadata['end_datetime']):
        source = cat['forecast']
    elif start_date >= pd.Timestamp(cat['hindcast'].metadata['start_datetime']) \
       and end_date <= pd.Timestamp(cat['hindcast'].metadata['end_datetime']):
        source = cat['hindcast']
    else:
        print('date range does not easily fit into forecast or hindcast')


    # # THIS WILL NEED TO BE REARRANGED
    # # determine filetype to send to `agg_for_date`
    # if "regulargrid" in model.lower():
    #     filetype = "regulargrid"
    # elif "2ds" in model.lower():
    #     filetype = "2ds"
    # else:
    #     filetype = "fields"
    #
    # start_datetime, end_datetime, filelocs = find_availability(ref_cat[model][timing], filetype=filetype)

    # urlpath is None or a list of filler files if the filepaths need to be determined
    if (
        source.urlpath is None
        or isinstance(source.urlpath, list)
    ) and "catloc" in source.metadata:
        if "pattern" in source.metadata:
            pattern = source.metadata["pattern"]
        else:
            pattern = None

        # # make sure necessary variables are present
        # assertion = f'You need to provide a `start_date` and `end_date` for finding the relevant model output locations.\nFor {model} and {timing}, the `overall_start_date` is: {cat.metadata["overall_start_datetime"]} `overall_end_date` is: {cat.metadata["overall_end_datetime"]}.'  # noqa
        # assert start_date is not None and end_date is not None, assertion

        # assertion = f'If timing is "hindcast", `treat_last_day_as_forecast` must be False because the forecast files are not available. `timing`=={timing}.'  # noqa
        # if timing == "hindcast":
        #     assert not treat_last_day_as_forecast, assertion

        # catloc = ref_cat[model][timing].metadata["catloc"]

        # loop over dates
        filelocs_urlpath = []
        for date in pd.date_range(start=start_date, end=end_date, freq="1D"):
            if date == end_date and source.metadata['timing'] == 'forecast':
                is_forecast = True
            else:
                is_forecast = False
            filelocs_urlpath.extend(
                agg_for_date(
                    date,
                    source.metadata['all_filepaths'],
                    source.metadata['filetype'],
                    is_forecast=is_forecast,
                    pattern=pattern,
                )
            )

        source_orig = source(urlpath=filelocs_urlpath)  # [:2])

    # urlpath is already available if the link is consistent in time
    else:
        source_orig = source

    # # COME BACK TO TRANSFORM PART
    # # open the skeleton transform cat entry and then alter
    # # a few things so can use it with source_orig
    # source_transform_loc = f"{self.cat_source_base}/transform.yaml"
    # source_transform = intake.open_catalog(source_transform_loc)["name"]
    # from copy import deepcopy
    #
    # # change new source information
    # # update source's info with model name since user would probably prefer this over timing?
    # # also other metadata to bring into user catalog
    # source_transform.name = f"{model}-{timing}"
    # # if treat_last_day_as_forecast:
    # #     source_transform.name += "-with_forecast"
    #     # source_transform.description += "-with_forecast"
    # # rename source_orig to match source_transform
    # source_orig.name = source_transform.name + "_orig"
    # source_transform.description = (
    #     f"Catalog entry for transform of dataset {source_orig.name}"
    # )
    #
    # # copy over axis and standard_names to transform_kwargs and metadata
    # # also fill in target
    # axis = deepcopy(source_orig.metadata["axis"])
    # snames = deepcopy(source_orig.metadata["standard_names"])
    # source_transform.metadata["axis"] = axis
    # source_transform.metadata["standard_names"] = snames
    # source_transform.__dict__["_captured_init_kwargs"]["transform_kwargs"][
    #     "axis"
    # ] = axis
    # source_transform.__dict__["_captured_init_kwargs"]["transform_kwargs"][
    #     "standard_names"
    # ] = snames
    #
    # # make source_orig the target since will be made available in same catalog
    # target = f"{source_orig.name}"
    # source_transform.__dict__["_captured_init_kwargs"]["targets"] = [target]
    # metadata = {
    #     "model": model,
    #     "timing": timing,
    #     "start_date": start_date.isoformat() if start_date is not None else None,
    #     "end_date": end_date.isoformat() if end_date is not None else None,
    #     # "treat_last_day_as_forecast": treat_last_day_as_forecast,
    #     "urlpath": deepcopy(source_orig.urlpath),
    #     "cat_source_base": self.cat_source_base,
    #     "cat_user_base": self.cat_user_base,
    #     "source_catalog_dir": self.source_catalog_dir,
    #     "source_catalog_name": self.source_catalog_name,
    #     "source_cat": self.source_cat,
    #     "user_catalog_dir": self.user_catalog_dir,
    #     "user_catalog_name": self.user_catalog_name,
    #     "start_datetime": start_datetime,
    #     "end_datetime": end_datetime,
    #     "all_filepaths": filelocs,
    # }
    # source_transform.metadata.update(metadata)
    # source_transform.metadata.update(source_orig.metadata)
    # source_transform.metadata.update(ref_cat[model].metadata)

    # return [source_orig, source_transform]
    return source_orig


def update_catalog(
    kwargs=None,
    model=None,
    timings=None,
    start_date=None,
    end_date=None,
    # treat_last_day_as_forecast=False,
):
    """Setup user catalog, multiple input approaches possible.

    This can be rerun to add more sources to `user_cat`.

    Parameters
    ----------
    kwargs: dict or list, optional
        This keyword provides two of the approaches for using this function.
        * kwargs can be a dict containing the keyword arguments for setting
          up a single source in `user_cat` with `_make_user_sources()`:
          model, timing, start_date, end_date, treat_last_day_as_forecast.
        * kwargs can be a list containing any number of dicts as described
          in the previous bullet point to set up multiple sources in
          `user_cat`.
        User can either input `kwargs` or the other keyword arguments, but
        not both.
    model: str, optional
        Name of model, e.g., CBOFS. User can either input `kwargs` or the
        other keyword arguments, but not both.
    timings: str, optional
        Which timing to use. Normally "forecast", "nowcast", or "hindcast", if
        available for model, but could have different names and/or options.
        Find model options available with `list(self.source_cat[model])` or
        `list(self.updated_cat[model])`.
    start_date, end_date: datetime-interpretable str or pd.Timestamp, optional
        If model has an aggregated link for timing, start_date and end_date
        are not used. Otherwise they should be input. Only year-month-day
        will be used in date. end_date is inclusive.
    treat_last_day_as_forecast: bool, optional
        CHANGE BEHAVIOR
        If True, then date is the last day of the time period being sought
        and the forecast files should be brought in along with the nowcast
        files, to get the model output the length of the forecast out in
        time. The forecast files brought in will have the latest timing
        cycle of the day that is available. If False, all nowcast files (for
        all timing cycles) are brought in.

    Returns
    -------
    Nothing, but sets up or adds to `self.user_cat`. For every model
    combination set up, there will be 2 sources: one ending with "_orig" and
    one without. The one without will be read in but refers to the other.
    Running this function also saves a user_catalog yaml file, or updates
    one if it alreadys exists. The location of the catalog file can be found
    as `cats.user_catalog_name`.

    Examples
    --------

    Set up single source into user catalog:

    >>> import model_catalogs as mc
    >>> cats = mc.Management()
    >>> cats.setup_cat(model='CBOFS', timing='forecast')

    Set up single source into user catalog, from a dict:
    >>> cats.setup_cat(dict(model='CBOFS', timing='forecast'))

    Set up multiple sources into user catalog, from a list:
    >>> cats.setup_cat([dict(model='DBOFS', timing='forecast'),
                        dict(model='TBOFS', timing='forecast')])
    """

    # keywords = [model, timings, start_date, end_date]
    # if any([keyword is not None for keyword in keywords]):
    #     assertion = "If inputting a set of keywords to setup user source, both `model` and `timing` are required."  # noqa: E501
    #     assert (model is not None) and (timings is not None), assertion
    #     assertion = "If inputting a set of keywords to setup user source, kwargs is not used and should be none."  # noqa: E501
    #     assert kwargs is None, assertion
    # if kwargs is not None:
    #     assertion = "If inputting kwargs, other keyword arguments should be None."
    #     assert any([keyword is None for keyword in keywords]), assertion

    # self.user_catalog_dir = self.cat_user_base
    # self.user_catalog_name = (
    #     f"{self.user_catalog_dir}/{self.time_ref.isoformat()}.yaml"
    # )

    # if isinstance(kwargs, dict):
    #     new_sources = self._make_user_sources(**kwargs)
    #     # sources.append(source)
    # elif isinstance(kwargs, list):
    #     new_sources = []
    #     for option_dict in kwargs:
    #         source = self._make_user_sources(**option_dict)
    #         new_sources.extend(source)
    # else:
    #     new_sources = self._make_user_sources(
    #         model=model,
    #         timing=timing,
    #         start_date=start_date,
    #         end_date=start_date,
    #         # treat_last_day_as_forecast=treat_last_day_as_forecast,
    #     )

    if timings is None:
        timings = ['forecast', 'hindcast']

    new_sources = []
    for timing in timings:
        source = _update_catalog(
            model=model,
            timing=timing,
            # start_date=start_date,
            # end_date=start_date,
            # treat_last_day_as_forecast=treat_last_day_as_forecast,
        )
        new_sources.append(source)

    # # if there is already a user_cat, pull out the existing sources and
    # # recreate with the new sources
    # if hasattr(self, "user_cat"):
    #     old_source_names = list(self.user_cat)
    #     sources = [
    #         self.user_cat[old_source_name] for old_source_name in old_source_names
    #     ]
    #     sources.extend(new_sources)
    # else:
    #     sources = new_sources

    metadata = {
        "source_catalog_dir": mc.SOURCE_CATALOG_DIR,
        "source_catalog_name": mc.SOURCE_CATALOG_NAME,
        "filetype": new_sources[0].metadata["filetype"]
        # "time_ref": self.time_ref.isoformat(),
    }
    # import pdb; pdb.set_trace()
    new_user_cat = make_catalog(
        new_sources,
        "User-catalog.",
        "User-made catalog.",
        metadata,
        [source._entry._driver for source in new_sources],
        cat_path=mc.SOURCE_CATALOG_NAME,
    )
    return new_user_cat


# class SetupCatalog:
#     """Class to manage the source catalog.
#
#     Attributes
#     ----------
#     catalog_path: str
#         Provide base location for catalogs. Default location is in the working directory,
#         in a directory called "catalogs".
#     cat_source_base: str
#         Provide base location for source catalogs. Default location is in the working directory,
#         in a directory called f'{self.catalog_path}/source_catalogs'.
#     cat_user_base: str
#         Provide base location for user catalogs. Default location is in the working directory,
#         in a directory called f'{self.catalog_path}/user_catalogs'.
#     time_ref: pd.Timestamp
#         The time when the class is initialized. This will be used to "version" any updated and
#         user catalogs that are created in this object.
#     source_catalog_name: str
#         Full relative path of the source catalog, as defined by
#         f'{self.cat_source_base}/{source_catalog_name}'.
#     source_catalog_dir: str
#         Subdirectory in source_catalogs specifying which source catalogs have been used for models.
#         Default location is in the working directory,
#         in a directory called f'{self.cat_source_base}/orig' and taking the most recent one.
#         Alternatively the user can input which date to use.
#     source_cat: intake Catalog object
#         Source catalog, containing references to all known models in the source_catalogs directory.
#         This is all hard-wired information about the models.
#     """
#
#     def __init__(
#         self,
#         catalog_path=f"{mc.__path__[0]}/catalogs",
#         source_catalog_name="source_catalog.yaml",
#         make_source_catalog=False,
#         source_ref_date=None,
#     ):
#         """Initialize a Management class object.
#
#         Parameters
#         ----------
#         catalog_path: str, optional
#             Provide base location for catalogs. Default location is in the working directory,
#             in a directory called "catalogs".
#         source_catalog_name: str, optional
#             Alternative name to use for source catalog. Default is 'source_catalog.yaml'.
#             Suffix isn't necessary.
#         make_source_catalog: boolean, optional
#             The source catalog is not meant to change often. So, if source_catalog_name in
#             catalog_path already exists and make_source_catalog is False, it will simply be
#             read in. Otherwise, the source catalog will be recreated.
#         source_ref_date: str, optional
#             Reference date for directory to use to find source_catalog files. If None,
#             code will choose most recent.
#
#         Returns
#         -------
#         Management class object which is initialized with `.source_cat`.
#
#         Examples
#         --------
#         >>> import model_catalogs as mc
#         >>> cat = mc.SetupCatalog()
#         >>> cat.source_cat
#         """
#
#         self.catalog_path = catalog_path
#         self.cat_source_base = f"{self.catalog_path}/source_catalogs"
#         self.cat_user_base = f"{self.catalog_path}/user_catalogs"
#         self.time_ref = pd.Timestamp.now()
#
#         # find most recent set of source_catalogs
#         if os.path.exists(f"{self.cat_source_base}/complete"):
#             which_dir = f"{self.cat_source_base}/complete"
#         else:
#             which_dir = f"{self.cat_source_base}/orig"
#             print(
#                 '"complete" model source files are not yet available. Run `.update_source_files()` to create this directory.'  # noqa: E501
#             )
#
#         self.source_catalog_dir = which_dir
#
#         # Read in already-available model source catalog
#         self.source_catalog_name = f"{self.cat_source_base}/{source_catalog_name}"
#         if os.path.exists(self.source_catalog_name) and not make_source_catalog:
#             self.source_cat = intake.open_catalog(self.source_catalog_name)
#
#         else:  # otherwise, make it
#             self.setup_source_catalog()
#
#     def setup_source_catalog(self):
#         """Setup source catalog for models.
#
#         Returns
#         -------
#         Nothing, but sets up `.source_cat`.
#         """
#
#         cat_source_description = "Source catalog for models."
#
#         # open catalogs
#         cat_locs = glob(f"{self.source_catalog_dir}/*.yaml")
#         cats = [intake.open_catalog(cat_loc) for cat_loc in cat_locs]
#
#         metadata = {"source_catalog_dir": self.source_catalog_dir}
#
#         self.source_cat = make_catalog(
#             cats,
#             self.source_catalog_name,
#             cat_source_description,
#             metadata,
#             intake.catalog.local.YAMLFileCatalog,
#         )
#
#     def update_source_files(self):
#         """Update model source files in 'source_catalogs/orig'.
#
#         This will add outer boundary files for the model domains and create
#         "complete" directory of model files that mirror those in "orig".
#
#         Returns
#         -------
#         Nothing, but resaves all model source catalogs into
#         f"{self.cat_source_base}/complete" with domain boundaries added.
#
#         Examples
#         --------
#
#         Add boundary calculations to source model files:
#         >>> cats.update_source_files()
#         """
#
#         if os.path.exists(f"{self.cat_source_base}/complete"):
#             print('"complete" source model catalog files already exist.')
#             return
#
#         # update default source_catalog model file location to complete
#         # instead of orig
#         self.source_catalog_dir_old = self.source_catalog_dir
#         self.source_catalog_dir = f"{self.cat_source_base}/complete"
#
#         models = list(self.source_cat)
#         timing = "forecast"
#
#         for model in models:
#
#             # read in model output
#             ds = self.source_cat[model][timing].to_dask()
#
#             # find metadata
#             # select lon/lat for use. There may be more than one and we also want the name.
#             if "alpha_shape" in self.source_cat[model].metadata:
#                 dd, alpha = self.source_cat[model].metadata["alpha_shape"]
#             else:
#                 dd, alpha = None, None
#             lonkey, latkey, bbox, wkt = find_bbox(ds, dd=dd, alpha=alpha)
#             # lonkey, latkey, bbox, wkt_low, wkt_high = find_bbox(ds, dd=dd, alpha=alpha)
#
#             # metadata for overall source_id0
#             metadata0 = {
#                 "geospatial_bounds": wkt,
#                 "bounding_box": bbox,
#             }
#
#             # add Dataset metadata to specific source metadata
#             # change metadata attributes to strings so catalog doesn't barf on them
#             for attr in ds.attrs:
#                 self.source_cat[model][timing].metadata[attr] = str(ds.attrs[attr])
#
#             # add 0th level metadata to 0th level model entry
#             self.source_cat[model].metadata.update(metadata0)
#
#             timings = list(self.source_cat[model])
#             sources = [self.source_cat[model][timing] for timing in timings]
#             make_catalog(
#                 sources,
#                 model,
#                 self.source_cat[model].description,
#                 self.source_cat[model].metadata,
#                 "opendap",
#                 f"{self.source_catalog_dir}/{model.lower()}.yaml",
#             )
#
#         self.setup_source_catalog()
#
#     def _make_user_sources(
#         self,
#         model,
#         timing,
#         start_date=None,
#         end_date=None,
#         # filetype="fields",
#         # treat_last_day_as_forecast=False,
#     ):
#         """Make a set of user catalog sources.
#
#         This is meant to be called by `setup_cat()`, not directly by user.
#
#         Parameters
#         ----------
#         model: str
#             Name of model, e.g., CBOFS
#         timing: str
#             Which timing to use. Normally "forecast", "nowcast", or "hindcast", if
#             available for model, but could have different names and/or options.
#             Find model options available with `list(self.source_cat[model])` or
#             `list(self.updated_cat[model])`.
#         start_date, end_date: datetime-interpretable str or pd.Timestamp, optional
#             If model has an aggregated link for timing, start_date and end_date
#             are not used. Otherwise they should be input. Only year-month-day
#             will be used in date. end_date is inclusive.
#         treat_last_day_as_forecast: bool, optional
#             CHANGE BEHAVIOR
#             If True, then date is the last day of the time period being sought and the forecast files
#             should be brought in along with the nowcast files, to get the model output the length of the
#             forecast out in time. The forecast files brought in will have the latest timing cycle of the
#             day that is available. If False, all nowcast files (for all timing cycles) are brought in.
#
#         Returns
#         -------
#         Source associated with the catalog entry.
#         """
#
#         model = model.upper()
#
#         # save source to already-made user catalog loc
#         self.user_catalog_dir = self.cat_user_base
#         self.user_catalog_name = (
#             f"{self.user_catalog_dir}/{self.time_ref.isoformat().replace(':','_')}.yaml"
#         )
#
#         if not isinstance(start_date, pd.Timestamp):
#             start_date = pd.Timestamp(start_date)
#         if not isinstance(end_date, pd.Timestamp):
#             end_date = pd.Timestamp(end_date)
#
#         # use updated_cat unless hasn't been run in which case use source_cat
#         if hasattr(self, "_updated_cat"):
#             ref_cat = self.updated_cat
#         else:
#             ref_cat = self.source_cat
#
#         # THIS WILL NEED TO BE REARRANGED
#         # determine filetype to send to `agg_for_date`
#         if "regulargrid" in model.lower():
#             filetype = "regulargrid"
#         elif "2ds" in model.lower():
#             filetype = "2ds"
#         else:
#             filetype = "fields"
#
#         start_datetime, end_datetime, filelocs = find_availability(ref_cat[model][timing], filetype=filetype)
#
#         # urlpath is None or a list of filler files if the filepaths need to be determined
#         if (
#             ref_cat[model][timing].urlpath is None
#             or isinstance(ref_cat[model][timing].urlpath, list)
#         ) and "catloc" in ref_cat[model][timing].metadata:
#             if "pattern" in ref_cat[model][timing].metadata:
#                 pattern = ref_cat[model][timing].metadata["pattern"]
#             else:
#                 pattern = None
#
#             # make sure necessary variables are present
#             assertion = f'You need to provide a `start_date` and `end_date` for finding the relevant model output locations.\nFor {model} and {timing}, the `overall_start_date` is: {ref_cat[model][timing].metadata["overall_start_datetime"]} `overall_end_date` is: {ref_cat[model][timing].metadata["overall_end_datetime"]}.'  # noqa
#             assert start_date is not None and end_date is not None, assertion
#
#             assertion = f'If timing is "hindcast", `treat_last_day_as_forecast` must be False because the forecast files are not available. `timing`=={timing}.'  # noqa
#             if timing == "hindcast":
#                 assert not treat_last_day_as_forecast, assertion
#
#             catloc = ref_cat[model][timing].metadata["catloc"]
#
#             # loop over dates
#             filelocs_urlpath = []
#             for date in pd.date_range(start=start_date, end=end_date, freq="1D"):
#                 if date == end_date and timing == 'forecast':
#                     is_forecast = True
#                 else:
#                     is_forecast = False
#                 filelocs_urlpath.extend(
#                     agg_for_date(
#                         date,
#                         filelocs,
#                         filetype,
#                         is_forecast=is_forecast,
#                         pattern=pattern,
#                     )
#                 )
#
#             source_orig = ref_cat[model][timing](urlpath=filelocs_urlpath)  # [:2])
#
#         # urlpath is already available if the link is consistent in time
#         else:
#             source_orig = ref_cat[model][timing]
#
#         # open the skeleton transform cat entry and then alter
#         # a few things so can use it with source_orig
#         source_transform_loc = f"{self.cat_source_base}/transform.yaml"
#         source_transform = intake.open_catalog(source_transform_loc)["name"]
#         from copy import deepcopy
#
#         # change new source information
#         # update source's info with model name since user would probably prefer this over timing?
#         # also other metadata to bring into user catalog
#         source_transform.name = f"{model}-{timing}"
#         # if treat_last_day_as_forecast:
#         #     source_transform.name += "-with_forecast"
#             # source_transform.description += "-with_forecast"
#         # rename source_orig to match source_transform
#         source_orig.name = source_transform.name + "_orig"
#         source_transform.description = (
#             f"Catalog entry for transform of dataset {source_orig.name}"
#         )
#
#         # copy over axis and standard_names to transform_kwargs and metadata
#         # also fill in target
#         axis = deepcopy(source_orig.metadata["axis"])
#         snames = deepcopy(source_orig.metadata["standard_names"])
#         source_transform.metadata["axis"] = axis
#         source_transform.metadata["standard_names"] = snames
#         source_transform.__dict__["_captured_init_kwargs"]["transform_kwargs"][
#             "axis"
#         ] = axis
#         source_transform.__dict__["_captured_init_kwargs"]["transform_kwargs"][
#             "standard_names"
#         ] = snames
#
#         # make source_orig the target since will be made available in same catalog
#         target = f"{source_orig.name}"
#         source_transform.__dict__["_captured_init_kwargs"]["targets"] = [target]
#         metadata = {
#             "model": model,
#             "timing": timing,
#             "start_date": start_date.isoformat() if start_date is not None else None,
#             "end_date": end_date.isoformat() if end_date is not None else None,
#             # "treat_last_day_as_forecast": treat_last_day_as_forecast,
#             "urlpath": deepcopy(source_orig.urlpath),
#             "cat_source_base": self.cat_source_base,
#             "cat_user_base": self.cat_user_base,
#             "source_catalog_dir": self.source_catalog_dir,
#             "source_catalog_name": self.source_catalog_name,
#             "source_cat": self.source_cat,
#             "user_catalog_dir": self.user_catalog_dir,
#             "user_catalog_name": self.user_catalog_name,
#             "start_datetime": start_datetime,
#             "end_datetime": end_datetime,
#             "all_filepaths": filelocs,
#         }
#         source_transform.metadata.update(metadata)
#         source_transform.metadata.update(source_orig.metadata)
#         source_transform.metadata.update(ref_cat[model].metadata)
#
#         return [source_orig, source_transform]
#
#     def setup_cat(
#         self,
#         kwargs=None,
#         model=None,
#         timing=None,
#         start_date=None,
#         end_date=None,
#         # treat_last_day_as_forecast=False,
#     ):
#         """Setup user catalog, multiple input approaches possible.
#
#         This can be rerun to add more sources to `user_cat`.
#
#         Parameters
#         ----------
#         kwargs: dict or list, optional
#             This keyword provides two of the approaches for using this function.
#             * kwargs can be a dict containing the keyword arguments for setting
#               up a single source in `user_cat` with `_make_user_sources()`:
#               model, timing, start_date, end_date, treat_last_day_as_forecast.
#             * kwargs can be a list containing any number of dicts as described
#               in the previous bullet point to set up multiple sources in
#               `user_cat`.
#             User can either input `kwargs` or the other keyword arguments, but
#             not both.
#         model: str, optional
#             Name of model, e.g., CBOFS. User can either input `kwargs` or the
#             other keyword arguments, but not both.
#         timing: str, optional
#             Which timing to use. Normally "forecast", "nowcast", or "hindcast", if
#             available for model, but could have different names and/or options.
#             Find model options available with `list(self.source_cat[model])` or
#             `list(self.updated_cat[model])`.
#         start_date, end_date: datetime-interpretable str or pd.Timestamp, optional
#             If model has an aggregated link for timing, start_date and end_date
#             are not used. Otherwise they should be input. Only year-month-day
#             will be used in date. end_date is inclusive.
#         treat_last_day_as_forecast: bool, optional
#             CHANGE BEHAVIOR
#             If True, then date is the last day of the time period being sought
#             and the forecast files should be brought in along with the nowcast
#             files, to get the model output the length of the forecast out in
#             time. The forecast files brought in will have the latest timing
#             cycle of the day that is available. If False, all nowcast files (for
#             all timing cycles) are brought in.
#
#         Returns
#         -------
#         Nothing, but sets up or adds to `self.user_cat`. For every model
#         combination set up, there will be 2 sources: one ending with "_orig" and
#         one without. The one without will be read in but refers to the other.
#         Running this function also saves a user_catalog yaml file, or updates
#         one if it alreadys exists. The location of the catalog file can be found
#         as `cats.user_catalog_name`.
#
#         Examples
#         --------
#
#         Set up single source into user catalog:
#
#         >>> import model_catalogs as mc
#         >>> cats = mc.Management()
#         >>> cats.setup_cat(model='CBOFS', timing='forecast')
#
#         Set up single source into user catalog, from a dict:
#         >>> cats.setup_cat(dict(model='CBOFS', timing='forecast'))
#
#         Set up multiple sources into user catalog, from a list:
#         >>> cats.setup_cat([dict(model='DBOFS', timing='forecast'),
#                             dict(model='TBOFS', timing='forecast')])
#         """
#
#         keywords = [model, timing, start_date, end_date]
#         if any([keyword is not None for keyword in keywords]):
#             assertion = "If inputting a set of keywords to setup user source, both `model` and `timing` are required."  # noqa: E501
#             assert (model is not None) and (timing is not None), assertion
#             assertion = "If inputting a set of keywords to setup user source, kwargs is not used and should be none."  # noqa: E501
#             assert kwargs is None, assertion
#         if kwargs is not None:
#             assertion = "If inputting kwargs, other keyword arguments should be None."
#             assert any([keyword is None for keyword in keywords]), assertion
#
#         self.user_catalog_dir = self.cat_user_base
#         self.user_catalog_name = (
#             f"{self.user_catalog_dir}/{self.time_ref.isoformat()}.yaml"
#         )
#
#         if isinstance(kwargs, dict):
#             new_sources = self._make_user_sources(**kwargs)
#             # sources.append(source)
#         elif isinstance(kwargs, list):
#             new_sources = []
#             for option_dict in kwargs:
#                 source = self._make_user_sources(**option_dict)
#                 new_sources.extend(source)
#         else:
#             new_sources = self._make_user_sources(
#                 model=model,
#                 timing=timing,
#                 start_date=start_date,
#                 end_date=start_date,
#                 # treat_last_day_as_forecast=treat_last_day_as_forecast,
#             )
#
#         # if there is already a user_cat, pull out the existing sources and
#         # recreate with the new sources
#         if hasattr(self, "user_cat"):
#             old_source_names = list(self.user_cat)
#             sources = [
#                 self.user_cat[old_source_name] for old_source_name in old_source_names
#             ]
#             sources.extend(new_sources)
#         else:
#             sources = new_sources
#
#         metadata = {
#             "source_catalog_dir": self.source_catalog_dir,
#             "source_catalog_name": self.source_catalog_name,
#             "time_ref": self.time_ref.isoformat(),
#         }
#
#         new_user_cat = make_catalog(
#             sources,
#             "User-catalog.",
#             "User-made catalog.",
#             metadata,
#             [source._entry._driver for source in sources],
#             cat_path=self.user_catalog_name,
#         )
#         self.user_cat = new_user_cat
#
#         return self.user_cat
