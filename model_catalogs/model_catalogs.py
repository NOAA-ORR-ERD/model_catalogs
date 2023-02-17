"""
Everything dealing with the catalogs.
"""

import warnings

from datetime import datetime
from pathlib import Path, PurePath

import cf_xarray  # noqa
import intake
import intake.source.derived
import pandas as pd
import requests
import yaml

from datetimerange import DateTimeRange
from dateutil.parser import parse
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
from intake_xarray.opendap import OpenDapSource

import model_catalogs as mc

from model_catalogs.process import DatasetTransform


DEFAULT = datetime(1970, 1, 1, 22, 22, 22)


def make_catalog(
    cats,
    full_cat_name,
    full_cat_description,
    full_cat_metadata,
    cat_driver,
    cat_path=None,
    save_catalog=True,
    return_cat: bool = True,
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

       * `intake.catalog.local.YAMLFileCatalog`
       * `'opendap'`

       If list, must be same length as cats and contains drivers that
       correspond to cats.
    cat_path: Path object, optional
       Path with catalog name to use for saving catalog. With or without yaml suffix. If not provided,
       will use `full_cat_name`.
    save_catalog : bool, optional
        Defaults to True, and saves to cat_path.
    return_cat : bool, optional
        Return catalog.

    Returns
    -------
    Intake Catalog
        A single catalog made from multiple catalogs or sources.

    Examples
    --------

    Make catalog:

    >>> make_catalog([list of Intake sources or catalogs], 'catalog name', 'catalog desc',
                     {}, 'opendap', save_catalog=False)
    """

    if cat_path is None and save_catalog:
        cat_path = Path(full_cat_name)
    # if save_catalog:
    #     cat_path = cat_path / full_cat_name.lower()
    #     # cat_path = f"{cat_path}/{full_cat_name.lower()}"
    # if save_catalog and ("yaml" not in str(cat_path)) and ("yml" not in str(cat_path)):
    #     cat_path = cat_path.with_suffix(".yaml")

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
            cat.name.upper() if isinstance(cat, Catalog) else cat.name,
            description=cat.description,
            driver=catd,
            args=cat._yaml()["sources"][cat.name]["args"],
            metadata=cat.metadata,
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
    if save_catalog:
        cat.save(cat_path.with_suffix(".yaml"))

    if return_cat:
        return cat


def open_catalog(
    cat_loc,
    return_cat=True,
    save_catalog=False,
    override=False,
    boundaries=False,
    save_boundaries=False,
):
    """Open an intake catalog file and set up code to apply processing/transform.

    Optionally calculate the boundaries of the model represented in cat_log.

    Note that saved boundaries files will be saved under the name inside the catalog, not the name of the file if you input a catalog path.

    Parameters
    ----------
    cat_loc : str, Catalog
        The catalog to open. cat_loc can be the representation of a path to a catalog file (string or Path) or it can be a Catalog object.
    return_cat : bool, optional
        Return catalog from function. Defaults to True.
    save_catalog : bool, optional
        Defaults to False, and saves to mc.CACHE_PATH_COMPILED(model).
    override : boolean, optional
        Use `override=True` to calculate boundaries of the model regardless of whether the file already exists.
    boundaries : boolean, optional
        If True, find previously-saved or calculate domain boundary of model.
    save_boundaries : bool, optional
        Defaults to False, and saves to mc.FILE_PATH_BOUNDARIES(model).
    """

    if isinstance(cat_loc, Catalog):
        cat_orig = cat_loc
    else:
        cat_loc = PurePath(cat_loc)
        cat_orig = intake.open_catalog(cat_loc)

    if boundaries:
        if mc.FILE_PATH_BOUNDARIES(cat_orig.name.lower()).is_file() and not override:
            # add previously-saved boundary info
            # this was calculated with mc.calculate_boundaries()
            with open(mc.FILE_PATH_BOUNDARIES(cat_orig.name.lower()), "r") as stream:
                boundary = yaml.safe_load(stream)
        else:
            boundary = mc.calculate_boundaries(
                cat_orig, save_files=save_boundaries, return_boundaries=True
            )[cat_orig.name]

        # add to cat_orig metadata
        cat_orig.metadata["bounding_box"] = boundary["bbox"]
        cat_orig.metadata["geospatial_bounds"] = boundary["wkt"]

    # get transform of each original catalog file, which points to
    # original file but applies metadata from original catalog file
    # to the resulting dataset after calling `to_dask()`
    source_transforms = [
        mc.transform_source(cat_orig[model_source]) for model_source in list(cat_orig)
    ]

    metadata = cat_orig.metadata
    metadata.update({"cat_path": cat_orig.path})

    # need to make catalog to transfer information properly from
    # source_orig to source_transform
    cat = mc.make_catalog(
        source_transforms,
        full_cat_name=cat_orig.name,  # model name
        full_cat_description=cat_orig.description,
        full_cat_metadata=metadata,
        cat_driver=mc.process.DatasetTransform,
        cat_path=mc.FILE_PATH_COMPILED(cat_orig.name),
        save_catalog=save_catalog,
        return_cat=True,
    )

    if return_cat:
        return cat


def setup(locs="mc_", override=False, boundaries=True):
    """Setup reference catalogs for models.

    Loops over catalogs that have been previously installed as data packages to intake that start with the string(s) in locs. The default is to read in the required GOODS model catalogs which are prefixed with `"mc_"`. Alternatively, one or more local catalog files can be input as strings or Paths.

    This function calls ``open_catalog`` which reads in previously-saved model boundary information (or calculates it if not available) and saves temporary catalog files for each model (called "compiled"), then this function links those together into the returned main catalog. For some models, reading in the original catalogs applies a "today" and/or "yesterday" date Intake user parameter that supplies two example model files that can be used for examining the model output for the example times. Those are rerun each time this function is rerun, filling the parameters using the proper dates.

    Note that saved compiled catalog files will be saved under the name inside the catalog, not the name of the file if you input a catalog path.

    Parameters
    ----------
    locs : str, Path, list
        This can be:

        * a string or Path describing where a Catalog file is located
        * a string of the prefix for selecting catalogs from the default intake catalog, ``intake.cat``. It is expected to be of the form "PREFIX_CATALOGNAME" with an underscore at the end followed by the catalog name, and there could be many catalogs with that `"PREFIX_"` set up.
        * a list of a combination of the previous options.

    override : boolean, optional
        Use `override=True` to compile the catalog files together regardless of freshness.
    boundaries : boolean, optional
        If True, find previously-saved or calculate domain boundary of model. Intended for testing.

    Returns
    -------
    Intake catalog
        Nested Intake catalog with a catalog for each input option. Each model in turn has one or more model_source available (e.g., "coops-forecast-agg", "coops-forecast-noagg").

    Examples
    --------

    Set up main catalog:

    >>> main_cat = mc.setup()

    Examine list of models available in catalog:

    >>> list(main_cat)

    Examine the model_sources for a specific model in the catalog:

    >>> list(main_cat['CBOFS'])

    Separate from ``model_catalogs`` you can check the default Intake catalog with:

    >>> list(intake.cat)
    """

    locs = mc.astype(locs, list)

    # arrange inputs into list of known Catalog instances and Paths to catalogs
    initial_cats = []
    for loc in locs:

        # initial_cats is a list of Catalogs in this case
        cats = [
            intake.cat[cat_name] for cat_name in list(intake.cat) if loc in cat_name
        ]

        # remove the prefix from the catalog name
        for cat in cats:
            cat.name = cat.name.lstrip(loc)

        # check for if loc is instead a path to a catalog
        if len(cats) == 0:
            # initial_cats is a list of one Path in this case
            # cats = [PurePath(loc)]
            # initial_cats is a list of one Catalog in this case
            cats = [intake.open_catalog(loc)]

        # now cats is a list of Catalog(s)
        initial_cats.extend(cats)

    cat_transform_locs = []
    for cat in list(initial_cats):

        # if isinstance(cat, PurePath):
        #     name = cat.stem
        # elif isinstance(cat, Catalog):
        name = cat.name

        # re-compile together catalog file if user wants to override possibly
        # existing file or if is not fresh
        if override or not mc.is_fresh(mc.FILE_PATH_COMPILED(name)):
            # override for open_catalog is about calculating boundaries
            open_catalog(
                cat,
                return_cat=False,
                save_catalog=True,
                boundaries=boundaries,
                save_boundaries=True,
                override=False,
            )
        cat_transform_locs.append(mc.FILE_PATH_COMPILED(name))

    # have to read these from disk in order to make them type
    # intake.catalog.local.YAMLFileCatalog
    # instead of intake.catalog.base.Catalog
    cats = [intake.open_catalog(loc) for loc in cat_transform_locs]

    # make master nested catalog
    main_cat = mc.make_catalog(
        cats,
        full_cat_name="MAIN-CATALOG",
        full_cat_description="Main catalog for models; a catalog of nested catalogs.",
        full_cat_metadata={},
        # full_cat_metadata={"source_catalog_dir": str(mc.CAT_PATH_ORIG)},
        cat_driver=intake.catalog.local.YAMLFileCatalog,
        cat_path=None,
        save_catalog=False,
    )

    return main_cat


def find_datetimes(source, find_start_datetime, find_end_datetime, override=False):
    """Find the start and/or end datetimes for source.

    For sources with static urlpaths, this opens the Dataset and checks the first time for `start_datetime` and the last time for `end_datetime`.

    Some NOAA OFS sources require aggregation: model_source names "coops-forecast-noagg" and "ncei-archive-noagg". For these, the available year and months of the thredd server subcatalogs are found with ``find_catrefs()``. `start_datetime` is found by further evaluating to make sure that files in the subcatalogs are both available on the page and that the days represented by model output files are consecutive (there are missing dates). `end_datetime` is found from the most recent subcatalog files since there aren't missing files and dates on the recent end of the time ranges.

    Uses ``cf-xarray`` to determine the time axis.

    Parameters
    ----------
    source : Intake source
        Model source for which to find start and/or end datetimes
    find_start_datetime : bool
        True to calculate start_datetime, otherwise returns None
    find_end_datetime : bool
        True to calculate end_datetime, otherwise returns None
    override : boolean, optional
        Use `override=True` to find catrefs regardless of freshness. This is passed in from
        ``find_availability()`` so has the same value as input there.

    Returns
    -------
    tuple
        Contains 'start_datetime' and 'end_datetime' where each are strings or can be None if they didn't need to be found.
    """

    # For any model/model_source pairs with static links or known file address,
    # which is all non-OFS models and OFS models that are already aggregated
    if "catloc" not in source.metadata:

        # flag to process.py for `to_dask()` call to skip detailed processing since just a quick check. Doesn't save ds to source._ds or cache.
        source._kwargs["skip_dask_processing"] = True
        ds = source.to_dask()
        # use one T in case there are more than one
        tkey = ds.cf.axes["T"][0]
        start_datetime = str(ds[tkey].values[0]) if find_start_datetime else None
        end_datetime = str(ds[tkey].values[-1]) if find_end_datetime else None
        ds.close()

    # for when we need to aggregate which is for model_source: ncei-archive-noagg and coops-forecast-noagg
    else:

        if "filetype" not in source.cat.metadata:
            raise KeyError(
                "If your model requires aggregation, it also requires `filetype` in the catalog-level metadata."
            )
        else:
            filetype = source.cat.metadata["filetype"]

        if not override and mc.is_fresh(
            mc.FILE_PATH_CATREFS(source.cat.name, source.name), source
        ):
            with open(
                mc.FILE_PATH_CATREFS(source.cat.name, source.name), "r"
            ) as stream:
                catrefs = yaml.safe_load(stream)["catrefs"]
        else:
            catrefs = mc.find_catrefs(source.metadata["catloc"])
            catrefs = sorted(catrefs)  # earliest first, most recent last
            with open(
                mc.FILE_PATH_CATREFS(source.cat.name, source.name), "w"
            ) as outfile:
                yaml.dump({"catrefs": catrefs}, outfile, default_flow_style=False)

        if find_start_datetime:
            # Getting start date #
            # first make sure the dates actually have model files available
            filelocs = []
            i = 0
            while len(filelocs) == 0:
                # print(catrefs[i])
                filelocs = sorted(
                    mc.find_filelocs(
                        catrefs[i], source.metadata["catloc"], filetype=filetype
                    )
                )
                i += 1

            # second make sure we only count when dates are consecutive, since servers tend to have some
            # spotty model output at the earliest dates get dates from file names
            df = mc.filedates2df(filelocs)

            # which differences in consecutive dates are over 1 day
            ddf = pd.Series(df.index).diff() > pd.Timedelta("1 day")
            if ddf.any():
                # first date after last jump in dates is desired start day
                start_datetime = str(df.index.where(ddf).dropna()[-1])

            # all dates were fine, so just use earliest fileloc
            else:
                start_datetime = str(df.index[0])
        else:
            start_datetime = None

        if find_end_datetime:
            # Getting end date #
            filelocs = mc.find_filelocs(
                catrefs[-1], source.metadata["catloc"], filetype=filetype
            )
            df = mc.filedates2df(filelocs)
            end_datetime = str(df.index[-1])
        else:
            end_datetime = None

    # save start/end to files
    if find_start_datetime:
        with open(mc.FILE_PATH_START(source.cat.name, source.name), "w") as outfile:
            yaml.dump(
                {"start_datetime": start_datetime}, outfile, default_flow_style=False
            )
    if find_end_datetime:
        with open(mc.FILE_PATH_END(source.cat.name, source.name), "w") as outfile:
            yaml.dump({"end_datetime": end_datetime}, outfile, default_flow_style=False)

    return start_datetime, end_datetime


def find_availability_source(source, override=False):
    """Find availabililty for source specifically.

    This function is called by `find_availability()` for each source. If ``source.status`` is False, input source is returned with None for `start_datetime` and `end_datetime`.

    Parameters
    ----------
    source : Intake source
        Source for which to find availability.

    Returns
    -------
    Intake source
        `start_datetime` and `end_datetime` are added to metadata of source.
    """

    # if server is not working, return input source with None for new metadata
    if not source.status:
        # raise RuntimeError(f"Server for source {source.cat.name}, {source.name}, is not working. Urlpath checked was {mc.astype(source.urlpath, list)[0]}.")
        warnings.warn(
            f"Server for source {source.cat.name}, {source.name}, is not working. Urlpath checked was {mc.astype(source.urlpath, list)[0]}.",
            RuntimeWarning,
        )
        start_datetime, end_datetime = None, None

    else:

        # check if start and end datetime files already exist and are new enough to use
        # check if already know the time and not stale
        # file times are given in UTC
        # If files are not stale, read in info from there
        if not override and mc.is_fresh(
            mc.FILE_PATH_START(source.cat.name, source.name), source
        ):
            with open(mc.FILE_PATH_START(source.cat.name, source.name), "r") as stream:
                start_datetime = yaml.safe_load(stream)["start_datetime"]
            find_start_datetime = False
        else:
            find_start_datetime = True  # need to still find the start_datetime

        if not override and mc.is_fresh(
            mc.FILE_PATH_END(source.cat.name, source.name), source
        ):
            with open(mc.FILE_PATH_END(source.cat.name, source.name), "r") as stream:
                end_datetime = yaml.safe_load(stream)["end_datetime"]
            find_end_datetime = False
        else:
            find_end_datetime = True  # need to still find the end_datetime

        # start and end temp could be None, depending on values of
        # find_start_datetime, find_end_datetime
        if find_start_datetime or find_end_datetime:
            start_temp, end_temp = find_datetimes(
                source, find_start_datetime, find_end_datetime, override=override
            )

        start_datetime = start_temp if find_start_datetime else start_datetime
        end_datetime = end_temp if find_end_datetime else end_datetime

    source.metadata["start_datetime"] = start_datetime
    source.metadata["end_datetime"] = end_datetime

    return source


def find_availability(cat_or_source, model_source=None, override=False, verbose=False):
    """Find availability for Catalog or Source.

    The code will check for previously-calculated availability. If found, the "freshness" of the information is checked as compared with ``mc.FRESH`` parameters specified in ``__init__``.

    Start and end datetimes are allowed to be calculated separately to save time.

    Note that for unaggregated models with forecasts (currently just model_source "coops-forecast-noagg"), this checks availability for the latest forecast, which goes forward in time from today. It is not possible to use this function to check for the case of a forecast forward in time from a past day.

    Parameters
    ----------
    cat_or_source : Intake catalog or source
        Catalog containing model_source sources for which to find availability, or single Source for which to find availability.
    model_source : str, list of strings, optional
        Specified model_source(s) for which to find the availability for a catalog. If unspecified and cat_or_source is a Catalog, loop over all sources in catalog and find availability for all.
    override : boolean, optional
        Use `override=True` to find availability regardless of freshness.
    verbose : boolean, optional
        If True, `start_datetime` and `end_datetime` found for each Source will be printed.

    Returns
    -------
    Intake catalog or source
        If a Catalog was input, a Catalog will be returned; if a Source was input, a Source will be returned. For the single input Source or all Sources in the input Catalog, `start_datetime` and `end_datetime` are added to metadata.

    Examples
    --------

    Set up source catalog, then find availability for all sources of CIOFS model:

    >>> main_cat = mc.setup()
    >>> cat = mc.find_availability(main_cat['CIOFS'], model_source=['coops-forecast-agg', 'coops-forecast-noagg'])

    Find availability for only model_source "coops-forecast-noagg" of CBOFS model, and print it:

    >>> source = mc.find_availability(main_cat['CBOFS']['coops-forecast-noagg'], verbose=True)
    coops-forecast-noagg: 2022-08-22 13:00:00 to  2022-09-25 12:00:00
    """

    # Check in case user input main_catalog which is not correct
    # if both the input obj and the items one layer down, contained in obj, are both catalogs, then
    # a nested catalog was input which is not correct
    if isinstance(cat_or_source, Catalog) and isinstance(
        cat_or_source[list(cat_or_source)[0]], Catalog
    ):
        raise ValueError(
            "A nested catalog was input, but should be either a catalog that contains sources instead of catalogs, or a source. For example, try `main_cat['CBOFS']` or `main_cat['CBOFS']['coops-forecast-agg']`."
        )

    # if Catalog was input
    if isinstance(cat_or_source, Catalog):
        # if no model_source input, loop through all
        model_sources = (
            list(cat_or_source)
            if model_source is None
            else mc.astype(model_source, list)
        )
        sources = []
        for model_source in model_sources:
            try:
                source = find_availability_source(
                    source=cat_or_source[model_source], override=override
                )
            except requests.exceptions.HTTPError:
                print(f"Found a server error with {model_source}")
                source = cat_or_source[model_source]
            # source = find_availability_source(
            #     source=cat_or_source[model_source], override=override
            # )
            sources.append(source)
            if verbose:
                print(
                    f"{source.name}: {source.metadata['start_datetime']} to {source.metadata['end_datetime']}"
                )

        # Make new catalog to remember the new metadata
        new_user_cat = mc.make_catalog(
            sources,
            full_cat_name=cat_or_source.name,
            full_cat_description=cat_or_source.description,
            full_cat_metadata=cat_or_source.metadata,
            cat_driver=[source._entry._driver for source in list(sources)],
            cat_path=None,
            save_catalog=False,
        )

        return new_user_cat

    # if Source input
    elif isinstance(cat_or_source, (OpenDapSource, DatasetTransform)):

        # doesn't make sense to input a model_source and source
        if model_source is not None:
            raise ValueError("A source was input, so `model_source` should be None.")

        source = find_availability_source(source=cat_or_source, override=override)

        if verbose:
            print(
                f"{source.name}: {source.metadata['start_datetime']} to  {source.metadata['end_datetime']}"
            )

        # Make new catalog to remember the new metadata, then extract source immediately
        out_source = mc.make_catalog(
            [source],
            full_cat_name=cat_or_source.cat.name,
            full_cat_description=cat_or_source.cat.description,
            full_cat_metadata=cat_or_source.cat.metadata,
            cat_driver=cat_or_source._entry._driver,
            cat_path=None,
            save_catalog=False,
        )[source.name]

        return out_source

    else:
        raise ValueError(
            f"Either an Intake catalog or Intake source should be input, but found object of type {type(cat_or_source)}."
        )


def transform_source(source_orig):
    """Set up transform of original catalog source

    Parameters
    ----------
    source_orig : Intake source
        Original source, which will be transformed

    Returns
    -------
    Intake source
        `source_transform`, the transformed version of source_orig. This source will point at the source of `source_orig` as the target.
    """

    # open the skeleton transform cat entry and then alter
    # a few things so can use it with source_orig
    source_transform = intake.open_catalog(mc.CAT_PATH_TRANSFORM)["name"]

    # Update name and description for transformed source
    source_transform.name = source_orig.name
    source_transform.description = (
        f"Catalog entry for transform of dataset {source_orig.name}"
    )

    # make path to source_orig the target
    source_transform._captured_init_kwargs["targets"] = [
        f"{source_orig.cat.path}:{source_orig.name}"
    ]
    source_transform.targets = [f"{source_orig.cat.path}:{source_orig.name}"]

    # add metadata from source_orig
    source_transform.metadata.update(source_orig.metadata)

    return source_transform


def select_date_range(
    cat_or_source,
    start_date,
    end_date=None,
    model_source=None,
    use_forecast_files=None,
    override=False,
):
    """For NOAA OFS unaggregated models: Update `urlpath` locations in `Source`.

    For other models, set up so that `start_date` and `end_date` are used to filter resulting Dataset in time. For all models, save `start_date` and `end_date` in the `Source` metadata.

    NOAA OFS model sources that require aggregation (currently for model_sources "coops-forecast-noagg" and "ncei-archive-noagg") need to have the specific file paths found for each file that will be read in. This function does that, based on the desired date range, and returns a `Source` with file locations in the `urlpath`. This function can also be used with any model that does not require this (because the model paths are either static or deterministic) but in those cases it does not need to be used; they will have the start and end dates applied to filter the resulting model output after ``to_dask()`` is called.

    Parameters
    ----------
    cat_or_source : Intake catalog or source
        Catalog containing model_source sources, or single Source.
    start_date: datetime-interpretable str or pd.Timestamp
        Date (and possibly time) of start to desired model date range. If input date does not include a time, times will be included from the start of the day. If a time is input in `start_date`, it is used to narrow the time range of the results.
    end_date: datetime-interpretable str, pd.Timestamp, or None; optional
        Date (and possibly time) of start to desired model date range. If input date does not include a time, times will be included from the start of the day. If a time is input in `start_date`, it is used to narrow the time range of the results. end_date can be None which indicates the user wants all available model output after start_date; this optional is not available for unaggregated historical NOAA OFS models which do not contain forecast files (i.e., model_source "ncei-archive-noagg").

        There are several use cases to specify:

        * if `start_date == end_date`, the full day of model output from the date is selected. If the date specified is today and all times for today are not yet available, output from forecast files will be used to fill out the day after the nowcast files end.
        * If `end_date is None`, all available model output will be retrieved starting at start_date. This option doesn't work for archival unaggregated NOAA OFS models currently.
        * If `end_date` is in the future, `use_forecast_files` is set to True and the forecast is read in, but stopped at `end_date`.
        * User can set `use_forecast_files=True` with an end_date in the past to get old forecast model results for end_date for unaggregated NOAA OFS models. This case is probably not well-used and is not regularly tested. The results from using this combination of inputs does not align with the results of ``mc.find_availability()`` since the forecast is not the latest.
    model_source: str, optional
        Which model_source to use. If ``mc.find_availability()`` has been run, the code will determine which model_source in the Catalog to use based on `start_date` and `end_date`. Otherwise a single model_source can be provided, or ``find_availability()`` will be run if needed. An exception is if there is only one model_source available for cat, that one will be used without specifying it.
    use_forecast_files : bool or None, optional
        This parameter is typically set by the code and is not used by the user. However, in one use case the user can input `use_forecast_files=True`: when they want to read in a forecast from the past for a NOAA OFS model. Otherwise do not use this parameter directly.
    override : boolean, optional
        Use `override=True` to find catrefs regardless of freshness.

    Returns
    -------
    Intake Source
        Intake `Source` associated with the catalog entry which now contains `source.metadata['start_date']` and `source.metadata['end_date']`. The values of `source.metadata['start/end_date']` will not necessarily be the same as the input `start_date` and `end_date`, but may be changed to return the desired output time range. For unaggregated NOAA OFS models, the returned Source will have updated `source.urlpath` to reflect the newly-found file paths of the selected date range.

    Examples
    --------

    Find model `'LMHOFS'` urlpaths for all of today through all available forecast, directly from source catalog without first searching for availability with ``mc.find_availability()``:

    >>> main_cat = mc.setup()
    >>> today = pd.Timestamp.today()
    >>> source = mc.select_date_range(main_cat["LMHOFS"]["coops-forecast-noagg"], start_date=today, end_date=None)

    Find urlpaths with ``select_date_range`` and have it run ``find_availability()``:

    >>> source = mc.select_date_range(main_cat['LMHOFS'], start_date=today, end_date=today)

    """

    # Check in case user input main_catalog which is not correct
    # if both the input obj and the items one layer down, contained in obj, are both catalogs, then
    # a nested catalog was input which is not correct
    if isinstance(cat_or_source, Catalog) and isinstance(
        cat_or_source[list(cat_or_source)[0]], Catalog
    ):
        raise ValueError(
            "A nested catalog was input, but should be either a catalog that contains sources instead of catalogs, or a source. For example, try `main_cat['CBOFS']` or `main_cat['CBOFS']['coops-forecast-agg']`."
        )

    # save these to determine if user input dates with times or not
    start_date_input, end_date_input = start_date, end_date

    # make sure start_date and end_date are both Timestamps
    start_date = mc.astype(start_date, pd.Timestamp)
    end_date = None if end_date is None else mc.astype(end_date, pd.Timestamp)

    today = pd.Timestamp.today()

    if end_date is None and model_source == "ncei-archive-noagg":
        raise KeyError(
            "model_source 'ncei-archive-noagg' does not have forecast files, so `end_date` should be a datetime representation."
        )

    if end_date is not None and end_date.date() < start_date.date():
        raise KeyError(
            f"`start_date` is {start_date} but needs to be earlier than `end_date` ({end_date})."
        )

    # upon input, use_forecast_files should only be True if end_date is not None and end_date < today
    # later, however, it will be set to True in several circumstances
    if use_forecast_files and (end_date is None or end_date.date() >= today.date()):
        warnings.warn(
            f"User should only input `use_forecast_files=True` to access a forecast from the past, however `end_date={end_date}`.",
            UserWarning,
        )

    # set start/end_date_sel, but end_date_sel is overwritten in one unusual case subsequently
    if start_date == end_date is not None:
        start_date_sel = str(start_date.date())
        end_date_sel = str(end_date.date())

    else:
        # If end_date_input contains the default input time options from dateutil, assume time not input
        # in which case use date only to retrieve the whole day of output when selecting
        # or end_date_input exactly matches start_date_input
        if (
            end_date_input is not None
            and parse(mc.astype(end_date_input, str), default=DEFAULT).strftime(
                "%H%M%S"
            )
            == "222222"
        ):
            # user didn't specify time or end_date is None
            end_date_sel = str(end_date.date())
        else:  # user specified time
            end_date_sel = str(end_date) if end_date is not None else None

        # same but for start_date at the beginning of the day
        if (
            parse(mc.astype(start_date_input, str), default=DEFAULT).strftime("%H%M%S")
            == "222222"
        ):
            start_date_sel = str(start_date.date())
        else:
            start_date_sel = str(start_date) if start_date is not None else None

    # set end_date_loop
    # if end_date is None, set use_forecast_files to True and end_date_loop to today
    # if end_date is after today, code set use_forecast_files to True and end_date_loop to today
    if (end_date is None) or (end_date.date() > today.date()):
        end_date_loop = today
        use_forecast_files = True
    # is end_date before today, set end_date_loop forward 1 day to get last few time steps of the day
    elif end_date.date() < today.date():
        # In only this case, since user inputs use_forecast_files as True,  end_date_loop should be end_date and should retrieve forecast from yesterday
        if use_forecast_files:
            end_date_loop = end_date
            # This is out of place! Overwrite end_date_sel with None
            end_date_sel = None
        else:
            end_date_loop = end_date + pd.Timedelta("1 day")
    # if end_date is today, set end_date_loop to end_date and use_forecast_files True to get the last few time steps of today
    elif end_date.date() == today.date():
        end_date_loop = end_date
        use_forecast_files = True

    # if Catalog was input, determine which model_source to use to get to a source
    if isinstance(cat_or_source, Catalog):

        cat = cat_or_source

        # if there is only one model_source, use it
        if model_source is None and len(list(cat)) == 1:
            model_source = list(cat)[0]

        # is model_source was input, make sure it is in the Catalog
        if model_source is not None and model_source not in list(cat):
            raise KeyError(
                f"User input model_source {model_source} but it is not a source in catalog {cat.name}."  # noqa: E501
            )

        elif model_source is None and any(
            [
                "start_datetime" not in cat[model_source].metadata
                or "end_datetime" not in cat[model_source].metadata
                for model_source in list(cat)
            ]
        ):
            warnings.warn(
                "`find_availability()` has not been run for this model — running now..."
            )
            cat = mc.find_availability(cat, verbose=True)
            # raise KeyError(
            #     "Run `mc.find_availability()` for this model before running this command. Otherwise input model_source that contains desired date range."  # noqa: E501
            # )

        # which source to use from catalog for desired date range
        if model_source is None:
            user_range = DateTimeRange(start_date, end_date)

            for model_source in list(cat):
                model_source_range = DateTimeRange(
                    cat[model_source].metadata["start_datetime"],
                    cat[model_source].metadata["end_datetime"],
                )
                try:  # use this model_source if it is in the date range
                    if user_range in model_source_range:
                        break
                except TypeError:
                    continue
            else:
                raise ValueError(
                    "date range does not fully fit into any model model_sources"
                )

        if use_forecast_files and model_source == "ncei-archive-noagg":
            raise KeyError(
                "model_source 'ncei-archive-noagg' does not have forecast files, so `use_forecast_files` should be False."
            )

        source = cat[model_source]

    # if Source input
    elif isinstance(cat_or_source, (OpenDapSource, DatasetTransform)):

        source = cat_or_source
        model_source = source.name

    # if server is not working, return input source with None for new metadata
    if not source.status:
        # raise RuntimeError(f"Server for source {source.cat.name}, {source.name}, is not working. Urlpath checked was {mc.astype(source.urlpath, list)[0]}.")
        warnings.warn(
            f"Server for source {source.cat.name}, {source.name}, is not working. Urlpath checked was {mc.astype(source.urlpath, list)[0]}.",
            RuntimeWarning,
        )
        return source

    # catch the models that require aggregation
    if "catloc" in source.metadata:
        pattern = source.metadata["pattern"] if "pattern" in source.metadata else None

        # check catalog
        if not mc.status(source.metadata["catloc"], suffix=""):
            raise RuntimeError(
                f"Catalog {source.metadata['catloc']} for source {source.cat.name}, {source.name}, is not working."
            )
            # warnings.warn(
            #     f"Catalog {source.metadata['catloc']} for source {source.cat.name}, {source.name}, is not working.",
            #     RuntimeWarning,
            # )
            # return None

        if not override and mc.is_fresh(
            mc.FILE_PATH_CATREFS(source.cat.name, source.name), source
        ):
            with open(
                mc.FILE_PATH_CATREFS(source.cat.name, source.name), "r"
            ) as stream:
                catrefs = yaml.safe_load(stream)["catrefs"]
        else:
            try:
                catrefs = mc.find_catrefs(source.metadata["catloc"])
            except Exception as e:
                print(e)
            catrefs = sorted(catrefs)  # earliest first, most recent last
            with open(
                mc.FILE_PATH_CATREFS(source.cat.name, source.name), "w"
            ) as outfile:
                yaml.dump({"catrefs": catrefs}, outfile, default_flow_style=False)

        # loop over dates
        filelocs_urlpath = []
        for date in pd.date_range(
            start=start_date.normalize(), end=end_date_loop, freq="1D"
        ):
            is_forecast = (
                True
                if date.date() == end_date_loop.date() and use_forecast_files
                else False
            )

            fname = mc.FILE_PATH_AGG_FILE_LOCS(
                source.cat.name, model_source, date, is_forecast
            )

            if not override and mc.is_fresh(fname, source):
                with open(fname, "r") as stream:
                    agg_filelocs = yaml.safe_load(stream)["agg_filelocs"]

            else:
                # translate date to catrefs to select which catref to use
                if len(catrefs[0]) == 3:
                    cat_ref_to_match = (
                        date.strftime("%Y"),
                        date.strftime("%m"),
                        date.strftime("%d"),
                    )
                elif len(catrefs[0]) == 2:
                    cat_ref_to_match = (date.strftime("%Y"), date.strftime("%m"))

                if cat_ref_to_match in catrefs:
                    ind = catrefs.index(cat_ref_to_match)
                else:
                    warnings.warn(
                        f"Probably the time range requested is not available for model {source.cat.name}, model_source {source.name}. Returning source now.",
                        RuntimeWarning,
                    )
                    return source

                filelocs = mc.find_filelocs(
                    catrefs[ind],
                    source.metadata["catloc"],
                    source.cat.metadata["filetype"],
                )

                agg_filelocs = mc.agg_for_date(
                    date,
                    filelocs,
                    source.cat.metadata["filetype"],
                    is_forecast,
                    pattern,
                )

                with open(fname, "w") as outfile:
                    yaml.dump(
                        {"agg_filelocs": agg_filelocs},
                        outfile,
                        default_flow_style=False,
                    )

            filelocs_urlpath.extend(agg_filelocs)

        # df is sorted and deduplicated
        df = mc.filedates2df(filelocs_urlpath)

        # Narrow the files used to the actual requested datetime range
        files_to_use = df[start_date_sel:end_date_sel]

        # get only unique files and change to list
        files_to_use = list(pd.unique(files_to_use["filenames"]))

        # This is how we input the newly found urlpaths in so they will be used
        # in the processing of the dataset, and overwrite the old urlpath
        source._captured_init_kwargs["transform_kwargs"]["urlpath"] = files_to_use

        # Then run the transform for urlpath to pass that info on
        source.update_urlpath()

    # Pass start and end dates to the transform so they can be implemented
    # there for static and deterministic model files (includes RTOFS) as well
    # as the OFS aggregated models.
    source._captured_init_kwargs["transform_kwargs"]["start_date"] = start_date_sel
    source._captured_init_kwargs["transform_kwargs"]["end_date"] = end_date_sel

    # store info in source_orig
    metadata = {
        "model_source": model_source,
        "start_date": start_date_sel,
        "end_date": end_date_sel,
    }
    source.metadata.update(metadata)
    # Add original overall model catalog metadata to this next version
    source.metadata.update(source.cat.metadata)

    return source
