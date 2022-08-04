"""
Everything dealing with the catalogs.
"""

import os

from copy import deepcopy

import cf_xarray  # noqa
import intake
import intake.source.derived
import pandas as pd
import yaml
from pathlib import Path
from datetimerange import DateTimeRange

from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry

import model_catalogs as mc


def make_catalog(
    cats,
    full_cat_name,
    full_cat_description,
    full_cat_metadata,
    cat_driver,
    cat_path=None,
    save_catalog=True,
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
    cat_path: Path object, optional
       Path with catalog name to use for saving catalog. With or without yaml suffix. If not provided,
       will use `full_cat_name`.
    save_catalog : bool, optional
        Defaults to True, and saves to cat_path.

    Returns
    -------

    Intake catalog.

    Examples
    --------

    Make catalog:

    >>> make_catalog([list of catalogs], 'catalog name', 'catalog desc', {}, 'opendap')
    """

    if cat_path is None and save_catalog:
        cat_path = Path(full_cat_name)
    if save_catalog:
        cat_path = cat_path / full_cat_name.lower()
        # cat_path = f"{cat_path}/{full_cat_name.lower()}"
    if save_catalog and ("yaml" not in str(cat_path)) and ("yml" not in str(cat_path)):
        cat_path = cat_path.with_suffix(".yaml")

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
        cat.save(cat_path)

    return cat


def setup(override=False):
    """Setup reference catalogs for models.

    Loops over hard-wired "orig" catalogs available in mc.CATALOG_PATH_DIR_ORIG, reads in previously-saved model boundary information, saves temporary catalog files for each model, and links those together into the returned master catalog. For some models, reading in the original catalogs applies a "today" and/or "yesterday" date Intake user parameter that supplies two example model files that can be used for examining the model output for the example times. Those are rerun each time this function is rerun, filling the parameters using the proper dates.

    Parameters
    ----------
    override : boolean, optional
        Use `override=True` to compile the catalog files together regardless of freshness.

    Returns
    -------
    Nested Intake catalog with a source for each model in mc.CATALOG_PATH_DIR_ORIG. Each source/model in turn has a source for each timing available (e.g., "forecast", "hindcast").

    Examples
    --------
    Set up master catalog:
    >>> cat = mc.setup()

    Examine list of sources/models available in catalog:
    >>> list(cat)

    Examine the sources for a specific model in the catalog:
    >>> list(cat['CBOFS'])
    """

    cat_transform_locs = []
    # Loop over all hard-wired original catalog files, one per model
    for cat_loc in mc.CATALOG_PATH_DIR_ORIG.glob("*.yaml"):

        base = mc.CATALOG_PATH_DIR_COMPILED
        filename = base / cat_loc.name

        # re-compile together catalog file if user wants to override possibly
        # existing file or if is not fresh
        if override or not mc.is_fresh(filename):

            cat_orig = intake.open_catalog(cat_loc)

            # add previously-saved boundary info
            # this was calculated with mc.calculate_boundaries()
            fname = mc.CATALOG_PATH_DIR_BOUNDARY / cat_loc.name
            with open(fname, "r") as stream:
                boundary = yaml.safe_load(stream)
            # add to cat_orig metadata
            cat_orig.metadata['bounding_box'] = boundary['bbox']
            cat_orig.metadata['geospatial_bounds'] = boundary['wkt']

            # get transform of each original catalog file, which points to
            # original file but applies metadata from original catalog file
            # to the resulting dataset after calling `to_dask()`
            source_transforms = [mc.transform_source(cat_orig[timing]) for timing in list(cat_orig)]

            # need to make catalog to transfer information properly from
            # source_orig to source_transform
            mc.make_catalog(
                source_transforms,
                full_cat_name=cat_orig.name,  # model name
                full_cat_description=cat_orig.description,
                full_cat_metadata=cat_orig.metadata,
                cat_driver=mc.process.DatasetTransform,
                cat_path=base,
                save_catalog=True
            )
        cat_transform_locs.append(filename)

    # have to read these from disk in order to make them type
    # intake.catalog.local.YAMLFileCatalog
    # instead of intake.catalog.base.Catalog
    cats = [intake.open_catalog(loc) for loc in cat_transform_locs]

    # make master nested catalog
    cat = mc.make_catalog(
        cats,
        full_cat_name=mc.SOURCE_CATALOG_NAME,
        full_cat_description="Source catalog for models.",
        full_cat_metadata={"source_catalog_dir": str(mc.CATALOG_PATH_DIR_ORIG)},
        cat_driver=intake.catalog.local.YAMLFileCatalog,
        cat_path=None,
        save_catalog=False
    )

    return cat


def calculate_boundaries(file_locs=None, save_files=True):
    """Calculate boundary information for all models.

    This loops over all catalog files available in mc.CATALOG_PATH_DIR_ORIG, tries first with forecast source and then with nowcast source if necessary to access the example model output files and calculate the bounding box and numerical domain boundary. The numerical domain boundary is calculated using `alpha_shape` with previously-chosen parameters stored in the original model catalog files. The bounding box and boundary string representation (as WKT) are then saved to files.

    The files that are saved by running this function have been previously saved into the repository, so this function should only be run if you suspect that a model domain has changed.

    Parameters
    ----------
    file_locs : Path, list of Paths, optional
        List of Path objects for model catalog files to read from. If not input, will use all catalog files available at mc.CATALOG_PATH_DIR_ORIG.glob("*.yaml").
    save_files : boolean, optional
        Whether to save files or not. Defaults to True. Saves to mc.CATALOG_PATH_DIR_BOUNDARY / cat_loc.name.

    Examples
    --------
    Calculate boundary information for all available models:
    >>> mc.calculate_boundaries()

    Calculate boundary information for CBOFS:
    >>> mc.calculate_boundaries([mc.CATALOG_PATH_DIR_ORIG / "cbofs.yaml"])
    """

    if file_locs is None:
        file_locs = mc.CATALOG_PATH_DIR_ORIG.glob("*.yaml")
    else:
        file_locs = mc.astype(file_locs, list)

    # loop over all orig catalogs
    for cat_loc in file_locs:

        # open model catalog
        cat_orig = intake.open_catalog(cat_loc)

        # try with forecast but if it doesn't work, use nowcast
        # this avoids problematic NOAA OFS aggregations when they are broken
        try:
            timing = "forecast"
            source_orig = cat_orig[timing]
            source_transform = transform_source(source_orig)

            # need to make catalog to transfer information properly from
            # source_orig to source_transform
            cat_transform = mc.make_catalog(
                source_transform,
                full_cat_name=cat_orig.name,  # model name
                full_cat_description=cat_orig.description,
                full_cat_metadata=cat_orig.metadata,
                cat_driver=mc.process.DatasetTransform,
                cat_path=None,
                save_catalog=False
            )

            # read in model output
            ds = cat_transform[timing].to_dask()

        except OSError:
            timing = "nowcast"
            source_orig = cat_orig[timing]
            source_transform = transform_source(source_orig)

            # need to make catalog to transfer information properly from
            # source_orig to source_transform
            cat_transform = mc.make_catalog(
                source_transform,
                full_cat_name=cat_orig.name,  # model name
                full_cat_description=cat_orig.description,
                full_cat_metadata=cat_orig.metadata,
                cat_driver=mc.process.DatasetTransform,
                cat_path=None,
                save_catalog=False
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
            with open(mc.boundary_filename(cat_loc.name), 'w') as outfile:
                yaml.dump({'bbox': bbox, 'wkt': wkt}, outfile, default_flow_style=False)


def find_datetimes(source, find_start_datetime, find_end_datetime, override=False):
    """Find the start and/or end datetimes for source.

    For sources with static urlpaths, this opens the Dataset and checks the first time for start_datetime and the last time for end_datetime. Some NOAA OFS models require aggregation: some forecasts, all nowcasts, and all hindcasts. For these, the available year and months of the thredd server subcatalogs are found with `find_catrefs()`. `start_datetime` is found by further evaluating to make sure that files in the subcatalogs are both available on the page and that the days represented by model output files are consecutive (there are missing dates). `end_datetime` is found from the most recent subcatalog files since there aren't missing files and dates on the recent end of the time ranges.

    Uses `cf-xarray` to determine the time axis.

    Parameters
    ----------
    source : Intake source
        Model source for which to find start and/or end datetimes
    find_start_datetime : bool
        True to calculate start_datetime, otherwise returns None
    find_start_datetime : bool
        True to calculate start_datetime, otherwise returns None
    override : boolean, optional
        Use `override=True` to find catrefs regardless of freshness. This is passed in from `find_availability()` so has the same value as input there.

    Returns
    -------
    (start_datetime, end_datetime) where each are strings or can be None if they didn't need to be found.
    """

    filetype = source.cat.metadata['filetype']

    # For any model/timing pairs with static links or known file address,
    # which is all non-OFS models and OFS models that are already aggregated
    if "catloc" not in source.metadata:

        try:
            ds = source.to_dask()
            # import pdb; pdb.set_trace()
            # use one T in case there are more than one
            start_datetime = str(ds[ds.cf.axes['T'][0]].values[0]) if find_start_datetime else None
            end_datetime = str(ds[ds.cf.axes['T'][0]].values[-1]) if find_end_datetime else None
            ds.close()
        except OSError:
            print(f"Model {source.cat.name} with timing {source.name} cannot connect to server.")
            return None, None

    # for when we need to aggregate which is OFS models nowcast and hindcast
    # and forecast if there is no pre-made aggregation
    else:
        if not override and mc.is_fresh(mc.catrefs_filename(source.cat.name, source.name)):
            with open(mc.catrefs_filename(source.cat.name, source.name), "r") as stream:
                catrefs = yaml.safe_load(stream)['catrefs']
        else:
            catrefs = mc.find_catrefs(source.metadata["catloc"])
            catrefs = sorted(catrefs)  # earliest first, most recent last
            with open(mc.catrefs_filename(source.cat.name, source.name), 'w') as outfile:
                yaml.dump({'catrefs': catrefs}, outfile, default_flow_style=False)

        if find_start_datetime:
            # Getting start date #
            # first make sure the dates actually have model files available
            filelocs = []
            i = 0
            while len(filelocs) == 0:
                # print(catrefs[i])
                filelocs = sorted(mc.find_filelocs(catrefs[i], source.metadata["catloc"], filetype=filetype))
                i += 1

            # second make sure we only count when dates are consecutive, since servers tend to have some spotty
            # model output at the earliest dates
            # get dates from file names
            all_dates = [pd.to_datetime(fileloc, format="%Y%m%d", exact=False) for fileloc in filelocs]
            unique_dates = list(set(all_dates))

            # if any dates are not consecutive, need to start after that date
            df = pd.Series(unique_dates)
            ddf = df.diff() > pd.Timedelta('1d')  # which differences in consecutive dates are over 1 day
            if ddf.any():
                # first date after last jump in dates is desired start day
                start_day = df.where(ddf).dropna().iloc[-1]
                # subset filelocs to match discovered consecutive dates
                df_filelocs = pd.Series(index=all_dates, data=filelocs)
                filelocs_ss = list(df_filelocs.where(df_filelocs.index >= start_day).dropna().values)

                # want first nowcast file (no forecast files available)
                start_datetime = str(mc.get_dates_from_ofs(filelocs_ss, filetype, "n", "first"))

            # all dates were fine, so just use earliest fileloc
            else:
                # running the following gives the actual first time. This might not be necessary in which case
                # want first nowcast file (no forecast files available)
                start_datetime = str(mc.get_dates_from_ofs(filelocs, filetype, "n", "first"))
                # # just use earliest day date
                # start_datetime = df.iloc[0]
        else:
            start_datetime = None

        if find_end_datetime:
            # Getting end date #
            filelocs = mc.find_filelocs(catrefs[-1], source.metadata["catloc"], filetype=filetype)
            # want last file
            if source.name == "hindcast":
                norf = "n"
            elif source.name == "nowcast":
                norf = "f"
            end_datetime = str(mc.get_dates_from_ofs(filelocs, filetype, norf, "last"))
        else:
            end_datetime = None

    # save start/end to files
    if find_start_datetime:
        with open(mc.start_filename(source.cat.name, source.name), 'w') as outfile:
            yaml.dump({'start_datetime': start_datetime}, outfile, default_flow_style=False)
    if find_end_datetime:
        with open(mc.end_filename(source.cat.name, source.name), 'w') as outfile:
            yaml.dump({'end_datetime': end_datetime}, outfile, default_flow_style=False)

    return start_datetime, end_datetime


def find_availability(cat, timings=None, override=False):
    """Find availability for model timings.

    The code will check for previously-calculated availability. If found, the "freshness" of the information is checked as compared with mc.FRESH parameters specified in `__init__`.

    Start and end datetimes are allowed to be calculated separately to save time.

    Parameters
    ----------
    cat : Intake catalog
        Catalog containing timing sources for which to find availability.
    timings : str, list of strings, optional
        Specified timing to find the availability for. If unspecified, loop over all timings and find availability for all.
    override : boolean, optional
        Use `override=True` to find availability regardless of freshness.

    Returns
    -------
    The input Intake catalog but with `start_datetime` and `end_datetime` added to metadata for the timings that were evaluated.

    Examples
    --------
    Setup source catalog, then find availability for all timings of CIOFS model:
    >>> source_cat = mc.setup()
    >>> cat = mc.find_availability(source_cat['CIOFS']))

    Find availability for only nowcast of CBOFS model:
    >>> cat = mc.find_availability(source_cat['CBOFS'], 'nowcast')
    """

    # if no timings input, loop through all
    timings = list(cat) if timings is None else mc.astype(timings, list)

    for timing in timings:
        # check if start and end datetime files already exist and are new enough to use
        # check if already know the time and not stale
        # file times are given in UTC
        # If files are not stale, read in info from there
        if not override and mc.is_fresh(mc.start_filename(cat.name, timing)):
            with open(mc.start_filename(cat.name, timing), "r") as stream:
                start_datetime = yaml.safe_load(stream)['start_datetime']
            find_start_datetime = False
        else:
            find_start_datetime = True  # need to still find the start_datetime

        if not override and mc.is_fresh(mc.end_filename(cat.name, timing)):
            with open(mc.end_filename(cat.name, timing), "r") as stream:
                end_datetime = yaml.safe_load(stream)['end_datetime']
            find_end_datetime = False
        else:
            find_end_datetime = True  # need to still find the end_datetime

        # start and end temp could be None, depending on values of
        # find_start_datetime, find_end_datetime
        if find_start_datetime or find_end_datetime:
            start_temp, end_temp = find_datetimes(cat[timing], find_start_datetime, find_end_datetime, override=override)

        start_datetime = start_temp if find_start_datetime else start_datetime
        end_datetime = end_temp if find_end_datetime else end_datetime

        cat[timing].metadata["start_datetime"] = start_datetime
        cat[timing].metadata["end_datetime"] = end_datetime

    # Make new catalog to remember the new metadata
    new_user_cat = mc.make_catalog(
        [cat[timing] for timing in list(cat)],
        full_cat_name=cat.name,
        full_cat_description=cat.description,
        full_cat_metadata=cat.metadata,
        cat_driver=[cat[timing]._entry._driver for timing in list(cat)],
        cat_path=None,
        save_catalog=False
    )

    return new_user_cat


def transform_source(source_orig):
    """Set up transform of original catalog source

    Parameters
    ----------
    source_orig : Intake source
        Original source, which will be transformed

    Returns
    -------
    source_transform, the transformed version of source_orig. This source will point at the source of source_orig as the target.
    """

    # open the skeleton transform cat entry and then alter
    # a few things so can use it with source_orig
    source_transform = intake.open_catalog(mc.SOURCE_TRANSFORM)["name"]

    # Update name and description for transformed source
    source_transform.name = source_orig.name
    source_transform.description = (
        f"Catalog entry for transform of dataset {source_orig.name}"
    )

    # make path to source_orig the target
    source_transform.__dict__["_captured_init_kwargs"]["targets"] = [f"{source_orig.cat.path}:{source_orig.name}"]

    # add metadata from source_orig
    source_transform.metadata.update(source_orig.metadata)

    # add yesterday if needed (some RTOFS models)
    if any(['yesterday' in d.values() for d in source_orig.describe()['user_parameters']]):
        yesterday = pd.Timestamp.today() - pd.Timedelta('1 day')
        # import pdb; pdb.set_trace()
        source_transform.__dict__["_captured_init_kwargs"]["transform_kwargs"]["yesterday"] = str(yesterday)[:10]

    return source_transform


def select_date_range(cat, timing=None, start_date=None, end_date=None, forecast_forward=True, override=False):
    """Add urlpath locations to existing catalog/source.

    MORE EXPLANATION

    Parameters
    ----------
    cat: Intake catalog
        An intake catalog for a specific model entry.
    timing: str, optional
        Which timing to use. If `find_availability` has been run, the code will
        determine whether `start_date`, `end_date` are in "forecast" or
        "hindcast". Otherwise timing must be provided for a single timing.
        Normally the options are "forecast", "nowcast", or "hindcast", and
        sometimes "hindcast-forecast-aggregation".
    start_date, end_date: datetime-interpretable str or pd.Timestamp, optional
        These two define the range of model output to include.
        If model has an aggregated link for timing, start_date and end_date
        are not used. Otherwise they should be input. Only year-month-day
        will be used in date. end_date is inclusive.
    forecast_forward : bool, optional
        Nowcast files are aggregated for the dates in the user-defined date range. However, if `forecast_forward==True`, the final date can have forecast files aggregated after the nowcast files to create a forecast going forward in time from the end date. The default is to include the forecast on the end (`forecast_forward==True`).
    override : boolean, optional
        Use `override=True` to find catrefs regardless of freshness.

    Returns
    -------
    Source associated with the catalog entry.

    Examples
    --------

    Find model 'LMHOFS' urlpaths directly from source catalog without first
    search for availability with `find_availability()`:
    >>> source_cat = mc.setup_source_catalog()
    >>> today = pd.Timestamp.today()
    >>> cat = source_cat["LMHOFS"]
    >>> source = mc.add_url_path(cat, timing="forecast",
                                 start_date=today, end_date=today)

    Find availability for model (for forecast and hindcast timings), then find
    urlpaths:
    >>> cat = mc.find_availability(model='LMHOFS')
    >>> today = pd.Timestamp.today()
    >>> source = mc.add_url_path(cat, start_date=today, end_date=today)

    """

    if timing is None and ('start_datetime' not in cat[timing].metadata or 'end_datetime' not in cat[timing].metadata):
        raise KeyError('Run `find_availability()` for this model before running this command. Otherwise input timing that contains desired date range.')

    # which source to use from catalog for desired date range
    if timing is None:
        user_range = DateTimeRange(start_date, end_date)

        for timing in list(cat):
            timing_range = DateTimeRange(cat[timing].metadata['start_datetime'], cat[timing].metadata['end_datetime'])
            try:  # use this timing if it is in the date range
                if user_range in timing_range:
                    break
            except TypeError:
                continue
        else:
            print("date range does not fully fit into any model timings")

    source = cat[timing]

    # catch the models that require aggregation
    if "catloc" in source.metadata:
        pattern = source.metadata["pattern"] if "pattern" in source.metadata else None

        if not override and mc.is_fresh(mc.catrefs_filename(source.cat.name, source.name)):
            with open(mc.catrefs_filename(source.cat.name, source.name), "r") as stream:
                catrefs = yaml.safe_load(stream)['catrefs']
        else:
            catrefs = mc.find_catrefs(source.metadata["catloc"])
            catrefs = sorted(catrefs)  # earliest first, most recent last
            with open(mc.catrefs_filename(source.cat.name, source.name), 'w') as outfile:
                yaml.dump({'catrefs': catrefs}, outfile, default_flow_style=False)

        # loop over dates
        filelocs_urlpath = []
        for date in pd.date_range(start=start_date, end=end_date, freq="1D"):
            is_forecast = True if date == pd.Timestamp(end_date) and forecast_forward else False

            # translate date to catrefs to select which catref to use
            if len(catrefs[0]) == 3:
                cat_ref_to_match = (date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'))
            elif len(catrefs[0]) == 2:
                cat_ref_to_match = (date.strftime('%Y'), date.strftime('%m'))

            ind = catrefs.index(cat_ref_to_match)

            filelocs = mc.find_filelocs(catrefs[ind], source.metadata["catloc"], cat.metadata["filetype"])

            filelocs_urlpath.extend(
                mc.agg_for_date(date, filelocs, cat.metadata["filetype"], is_forecast, pattern)
            )

        # This is how we input the newly found urlpaths in so they will be used
        # in the processing of the dataset, and overwrite the old urlpath
        source.__dict__["_captured_init_kwargs"]["transform_kwargs"]["urlpath"] = filelocs_urlpath

    # urlpath is already available if the link is consistent in time
    else:
        print(
            "`start_date` and `end_date` were not used since static link available."
        )  # noqa: E501
        # source_orig = source

    # store info in source_orig
    # source_orig.metadata["model"] = model
    metadata = {"timing": timing,
                "start_date": start_date if start_date is not None else None,
                "end_date": end_date if end_date is not None else None,
                }
    source.metadata.update(metadata)
    # Add original overall model catalog metadata to this next version
    source.metadata.update(cat.metadata)

    return source
