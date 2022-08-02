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
import tempfile
from pathlib import Path

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
    from pathlib import Path

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
            cat.name.upper(),
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


# def setup_source_catalog(override=False):
#     """Setup source catalog for models.
#
#     Parameters
#     ----------
#     override: bool
#         Will use model catalog files available in "complete" directory if it is
#         available, or if `override==True` will always use "orig" directory to
#         set up source catalog.
#
#     Returns
#     -------
#     Intake catalog `source_cat`.
#     """
#
#     cat_source_description = "Source catalog for models."
#
#     # find most recent set of source_catalogs
#     if not os.path.exists(mc.CATALOG_PATH_DIR):
#         print(
#             '"complete" model source files are not yet available. Run `model_catalogs.complete_source_catalog()` to create this directory.'  # noqa: E501
#         )
#         cat_dir = mc.CATALOG_PATH_DIR_ORIG
#     elif override:
#         cat_dir = mc.CATALOG_PATH_DIR_ORIG
#     else:
#         cat_dir = mc.CATALOG_PATH_DIR
#
#     # # open catalogs
#     # cats = []
#     # for cat_loc in cat_dir.glob("*.yaml"):
#     #     cat = intake.open_catalog(cat_loc)
#     #     # cat.metadata['orig_catalog_location'] = str(cat_loc)
#     #     cats.append(cat)
#     cats = [intake.open_catalog(cat_loc) for cat_loc in cat_dir.glob("*.yaml")]
#
#     metadata = {"source_catalog_dir": str(cat_dir)}
#
#     return make_catalog(
#         cats,
#         mc.SOURCE_CATALOG_NAME,
#         cat_source_description,
#         metadata,
#         intake.catalog.local.YAMLFileCatalog,
#         mc.CATALOG_PATH,
#     )


def setup():
    """Setup reference catalogs for models.

    Loops over hard-wired "orig" catalogs available in mc.CATALOG_PATH_DIR_ORIG, reads in previously-saved model boundary information, saves temporary catalog files for each model, and links those together into the returned master catalog. For some models, reading in the original catalogs applies a "today" and/or "yesterday" date Intake user parameter that supplies two example model files that can be used for examining the model output for the example times. Those are rerun each time this function is rerun, filling the parameters using the proper dates.

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

    # store intermediary catalog files in temp directory
    base = Path(tempfile.gettempdir())

    cat_transform_locs = []
    # Loop over all hard-wired original catalog files, one per model
    for cat_loc in mc.CATALOG_PATH_DIR_ORIG.glob("*.yaml"):

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
        loc = (base / cat_orig.name.lower()).with_suffix(".yaml")
        cat_transform_locs.append(loc)

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


def calculate_boundaries():
    """Calculate boundary information for all models.

    This loops over all catalog files available in mc.CATALOG_PATH_DIR_ORIG, tries first with forecast source and then with nowcast source if necessary to access the example model output files and calculate the bounding box and numerical domain boundary. The numerical domain boundary is calculated using `alpha_shape` with previously-chosen parameters stored in the original model catalog files. The bounding box and boundary string representation (as WKT) are then saved to files.

    The files that are saved by running this function have been previously saved into the repository, so this function should only be run if you suspect that a model domain has changed.

    Examples
    --------
    Calculate boundary information for all available models:
    >>> mc.calculate_boundaries()
    """

    # loop over all orig catalogs
    for cat_loc in mc.CATALOG_PATH_DIR_ORIG.glob("*.yaml"):

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
        fname = mc.CATALOG_PATH_DIR_BOUNDARY / cat_loc.name
        with open(fname, 'w') as outfile:
            yaml.dump({'bbox': bbox, 'wkt': wkt}, outfile, default_flow_style=False)


def find_availability(model, override=False, override_updated=False):
    """Find availability for model for 'forecast' and 'hindcast'.

    Parameters
    ----------
    model: str
        Name of model, e.g., CBOFS
    override: bool
        Will use model catalog files available in "complete" directory if it is
        available, or if `override==True` will always use "orig" directory to
        set up source catalog.
    override_updated: bool
        Will use model "updated" catalog file if available in "updated"
        directory if it is not stale, or if `override==True` will remake updated
        catalog file regardless.

    Returns
    -------
    Intake catalog with some added metadata about the availability.

    Examples
    --------
    >> cat = mc.find_availability(model='DBOFS')
    """

    model = model.upper()

    ran_forecast, ran_hindcast = False, False

    complete_path = (mc.CATALOG_PATH_UPDATED / model.lower()).with_suffix(".yaml")
    # complete_path = f"{mc.CATALOG_PATH_UPDATED}/{model.lower()}.yaml"
    if os.path.exists(complete_path):
        cat = intake.open_catalog(complete_path)
    else:
        ref_cat = setup_source_catalog(override=override)
        cat = ref_cat[model]

    # deal with RTOFS completely separately
    if "RTOFS" in model:
        ds = cat["forecast"].to_dask()
        start_datetime = str(ds.time.values[0])
        end_datetime = str(ds.time.values[-1])
        cat["forecast"].metadata["start_datetime"] = start_datetime
        cat["forecast"].metadata["end_datetime"] = end_datetime
        cat_metadata = cat.metadata
        metadata = {
            "catalog_path": str(mc.CATALOG_PATH),
            # "source_catalog_name": mc.SOURCE_CATALOG_NAME,
            # "filetype": cat.metadata["filetype"]
        }
        cat_metadata.update(metadata)
        new_user_cat = mc.make_catalog(
            cat["forecast"],
            f"{model.upper()}",
            f"Model {model} with availability included.",
            cat_metadata,
            cat["forecast"]._entry._driver,
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

    timings = ["forecast", "hindcast"]
    # if both aren't available for cat, then this chooses those that are
    # hindcast isn't available for regulargrid
    timings = list(set(list(cat)).intersection(timings))

    new_sources = []
    for timing in timings:

        metadata = deepcopy(cat[timing].metadata)  # save metadata

        # forecast: don't need to check for consecutive dates bc files are by day
        # just find first file from earliest catref and last file from last catref
        if "stale" in cat[timing].metadata:
            stale = pd.Timedelta(cat[timing].metadata["stale"])
        else:
            stale = pd.Timedelta("1 minute")
        if "time_last_checked" in cat[timing].metadata:
            time_last_checked = pd.Timestamp(cat[timing].metadata["time_last_checked"])
        else:
            time_last_checked = pd.Timestamp.today() - pd.Timedelta(
                "30 days"
            )  # just a big number
        dt = pd.Timestamp.now() - time_last_checked

        if timing == "forecast" and (dt > stale or override_updated):

            if "catloc" in cat[timing].metadata:
                catloc = cat[timing].metadata["catloc"]
                catrefs = mc.find_catrefs(catloc)

                # find start_datetime. Have to loop bc there are fewer files for
                # e.g. filetype=='regulargrid'
                for catref in catrefs[::-1]:
                    filelocs = mc.find_filelocs(catref, catloc, filetype=filetype)
                    if len(filelocs) == 0:
                        continue
                start_datetime = mc.get_dates_from_ofs(filelocs, filetype, "n", "first")

                # find end_datetime
                filelocs = mc.find_filelocs(catrefs[0], catloc, filetype=filetype)
                end_datetime = mc.get_dates_from_ofs(filelocs, filetype, "f", "last")

            else:
                ds = cat["forecast"].to_dask()
                start_datetime = str(ds.time.values[0])
                end_datetime = str(ds.time.values[-1])

            ran_forecast = True
            # time_last_checked = pd.Timestamp.now()

        elif timing == "hindcast" and (dt > stale or override_updated):

            catloc = cat[timing].metadata["catloc"]
            catrefs = mc.find_catrefs(catloc)

            # Find start_datetime by checking catrefs from the old end [-1]
            for catref in catrefs[::-1]:
                filelocs = mc.find_filelocs(catref, catloc, filetype=filetype)
                if len(filelocs) == 0:
                    continue

                # determine unique dates
                dates = sorted(
                    list(
                        set(
                            [
                                pd.Timestamp(fileloc.split("/")[-1].split(".")[4])
                                for fileloc in filelocs
                            ]
                        )
                    )
                )

                # determine consecutive dates
                dates = [
                    dates[i]
                    for i in range(len(dates) - 2)
                    if (dates[i] + pd.Timedelta("1 day") in dates)
                    and (dates[i] + pd.Timedelta("2 days") in dates)
                ]
                if len(dates) > 0:
                    # keep filelocs if their date matches one in dates
                    # filelocs that don't exceed date range found
                    filelocs = [
                        fileloc
                        for fileloc in filelocs
                        if dates[0]
                        <= pd.Timestamp(fileloc.split("/")[-1].split(".")[4])
                        <= dates[-1]
                    ]
                    start_datetime = mc.get_dates_from_ofs(
                        filelocs, filetype, "n", "first"
                    )
                    break

            # find end_datetime, no need to search through files on this end of time
            filelocs = mc.find_filelocs(catrefs[0], catloc, filetype=filetype)
            end_datetime = mc.get_dates_from_ofs(filelocs, filetype, "n", "last")

            ran_hindcast = True
            # time_last_checked = pd.Timestamp.now()
        else:
            start_datetime = cat[timing].metadata["start_datetime"]
            end_datetime = cat[timing].metadata["end_datetime"]

        # stale parameter: 4 hours for forecast, 1 day for hindcast
        if timing == "forecast":
            stale = "4 hours"
        elif timing == "hindcast":
            stale = "1 day"

        # replace model, timing metadata to exclude Dataset attributes
        cat[timing].metadata = metadata

        metadata = {
            "model": model,
            "timing": timing,
            "filetype": filetype,
            "time_last_checked": str(pd.Timestamp.now()),
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
            "catalog_path": str(mc.CATALOG_PATH),
            # "source_catalog_name": mc.SOURCE_CATALOG_NAME,
            "filetype": new_sources[0].metadata["filetype"],
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


def add_url_path(cat, timing=None, start_date=None, end_date=None):
    """Add urlpath locations to existing catalog/source.

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

    # either `find_availability` needs to have been run and therefore certain
    # metadata present in cat (start_datetime, end_datetime), or don't need to
    #  have run `find_availability` but need to input which "timing" to use.
    dates = set(cat["forecast"].metadata).intersection(
        set(["start_datetime", "end_datetime"])
    )  # noqa
    assertion = 'Either `find_availability` needs to have been run and therefore certain metadata present in cat (start_datetime, end_datetime), or do not need to have run `find_availability` but need to input which "timing" to use.'  # noqa
    assert len(dates) > 1 or timing is not None, assertion

    # model = model.upper()
    model = cat.name.upper()

    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)

    # which source to use from catalog for desired date range
    if timing is None:
        if start_date >= pd.Timestamp(
            cat["forecast"].metadata["start_datetime"]
        ).normalize() and end_date <= pd.Timestamp(
            cat["forecast"].metadata["end_datetime"]
        ):
            timing = "forecast"
        elif (
            "hindcast" in list(cat)
            and start_date
            >= pd.Timestamp(cat["hindcast"].metadata["start_datetime"]).normalize()
            and end_date <= pd.Timestamp(cat["hindcast"].metadata["end_datetime"])
        ):
            timing = "hindcast"
        else:
            print("date range does not easily fit into forecast or hindcast")

    source = cat[timing]

    # RTOFS has special issues to form the paths for the model output available right now
    if "RTOFS-EAST" in model or "RTOFS-ALASKA" in model or "RTOFS-WEST" in model:
        # RTOFS needs to have yesterday's date input, then it is able to
        # create the necessary file names to get the model output
        source_orig = cat[timing](
            yesterday=pd.Timestamp.today() - pd.Timedelta("1 day")
        )

    elif "RTOFS-GLOBAL" in model or "RTOFS-GLOBAL_2D" in model:
        source_orig = cat[timing]

    # urlpath is None or a list of filler files if the filepaths need to be determined
    elif (
        source.urlpath is None or isinstance(source.urlpath, list)
    ) and "catloc" in source.metadata:

        if "pattern" in source.metadata:
            pattern = source.metadata["pattern"]
        else:
            pattern = None

        catloc = source.metadata["catloc"]
        catrefs = mc.find_catrefs(catloc)

        # loop over dates
        filelocs_urlpath = []
        for date in pd.date_range(start=start_date, end=end_date, freq="1D"):
            if date == end_date and (
                (
                    "timing" in source.metadata
                    and source.metadata["timing"] == "forecast"
                )
                or (timing == "forecast")
            ):
                is_forecast = True
            else:
                is_forecast = False

            # translate date to catrefs to select which catref to use
            if len(catrefs[0]) == 3:
                cat_ref_to_match = (
                    str(date.year),
                    str(date.month).zfill(2),
                    str(date.day).zfill(2),
                )
            elif len(catrefs[0]) == 2:
                cat_ref_to_match = (str(date.year), str(date.month).zfill(2))

            ind = catrefs.index(cat_ref_to_match)

            filelocs = mc.find_filelocs(
                catrefs[ind], catloc, filetype=cat.metadata["filetype"]
            )
            filelocs_urlpath.extend(
                mc.agg_for_date(
                    date,
                    filelocs,
                    cat.metadata["filetype"],
                    # source.metadata['filetype'],
                    is_forecast=is_forecast,
                    pattern=pattern,
                )
            )

        source_orig = source(urlpath=filelocs_urlpath)  # [:2])

    # urlpath is already available if the link is consistent in time
    else:
        print(
            "`start_date` and `end_date` were not used since static link available."
        )  # noqa: E501
        source_orig = source

    # save "orig" source to a tempfile
    # import tempfile
    # temp_dir = tempfile.TemporaryDirectory()
    # print(temp_dir.name)
    # fp = tempfile.NamedTemporaryFile()
    # temp_cat = make_catalog(
    #                 source_orig,
    #                 "User-catalog",
    #                 "User-made catalog",
    #                 {},#metadata,
    #                 source_orig._entry._driver,
    #                 cat_path=mc.CATALOG_PATH_TMP,
    #             )

    # store info in source_orig
    # source_orig.metadata["model"] = model
    source_orig.metadata["timing"] = timing
    source_orig.metadata["start_date"] = (
        start_date.isoformat() if start_date is not None else None,
    )
    source_orig.metadata["end_date"] = (
        end_date.isoformat() if end_date is not None else None,
    )
    # Add original overall model catalog metadata to this next version
    source_orig.metadata.update(cat.metadata)

    sources = transform_source(source_orig)

    new_cat = make_catalog(
        sources,
        "User-catalog.",
        "User-made catalog.",
        sources[1].metadata,  # this is where the most metadata is, but probably not important for cat  # noqa: E501
        [source._entry._driver for source in sources],
        cat_path=mc.CATALOG_PATH_TMP,
    )

    return new_cat
