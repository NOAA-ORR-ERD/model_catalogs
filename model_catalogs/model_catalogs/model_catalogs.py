"""
Everything dealing with the catalogs.
"""

import os

from glob import glob

import cf_xarray  # noqa
import intake
import intake.source.derived
import pandas as pd

from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry

import model_catalogs as mc
# import multiprocessing
# from joblib import Parallel, delayed
# num_cores = multiprocessing.cpu_count()


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
    else:
        cat_path = f"{cat_path}/{full_cat_name.lower()}"
    if ("yaml" not in cat_path) and ("yml" not in cat_path):
        cat_path = f"{cat_path}.yaml"
    # import pdb; pdb.set_trace()
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
            # metadata=cat._yaml()["sources"][cat.name]["metadata"],
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


def setup_source_catalog():
    """Setup source catalog for models.

    Returns
    -------
    Source catalog `source_cat`.
    """

    cat_source_description = "Source catalog for models."

    # find most recent set of source_catalogs
    if not os.path.exists(mc.CATALOG_PATH_DIR):
        print(
            '"complete" model source files are not yet available. Run `model_catalogs.complete_source_catalog()` to create this directory.'  # noqa: E501
        )
        cat_dir = mc.CATALOG_PATH_DIR_ORIG
    else:
        cat_dir = mc.CATALOG_PATH_DIR

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
        mc.CATALOG_PATH
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
        # print(model)
        # save original metadata so as to not include Dataset attributes
        from copy import deepcopy
        metadata = deepcopy(source_cat[model][timing].metadata)
        # metadata = source_cat[model][timing].metadata

        # read in model output
        ds = source_cat[model][timing].to_dask()

        # find metadata
        # select lon/lat for use. There may be more than one and we also want the name.
        if "alpha_shape" in source_cat[model].metadata:
            dd, alpha = source_cat[model].metadata["alpha_shape"]
        else:
            dd, alpha = None, None
        lonkey, latkey, bbox, wkt = mc.find_bbox(ds, dd=dd, alpha=alpha)
        # lonkey, latkey, bbox, wkt_low, wkt_high = find_bbox(ds, dd=dd, alpha=alpha)

        # metadata for overall source_id0
        metadata0 = {
            "geospatial_bounds": wkt,
            "bounding_box": bbox,
        }
        ds.close()

        # # add Dataset metadata to specific source metadata
        # # change metadata attributes to strings so catalog doesn't barf on them
        # for attr in ds.attrs:
        #     source_cat[model][timing].metadata[attr] = str(ds.attrs[attr])
        # replace model, timing metadata to exclude Dataset attributes
        source_cat[model][timing].metadata = metadata

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
            f"{mc.CATALOG_PATH_DIR}",
        )
        # import pdb; pdb.set_trace()

    return setup_source_catalog()


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
    model = cat.name.upper()

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
    # determine filetype to send to `agg_for_date`
    if "regulargrid" in model.lower():
        filetype = "regulargrid"
    elif "2ds" in model.lower():
        filetype = "2ds"
    else:
        filetype = "fields"

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
        catloc = source.metadata['catloc']
        catrefs = mc.find_catrefs(catloc)
        # loop over dates
        filelocs_urlpath = []
        for date in pd.date_range(start=start_date, end=end_date, freq="1D"):
            if date == end_date and source.metadata['timing'] == 'forecast':
                is_forecast = True
            else:
                is_forecast = False

            # translate date to catrefs to select which catref to use
            if len(catrefs[0]) == 3:
                cat_ref_to_match = (str(date.year), str(date.month).zfill(2), str(date.day).zfill(2))
            elif len(catrefs[0]) == 2:
                cat_ref_to_match = (str(date.year), str(date.month).zfill(2))
            ind = catrefs.index(cat_ref_to_match)

            filelocs = mc.find_filelocs(catrefs[ind], catloc, filetype=filetype)
            # filepaths = source.metadata['all_filepaths']
            # base = source.metadata['all_filepaths_base']
            # # add base onto filepaths
            # filepaths = [f"{base}{filepath}.nc" for filepath in filepaths]
            # import pdb; pdb.set_trace()
            filelocs_urlpath.extend(
                mc.agg_for_date(
                    date,
                    filelocs,
                    source.metadata['filetype'],
                    is_forecast=is_forecast,
                    pattern=pattern,
                )
            )

        source_orig = source(urlpath=filelocs_urlpath)  # [:2])

    # urlpath is already available if the link is consistent in time
    else:
        source_orig = source

    return source_orig
