import cf_xarray
import xarray as xr
from intake.source.derived import DerivedSource
# from intake.source.derived import GenericTransform
from intake.source import import_name
import numpy as np


# class DerivedSource(DataSource):
#     """Base source deriving from another source in the same catalog

#     Target picking and parameter validation are performed here, but
#     you probably want to subclass from one of the more specific
#     classes like ``DataFrameTransform``.
#     """

#     input_container = "other"  # no constraint
#     container = 'other'  # to be filled in per instance at access time
#     required_params = []  # list of kwargs that must be present
#     optional_params = {}  # optional kwargs with defaults

#     def __init__(self, targets, target_chooser=first, target_kwargs=None,
#                  cat_kwargs=None,
#                  container=None, metadata=None, **kwargs):
#         """

#         Parameters
#         ----------
#         targets: list of string or DataSources
#             If string(s), refer to entries of the same catalog as this Source
#         target_chooser: function to choose between targets
#             function(targets, cat) -> source, or a fully-qualified dotted string pointing
#             to it
#         target_kwargs: dict of dict with keys matching items of targets
#         cat_kwargs: to pass to intake.open_catalog, if the target is in
#             another catalog
#         container: str (optional)
#             Assumed output container, if known/different from input

#         [Note: the exact form of target_kwargs and cat_kwargs may be
#         subject to change]
#         """
#         self.targets = targets
#         self._chooser = (target_chooser if callable(target_chooser)
#                          else import_name(target_chooser))
#         self._kwargs = target_kwargs or {}
#         self._source = None
#         self._params = kwargs
#         self._cat_kwargs = cat_kwargs or {}
#         if container:
#             self.container = container
#         self._validate_params()
#         super().__init__(metadata=metadata)

#     def _validate_params(self):
#         """That all required params are present and that optional types match"""
#         assert set(self.required_params) - set(self._params) == set()
#         for par, val in self.optional_params.items():
#             if par not in self._params:
#                 self._params[par] = val

#     def _pick(self):
#         """ Pick the source from the given targets """
#         self._source = self._chooser(self.targets, self.cat, self._kwargs,
#                                      self._cat_kwargs)
#         if self.input_container != "other":
#             assert self._source.container == self.input_container

#         self.metadata['target'] = self._source.metadata
#         if self.container is None:
#             self.container = self._source.container


class GenericTransform(DerivedSource):
    required_params = ["transform", "transform_kwargs"]
    optional_params = {"allow_dask": True}
    """
    Perform an arbitrary function to transform an input

        transform: function to perform transform
            function(container_object) -> output, or a fully-qualified dotted string pointing
            to it
        transform_params: dict
            The keys are names of kwargs to pass to the transform function. Values are either
            concrete values to pass; or param objects which can be made into widgets (but
            must have a default value) - or a spec to be able to make these objects.
        allow_dask: bool (optional, default True)
            Whether to_dask() is expected to work, which will in turn call the
            target's to_dask()
    """

    def _validate_params(self):
        super()._validate_params()
        transform = self._params["transform"]
        self._transform = (transform if callable(transform)
                           else import_name(transform))

    def _get_schema(self):
        """We do not know the schema of a generic transform"""
        self._pick()
        return Schema()

    def to_dask(self):
        self._get_schema()
        if not self._params['allow_dask']:
            raise ValueError("This transform is not compatible with Dask"
                             "because it has use_dask=False")
        return self._transform(self._source.to_dask(), **self._params["transform_kwargs"])

    def read(self):
        self._get_schema()
        return self._transform(self._source.read(), **self._params["transform_kwargs"])

    def add_metadata(self):
        pass



class DatasetTransform(GenericTransform):
    """Transform where the input and output are both Dask-compatible Datasets

    This derives from GenericTransform, and you must supply ``transform`` and
    any ``transform_kwargs``.
    """

    input_container = "xarray"# "source"# "catalog"# "xarray"
    container = "xarray"# "source"# "catalog"# "xarray"
    optional_params = {}
    _ds = None
    # print('TEST')

    # add_metadata()

    # def __init__(self, metadata=None, targets=None, transform=None, transform_kwargs=None):
    #     import pdb; pdb.set_trace()


    # def __init__(self, metadata=None, targets=None, transform=None, transform_kwargs=None):
    #     # Do init here with a and b
    #     super(DatasetTransform, self).__init__(
    #                                         metadata=metadata, targets=None, transform=None, transform_kwargs=None
    #                                 )

#     def add_metadata(self):
#         self.metadata.update(self.cat[self.__dict__['targets'][0]].metadata)
#         self.description = self.cat[self.__dict__['targets'][0]].description

#         return self


    # # class C(B):
    # def __init__(self):
    #     super(DatasetTransform, self).__init__(metadata='blah')


    # def __init__(self, metadata=None, targets=None, transform=None, transform_kwargs=None):
    #     super().__init__()
    #     import pdb; pdb.set_trace()
    #     # print('rob is wrong')
    # super(self).__init__(self.description = self.cat[self.__dict__['targets'][0]].description)

    # import pdb; pdb.set_trace()
    # super().__init__():
    # def __init__(self, metadata='blah', targets='blah2'):
    #     pass
    #     # self.metadata.update(self.cat[self.__dict__['targets'][0]].metadata)
    #     # self.description = self.cat[self.__dict__['targets'][0]].description

    def to_dask(self):
        # self.metadata.update(self.cat[self.__dict__['targets'][0]].metadata)
        # # import pdb; pdb.set_trace()
        # self.description = self.cat[self.__dict__['targets'][0]].description
        if self._ds is None:
            self._pick()
            self._ds = self._transform(self._source.to_dask(),
                                       **self._params["transform_kwargs"])
        return self._ds

    # def _get_schema(self):
    #     """load metadata only if needed"""
    #     metadata = {
    #         'dims': dict(self._ds.dims),
    #         'data_vars': {k: list(self._ds[k].coords)
    #                       for k in self._ds.data_vars.keys()},
    #         'coords': tuple(self._ds.coords.keys()),
    #     }
    #     return Schema(datashape=None,
    #                   dtype=self._ds.dtypes,
    #                   shape=None,
    #                   npartitions=self._ds.npartitions,
    #                   metadata=metadata)
    #     # return Schema(dtype=self._ds.dtypes,
    #     #               shape=(None, len(self._ds.columns)),
    #     #               npartitions=self._ds.npartitions,
    #     #               metadata=self.metadata)

    def read(self):
        return self.to_dask()

#     def metadata(self):
#         self.metadata.update(**self._params["transform_kwargs"])
#         return self.metadata
#         # self.metadata


#

def add_attributes(ds, axis, standard_names):
    """Also calculate depths?
    Do currents needs to be rotated?

    """
    # axis = eval(axis)
    # standard_names = eval(standard_names)

    # set axis attributes (time, lon, lat, z potentially)
    for ax_name, var_names in axis.items():
        if not isinstance(var_names, list):
            var_names = [var_names]
        for var_name in var_names:
            # var_name needs to be a coord to have attributes
            if var_name not in ds.coords:
                ds[var_name] = (var_name, np.arange(ds.sizes[var_name]),
                                {"axis": ax_name})
            else:
                ds[var_name].attrs["axis"] = ax_name

    # set standard_names for all variables
    for stan_name, var_name in standard_names.items():
        ds[var_name].attrs['standard_name'] = stan_name

    # this won't run for e.g. GFS which has multiple time variables
    # but also doesn't need to have the calendar updated
    try:
        attrs = ds[ds.cf['T'].name].attrs
        if ('calendar' in attrs) and (attrs['calendar'] == 'gregorian_proleptic'):
            attrs['calendar'] = "proleptic_gregorian"
            ds[ds.cf['T'].name].attrs = attrs
    except KeyError:
        pass

    return xr.decode_cf(ds)


# def nothing(ds):
#     # ds.time.calendar = "proleptic_gregorian"
#     return ds#xr.decode_cf(ds)
