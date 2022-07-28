"""
Test process functions.
"""

import model_catalogs as mc

# what controls urlpath?
# NOT THIS:  source_transform._captured_init_kwargs['metadata']['urlpath']
# THIS DOES NOT SINCE NOT ALREADYD LOADED!!!!! source_transform._source.urlpath

def astype(value, type_):
    if not isinstance(value, type_):
        if type_ == list and type(value) == str:
            return [value]
        return type_(value)
    return value

# @pytest.mark.slow
def test_process():
    """Test all models for transform."""
    # CHECK:
    # cf-xarray identification

    source_cat = mc.setup_source_catalog()

    for model in ['CBOFS']:# list(source_cat):
        for timing in ['hindcast']:# list(source_cat[model]):
            print(model, timing)
            cat = source_cat[model]
            source_orig = cat[timing]
            
            # use local version of model output
            source_orig._captured_init_kwargs['urlpath'] = [f'model_catalogs/notebooks/test_files/{model}_{timing}.nc']
            
            cat_transform = mc.transform_source(source_orig)
            source_transform = cat_transform[model]
            # Replace dynamic urlpath with local static version
            # import pdb; pdb.set_trace()
            # source_transform._source.urlpath = f'model_catalogs/notebooks/test_files/{model}_{timing}.nc'
            ds = source_transform.to_dask()
            
            # check axis attributes have been assigned
            checks = [ds[var_name].attrs['axis'] == axis for axis, var_names in source_transform.metadata['axis'].items() for var_name in astype(var_names, list) if var_name in ds.dims]
            assert all(checks)
            
            # check standard_names attributes have been assigned
            checks = [ds[var_name].attrs['standard_name'] == st_name for st_name, var_names in source_transform.metadata['standard_names'].items() for var_name in astype(var_names, list) if var_name in ds.data_vars]
            assert all(checks)
            
            # check cf-xarray
            # AXIS X and Y won't be defined for unstructured model
            assert sorted(list(ds.cf.axes.keys())) == ['T', 'X', 'Y', 'Z']
            assert sorted(list(ds.cf.coordinates.keys())) == ['latitude', 'longitude', 'time', 'vertical']

            ds.close()
