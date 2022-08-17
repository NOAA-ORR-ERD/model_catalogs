# How to Update Boundaries

The boundary information for all known model domains (that is, one per catalog file in `orig`) has been previously calculated and saved into the `model_catalogs` repository since it should not change frequently. However, sometimes a new model will be added or an existing model will change, and this demonstrates how to calculate and save the new information.

## Run for all models

Run this way to run through all the model catalog files, calculate the boundary information, and save them each to files at `mc.CAT_PATH_BOUNDARY / cat_loc.name`, for example `mc.CAT_PATH_BOUNDARY / 'DBOFS.yaml'` for the DBOFS model. If you want to then update the repository with the newly-calculated boundary information, you can commit these files to the repository (if you are set up to develop the package).

```
import model_catalogs as mc
mc.calculate_boundaries(save_files=True)
```

## Run for one model

Run this way to run a single model. For example, perhaps you want to make sure that the CBOFS boundary information hasn't changed, so you run the boundary calculation and return the information without saving it to disk and then compare it with the existing boundary information.

```
boundaries = mc.calculate_boundaries([mc.CAT_PATH_ORIG / "cbofs.yaml"], save_files=False, return_boundaries=True)
```
