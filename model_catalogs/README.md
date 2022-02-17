# GOODS catalog

## Get code, setup environment, install package

Clone the repo:
``` bash
$ git clone http://git.axiom/NOAA-ORR-ERD/LibGOODS.git
```

In the `LibGOODS/model_catalogs` directory, install conda environment:
``` bash
$ conda env create -f environment.yml
```

Install `model_catalogs` into new environment (still in `LibGOODS/model_catalogs` directory):
``` bash
$ conda activate model_catalogs
$ pip install -e .
```

## Run demo

You can then open Jupyter lab from a terminal window with:
``` bash
$ jupyter lab
```

Then double-click the "demo.ipynb" notebook and run through the cells with "shift-enter".

## Run tests

Run tests that haven't been marked as "slow" with
``` bash
$ pytest
```

Run all tests, including slow tests, with:
``` bash
$ pytest --runslow
```
