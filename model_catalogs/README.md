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

Alternatively, if you have an existing environment you want to add to:
``` bash
$ conda install --file conda_requirements.txt
$ pip install -r pip_requirements.txt
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
Note that the slow tests are not run during CI.

Also note that when running tests locally, the conda environment is apparently not used for the tests unless you prefix the command as follows, where `model_catalogs` is the default name of the conda environment:

``` base
conda run -n model_catalogs pytest --runslow
```

## Set up to check linting locally

Install additional packages:
``` bash
$ conda install --file requirements-dev.txt
```

To then check code before committing and pushing it to github, locally run
``` bash
$ pre-commit run --all-files
```
These checks can change your files so it is best to check the changes before committing.
