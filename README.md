model_catalogs
==============================
[![Build Status](https://img.shields.io/github/workflow/status/NOAA-ORR-ERD/model_catalogs/Tests?logo=github&style=for-the-badge)](https://github.com/NOAA-ORR-ERD/model_catalogs/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/NOAA-ORR-ERD/model_catalogs.svg?style=for-the-badge)](https://codecov.io/gh/NOAA-ORR-ERD/model_catalogs)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/model_catalogs/latest.svg?style=for-the-badge)](https://model_catalogs.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/workflow/status/NOAA-ORR-ERD/model_catalogs/linting%20with%20pre-commit?label=Code%20Style&style=for-the-badge)](https://github.com/NOAA-ORR-ERD/model_catalogs/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/model_catalogs.svg?style=for-the-badge)](https://anaconda.org/conda-forge/model_catalogs)

Provides access through catalogs to a set of ocean models.

## Get code, setup environment, install package

### To use provided environment

Clone the repo:
``` bash
$ git clone http://github.com/NOAA-ORR-ERD/model_catalogs.git
```

In the `model_catalogs` directory, install conda environment:
``` bash
$ conda env create -f environment.yml
```

Alternatively, if you have an existing environment you want to add to:
``` bash
$ conda install --file conda_requirements.txt
$ pip install -r pip_requirements.txt
```

Install `model_catalogs` into new environment (still in `model_catalogs` directory):
``` bash
$ conda activate model_catalogs
$ pip install -e .
```

### To install with LibGOODS requirements

Clone the repo:
``` bash
$ git clone http://github.com/NOAA-ORR-ERD/LibGOODS.git
```

Navigate to the LibGOODS directory and then:
``` bash
conda create --name libgoods_env  # create new environment, if you want
conda activate libgoods_env  # activate whichever environment you want to use
conda install -c conda-forge mamba  # mamba installs packages fast
mamba install -c conda-forge --file libgoods/conda_requirements.txt  # install LibGOODS conda requirements
mamba install -c conda-forge --file model_catalogs/conda_requirements.txt  # install model_catalogs conda requirements
pip install -r model_catalogs/pip_requirements.txt  # install model_catalogs pip requirements
```

Install `model_catalogs` locally into environment:
``` bash
cd model_catalogs
pip install -e .
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
