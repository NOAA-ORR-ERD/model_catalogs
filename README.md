model_catalogs
==============================
[![Build Status](https://img.shields.io/github/workflow/status/NOAA-ORR-ERD/model_catalogs/Tests?logo=github&style=for-the-badge)](https://github.com/NOAA-ORR-ERD/model_catalogs/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/NOAA-ORR-ERD/model_catalogs.svg?style=for-the-badge)](https://codecov.io/gh/NOAA-ORR-ERD/model_catalogs)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/model_catalogs/latest.svg?style=for-the-badge)](https://model_catalogs.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/workflow/status/NOAA-ORR-ERD/model_catalogs/linting%20with%20pre-commit?label=Code%20Style&style=for-the-badge)](https://github.com/NOAA-ORR-ERD/model_catalogs/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/model_catalogs.svg?style=for-the-badge)](https://anaconda.org/conda-forge/model_catalogs)
[![Python Package Index](https://img.shields.io/pypi/v/model_catalogs.svg?style=for-the-badge)](https://pypi.org/project/model_catalogs)


Provides access through Intake catalogs to a set of ocean models, especially the NOAA OFS models.

Specific functionality includes:
* Sets up an `Intake` catalog for known models to provide direct access to model output.
* Provides access to model output as an `xarray` Dataset.
* Models are known by their catalog files; see set [here](https://github.com/NOAA-ORR-ERD/model_catalogs/tree/main/model_catalogs/catalogs/orig). They include
  * NOAA OFS Models:
    * CBOFS
    * CIOFS
    * CREOFS
    * DBOFS
    * GOMOFS
    * LEOFS
    * LMHOFS
    * LOOFS
    * NGOFS2
    * NYOFS
    * SFBOFS
    * TBOFS
    * WCOFS
    * Full 3D fields, or regularly gridded or 2D versions when available
  * GFS models
  * Global GOFS HYCOM
* Multiple time ranges and sources of model output are provided when known. For example for the NOAA OFS models there are both forecast and historical sources for all models, and some have others as well.
* `model_catalogs` knows how to aggregate NOAA OFS model output between nowcast and forecast files.
* Known models have cleaned up and filled-in metadata so they are easy to work with in `xarray` and with `cf-xarray`.
  * `cf-xarray` will understand dimension and coordinate names, as well as a set of standard_names mapped to the variables.
* Metadata about models is included in the `Intake` catalogs, such as:
  * polygon boundary of numerical domain
  * grid parameters
  * arguments for optimal read-in with `xarray`
* Can request the availability of each model source.


## Installation

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
$ conda install --file conda-requirements.txt
$ pip install -r pip-requirements.txt
```

Install `model_catalogs` into new environment (still in `model_catalogs` directory):
``` bash
$ conda activate model_catalogs
$ pip install -e .
```

### To install alongside LibGOODS requirements

Clone the `LibGOODS` repo:
``` bash
$ git clone http://github.com/NOAA-ORR-ERD/LibGOODS.git
```

Navigate to the `LibGOODS` directory and then:
``` bash
conda create --name libgoods_env  # create new environment, if you want
conda activate libgoods_env  # activate whichever environment you want to use
conda install -c conda-forge mamba  # mamba installs packages fast
mamba install -c conda-forge --file libgoods/conda_requirements.txt  # install LibGOODS conda requirements
```

Clone the `model_catalogs` repo in a good location:
``` bash
$ git clone http://github.com/NOAA-ORR-ERD/model_catalogs.git
```

Navigate to the `model_catalogs` directory, then:
``` bash
mamba install -c conda-forge --file conda-requirements.txt  # install model_catalogs conda requirements
pip install -r pip-requirements.txt  # install model_catalogs pip requirements
```

Install `model_catalogs` locally into environment:
``` bash
pip install -e .
```

### Install Optional Dependencies

Install additional dependencies for full functionality and running the demonstration notebooks. Activate your Python environment, then:

``` bash
$ mamba install -c conda-forge --file model_catalogs/conda-requirements-opt.txt
```
or use `conda` in place of `mamba` if you don't have `mamba` installed.


## Run demo

You can then open Jupyter lab from a terminal window with:
``` bash
$ jupyter lab
```

Then double-click the "demo.ipynb" notebook and run through the cells with "shift-enter".

## Develop Package

To develop the code, follow instructions above for "To use provided environment". Then you can install additional dependencies for development and testing with
``` bash
$ conda install --file requirements-dev.txt
```

### Run tests

Run tests that haven't been marked as "slow" with
``` bash
$ pytest
```

Run all tests, including slow tests, with:
``` bash
$ pytest --runslow
```
Note that the slow tests are not run during CI.

<!-- Also note that when running tests locally, the conda environment is apparently not used for the tests unless you prefix the command as follows, where `model_catalogs` is the default name of the conda environment:

``` base
conda run -n model_catalogs pytest --runslow
``` -->

### Check precommits locally before pushing

To then check code before committing and pushing it to github, locally run
``` bash
$ pre-commit run --all-files
```
These checks can change your files so it is best to check the changes before pushing to github.
