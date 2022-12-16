model_catalogs
==============================
[![Build Status](https://img.shields.io/github/actions/workflow/status/NOAA-ORR-ERD/model_catalogs/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/axiom-data-science/model_catalogs/actions/workflows/test.yaml)
[![Code Coverage](https://img.shields.io/codecov/c/github/NOAA-ORR-ERD/model_catalogs.svg?style=for-the-badge)](https://codecov.io/gh/NOAA-ORR-ERD/model_catalogs)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/model_catalogs/latest.svg?style=for-the-badge)](https://model_catalogs.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/actions/workflow/status/NOAA-ORR-ERD/model_catalogs/linting.yaml?branch=main&label=Code%20Style&style=for-the-badge)](https://github.com/NOAA-ORR-ERD/model_catalogs/actions/workflows/linting.yaml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/model_catalogs.svg?style=for-the-badge)](https://anaconda.org/conda-forge/model_catalogs)
[![Python Package Index](https://img.shields.io/pypi/v/model_catalogs.svg?style=for-the-badge)](https://pypi.org/project/model_catalogs)


Provides access through Intake catalogs to a set of ocean models, especially the NOAA OFS models. In particular, this package is good for working with unaggregated NOAA OFS models.

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
  * RTOFS models
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

### PyPI

To install from PyPI:
``` base
pip install model_catalogs
```

### conda-forge

To install from conda with channel conda-forge:
``` base
conda install -c conda-forge model_catalogs
```


### Install Optional Dependencies

Install additional dependencies for full functionality and running the demonstration notebooks. Activate your Python environment, then:

``` bash
$ mamba install -c conda-forge --file conda-requirements-opt.txt
```
or use `conda` in place of `mamba` if you don't have `mamba` installed.


## Develop Package

### Choose environment approach

#### Use provided environment

Clone the repo:
``` bash
$ git clone http://github.com/NOAA-ORR-ERD/model_catalogs.git
```

In the `model_catalogs` directory, install conda environment:
``` bash
$ conda env create -f environment.yml
```

Install `model_catalogs` into new environment (still in `model_catalogs` directory):
``` bash
$ conda activate model_catalogs
$ pip install -e .
```

#### Use other environment

Alternatively, if you have an existing environment you want to add to, clone the repo:
``` bash
$ git clone http://github.com/NOAA-ORR-ERD/model_catalogs.git
$ cd model_catalogs
```

Make sure the desired environment is activated and then:
``` bash
$ conda install -c conda-forge --file conda-requirements.txt
$ pip install -r pip-requirements.txt
```

Install `model_catalogs` into the environment (still in `model_catalogs` directory):
``` bash
$ pip install -e .
```

### Install development packages

To develop the code, follow instructions above for "Use provided environment" or "Use other environment" as appropriate. Then you can install additional dependencies for development and testing with
``` bash
$ conda install -c conda-forge --file conda-requirements-dev.txt
```


#### Run tests

Run tests that haven't been marked as "slow" with
``` bash
$ pytest
```

Run all tests, including slow tests, with:
``` bash
$ pytest --runslow
```


#### Check precommits locally before pushing

To then check code before committing and pushing it to github, locally run
``` bash
$ pre-commit run --all-files
```
These checks can change your files so it is best to check the changes before pushing to github.


#### Compile docs

Compile the docs locally after having installed the developer packages (see "Install development packages") or after making the docs environment with
``` bash
$ conda env create -f docs/environment.yml
```
and activating that environment.

Navigate to the docs folder and build the html documentation with
``` bash
$ make html
```

Finally you can make sure the documentation looks right by opening "_build/html/index.html".
