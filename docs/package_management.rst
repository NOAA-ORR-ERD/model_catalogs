Developer Guide
===============

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation


Get set up and write new code
-----------------------------

1. Fork ``model_catalogs`` to your GitHub account.
#. Clone the package from your fork on GitHub (following `these instructions <https://github.com/NOAA-ORR-ERD/model_catalogs#develop-package>`_).
#. Make your code changes in a branch.
#. You'll need to lint your code to pass the precommit tests. `Install the development packages <https://github.com/NOAA-ORR-ERD/model_catalogs#install-development-packages>`_ and `run with <https://github.com/NOAA-ORR-ERD/model_catalogs#check-precommits-locally-before-pushing>`_ `pre-commit run --all-files``.
#. `Run tests <https://github.com/NOAA-ORR-ERD/model_catalogs#run-tests>`_, including the slow tests, to make sure your code doesn't break anything.
#. When your code is ready, open a pull request from your fork and branch back to the base repo main branch.
#. Check the pull request checklist and be sure that the docs notebooks still run correctly and that you update "What's New" in "docs" with your changes.


Release steps
-------------

1. Make sure docs notebooks all run still and save to repo if there are changes.
#. Update the "What's New" doc page to the newest release version.
#. Go to the `package releases <https://github.com/NOAA-ORR-ERD/model_catalogs/releases>`_ and add a new one to match the What's New page.
#. PyPI will update to this version through a GitHub Action when the release is made.
