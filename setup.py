"""
Setup for model_catalogs.
"""

from setuptools import setup


setup(
    use_scm_version={
        # "root": "..",  # this is because the .git dir is back a dir
        "relative_to": __file__,
        "write_to": "model_catalogs/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    }
)
