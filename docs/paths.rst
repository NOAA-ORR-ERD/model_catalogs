Catalog Paths
=============

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

Some information is cached in the user application cache directory when ``model_catalogs`` is run to save time if the same model/model_source combination is requested while it is still considered fresh information. All paths (as ``pathlib`` objects) and freshness definitions are stored in ``model_catalogs.__init__.py``. As an example, the base cache path is accessible as ``mc.CACHE_PATH``.
