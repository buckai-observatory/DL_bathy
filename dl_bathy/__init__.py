"""DL_bathy: deep-learning satellite-derived bathymetry from Sentinel-2 imagery."""

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("dl-bathy")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0+unknown"
