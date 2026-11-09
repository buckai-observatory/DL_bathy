"""Sentinel-2 band normalisation shared by `sdb split` and inference.py.

Patches written by `sdb split` hold normalised reflectance, and the model is
trained on those values, so inference must apply exactly the same transform
to raw scenes.
"""
import numpy as np


def normalize_reflectance(cube, log_scale=True, band_normalization=10000, epsilon=1e-6):
    """
    Normalise Sentinel-2 L2A reflectance the way `sdb split` does.

    Parameters
    ----------
    cube : np.ndarray
        Raw L2A digital numbers (reflectance x 10000), any shape.
    log_scale : bool
        ``Logcube`` setting: ``log(cube / band_normalization + epsilon)`` if
        True, else ``cube / band_normalization``.
    band_normalization : float
        ``band_normalization`` setting.
    epsilon : float
        ``Epsilon`` setting, which keeps the log finite at zero reflectance.

    Returns
    -------
    np.ndarray
        Normalised values, same shape. Pixels with no valid log (negative
        reflectance) become NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if log_scale:
            return np.log(cube / band_normalization + epsilon)
        return cube / band_normalization
