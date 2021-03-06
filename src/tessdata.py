"""Contains the TESSData class for working with TESS datasets with
1..n flux and error arrays (using different aperture size, etc.)

"""
from __future__ import division
import numpy as np

from numpy import array, atleast_2d, zeros, ones, ones_like, argmax, all, isfinite, tile, extract, zeros_like
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.signal import medfilt

from .ls import fasper
from .utils import medsig, fold
from .core import *

class TESSData(object):
    """Encapsulates the TESS data.

    Notes
    -----

    We remove the cadences where either time, x, or y is nan since input
    nans screw up the GP fitting and reduction. This must be accounted
    for when saving the data.

    Parameters
    ----------
    TIC     : int
              TIC number of the star
    time    : array_like
              ndarray or list containing the time values
    cadence : array_like
              ndarray or list containing the cadence values
    quality : array_like
              ndarray or list containing the quality values
    fluxes  : array_like
              ndarray or list containing the flux values
    errors  : array_like
              ndarray or list containing the flux uncertainties
    x       : array_like
              ndarray or list containing the x positions
    y       : array_like
              ndarray or list containing the y positions
    ftype   : string
              'sap' or 'pdc'

    Attributes
    ----------
    npoints     : int
                  Number of datapoints
    is_periodic : bool
                  Does the flux show clear periodic variability
    ls_period   : float
                  Period of the strongest periodic variability detected
    ls_power    : float
                  Lomb-Scargle power of the strongest periodic variability 
                  detected
    """
    
    def __init__(self, tic, time, cadence, quality, fluxes, errors, x, y, \
                     primary_header=None, data_header=None, sector=None, \
                     ftype='pdc'):
        self.tic = tic
        self.sector = sector
        self.nanmask = nm = isfinite(time) & isfinite(x) & isfinite(y)
        self.time = extract(nm, time)
        self.cadence =  extract(nm, cadence)
        self.quality =  extract(nm, quality).astype(np.int32)
        self.fluxes =  extract(nm, fluxes)
        self.errors = extract(nm, errors)
        self.x = extract(nm,x)
        self.y = extract(nm,y)
        self.ftype = ftype
        self.primary_header = primary_header
        self.data_header = data_header

        self.npoints = self.fluxes.shape

        self.is_periodic = False
        self.ls_period = None
        self.ls_power = None

        qmask = all(isfinite(self.fluxes),0) & (self.quality==0)
        self.mflags   = zeros(self.npoints, np.uint8)
        self.mflags[~qmask] |= M_QUALITY

    def mask_periodic_signal(self, center, period, duration):
        self.pmask = np.abs(fold(self.time, period, center, shift=0.5) - 0.5)*period > 0.5*duration
        self.mflags[~self.pmask] |= M_PERIODIC
        
