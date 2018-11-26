"""Module for reading K2 data from different sources and writing k2sc FITS files.

    This module contains severals "readers" that can read a file containing a K2 light
    curve, and return a properly initialised K2Data instance. The module also contains
    a writer class for writing FITS files containing the K2 data and the detrending
    time series.

    Copyright (C) 2016  Suzanne Aigrain

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import warnings
import numpy as np
import re
import astropy.io.fits as pf
from os.path import basename, splitext
from datetime import datetime
from collections import namedtuple

from .tessdata import TESSData

warnings.resetwarnings()
warnings.filterwarnings('ignore', category=UserWarning, append=True)

## ===  READERS  ===
## =================

class DataReader(object):
    extensions = []
    ndatasets = 0
    fn_out_template = None
    
    def __init__(self, fname):
        raise NotImplementedError

    @classmethod
    def read(cls, fname, **kwargs):
        raise NotImplementedError

    @classmethod
    def can_read(cls, fname):
        raise NotImplementedError
                
    @classmethod
    def is_extension_valid(cls, fname):
        return splitext(basename(fname))[1].lower() in cls.extensions
        
class MASTReader(DataReader):
    extensions = ['.fits', '.fit']
    ndatasets = 1
    allowed_types = ['sap', 'pdc']
    fkeys = dict(sap = 'sap_flux', pdc = 'pdcsap_flux')
    fn_out_template = 'TIC_{:015d}_{:3s}_k2sc.fits'

    @classmethod
    def read(cls, fname, sid, **kwargs):
        ftype = kwargs.get('type', 'pdc').lower()
        assert ftype in cls.allowed_types, 'Flux type must be either `sap` or `pdc`'
        fkey = cls.fkeys[ftype]

        data  = pf.getdata(fname, 1)
        phead = pf.getheader(fname, 0)
        dhead = pf.getheader(fname, 1)

        tic = int(phead['TICID'])

        try:
            [h.remove('CHECKSUM') for h in (phead,dhead)]
            [phead.remove(k) for k in 'CREATOR PROCVER FILEVER TIMVERSN'.split()]
        except:
            pass # this can be an issue on some custom file formats

        try:
            sector = phead['sector']
        except:
            sector = kwargs.get('sector', None)

        return TESSData(tic,
                      time    = data['time'].flatten(),
                      cadence = data['cadenceno'].flatten(),
                      quality = data['quality'].flatten(),
                      fluxes  = data[fkey].flatten(),
                      errors  = data[fkey+'_err'].flatten(),
                      x       = data['pos_corr1'].flatten(),
                      y       = data['pos_corr2'].flatten(),
                      primary_header = phead,
                      data_header = dhead,
                      sector = sector,
                      ftype = ftype)
    
    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        if not ext_ok:
            return False
        else:
            h = pf.getheader(fname, 1)
            fmt_ok = 'SAP_FLUX' in h.values()
            return fmt_ok

## ===  WRITERS  ===
## =================

class FITSWriter(object):
    @classmethod
    def write(cls, fname, splits, data, dtres):

        def unpack(arr):
            aup = np.full(data.nanmask.size, np.nan)
            aup[data.nanmask] = arr
            return arr

        C = pf.Column
        cols = [C(name='time',     format='D', array=unpack(data.time)),
                C(name='cadence',  format='J', array=unpack(data.cadence)),
                C(name='quality',  format='J', array=unpack(data.quality)),
                C(name='x',        format='D', array=unpack(data.x)),
                C(name='y',        format='D', array=unpack(data.y)),
                C(name='flux',     format='D', array=unpack(data.fluxes)),
                C(name='error',    format='D', array=unpack(data.errors)),
                C(name='mflags',   format='B', array=unpack(data.mflags)),
                C(name='trtime',   format='D', array=unpack(dtres.tr_time)),
                C(name='trposi',   format='D', array=unpack(dtres.tr_position))]
        hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols), header=data.data_header)
        hdu.header['extname'] = 'TESSK2SC'
        hdu.header['object'] = data.tic
        hdu.header['ticid']   = data.tic
        hdu.header['splits'] = str(splits)
        hdu.header['cdppr'] = dtres.cdpp_r
        hdu.header['cdppt'] = dtres.cdpp_t
        hdu.header['cdppc'] = dtres.cdpp_c
        hdu.header['dt_warn'] = dtres.warn
        hdu.header['ker_name'] = dtres.detrender.kernel.name
        hdu.header['ker_pars'] = ' '.join(dtres.detrender.kernel.names)
        hdu.header['ker_eqn']  = dtres.detrender.kernel.eq
        hdu.header['ker_hps'] = str(dtres.detrender.tr_pv).replace('\n', '')

        for h in (data.primary_header,hdu.header):
            h['origin'] = 'SPLOX: Stars and Planets at Oxford'
            h['program'] = 'k2SC v1.0'
            h['date']   = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')
        
        primary_hdu = pf.PrimaryHDU(header=data.primary_header)
        hdu_list = pf.HDUList([primary_hdu, hdu])
        hdu_list.writeto(fname, overwrite=True)

readers = [MASTReader]

def select_reader(fname):
    for R in readers:
        if R.can_read(fname):
            return R
    return None
