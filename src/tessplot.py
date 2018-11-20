#!/usr/bin/env python
import math as mt
import numpy as np
import astropy.io.fits as pf
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pl

from os.path import join, abspath, basename
from numpy import arange, zeros, nanmax, nanmin, nanmedian
from numpy.random import uniform
from time import sleep
from IPython.display import display
from glob import glob
from copy import copy

from glob import glob
from os.path import join, basename, splitext
from matplotlib.backends.backend_pdf import PdfPages

sb.set_style('white')
sb.set_context('paper')
pl.rc('figure', figsize=(14,5), dpi=100)

c_ob = '#002147'
c_bo = '#BF5700'

def normalise(a):
    return a/nanmedian(a)

def plot_lc(data):
    time = data['time']
    mask = (data['mflags'] == 0)
    fraw = data['flux']
    ctim = data['trtime']
    cpos = data['trposi']
    xpos = data['x']
    ypos = data['y']
    
    m = np.nanmedian(fraw)
    ctot = cpos + ctim - m
    fcorr1 = fraw - cpos + m
    fcorr2 = fraw - ctim + m
    fcorr = fcorr1 - ctim + m
    fcorr_ppm = 1e6 * ((fcorr / m) - 1)
    
    gs = pl.GridSpec(6, 1, height_ratios=[0.5,1,1,1,1,1], bottom=0.065, \
                         top=0.99, \
                         left=0.09, right=0.99, hspace=0.02, wspace=0.01)
    ax0 = pl.subplot(gs[0,:])
    ax5 = pl.subplot(gs[1,:])
    ax1 = pl.subplot(gs[2,:], sharex = ax5)
    ax2 = pl.subplot(gs[3,:], sharex = ax1, sharey = ax1)
    ax3 = pl.subplot(gs[4,:], sharex = ax1, sharey = ax1)
    ax4 = pl.subplot(gs[5,:], sharex = ax1, sharey = ax1)
    
    ax5.plot(time[mask], xpos[mask], lw = 0.5, label = 'x')
    ax5.plot(time[mask], ypos[mask], lw = 0.5, label = 'y')
    pl.setp(ax5, ylabel='pixels')
    ax5.legend(loc = 0)
    ax1.plot(time[mask], fraw[mask], lw = 0.5, label = 'raw flux')
    ax1.plot(time[mask], ctot[mask], lw = 0.5, label = 'full GP')
    pl.setp(ax1, ylabel='e-/s')
    ax1.legend(loc = 0)
    ax2.plot(time[mask], fcorr1[mask], lw = 0.5, label = 'xy-corrected')
    ax2.plot(time[mask], ctim[mask], lw = 0.5, label = 'time GP')
    pl.setp(ax2, ylabel='e-/s')
    ax2.legend(loc = 0)
    ax3.plot(time[mask], fcorr2[mask], lw = 0.5, label = 'time-corrected')
    ax3.plot(time[mask], cpos[mask], lw = 0.5, label = 'xy GP')
    pl.setp(ax3, ylabel='e-/s')
    ax3.legend(loc = 0)
    ax4.plot(time[mask], fcorr[mask], lw = 0.5, label = 'residuals')
    pl.setp(ax4, ylabel='e-/s')
    ax4.legend(loc = 0)
    # ax5.plot(time[mask], fcorr_ppm[mask], lw = 0.5, label = 'residuals')
    # pl.setp(ax5, ylabel='ppm')
    # ax5.legend(loc = 0)
    pl.xlim(time[mask].min(), time[mask].max())
    
    return ax0

def create_page():
    fig = pl.figure(figsize=(8.3,8.3))
    return fig

def make_plot(k2sc_file, savedir = None):
    fname = k2sc_file
    root, name = fname.split
    tic = int(name.split('_')[1])
    if savedir is None:
        savedir = root
    out_name = join(savedir,'{:s}.pdf'.format(splitext(basename(fname))[0]))

    data = pf.getdata(fname, 1)
    hd = pf.getheader(fname, 1)
    
    with PdfPages(out_name) as pdf:
        fig1 = create_page()
        ax0 = plot_lc(data)
        ax0.text(0.01,0.80, 'TIC {:d}'.format(tic), size = 13, weight = 'bold')
        ax0.text(0.01,0.50, 'CDPP (ppm): raw {:4.0f}, xy-corr {:4.0f}, detrended {:4.0f}'.format(hd['cdppr'], hd['cdppt'],hd['cdppc']), size=11)
        if hd['ker_name'] == 'QuasiPeriodicKernel':
            pars = np.fromstring(hd.get('ker_hps1').strip('[]'), sep=' ')
            ax0.text(0.01,0.20, '{:s} (P={:.1f} days)'.format(hd['ker_name'], pars[2]), size=11)
        else:
            ax0.text(0.01,0.20, '{:s}'.format(hd['ker_name']), size=11)
        pl.setp(ax0, frame_on=False, xticks=[], yticks=[])
        pdf.savefig(fig1)
