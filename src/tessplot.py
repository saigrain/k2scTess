#!/usr/bin/env python
#!/usr/bin/env python
import math as mt
import numpy as np
import astropy.io.fits as pf
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pl

from os.path import join, basename, split, splitext
from numpy.random import uniform
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages

from .core import *
from .utils import medsig

sb.set_style('white')
sb.set_context('paper')
pl.rc('figure', figsize=(14,5), dpi=100)

c_ob = '#002147'
c_bo = '#BF5700'

def normalise(a):
    return a/np.nanmedian(a)

def plot_lc(data, zoom  = False):
    time = data['time']
    flags = data['mflags']
    lcond = flags == 0
    lqual = flags == M_QUALITY
    loutl = (flags == M_OUTLIER_U) | (flags == M_OUTLIER_D)
    lperm = flags == M_PERIODIC
    mask = ~(lcond | lqual | loutl | lperm)

    fraw = data['flux']
    ctim = data['trtime']
    cpos = data['trposi']
    xpos = data['x']
    ypos = data['y']
    
    m = np.nanmedian(fraw[lcond])
    ctot = cpos + ctim - m
    fcorr1 = fraw - cpos + m
    fcorr2 = fraw - ctim + m
    fcorr = fcorr1 - ctim + m
    fcorr_ppm = 1e6 * ((fcorr / m) - 1)
    
    gs = pl.GridSpec(7, 1, height_ratios=[0.6,1,1,1,1,1,1], bottom=0.065, \
                         top=0.99, \
                         left=0.09, right=0.99, hspace=0.02, wspace=0.01)
    ax0 = pl.subplot(gs[0,:])
    ax1 = pl.subplot(gs[1,:])
    ax2 = pl.subplot(gs[2,:], sharex = ax1)
    ax3 = pl.subplot(gs[3,:], sharex = ax1)
    ax4 = pl.subplot(gs[4,:], sharex = ax1, sharey = ax3)
    ax5 = pl.subplot(gs[5,:], sharex = ax1, sharey = ax3)
    ax6 = pl.subplot(gs[6,:], sharex = ax1)
    
    ax1.plot(time[mask], xpos[mask], 'k.', ms = 3, alpha = 0.3, \
                 label = 'normal data')
    ax2.plot(time[mask], ypos[mask], 'k.', ms = 3, alpha = 0.3)
    ax3.plot(time[mask], fraw[mask], 'k.', ms = 3, alpha = 0.3, \
                 label = 'raw flux')
    ax3.plot(time[mask], ctot[mask], 'C2-', lw = 0.5, label = 'full GP')
    ax4.plot(time[mask], fcorr1[mask], 'k.', ms = 3, alpha = 0.3, \
                 label = 'xy-corrected')
    ax4.plot(time[mask], ctim[mask], 'C2-', lw = 0.5, label = 'time GP')
    ax5.plot(time[mask], fcorr2[mask], 'k.', ms = 3, alpha = 0.3, \
                 label = 'time-corrected')
    ax5.plot(time[mask], cpos[mask], 'C2-', lw = 0.5, label = 'xy GP')
    ax6.plot(time[mask], fcorr[mask], 'k.', ms = 3, alpha = 0.3, \
                 label = 'fully corrected')

    if lperm.sum() > 0:
        ax1.plot(time[lperm], xpos[lperm], 'C0.', ms = 5, alpha = 0.7, \
                    label = 'periodic mask')
        ax2.plot(time[lperm], ypos[lperm], 'C0.', ms = 5, alpha = 0.7)
        ax3.plot(time[lperm], fraw[lperm], 'C0.', ms = 5, alpha = 0.7)
        ax4.plot(time[lperm], fcorr1[lperm], 'C0.', ms = 5, alpha = 0.7)
        ax5.plot(time[mask], fcorr2[mask], 'C0.', ms = 5, alpha = 0.7)
        ax6.plot(time[mask], fcorr[mask], 'C0.', ms = 5, alpha = 0.7)
        
    ax1.plot(time[lqual], xpos[lqual], 'C1.', ms = 5, alpha = 0.7, \
                 label = 'bad quality flag')
    ax2.plot(time[lqual], ypos[lqual], 'C1.', ms = 5, alpha = 0.7)
    ax3.plot(time[lqual], fraw[lqual], 'C1.', ms = 5, alpha = 0.7)
    ax4.plot(time[lqual], fcorr1[lqual], 'C1.', ms = 5, alpha = 0.7)
    ax5.plot(time[lqual], fcorr2[lqual], 'C1.', ms = 5, alpha = 0.7)
    ax6.plot(time[lqual], fcorr[lqual], 'C1.', ms = 5, alpha = 0.7)

    ax1.plot(time[lcond], xpos[lcond], 'C2.', ms = 5, alpha = 0.7, \
                 label = 'training set')                 
    ax2.plot(time[lcond], ypos[lcond], 'C2.', ms = 5, alpha = 0.7)          
    ax3.plot(time[lcond], fraw[lcond], 'C2.', ms = 5, alpha = 0.7)
    ax4.plot(time[lcond], fcorr1[lcond], 'C2.', ms = 5, alpha = 0.7)
    ax5.plot(time[lcond], fcorr2[lcond], 'C2.', ms = 5, alpha = 0.7)
    ax6.plot(time[lcond], fcorr[lcond], 'C2.', ms = 5, alpha = 0.7)

    pl.setp(ax1, ylabel='x (pix)')
    pl.setp(ax2, ylabel='y (pix)')
    pl.setp(ax3, ylabel='flux (e-/s)')
    pl.setp(ax4, ylabel='flux (e-/s)')
    pl.setp(ax5, ylabel='flux (e-/s)')
    pl.setp(ax6, ylabel='flux (e-/s)')
    pl.setp(ax6, xlabel='time (d)')

    ax1.legend(loc = 0)
    ax3.legend(loc = 0)
    ax4.legend(loc = 0)
    ax5.legend(loc = 0)
    ax6.legend(loc = 0)
    
    pl.xlim(np.nanmin(time), np.nanmax(time))

    if zoom:
        med, sig = medsig(xpos[mask])
        ymi, yma = xpos[lcond].min(), xpos[lcond].max()
        yr = yma-ymi
        ymax = max(med + 5 * sig, yma + 0.1 * yr)
        ymin = min(med - 5 * sig, ymi - 0.1 * yr)
        pl.setp(ax1, ylim = (ymin, ymax))
        med, sig = medsig(ypos[mask])
        ymi, yma = ypos[lcond].min(), ypos[lcond].max()
        yr = yma-ymi
        ymax = max(med + 5 * sig, yma + 0.1 * yr)
        ymin = min(med - 5 * sig, ymi - 0.1 * yr)
        pl.setp(ax2, ylim = (ymin, ymax))
        med, sig = medsig(fraw[mask])
        ymi, yma = fraw[lcond].min(), fraw[lcond].max()
        yr = yma-ymi
        ymax = max(med + 5 * sig, yma + 0.1 * yr)
        ymin = min(med - 5 * sig, ymi - 0.1 * yr)
        pl.setp(ax3, ylim = (ymin, ymax))
        med, sig = medsig(fcorr[mask])
        ymi, yma = fcorr[lcond].min(), fcorr[lcond].max()
        yr = yma-ymi
        ymax = max(med + 5 * sig, yma + 0.1 * yr)
        ymin = min(med - 5 * sig, ymi - 0.1 * yr)
        pl.setp(ax6, ylim = (ymin, ymax))
        
    
    return ax0

def create_page():
    fig = pl.figure(figsize=(8.3,8.3))
    return fig

def make_plot(fname):
    outfile = fname.replace('fits','pdf')
    root, name = split(fname)
    tic = int(name.split('_')[1])
    data = pf.getdata(fname, 1)
    hd = pf.getheader(fname, 1)
    
    with PdfPages(outfile) as pdf:
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

    with PdfPages(outfile.replace('.pdf','_zoom.pdf')) as pdf:
        fig1 = create_page()
        ax0 = plot_lc(data, zoom = True)
        ax0.text(0.01,0.80, 'TIC {:d}'.format(tic), size = 13, weight = 'bold')
        ax0.text(0.01,0.50, 'CDPP (ppm): raw {:4.0f}, xy-corr {:4.0f}, detrended {:4.0f}'.format(hd['cdppr'], hd['cdppt'],hd['cdppc']), size=11)
        if hd['ker_name'] == 'QuasiPeriodicKernel':
            pars = np.fromstring(hd.get('ker_hps1').strip('[]'), sep=' ')
            ax0.text(0.01,0.20, '{:s} (P={:.1f} days)'.format(hd['ker_name'], pars[2]), size=11)
        else:
            ax0.text(0.01,0.20, '{:s}'.format(hd['ker_name']), size=11)
        pl.setp(ax0, frame_on=False, xticks=[], yticks=[])
        pdf.savefig(fig1)
