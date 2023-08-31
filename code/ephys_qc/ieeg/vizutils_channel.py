#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Basic visualization functions for plotting PSTHs for LFP and iEEG data.

"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def get_bootstrappval(x, alternative='two-sided'):
    ''' Calculate bootstrap p-value
    the null hypothesis is that the mean of the distribution 
    from which x is sampled is zero. '''

    x = np.array(x)
    assert x.ndim == 1
    assert alternative in ['two-sided', 'one-sided']
    
    if np.mean(x) > 0:
        p_1 = (np.sum(x<=0)+1.)/(x.size+1.)
        p_2 = (np.sum(x>0)+1.)/(x.size+1.)
    elif np.mean(x) < 0:
        p_1 = (np.sum(x<0)+1.)/(x.size+1.)
        p_2 = (np.sum(x>=0)+1.)/(x.size+1.)
    else: # rare but happens in this electrophys data 
        p_1 = (np.sum(x<=0)+1.)/(x.size+1.)
        p_2 = (np.sum(x>=0)+1.)/(x.size+1.)

    # Return the appropriate p-value based on the 'alternative' parameter
    if alternative=='two-sided':
        return 2.*np.min([p_1,p_2])
    elif alternative=='one-sided':
        return np.min([p_1,p_2])


get_NulltestPvals_cols = lambda x, x_pmean: [ (np.sum(x[:,p_ii]>=x_pmean[p_ii])+1.)/float(x.shape[0]+1.) if x_pmean[p_ii] > 0 else
                                              (np.sum(x[:,p_ii]<=x_pmean[p_ii])+1.)/float(x.shape[0]+1.) for p_ii in range(x.shape[1]) ]


import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gcd(a, b):
    while b:
        a, b = b, a%b
    return a


def arrange_man(start, stop, step_size):
    # More accurate on step than np.arange and includes the end-point as well. 
    n = (stop-start)/float(step_size) + 1
    if np.isclose(n, np.round(n)):
        n = np.round(n)
    return np.linspace(start, stop, int(n))


def get_psth(rasters, binsize=1., ylim=None):

    if not isinstance(rasters, list):
        rasters = [rasters]

    # Initialize PSTH
    psth = dict()
    psth['data'] = dict()

    # Compute the PSTH
    for cond_id in range(len(rasters)):
        psth['data'][cond_id] = dict()
        raster = rasters[cond_id]
        mean_psth = np.mean(raster, axis=0) / binsize
        std_psth = np.std(raster, axis=0) / binsize
        sem_psth = std_psth / np.sqrt(float(np.shape(raster)[0]))

        psth['data'][cond_id]['mean'] = mean_psth
        psth['data'][cond_id]['sem'] = sem_psth

    return psth


def plot_psth(psth, t_trace, binsize, window=[-1,1], window2=None, figsize=(2,1.5),
              colors=['#F5A21E','#134B64','#EF3E34','#02A68E','C4','C5','C8'], 
              legends=None, cmap='Greys', fig_title=None, save_loc=None, 
              ylim=None, ylabel=None, save_svg=False):
    
    xtic_len = gcd(abs(window[0]), window[1])
    if xtic_len>0.05: # against very low xtic_len
        if xtic_len==1: xtic_len=0.5 
        xtics_def = np.arange(window[0], window[1] + xtic_len, xtic_len)
        xtics = [ str(round(i,2)) for i in xtics_def]
        xtics_loc = [(j - window[0]) / binsize -0.5 for j in xtics_def]
    else:
        xtic_len = 0.1
        xtics_def = np.arange(window[0], window[1] + xtic_len, xtic_len)
        xtics = [ str(round(i,2)) for i in xtics_def]
        xtics_loc = [(j - window[0]) / binsize -0.5 for j in xtics_def]

    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    if fig_title:
        ax.set_title(fig_title, fontsize=6)    

    y_min = np.nanmin([np.min(psth['data'][psth_idx]['mean']-psth['data'][psth_idx]['sem']) for psth_idx in psth['data']])
    y_max = np.nanmax([np.max(psth['data'][psth_idx]['mean']+psth['data'][psth_idx]['sem']) for psth_idx in psth['data']])

    scale = 0.05
    y_min = (1.0-np.sign(y_min)*scale)*y_min
    y_max = (1.0+np.sign(y_max)*scale)*y_max
    
    time_bins_x = [(j - window[0]) / binsize -0.5 for j in t_trace]
    time_bins = np.convolve(time_bins_x, np.ones(2), 'valid') / 2.0
    
    x_0 = (-window[0]) / binsize - 0.5
    if ylim:
        ax.plot([x_0,x_0], ylim, color='r', ls='--')
    else:
        ax.plot([x_0,x_0], [y_min,y_max], color='r', ls='--',
                 zorder=1000, lw=1.2)

    
    for ii in range(len(psth['data'])):
        if legends:
            ax.plot(time_bins, psth['data'][ii]['mean'], '.-',
                     color=colors[ii % len(colors)], lw=1.1, markersize=3,
                     label=legends[ii])
        else:
            ax.plot(time_bins, psth['data'][ii]['mean'], '.-',
                     color=colors[ii % len(colors)], lw=1.1, markersize=3)
        

    ax.set_xlabel('Time (s)', fontsize=8, labelpad=1.5)
    ax.set_ylabel('Mean HFB response\n(z-scored)', fontsize=8, labelpad=1.5)
    
    ax.set_xticks(xtics_loc)
    ax.set_xticklabels(xtics,fontsize=8)
    
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([y_min, y_max])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='y', right=False, length=3, pad=1, labelsize=7)
    ax.tick_params(axis='x', top=False, length=3, pad=1, labelsize=7)
    
    divider = make_axes_locatable(ax)
    colsize, padsize = '2%', 0.1
    cax = divider.append_axes("right", size=colsize, pad=padsize)
    cax.remove()
    
    # ax.margins(x=0.0, y=None)
    if legends:
        ax.legend(frameon=False, fontsize=6, handlelength=1., handletextpad=0.5,
                   columnspacing=1.0, borderaxespad=0.5, labelspacing=0.15)
        
    plt.subplots_adjust(hspace=0.1)
    if save_loc: 
        plt.savefig(save_loc, dpi=300, bbox_inches='tight')
        if save_svg: plt.savefig(save_loc[:-4]+'.svg',
                                 format='svg', bbox_inches='tight')
    
