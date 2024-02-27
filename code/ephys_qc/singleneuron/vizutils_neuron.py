#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Basic visualization functions for plotting rasters and PSTHs.

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


def arrange_man(start, stop, step_size):
    # More accurate on step than np.arange and includes the end-point as well. 
    n = (stop-start)/float(step_size) + 1
    if np.isclose(n, np.round(n)):
        n = np.round(n)
    return np.linspace(start, stop, int(n))


def plot_cellraster(spiketimes, binsize, window=[-1,1], window2=None, cut_line=None,
                    colors=['#F5A21E','#134B64','#EF3E34','#02A68E','C4','C5','C8', 'C6', 'C9', 'C7'], 
                    psth_inds=None, legends=None, figsize=(2,2.5),
                    cmap='Greys', fig_title=None, save_fname=None, ylim=None, 
                    ylabel=None, save_svg=False):

    
    spikes_use = np.array( [tii[ np.logical_and( tii >= window[0] , tii <= window[1] ) ]
                            for tii in spiketimes ], dtype=object)
    
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    
    if psth_inds is None:
        ax1.eventplot(spikes_use, colors='k', lineoffsets=1, linelengths=1, linewidths=1)
    else:
        if len(psth_inds) > len(colors):
            raise SystemExit('The number of psth_inds is greater than the number of available colors!')
        
        spikes_use_sorted = []
        colors_use = []
        for pii, pii_inds in enumerate(psth_inds):
            spikes_use_sorted.append(spikes_use[pii_inds])
            colors_use.extend([colors[pii]]*len(pii_inds))
            
        spikes_use_sorted = np.concatenate(spikes_use_sorted)
        ax1.eventplot(spikes_use_sorted, colors=colors_use, lineoffsets=1, linelengths=1, linewidths=1)
            
    ax1.margins(y=0.0)
    ax1.axvline(0, color='r', linestyle='--', lw=1.1)
    
    if cut_line is not None: 
        print('check whether origin is upper or lower!')
        if not isinstance(cut_line, list):
            ax1.axhline(cut_line - 0.5, color='#304163', linestyle='--',lw=1)
        else:
            for axl_ii in cut_line:
                ax1.axhline(axl_ii - 0.5, color='#304163', linestyle='--',lw=1)
    
    
    if window2 is not None:
        ax1.axvline((window2), color='#28385e', linestyle='--',lw=1)
    
    if ylabel is not None:
        ax1.set_ylabel(ylabel, fontsize=8, labelpad=1.5)
    else: 
        ax1.set_ylabel('Trials (re-sorted)', fontsize=8, labelpad=1.5)
    
    # to start counting from 1 rather than 0 on the y-axis of raster figure    
    interval_size = 10
    if len(spikes_use)/interval_size == len(spikes_use)//interval_size:
        ytick_vals = np.arange(0, len(spikes_use)+1, interval_size) 
    else:
        ytick_vals = np.arange(0, len(spikes_use), interval_size) 
        
    ytick_labels = ytick_vals.copy()
    ytick_labels[0] = ytick_labels[0] + 1
    ytick_vals[1:] = ytick_vals[1:] - 1
    
    ax1.set_yticks(ytick_vals)
    ax1.set_yticklabels(ytick_labels) # to start counting from 1 rather than 0. 
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
    ax1.tick_params(axis='x', which='both', top=False, bottom=False, length=3, pad=1, labelsize=7)
    ax1.tick_params(axis='y', which='both', right=False, length=3, pad=1, labelsize=7)
    
    if fig_title is not None:
        ax1.set_title(fig_title, fontsize=6)    
    
    
    tbins = arrange_man(window[0], window[1], binsize)
    tbin_center = np.convolve(tbins, np.ones(2), 'valid') / 2.0
    
    if psth_inds is None:
        psth_inds = [ np.arange(len(spikes_use)) ]
    
    
    psth = {}
    for pii, pii_inds in enumerate(psth_inds):
        
        psth[pii] = {}
        
        psth_block = []
        for sp_ii in spikes_use[pii_inds]:
            spikes_inbins, _ = np.histogram(sp_ii,tbins)
            psth_block.append(spikes_inbins)
    
        psth_block = np.asarray(psth_block)
    
        mean_psth = np.mean(psth_block, axis=0) / binsize
        std_psth = np.std(psth_block, axis=0) / binsize
        sem_psth = std_psth / np.sqrt(psth_block.shape[0])
        
        psth[pii]['mean'] = mean_psth
        psth[pii]['sem'] = sem_psth

    
    y_min = np.nanmin([np.min(psth[psth_idx]['mean']-psth[psth_idx]['sem']) for psth_idx in psth])
    y_max = np.nanmax([np.max(psth[psth_idx]['mean']+psth[psth_idx]['sem']) for psth_idx in psth])
    
    scale = 0.05
    y_min = (1.0-np.sign(y_min)*scale)*y_min
    y_max = (1.0+np.sign(y_max)*scale)*y_max
    
    x_0 = 0
    if ylim is not None:
        ax2.plot([x_0,x_0], ylim, color='r', ls='--', lw=1.2)
    else:
        ax2.plot([x_0,x_0], [y_min,y_max], color='r', ls='--',
                 zorder=1000, lw=1.2)
    
    for ii in range(len(psth)):
        if legends:
            # ax2.plot(tbin_center, psth[ii]['mean'], '.-',
                     # color=colors[ii % len(colors)], lw=1.1, markersize=3,  
                     # label=legends[ii])

            ax2.errorbar(tbin_center, psth[ii]['mean'], yerr=psth[ii]['sem'],
                         color=colors[ii % len(colors)],
                         fmt='.-', lw=1.1, markersize=3, 
                         elinewidth=0.7, capthick=0.6, capsize=1.3,
                         label=legends[ii])

        else:
            ax2.plot(tbin_center, psth[ii]['mean'], '.-',
                     color='#676767', lw=1.1, markersize=3)
            
    ax2.set_xlabel('Time (s)', fontsize=8, labelpad=1.5)
    ax2.set_ylabel('Firing rate (Hz)', fontsize=8, labelpad=1.5)
    
    if ylim is not None:
        ax2.set_ylim(ylim)
    else:
        ax2.set_ylim([y_min, y_max])
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax2.tick_params(axis='y', right=False, length=3, pad=1, labelsize=7)
    ax2.tick_params(axis='x', top=False, length=3, pad=1, labelsize=7)
    
    ax2.margins(x=0.0,y=None)
    ax2.set_xlim(window)
    if legends:
        ax2.legend(frameon=False, fontsize=6, handlelength=1., handletextpad=0.5,
                   columnspacing=1.0, borderaxespad=0.5, labelspacing=0.15)
    
    plt.subplots_adjust(hspace=0.1)
    if save_fname: 
        plt.savefig(save_fname,dpi=300,bbox_inches='tight')
        if save_svg: plt.savefig(save_fname[:-4]+'.svg',format='svg')


