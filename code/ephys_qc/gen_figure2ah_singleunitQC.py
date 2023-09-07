#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plots SU QC metrics as shown in Figure 2 a-h. 

"""

# Required Libraries
import os
import numpy as np
from pynwb import NWBHDF5IO
from ephys_utills import get_color
import matplotlib.pyplot as plt
from glob import glob
import argparse

# Set SVG font type for Matplotlib
plt.rcParams['svg.fonttype'] = 'none'


def main(nwb_input_dir):
    
    nwb_session_files = sorted(glob(os.path.join(nwb_input_dir, 'sub-*/*.nwb')))

    # Initialize lists to hold various data
    n_units_per_wire = []
    firing_rate = []
    perc_isibelow = []
    peak_snr = []
    mean_snr = []
    pairs_proj_dist = []
    isol_dist = []
    cv2_vals = []
    cell_brainareas = []
    cell_brainareas_lr = []
    
    # Loop through each NWB file
    for session_ii in nwb_session_files:
        print(f'processing {os.path.basename(session_ii)}...')
    
        # Open the NWB file and read its content
        with NWBHDF5IO(session_ii,'r') as nwb_io: 
            nwbfile = nwb_io.read()
        
            # Extract trial information and encoding times
            trials_df = nwbfile.trials.to_dataframe()
            enc_start_time = trials_df[trials_df['stim_phase']=='encoding']['start_time'].values[0]
            enc_stop_time = trials_df[trials_df['stim_phase']=='encoding']['stop_time'].values[0]
            
            # Extract electrode information
            electrodes_df = nwbfile.electrodes.to_dataframe()
        
            # Extracting QC metrics
            proj_dist_ii = electrodes_df['pairwise_distances'].to_list()
            proj_dist_ii_use = []
            for dii in proj_dist_ii:
                if dii != 'NA' and dii != '':
                    proj_dist_ii_use.extend( list(map(float,dii.split('_'))) )
            pairs_proj_dist.extend(proj_dist_ii_use)
        
            # get information about single units
            units_df = nwbfile.units.to_dataframe()  # see units_df.colnames
        
            electrodes_ii = units_df["electrode_id"]
            chans, n_per_wire = np.unique(electrodes_ii, return_counts=True)
            n_units_per_wire.extend(n_per_wire)
        
            for indx_ii, series_ii in units_df.iterrows():
                
                spike_times_ii = series_ii['spike_times']
                sp_use = np.logical_and(spike_times_ii >= enc_start_time,
                                        spike_times_ii <= enc_stop_time) 
                
                if len(spike_times_ii[sp_use]) == 0:
                    continue # --- no spike during encoding --- 
                    
                firing_rate.append( len(spike_times_ii[sp_use]) / (enc_stop_time-enc_start_time) )
                
                perc_isibelow.append(series_ii['isibelow'])
                peak_snr.append(series_ii['peakSNR'])
                mean_snr.append(series_ii['meanSNR'])
                isol_dist.append(series_ii['isolationdist'])
                cv2_vals.append(series_ii['cv2'])    
    
                b_area_init = electrodes_df.loc[series_ii['electrode_id'],'location']
    
                cell_brainareas.append(b_area_init.removeprefix('Right ').removeprefix('Left '))         
    
                if b_area_init.startswith('Left '):
                    cell_brainareas_lr.append(b_area_init.removeprefix('Left ')+' left')         
                elif b_area_init.startswith('Right '):
                    cell_brainareas_lr.append(b_area_init.removeprefix('Right ')+' right')         
    
        
    # --- Plotting QC figures based on above data across electrodes and session ---
    fcol = '#009cccff'
    
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(6.3,3), constrained_layout=True )
    axs = axs.flat 
    
    n_unq, n_cnt = np.unique(n_units_per_wire, return_counts=True)
    
    # Histogram of the number of units identified on each active wire 
    # (only wires with at least one unit identified are counted).
    axs[0].bar(n_unq, n_cnt, color=fcol)
    axs[0].set_xticks(n_unq)
    axs[0].set_xlabel('Number of units per wire', fontsize=7, labelpad=2)
    axs[0].set_ylabel('Number of wires', fontsize=7, labelpad=2)
    
    
    # Histogram of mean firing rates.
    axs[1].hist(firing_rate, 30, color=fcol, edgecolor='w', lw=0.2)
    axs[1].set_xlim([0.0,30])
    axs[1].set_xlabel('Firing rate (spikes/s)', fontsize=7, labelpad=2)
    axs[1].set_ylabel('Number of units', fontsize=7, labelpad=2)
    axs[1].set_title('Mean firing rate', fontsize=7, y=1.0, pad=-5)
    
    
    # Histogram of proportion of inter-spike intervals (ISIs) 
    # which are shorter than 3 ms.
    perc_isibelow = np.asarray(perc_isibelow)
    axs[2].hist(perc_isibelow[perc_isibelow<3], 20, 
                color=fcol, edgecolor='w', lw=0.2)
    axs[2].set_xlabel('Percentage of ISI < 3 ms', fontsize=7, labelpad=2)
    axs[2].set_ylabel('Number of units', fontsize=7, labelpad=2)
    axs[2].set_title('ISI refractoriness', fontsize=7, y=1.0, pad=-5)
    
    
    # Histogram of the signal-to-noise ratio (SNR) of 
    # the mean waveform peak of each unit. 
    axs[3].hist(peak_snr, 20, color=fcol, edgecolor='w', lw=0.2)
    axs[3].set_xlim([0.0,30])
    axs[3].set_xlabel('Peak SNR', fontsize=7, labelpad=2)
    axs[3].set_ylabel('Number of units', fontsize=7, labelpad=2)
    axs[3].set_title('Waveform\npeak SNR', fontsize=7, y=1.0, pad=-10)
    
    
    # Histogram of the SNR of the entire waveform of all units. 
    axs[4].hist(mean_snr, 20, color=fcol, edgecolor='w', lw=0.2)
    axs[4].set_xlim([0.0,10])
    axs[4].set_xticks(np.arange(0,11,2))
    axs[4].set_xlabel('SNR', fontsize=7, labelpad=2)
    axs[4].set_ylabel('Number of units', fontsize=7, labelpad=2)
    axs[4].set_title('Waveform\nmean SNR', fontsize=7, y=1.0, pad=-10)
    
    
    # Pairwise distance between all possible pairs of units 
    # on all wires where more than 1 cluster was isolated. 
    axs[5].hist(pairs_proj_dist, 20, color=fcol, edgecolor='w', lw=0.2)
    axs[5].set_xlabel('Projection distance (SD)', fontsize=7, labelpad=2)
    axs[5].set_ylabel('Number of pairs of units', fontsize=7, labelpad=2)
    axs[5].set_title('Projection test', fontsize=7, y=1.0, pad=-5)
    
    
    # Isolation distance of all units for which this metric was defined.
    isol_dist = np.asarray(isol_dist)
    isol_dist = isol_dist[~np.isnan(isol_dist)]
    isol_dist = isol_dist[isol_dist<1e+18] # remove a wierd outlier.
    
    axs[6].hist(np.log10(isol_dist), 20, color=fcol, edgecolor='w', lw=0.2)
    axs[6].set_xlabel('Isolation distance (log 10)', fontsize=7, labelpad=2)
    axs[6].set_ylabel('Number of units', fontsize=7, labelpad=2)
    
    
    for ax in axs[:-1]:
        ax.tick_params(axis='x', labelsize=7, length=2, pad=1)
        ax.tick_params(axis='y', labelsize=7, length=2, pad=1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    
    labels, sizes = np.unique(cell_brainareas,return_counts=True)
    labels_lr, sizes_lr = np.unique(cell_brainareas_lr,return_counts=True)
    
    abbv_areas = ['ACC', 'vmPFC', 'preSMA' ]
    
    colors = [ get_color(aii) for aii in labels ]   
    colors_lr = ['gray','lightgray']*4
    pie_size = 0.40
    
    
    labels = [ aii if aii in abbv_areas else aii.capitalize() for aii in labels ]
    
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return "{:d}".format(absolute)
    
    wedges, texts, autotexts =  axs[7].pie(sizes, 
                                           radius=1,
            autopct=lambda pct: func(pct, sizes),
            colors = colors,
            wedgeprops=dict(width=pie_size, edgecolor='w'),
            pctdistance=0.80,
            )
    
    plt.setp(autotexts, size=5, color="w", weight="bold")
    plt.setp(texts, size=7)
    
    
    wedges_sub, texts_sub, autotexts_sub = axs[7].pie(sizes_lr, 
            radius=1-pie_size,
            autopct=lambda pct: func(pct, sizes_lr),
            colors = colors_lr,
            wedgeprops=dict(width=pie_size, edgecolor='w'),
            pctdistance=0.80
            )
    
    plt.setp(autotexts_sub, size=4)
    axs[7].axis('equal')
    
    axs[7].legend(wedges+wedges_sub[:2], labels+['Left','Right'],
              loc="lower right", fontsize=6, ncol=2,
              frameon=False, borderpad=0, labelspacing=0.02,
              handlelength=1, handletextpad=0.2,
              borderaxespad=0.1, columnspacing=0.5, edgecolor=None
              # bbox_to_anchor=(1, 0, 0.5, 1),
              )
    
    plt.savefig('SU_qc.png', dpi=300)
    plt.savefig('SU_qc.svg', format='svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to plot the SU QC figure.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    
    args = parser.parse_args()
    main(args.nwb_input_dir)
    
    
'''
python gen_figure2ah_singleunitQC.py --nwb_input_dir /path/to/nwb_files/

'''
