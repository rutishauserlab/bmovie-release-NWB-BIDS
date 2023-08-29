#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load NWB files to calculate the ratio of memory selective neurons 
and plot Figure 6 a, b.

"""

import os
import numpy as np
from tqdm import tqdm
from pynwb import NWBHDF5IO
import argparse

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from vizutils_neuron import plot_cellraster, get_bootstrappval, get_NulltestPvals_cols


class Cell:
    def __init__(self):
        self.id = None
        self.brainarea = None
        self.newold_recog = None
        self.diff_val = None
        self.bs_pval = None
        
    def ms_test(self, n, run_permutation=False):
        """
        A bootstrap test for new old test
        :param n: number of bootstraps
        :return: a p value of the bootstrap test
        """

        this_cell_events = self.newold_recog
        this_labels = np.array([ eii.label for eii in this_cell_events ])
        this_responses = np.array([ 'new' if eii.actual_response <= 3. else 'old' 
                          for eii in this_cell_events ])
        this_spike_rates = np.array([ eii.spike_rate for eii in this_cell_events ])

        # Calculate the spike rates for new and old stimuli with correct answers
        old_inds = np.logical_and(this_labels=='old', 
                                  this_responses=='old')

        new_inds = np.logical_and(this_labels=='new', 
                                  this_responses=='new')

        old = this_spike_rates[old_inds]
        new = this_spike_rates[new_inds]

        n_old, n_new = len(old), len(new)
        bal_n = np.min([n_new, n_old])

        if run_permutation:
            perm_inds = np.random.choice(len(this_spike_rates), 
                               n_old+n_new, replace=False)
    
            old = this_spike_rates[perm_inds][:n_old]
            new = this_spike_rates[perm_inds][n_old:]
  
            rng = np.random.default_rng()
        else:
            # set a seed to get the same results each time
            rng = np.random.default_rng(13)

        inds_new = rng.integers(n_new, size=(bal_n,n))
        inds_old = rng.integers(n_old, size=(bal_n,n))
        diff_vals = np.mean(new[inds_new],0) - np.mean(old[inds_old],0)
        return np.mean(new)-np.mean(old), get_bootstrappval(diff_vals)
    
    
class Event:
    def __int__(self):
        self.event_type = None
        self.spike_timestamps = None
        self.label = None
        self.actual_response = None
        self.response_correct = None
        self.spike_rate = None

    def win_spike_rate(self, spike_timestamps, start, end):
        """
        Calculate the spike rate of a given window
        :return: spike rate
        """
        spike_count = np.logical_and(spike_timestamps >= start,
                                     spike_timestamps <= end )
        frate = sum(spike_count)/(end-start)
        return frate
    

def main(nwb_input_dir):
    
    session_ids = [ f for f in sorted(os.listdir(nwb_input_dir)) if f.endswith('.nwb') ]
    
    offset_pre = 1.
    offset_post = 2.
        
    # ----- Load all cells -----
    all_cells_newold = []
    keep_cells_newold = []
    keep_cells_areas = []
    
    cnt_cells_tot = 0
    for session_ii in session_ids:
        
        print(f'processing {session_ii}...')
        filepath = os.path.join(nwb_input_dir,session_ii)
        
        # hdf file associated with nwbfile --- 
        with NWBHDF5IO(filepath,'r') as nwb_io: 
            nwbfile = nwb_io.read()
        
            trials_df = nwbfile.trials.to_dataframe()
            recog_trials_df = trials_df.loc[trials_df.stim_phase=='recognition']
            enc_start_time = trials_df[trials_df['stim_phase']=='encoding']['start_time'].values[0]
            enc_stop_time = trials_df[trials_df['stim_phase']=='encoding']['stop_time'].values[0]
            
            # get information about electrodes
            electrodes_df = nwbfile.electrodes.to_dataframe()
            # get information about single units
            units_df = nwbfile.units.to_dataframe()  # see units_df.colnames
        
        for indx_ii, series_ii in units_df.iterrows():
    
            spike_times_ii = series_ii['spike_times']
    
            sp_use = np.logical_and(spike_times_ii >= enc_start_time,
                                    spike_times_ii <= enc_stop_time) 
            if sum(sp_use) == 0:
                continue # --- no spike during encoding ---
    
            newold_events = []
            for no_cnt, no_series_ii in recog_trials_df.iterrows():
                sp_use = np.logical_and(spike_times_ii >= no_series_ii['start_time']-offset_pre,
                                        spike_times_ii <= no_series_ii['start_time']+offset_post) 
                
                this_event = Event()
                this_event.event_type = no_series_ii['stim_phase']
                this_event.label = no_series_ii['stimulus_file'][:3]
                this_event.spike_timestamps = spike_times_ii[sp_use]-no_series_ii['start_time']
    
                this_event.actual_response = no_series_ii['actual_response']
                this_event.response_correct = bool(no_series_ii['response_correct'])
                this_event.spike_rate = this_event.win_spike_rate(this_event.spike_timestamps, 0.2, 1.7)
                newold_events.append(this_event)
                
                
            # ----- Check new/old task performance -----
            this_labels = np.array([ eii.label for eii in newold_events ])
            this_responses = np.array([ 'new' if eii.actual_response <= 3. else 'old' 
                              for eii in newold_events ])
    
            # Calculate the spike rates for new and old stimuli with correct answers
            old_inds = np.logical_and(this_labels=='old', this_responses=='old')
            new_inds = np.logical_and(this_labels=='new', this_responses=='new')
            
            if sum(old_inds) < 10 or sum(new_inds) < 10:
                print(f'Correct old replies: {sum(old_inds)}, new replies: {sum(new_inds)}...Skipped!')
                break # to skip this session
                
            this_cell = Cell()
            this_cell.id = series_ii['unit_id_session'] 
            this_cell.brainarea = electrodes_df.iloc[series_ii['electrode_id']]['location'] 
            this_cell.newold_recog = newold_events
            
            diff_val, bs_pval = this_cell.ms_test(10000)
            this_cell.diff_val = diff_val
            this_cell.bs_pval = bs_pval
            
            all_cells_newold.append(this_cell)
            cnt_cells_tot += 1
            
            if bs_pval is not None and bs_pval < 0.05:
                keep_cells_newold.append( (diff_val, bs_pval, this_cell) )
            
            this_area = this_cell.brainarea
                
            if this_area.startswith('Left '):
                keep_cells_areas.append( this_area.replace('Left ', '')  )
            elif this_area.startswith('Right '):
                keep_cells_areas.append( this_area.replace('Right ', '')  )
            else:
                raise ValueError(f'Brain area: {this_area} is not implemented here!')
                
                
    # --- Make tables about the brain area distribution of new/old cells ---
    keep_cells_areas = np.asarray(keep_cells_areas)
    
    areas_alllist_newold = []
    for cii in keep_cells_newold:
    
        cell_ii = cii[2]
        bii = cell_ii.brainarea    
    
        if bii.startswith('Left '):
            areas_alllist_newold.append( bii.replace('Left ', '')  )
        elif bii.startswith('Right '):
            areas_alllist_newold.append( bii.replace('Right ', '')  )
        else:
            raise ValueError(f'Brain area: {bii} is not implemented here!')
    
    print()
    print(list(zip(*np.unique(areas_alllist_newold, return_counts=True))))
    print(list(zip(*np.unique(keep_cells_areas, return_counts=True))))
    
    cell_significant = np.unique(areas_alllist_newold, return_counts=True)[1]
    
    # --- Perform permutation test ---
    n_null = 2000
    
    cells_newold_info = np.full((cnt_cells_tot,n_null), fill_value=np.nan)
    for cii, cell_ii in enumerate(tqdm(all_cells_newold)):
        for nii in range(n_null):
            diff_val_null, bs_pval_null = cell_ii.ms_test(5000, run_permutation=True)
            if bs_pval_null is not None:
                cells_newold_info[cii, nii] = bs_pval_null < 0.05
        
    cells_newold_info_n = np.nan_to_num(cells_newold_info, nan=0)
    cells_newold_info_n = cells_newold_info_n.astype(bool)
    
    
    # --- Perform permutation test results to assess the area-based p-values ---
    areas = ['ACC', 'amygdala', 'hippocampus', 'preSMA', 'vmPFC']
    
    area_counts_null_dist = np.zeros((n_null, len(areas)))
    for nii in range(n_null):
        this_nii = keep_cells_areas[cells_newold_info_n[:,nii]]
        for aii, area_ii in enumerate(areas):
            area_counts_null_dist[nii, aii] = sum(area_ii==this_nii)
        
    pval_nulltest = get_NulltestPvals_cols(area_counts_null_dist, cell_significant)
    
    print(areas)
    print(np.array(pval_nulltest).round(5))
        
    
    # --- Plot and save results ---
    plot_cells_ids = ['P53CS_R1_11_1_3077', 'P58CS_R1_34_2_3239']
    plot_cells = [cii for pii in plot_cells_ids \
                  for cii in keep_cells_newold  if pii in cii[2].id ]
            
    output_dir = 'neuron_figs'
    os.makedirs(output_dir, exist_ok=True)
    
    for cii in plot_cells:
            
        cell_ii = cii[2]
        newold_ii = cell_ii.newold_recog
        stimes_ii = np.array([ tii.spike_timestamps for tii in newold_ii if tii.response_correct ], dtype=object)
        newold_label = np.array([ tii.label for tii in newold_ii if tii.response_correct ])
        psth_inds = [np.where(newold_label=='new')[0], np.where(newold_label=='old')[0]]
        
        plot_cellraster(stimes_ii, 0.25, window=[-1,2.0], 
                        fig_title=', '.join(cell_ii.id.split('_')[:2])+f'\n{cell_ii.brainarea}' + \
                        f'\n{cii[0]:.3f}, {cii[1]:.4f}', 
                        psth_inds=psth_inds, legends=['novel','familiar'], 
                        save_svg=True, figsize=(2,2.5),
                        save_fname=os.path.join(output_dir,f'recognition_{cell_ii.id}.png') )
        
        plt.close('all')
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to calculate the ratio of memory selective neurons.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    
    args = parser.parse_args()
    main(args.nwb_input_dir)
    
    
'''
python examine_neurons_recognitiontask.py --nwb_input_dir /path/to/nwb_files/ 

e.g.:
python examine_neurons_recognitiontask.py --nwb_input_dir /media/umit/easystore/bmovie_NWBfiles

'''


