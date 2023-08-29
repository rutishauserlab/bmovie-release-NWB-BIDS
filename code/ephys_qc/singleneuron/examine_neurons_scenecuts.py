#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load NWB files to calculate the ratio of event selective neurons
and plot Figure 7 a, b. 

"""

import os
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from vizutils_neuron import plot_cellraster, get_bootstrappval, get_NulltestPvals_cols


class Cell:
    def __init__(self):
        self.id = None
        self.brainarea = None
        self.diff_val = None
        self.bs_pval = None
        self.scene_changes = None
    
    def scenechange_test(self, n, run_permutation=False):
        """
        A bootstrap test for new old test
        :param n: number of bootstraps
        :return: a p value of the bootstrap test
        """
        
        this_cell_events = self.scene_changes
        this_labels = np.array([ eii.label for eii in this_cell_events ])
        this_spike_rates = np.array([ eii.spike_rate for eii in this_cell_events ])

        cont_inds = this_labels == 'cont'
        change_inds = this_labels == 'change'

        n_change, n_cont = sum(change_inds), sum(cont_inds)

        if run_permutation:
            perm_inds = np.random.choice(len(this_spike_rates), 
                               n_change+n_cont, replace=False)
    
            change = this_spike_rates[perm_inds][:n_change]
            cont = this_spike_rates[perm_inds][n_change:]
  
            rng = np.random.default_rng()

            if np.mean(change)-np.mean(cont) <= 0:
                return np.mean(change)-np.mean(cont), 1.0
        else:
            cont = this_spike_rates[cont_inds]
            change = this_spike_rates[change_inds]
            # set a seed to get the same results each time
            rng = np.random.default_rng(13)

        inds_change = rng.integers(n_change, size=(n_change,n))
        inds_cont = rng.integers(n_cont, size=(n_cont,n))
        diff_vals = np.mean(change[inds_change],0) - np.mean(cont[inds_cont],0)
    
        return np.mean(change)-np.mean(cont), get_bootstrappval(diff_vals)

    
class Event:
    def __int__(self):
        self.event_type = None
        self.spike_timestamps = None
        self.label = None
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
    
    

def main(nwb_input_dir, scenecuts_file):

    session_ids = [ f for f in sorted(os.listdir(nwb_input_dir)) if f.endswith('.nwb') ]
        
    # --- Load scene cuts info ---
    cuts_df_init = pd.read_csv(scenecuts_file)
    cuts_df = cuts_df_init
    cuts_df.reset_index(drop=True, inplace=True)
    
    new_scenes = np.where(np.diff(cuts_df['scene_id']))[0] + 1
    scene_change_info = [ 'change' if ii in new_scenes else 'cont' for ii in range(len(cuts_df)) ]
    scene_change_info[0] = 'change'
    
    scene_cut_frames = cuts_df['shot_start_fr'].to_numpy(dtype=int)
    # scene_cut_times = cuts_df['shot_start_t'].to_numpy()
    
    offset_pre = 1.
    offset_post = 2.
    
    # ----- Load all cells -----
    all_cells = []
    
    keep_cells_change = []
    keep_cells_areas = []
    
    cnt_cells_tot = 0
    for session_ii in session_ids:
        
        print(f'processing {session_ii}...')
        filepath = os.path.join(nwb_input_dir,session_ii)
        
        # hdf file associated with nwbfile --- 
        with NWBHDF5IO(filepath,'r') as nwb_io: 
            nwbfile = nwb_io.read()
        
            # scene cut times
            frame_times = np.column_stack((nwbfile.stimulus['movieframe_time'].data[:], 
                                nwbfile.stimulus['movieframe_time'].timestamps[:] )).astype(float)
            cut_times_su = frame_times[scene_cut_frames-1,1] # -1 is to pythonize
            
            
            trials_df = nwbfile.trials.to_dataframe()
            enc_start_time = trials_df[trials_df['stim_phase']=='encoding']['start_time'].values[0]
            enc_stop_time = trials_df[trials_df['stim_phase']=='encoding']['stop_time'].values[0]
            
            # get information about electrodes
            electrodes_df = nwbfile.electrodes.to_dataframe()
            # get information about single units
            units_df = nwbfile.units.to_dataframe()  # see units_df.colnames
        
    
        event_times = cut_times_su
        assert len(event_times) == len(scene_change_info)
    
        for indx_ii, series_ii in units_df.iterrows():
    
            spike_times_ii = series_ii['spike_times']
    
            sp_use = np.logical_and(spike_times_ii >= enc_start_time,
                                    spike_times_ii <= enc_stop_time) 
            if sum(sp_use) == 0:
                continue # --- no spike during encoding ---
    
            scenechange_events = []
            for cii, tii in enumerate(event_times):
                sp_use = np.logical_and(spike_times_ii >= tii-offset_pre,
                                        spike_times_ii <= tii+offset_post) 
                
                this_event = Event()
                this_event.event_type = 'scene_change'
                this_event.label = scene_change_info[cii]
                this_event.spike_timestamps = spike_times_ii[sp_use]-tii
                this_event.spike_rate = this_event.win_spike_rate(this_event.spike_timestamps, 0.0, 1.0)
                scenechange_events.append(this_event)
    
            # ---
            this_cell = Cell()
            this_cell.id = series_ii['unit_id_session'] 
            this_cell.brainarea = electrodes_df.iloc[series_ii['electrode_id']]['location'] 
            this_cell.scene_changes = scenechange_events
    
            all_cells.append(this_cell)
            cnt_cells_tot += 1
            
            diff_val, bs_pval = this_cell.scenechange_test(10000)
            this_cell.diff_val = diff_val
            this_cell.bs_pval = bs_pval
            
            if diff_val > 0 and bs_pval < 0.05:
                keep_cells_change.append( (diff_val, bs_pval, this_cell) )
            
            this_area = this_cell.brainarea
            if this_area.startswith('Left '):
                keep_cells_areas.append( this_area.replace('Left ', '')  )
            elif this_area.startswith('Right '):
                keep_cells_areas.append( this_area.replace('Right ', '')  )
            else:
                raise ValueError(f'Brain area: {this_area} is not implemented here!')
            
    
    # --- Make tables about the brain area distribution of new/old cells ---
    keep_cells_areas = np.asarray(keep_cells_areas)
    
    areas_alllist_change = []
    for cii in keep_cells_change:
    
        cell_ii = cii[2]
        bii = cell_ii.brainarea    
    
        if bii.startswith('Left '):
            areas_alllist_change.append( bii.replace('Left ', '')  )
        elif bii.startswith('Right '):
            areas_alllist_change.append( bii.replace('Right ', '')  )
        else:
            raise ValueError(f'Brain area: {bii} is not implemented here!')
    
    print()
    print(list(zip(*np.unique(areas_alllist_change, return_counts=True))))
    print(list(zip(*np.unique(keep_cells_areas, return_counts=True))))
    
    cell_significant = np.unique(areas_alllist_change, return_counts=True)[1]
    
    # --- Perform permutation test ---
    n_null = 2000
    cells_info = np.full((cnt_cells_tot,n_null), fill_value=np.nan)
    
    for cii, cell_ii in enumerate(tqdm(all_cells)):
        for nii in range(n_null):
            diff_val_null, bs_pval_null = cell_ii.scenechange_test(5000, run_permutation=True)
            if diff_val_null > 0:
                cells_info[cii, nii] = bs_pval_null < 0.05
        
    cells_info_n = np.nan_to_num(cells_info, nan=0)
    cells_info_n = cells_info_n.astype(bool)
    
    
    # --- Save permutation test results to assess the area-based p-values ---
    areas = ['ACC', 'amygdala', 'hippocampus', 'preSMA', 'vmPFC']
    
    area_counts_null_dist = np.zeros((n_null, len(areas)))
    for nii in range(n_null):
        this_nii = keep_cells_areas[cells_info_n[:,nii]]
        for aii, area_ii in enumerate(areas):
            area_counts_null_dist[nii, aii] = sum(area_ii==this_nii)
        
    pval_nulltest = get_NulltestPvals_cols(area_counts_null_dist, cell_significant)
    
    print(areas)
    print(np.array(pval_nulltest).round(5))
    
    
    # --- Plot and save results ---
    plot_cells_ids = ['P42CS_R1_47_2_1347', 'P47CS_R1_78_1_2233']
    plot_cells = [cii for pii in plot_cells_ids \
                  for cii in keep_cells_change  if pii in cii[2].id ]
        
    output_dir = 'neuron_figs'
    os.makedirs(output_dir, exist_ok=True)
    
    for cii in plot_cells:
            
        cell_ii = cii[2]
        schange_ii = cell_ii.scene_changes
        stimes_ii = np.array([ tii.spike_timestamps for tii in schange_ii ], dtype=object)
        schange_label = np.array([ tii.label for tii in schange_ii ])
        psth_inds = [np.where(schange_label=='change')[0], np.where(schange_label=='cont')[0]]
    
        plot_cellraster(stimes_ii, 0.25, window=[-0.5,1.0], 
                        fig_title=', '.join(cell_ii.id.split('_')[:2])+f'\n{cell_ii.brainarea}' + \
                            f'\n{cii[0]:.3f}, {cii[1]:.4f}', 
                        psth_inds=psth_inds, legends=['scene change','continuity cut'], 
                        save_svg=True, figsize=(1.5,2.5),
                        save_fname=os.path.join(output_dir,f'scenechange_{cell_ii.id}.png') )
    
        plt.close('all')
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to calculate the ratio of event selective neurons.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    parser.add_argument('--scenecuts_file', type=str, required=True, help='Scene cuts annotations file (provided with datasharing).')
    
    args = parser.parse_args()
    main(args.nwb_input_dir, args.scenecuts_file)
    
    
'''
python examine_neurons_scenecuts.py --nwb_input_dir /path/to/nwb_files/ --scenecuts_file /path/to/scenecut_info.csv

e.g.:
python examine_neurons_scenecuts.py --nwb_input_dir /media/umit/easystore/bmovie_NWBfiles --scenecuts_file scenecut_info.csv

'''


    
