#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load NWB files and preprocessed HFB signals to calculate the ratio of event selective channels
and plot Figure 7 c, d.

"""


import os
import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO
from scipy.stats import mode

from glob import glob
from tqdm import tqdm
import mne
import argparse

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from vizutils_channel import get_bootstrappval, get_NulltestPvals_cols
from vizutils_channel import get_psth, plot_psth, arrange_man


class Channel:
    def __init__(self):
        self.session_id = None
        self.id = None
        self.brainarea = None
        self.scene_changes = None

    def ms_test(self, n, run_permutation=False):
        """
        A bootstrap test for new old test
        :param n: number of bootstraps
        :return: a p value of the bootstrap test
        """

        this_chan_events = self.scene_changes
        this_labels = np.array([ eii.label for eii in this_chan_events ])
        this_meanresp = np.array([ eii.mean_lfpresp for eii in this_chan_events ])

        cont_inds = this_labels == 'cont'
        change_inds = this_labels == 'change'
        n_change, n_cont = sum(change_inds), sum(cont_inds)

        if run_permutation:
            perm_inds = np.random.choice(len(this_meanresp), 
                               n_change+n_cont, replace=False)
    
            change = this_meanresp[perm_inds][:n_change]
            cont = this_meanresp[perm_inds][n_change:]
            rng = np.random.default_rng()
            
            if np.mean(change)-np.mean(cont) <= 0:
                return np.mean(change)-np.mean(cont), 1.0
        else:
            cont = this_meanresp[cont_inds]
            change = this_meanresp[change_inds]
            # set a seed to get the same results each time
            rng = np.random.default_rng(13)

        inds_change = rng.integers(n_change, size=(n_change,n))
        inds_cont = rng.integers(n_cont, size=(n_cont,n))
        diff_vals = np.mean(change[inds_change],0) - np.mean(cont[inds_cont],0)

        return np.mean(change)-np.mean(cont), get_bootstrappval(diff_vals)

    
class EventCh:
    def __int__(self):
        self.event_type = None
        self.lfp_time = None
        self.lfp_data = None
        self.label = None
        self.mean_lfpresp = None

    def meanresp_fixwin(self, signal_data, signal_time, start, end):
        t_use = np.logical_and(signal_time >= start, signal_time <= end)
        mean_sig = np.mean(signal_data[t_use], axis=-1) # mean across time 
        return mean_sig



def main(nwb_input_dir, lfp_process_dir, scenecuts_file):

    nwb_session_files = sorted(glob(os.path.join(nwb_input_dir, 'sub-*/*.nwb')))
        
    # Load scene cuts info
    cuts_df_init = pd.read_csv('scenecut_info.csv')
    cuts_df = cuts_df_init#.iloc[1:-1]
    cuts_df.reset_index(drop=True, inplace=True)
    
    new_scenes = np.where(np.diff(cuts_df['scene_id']))[0] + 1
    scene_change_info = [ 'change' if ii in new_scenes else 'cont' for ii in range(len(cuts_df)) ]
    scene_change_info[0] = 'change'
    
    scene_cut_frames = cuts_df['shot_start_fr'].to_numpy(dtype=int)
    # scene_cut_times = cuts_df['shot_start_t'].to_numpy()
    
    for ch_type in ['macro', 'micro']:
    
        task2load = 'enc'
        band2load = 'hfb'    
        binsize = 0.250
        offset_pre = 0.5
        offset_post = 1.0
        
        # need to remove 10 sec time padding added in prep_filterLFP.py to reduce edge effects
        recog_offset = 10.0 # sec. 
        
        print(f'analyzing {ch_type}...')
        
        keep_chs_scenecut = []
        lfp_freq = None
        
        keep_chs_areas = []
        all_chs_scenecut = []
        
        cnt_chs_tot = 0
        for sii, session_ii in enumerate(nwb_session_files):
            print(f'processing {os.path.basename(session_ii)}...')
            
            with NWBHDF5IO(session_ii,'r') as nwb_io: 
                nwbfile = nwb_io.read()
                session_id = nwbfile.identifier
        
                # scene cut times
                frame_times = np.column_stack((nwbfile.stimulus['movieframe_time'].data[:], 
                                    nwbfile.stimulus['movieframe_time'].timestamps[:] )).astype(float)
                cut_times_su = frame_times[scene_cut_frames-1,1] # -1 is to pythonize
                
        
            lfp_file = glob(os.path.join(lfp_process_dir, f'{session_id}*{task2load}*{ch_type}*{band2load}*'))
            
            if len(lfp_file) == 0:
                print(f'skipped {session_id}')
                continue
            
            assert len(lfp_file) <= 1
            lfp_file = lfp_file[0]
        
            lfp_mne = mne.io.read_raw_fif(lfp_file, preload=True, verbose='error')
            lfp_data = lfp_mne.get_data()
            lfp_time = lfp_mne.times
            lfp_chnames = lfp_mne.ch_names
            lfp_freq_this = lfp_mne.info['sfreq']
            if lfp_freq is None:
                lfp_freq = lfp_freq_this
            else:
                assert lfp_freq == lfp_freq_this
                
            # remove padding added during preprocessing
            lfp_time = lfp_time - recog_offset
            
            # try either scene_cut_times or cut_times_su
            event_times = cut_times_su
            assert len(event_times) == len(scene_change_info)
            
            for ch_ii, ch_name_ii in enumerate(lfp_chnames):
        
                ch_data = lfp_data[ch_ii]
                
                scenechange_events = []
                for cii, tii in enumerate(event_times):
                    sp_use = np.logical_and(lfp_time >= tii-offset_pre,
                                            lfp_time <= tii+offset_post) 
                    
                    this_event = EventCh()
                    this_event.event_type = 'scene_change'
                    this_event.label = scene_change_info[cii]
                    this_event.lfp_time = lfp_time[sp_use]-tii
                    this_event.lfp_data = ch_data[sp_use] 
                    this_event.mean_lfpresp = this_event.meanresp_fixwin(this_event.lfp_data,
                                                                     this_event.lfp_time, 0.0, 1.0)            
                    scenechange_events.append(this_event)
        

                this_ch = Channel()
                this_ch.session_id = session_id
                this_ch.id = ch_name_ii
                this_ch.brainarea = ch_name_ii[1:-1]
                this_ch.scene_changes = scenechange_events
                
                all_chs_scenecut.append(this_ch)
                cnt_chs_tot += 1
        
                diff_val, bs_pval = this_ch.ms_test(10000)
                this_ch.diff_val = diff_val
                this_ch.bs_pval = bs_pval
                
                if diff_val > 0 and bs_pval < 0.05:
                    keep_chs_scenecut.append( (diff_val, bs_pval, this_ch) )
                    
                keep_chs_areas.append(ch_name_ii[1:-1])
        
        
        # --- Make tables about the brain area distribution of event selective channels ---
        keep_chs_areas = np.asarray(keep_chs_areas)
        
        areas_alllist_scenecut = []
        for cii in keep_chs_scenecut:
            ch_ii = cii[2]
            areas_alllist_scenecut.append( ch_ii.brainarea )    
        
        print()
        print(list(zip(*np.unique(areas_alllist_scenecut, return_counts=True))))
        print(list(zip(*np.unique(keep_chs_areas, return_counts=True))))
        
        chs_significant = np.unique(areas_alllist_scenecut, return_counts=True)[1]
        
        # --- Perform permutation test ---
        n_null = 2000
        chs_scenecut_info = np.full((cnt_chs_tot,n_null), fill_value=np.nan)
        for cii, ch_ii in enumerate(tqdm(all_chs_scenecut)):
            for nii in range(n_null):
                diff_val_null, bs_pval_null = ch_ii.ms_test(5000, run_permutation=True)
                if diff_val_null > 0:
                    chs_scenecut_info[cii, nii] = bs_pval_null < 0.05
            
        chs_scenecut_info_n = np.nan_to_num(chs_scenecut_info, nan=0)
        chs_scenecut_info_n = chs_scenecut_info_n.astype(bool)
        
        
        # --- Perform permutation test results to assess the area-based p-values ---
        areas = ['ACC', 'AMY', 'HIP', 'OFC', 'SMA' ] # OFC = vmPFC; SMA = preSMA
        
        area_counts_null_dist = np.zeros((n_null, len(areas)))
        for nii in range(n_null):
            this_nii = keep_chs_areas[chs_scenecut_info_n[:,nii]]
            for aii, area_ii in enumerate(areas):
                area_counts_null_dist[nii, aii] = sum(area_ii==this_nii)
        
        pval_nulltest = get_NulltestPvals_cols(area_counts_null_dist, chs_significant)
        
        print(areas)
        print(np.array(pval_nulltest).round(5))
        
        
        # --- Plot and save results ---
        if ch_type == 'micro':
            plot_chs_ids = ['P51CS_R2_RSMA8']
        else:
            plot_chs_ids = ['P62CS_R2_RHIP6']
        
        plot_chs = [cii for pii in plot_chs_ids for cii in keep_chs_scenecut \
                    if pii in cii[2].session_id+'_'+cii[2].id ]
                
        output_dir = 'channel_figs'
        os.makedirs(output_dir, exist_ok=True)
        
        time_sample_size = int(lfp_freq*(offset_post + offset_pre))
        
        for cii in plot_chs:
            
            ch_ii = cii[2]
            scut_ii = ch_ii.scene_changes
            
            appx_times = [ len(tii.lfp_time) for tii in scut_ii ]
            appx_times_mode = mode(appx_times, keepdims=False).mode
            assert time_sample_size == appx_times_mode
            
            lfp_cont_resp = []
            lfp_change_resp = []
            
            for tii in scut_ii:
                
                if len(tii.lfp_time) == time_sample_size+1:
                    tii_time = np.convolve(tii.lfp_time, np.ones(2)/2, mode='valid')
                    tii_vals = np.convolve(tii.lfp_data, np.ones(2)/2, mode='valid')
                    
                elif len(tii.lfp_time) < time_sample_size:
                    tii_time = np.linspace(-offset_pre, offset_post, time_sample_size)
                    assert len(tii_time) == time_sample_size
                    tii_vals = np.full(time_sample_size, fill_value=np.mean(tii.lfp_data), 
                                       dtype=float)
                    tii_vals[:len(tii.lfp_data)] = tii.lfp_data.copy()
                
                elif len(tii.lfp_time) == time_sample_size:
                    tii_time = tii.lfp_time.copy()
                    tii_vals = tii.lfp_data.copy()
                
                else:
                    raise SystemExit('This case was not expected!')
                    
                if tii.label=='cont':
                    lfp_cont_resp.append(tii_vals)
                elif tii.label=='change':
                    lfp_change_resp.append(tii_vals)
                    
            
            lfp_cont_resp = np.vstack(lfp_cont_resp)
            lfp_change_resp = np.vstack(lfp_change_resp)
        
            tbins = arrange_man(-offset_pre, offset_post, binsize)
            tbins_2d = [ [tbins[tbii], tbins[tbii+1]] for tbii in range(len(tbins)) if tbii<len(tbins)-1 ]
            
            tindex_chunks = np.zeros(len(tii_time))
            for tcnt, tbii in enumerate(tbins_2d):
                tindex_chunks[np.logical_and(tii_time >= tbii[0], tii_time <= tbii[1])] = tcnt
        
        
            lfp_change_resp_sp = np.stack(np.split(lfp_change_resp, np.where(np.diff(tindex_chunks))[0]+1, axis=1))
            lfp_cont_resp_sp = np.stack(np.split(lfp_cont_resp, np.where(np.diff(tindex_chunks))[0]+1, axis=1))
        
            lfp_change_resp_sp = lfp_change_resp_sp.mean(-1).T
            lfp_cont_resp_sp = lfp_cont_resp_sp.mean(-1).T
            psth_mat = get_psth([lfp_change_resp_sp, lfp_cont_resp_sp], binsize=binsize)
                
            plot_psth(psth_mat, tbins, binsize, 
                      window=[-0.5,1.0], window2=None, figsize=(1.5,1.2),
                      legends = ['scene change','continuity cut'], 
                      fig_title=f'{ch_ii.session_id}-{ch_ii.id} \n {cii[0]:.3f}, {cii[1]:.3f}',
                      save_loc = os.path.join(output_dir, f'scenecut_{ch_type}_{ch_ii.session_id}_{ch_ii.id}.png'),
                      save_svg=True,
                      )
            
            plt.close('all')
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files and preprocessed HFB signals to calculate the ratio of event selective channels.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    parser.add_argument('--lfp_process_dir', type=str, required=True, help='Directory containing preprocessed HFB signals (by prep_filterLFP.py).')
    parser.add_argument('--scenecuts_file', type=str, required=True, help='Scene cuts annotations file (provided with datasharing).')
    
    args = parser.parse_args()
    main(args.nwb_input_dir, args.lfp_process_dir, args.scenecuts_file)
    

'''
python examine_channels_scenecuts.py --nwb_input_dir /path/to/nwb_files/ --lfp_process_dir /path/to/lfp_prep_dir --scenecuts_file /path/to/scenecut_info_file

e.g.:
python examine_channels_scenecuts.py --nwb_input_dir /media/umit/easystore/bmovie_dandi/000623 --lfp_process_dir /media/umit/easystore/lfp_prep --scenecuts_file scenecut_info.csv

'''
