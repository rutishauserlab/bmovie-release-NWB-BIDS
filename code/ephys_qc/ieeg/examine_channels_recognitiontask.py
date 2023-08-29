#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load NWB files and preprocessed HFB signals to calculate the ratio of event selective channels 
and plot Figure 6 c,d. 

"""


import os
import numpy as np

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
        self.newold_recog = None
        
    def ms_test(self, n, run_permutation=False):
        """
        A bootstrap test for new old test
        :param n: number of bootstraps
        :return: a p value of the bootstrap test
        """

        this_chan_events = self.newold_recog
        this_labels = np.array([ eii.label for eii in this_chan_events ])
        this_responses = np.array([ 'new' if eii.actual_response <= 3. else 'old' 
                          for eii in this_chan_events ])
        this_meanresp = np.array([ eii.mean_lfpresp for eii in this_chan_events ])

        # Calculate the spike rates for new and old stimuli with correct answers
        old_inds = np.logical_and(this_labels=='old', 
                                  this_responses=='old')

        new_inds = np.logical_and(this_labels=='new', 
                                  this_responses=='new')
        
        n_old, n_new = sum(old_inds), sum(new_inds)
        bal_n = np.min([n_new, n_old])

        if run_permutation:
            perm_inds = np.random.choice(len(this_meanresp), 
                               n_old+n_new, replace=False)
            old = this_meanresp[perm_inds][:n_old]
            new = this_meanresp[perm_inds][n_old:]
            rng = np.random.default_rng()
        else:
            old = this_meanresp[old_inds]
            new = this_meanresp[new_inds]
            # set a seed to get the same results each time
            rng = np.random.default_rng(13)

        inds_new = rng.integers(n_new, size=(bal_n,n))
        inds_old = rng.integers(n_old, size=(bal_n,n))
        diff_vals = np.mean(new[inds_new],0) - np.mean(old[inds_old],0)
            
        return np.mean(new)-np.mean(old), get_bootstrappval(diff_vals)
    

class EventCh:
    def __int__(self):
        self.event_type = None
        self.lfp_time = None
        self.lfp_data = None
        self.label = None
        self.actual_response = None
        self.response_correct = None
        self.mean_lfpresp = None

    def meanresp_fixwin(self, signal_data, signal_time, start, end):
        t_use = np.logical_and(signal_time >= start, signal_time <= end)
        mean_sig = np.mean(signal_data[t_use], axis=-1) # mean across time 
        return mean_sig



def main(nwb_input_dir, lfp_process_dir):
    
    session_ids = [ f for f in sorted(os.listdir(nwb_input_dir)) if f.endswith('.nwb') ]
    session_names = [ os.path.splitext(f)[0] for f in session_ids ] 
    
    for ch_type in ['macro', 'micro']:
        
        task2load = 'recog'
        band2load = 'hfb' 
        binsize = 0.250
        offset_pre = 1.0
        offset_post = 2.0
        
        # need to remove 10 sec time padding added in prep_filterLFP.py to reduce edge effects
        edge_offset = 10.0 # sec. 
        
        print(f'analyzing {ch_type}...')
        
        keep_chs_newold = []
        lfp_freq = None
        
        keep_chs_areas = []
        all_chs_newold = []
        
        cnt_chs_tot = 0
        for sii, session_ii in enumerate(session_names):
        
            print(f'processing {session_ii}...')
            filepath = os.path.join(nwb_input_dir,session_ii+'.nwb')
            
            # hdf file associated with nwbfile --- 
            with NWBHDF5IO(filepath,'r') as nwb_io: 
                # read the file
                nwbfile = nwb_io.read()
        
                # load info about movie watching and new/old task time blocks.
                trials_df = nwbfile.trials.to_dataframe()
        
                recog_trials_df = trials_df.loc[trials_df.stim_phase=='recognition'].copy()
                recog_start_time = recog_trials_df['start_time'].values[0]
        
                recog_trials_df['start_time'] = recog_trials_df['start_time'].values \
                    - recog_start_time 
                recog_trials_df['stop_time'] = recog_trials_df['stop_time'].values \
                    - recog_start_time 
            
            
            lfp_file = glob(os.path.join(lfp_process_dir, f'{session_ii}*{task2load}*{ch_type}*{band2load}*'))
            
            if len(lfp_file) == 0:
                print(f'skipped {session_ii}')
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
            lfp_time = lfp_time - edge_offset
            
            for ch_ii, ch_name_ii in enumerate(lfp_chnames):
        
                ch_data = lfp_data[ch_ii]
                
                newold_events = []
                for nii, (no_cnt, no_series_ii) in enumerate(recog_trials_df.iterrows()):
                    
                    t_use = np.logical_and(lfp_time >= no_series_ii['start_time'] - offset_pre,
                                           lfp_time <= no_series_ii['start_time'] + offset_post)
                    
                    this_event = EventCh()
                    this_event.event_type = no_series_ii['stim_phase']
                    this_event.label = no_series_ii['stimulus_file'][:3]
                    this_event.lfp_time = lfp_time[t_use] - no_series_ii['start_time']
                    this_event.lfp_data = ch_data[t_use] 
                    
                    this_event.actual_response = no_series_ii['actual_response']
                    this_event.response_correct = bool(no_series_ii['response_correct'])
                    this_event.mean_lfpresp = this_event.meanresp_fixwin(this_event.lfp_data,
                                                                     this_event.lfp_time, 
                                                                     0.2, 1.7)
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
                    
                this_ch = Channel()
                this_ch.session_id = session_ii
                this_ch.id = ch_name_ii
                this_ch.brainarea = ch_name_ii[1:-1]
                this_ch.newold_recog = newold_events
                
                diff_val, bs_pval = this_ch.ms_test(10000)
                this_ch.diff_val = diff_val
                this_ch.bs_pval = bs_pval        
                
                all_chs_newold.append(this_ch)
                cnt_chs_tot += 1
                
                if bs_pval is not None and bs_pval < 0.05:
                    keep_chs_newold.append( (diff_val, bs_pval, this_ch) )
                    
                keep_chs_areas.append(ch_name_ii[1:-1])
        
        
        # --- Make tables about the brain area distribution of memory selective channels ---
        keep_chs_areas = np.asarray(keep_chs_areas)
        
        areas_alllist_newold = []
        for cii in keep_chs_newold:
            ch_ii = cii[2]
            areas_alllist_newold.append( ch_ii.brainarea )    
        
        print()
        print(list(zip(*np.unique(areas_alllist_newold, return_counts=True))))
        print(list(zip(*np.unique(keep_chs_areas, return_counts=True))))
        
        chs_significant = np.unique(areas_alllist_newold, return_counts=True)[1]
        
        # --- Perform permutation test ---
        n_null = 2000
        chs_newold_info = np.full((cnt_chs_tot,n_null), fill_value=np.nan)
        for cii, ch_ii in enumerate(tqdm(all_chs_newold)):
            for nii in range(n_null):
                diff_val_null, bs_pval_null = ch_ii.ms_test(5000, run_permutation=True)
                if bs_pval_null is not None:
                    chs_newold_info[cii, nii] = bs_pval_null < 0.05
            
        chs_newold_info_n = np.nan_to_num(chs_newold_info, nan=0)
        chs_newold_info_n = chs_newold_info_n.astype(bool)
        
        
        # --- Perform permutation test results to assess the area-based p-values ---
        areas = ['ACC', 'AMY', 'HIP', 'OFC', 'SMA' ] # OFC = vmPFC; SMA = preSMA
        
        area_counts_null_dist = np.zeros((n_null, len(areas)))
        for nii in range(n_null):
            this_nii = keep_chs_areas[chs_newold_info_n[:,nii]]
            for aii, area_ii in enumerate(areas):
                area_counts_null_dist[nii, aii] = sum(area_ii==this_nii)
        
        pval_nulltest = get_NulltestPvals_cols(area_counts_null_dist, chs_significant)
        
        print(areas)
        print(np.array(pval_nulltest).round(5))
        
        
        # --- Plot and save results ---
        if ch_type == 'micro':
            plot_chs_ids = ['P58CS_R1_RACC2']
        else:
            plot_chs_ids = ['P42CS_R2_RSMA3']
        
        plot_chs = [cii for pii in plot_chs_ids for cii in keep_chs_newold \
                    if pii in cii[2].session_id+'_'+cii[2].id ]
                
        output_dir = 'channel_figs'
        os.makedirs(output_dir, exist_ok=True)
        
        time_sample_size = int(lfp_freq*(offset_post + offset_pre))
        
        for cii in plot_chs:
            
            ch_ii = cii[2]
            newold_ii = ch_ii.newold_recog
            
            appx_times = [ len(tii.lfp_time) for tii in newold_ii ]
            appx_times_mode = mode(appx_times, keepdims=False).mode
            assert time_sample_size == appx_times_mode
            
            lfp_oldresp = []
            lfp_newresp = []
            
            for tii in newold_ii:
                
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
                    
                if tii.response_correct and tii.label=='old':
                    lfp_oldresp.append(tii_vals)
                elif tii.response_correct and tii.label=='new':
                    lfp_newresp.append(tii_vals)
                    
            lfp_newresp = np.vstack(lfp_newresp)
            lfp_oldresp = np.vstack(lfp_oldresp)
        
            tbins = arrange_man(-offset_pre, offset_post, binsize)
            tbins_2d = [ [tbins[tbii], tbins[tbii+1]] for tbii in range(len(tbins)) if tbii<len(tbins)-1 ]
            
            tindex_chunks = np.zeros(len(tii_time))
            for tcnt, tbii in enumerate(tbins_2d):
                tindex_chunks[np.logical_and(tii_time >= tbii[0], tii_time <= tbii[1])] = tcnt
        
            lfp_newresp_sp = np.stack(np.split(lfp_newresp, np.where(np.diff(tindex_chunks))[0]+1, axis=1))
            lfp_oldresp_sp = np.stack(np.split(lfp_oldresp, np.where(np.diff(tindex_chunks))[0]+1, axis=1))
        
            lfp_newresp_sp = lfp_newresp_sp.mean(-1).T
            lfp_oldresp_sp = lfp_oldresp_sp.mean(-1).T
            psth_mat = get_psth([lfp_newresp_sp, lfp_oldresp_sp], binsize=binsize)
            
            plot_psth(psth_mat, tbins, binsize, 
                      window=[-1.0,2.0], window2=None, figsize=(2,1.5),
                      legends = ['novel', 'familiar'], 
                      fig_title=f'{ch_ii.session_id}-{ch_ii.id} \n {cii[0]:.3f}, {cii[1]:.3f}',
                      save_loc = os.path.join(output_dir,f'recognition_{ch_type}_{ch_ii.session_id}_{ch_ii.id}.png'),
                      save_svg=True,
                      )
        
            plt.close('all')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files and preprocessed HFB signals to calculate the ratio of memory selective channels.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    parser.add_argument('--lfp_process_dir', type=str, required=True, help='Directory containing preprocessed HFB signals (by prep_filterLFP.py).')
    
    args = parser.parse_args()
    main(args.nwb_input_dir, args.lfp_process_dir)
    

'''
python examine_channels_recognitiontask.py --nwb_input_dir /path/to/nwb_files/ --lfp_process_dir /path/to/lfp_prep

e.g.:
python examine_channels_recognitiontask.py --nwb_input_dir /media/umit/easystore/bmovie_NWBfiles --lfp_process_dir /media/umit/easystore/lfp_prep

'''
