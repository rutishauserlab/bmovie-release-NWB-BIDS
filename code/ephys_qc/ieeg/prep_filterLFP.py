#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load NWB files to extract and preprocess iEGG signals, both micros and macros, to 
obtain the high-frequency broadband (HFB) time-course per channel.
"""


import os
import numpy as np
from pynwb import NWBHDF5IO

import mne
from scipy.stats import zscore
import argparse

# note that in micros and macros data, 
# 'OFC' corresponds to 'vmPFC' and 'SMA' to 'preSMA'
keep_channel_areas = [ 'AMY', 'HIP', 'ACC', 'SMA', 'OFC' ]


def main(nwb_input_dir, lfp_process_dir):
    
    session_ids = [ f for f in sorted(os.listdir(nwb_input_dir)) if f.endswith('.nwb') ]
    
    os.makedirs(lfp_process_dir, exist_ok=True)

    # We might want to take some these as inputs for a more general application    
    bandfreq = {'hfb':[70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
                # 'gamma':[20, 30, 40, 50, 61, 70], # can other bands here
                }
    
    # the padding length used while generating the NWB files.
    LFP_PAD_us =  10. # in second
    
    # add 10 sec time padding to reduce edge effects, this is different than LFP_PAD_us.
    edge_offset = 10.0 # sec. 
    srate = 500
    
    for data2load in ['LFP_micro', 'LFP_macro']:
    
        etype = data2load.split('_')[1]
        
        for session_ii in session_ids:
        
            print(f'\nprocessing {session_ii}...')
        
            filepath = os.path.join(nwb_input_dir,session_ii)
            with NWBHDF5IO(filepath,'r') as nwb_io: 
                nwbfile = nwb_io.read()
        
                # load info about movie watching and new/old task time blocks.
                trials_df = nwbfile.trials.to_dataframe()
                recog_trials_df = trials_df.loc[trials_df.stim_phase=='recognition']
            
                # add 10 sec time padding to reduce edge effects
                enc_start_time = trials_df[trials_df['stim_phase']=='encoding']['start_time'].values[0] - edge_offset
                enc_stop_time = trials_df[trials_df['stim_phase']=='encoding']['stop_time'].values[0] + edge_offset
            
                # add 10 sec time padding to reduce edge effects
                recog_start_time = recog_trials_df['start_time'].values[0] - edge_offset
                recog_stop_time = recog_trials_df['stop_time'].values[-1] + edge_offset
            
                try:
                    lfp_elect_series = nwbfile.processing['ecephys'][data2load]['ElectricalSeries']
                except KeyError:
                    print(f"\n\tCouldn't find {data2load}. Skipped.\n")
                    continue
        
                lfp_data = lfp_elect_series.data[:].T
                # print(lfp_data.shape)
        
                lfp_channel_info_df = lfp_elect_series.electrodes.to_dataframe()
                channel_names_init = lfp_channel_info_df['origchannel_name'].to_numpy(dtype=str)
                assert len(channel_names_init) == lfp_data.shape[0]
        
        
                use_chns = [ True if chii[1:4] in keep_channel_areas else False for chii in channel_names_init ]
                channel_names = channel_names_init[use_chns].tolist()
                lfp_data = lfp_data[use_chns]
                assert len(channel_names) == lfp_data.shape[0]
            
                if lfp_elect_series.rate is None:
                    lfp_time = lfp_elect_series.timestamps[:]
                    sfreq = 1./np.diff(lfp_time).mean()  # in Hertz
                else:
                    lfp_time = np.arange(0,lfp_data.shape[1])/(lfp_elect_series.rate) \
                        + lfp_elect_series.starting_time
                    sfreq = lfp_elect_series.rate
                assert len(lfp_time) == lfp_data.shape[1]        
        
        
            # Notice that we used 10 sec padding for LFP data in the NWB files. 
            # This is different than edge_offset padding above.
            # Due to this padding the movie starts at t=10.0 sec. 
            # We need to correct for this 10 sec here so that the movie starts at t=0.0 sec.
            lfp_time -= LFP_PAD_us
            
        
            # to reduce data size just process experiment time interval
            for task_ii, task_start, task_stop in [('enc', enc_start_time, enc_stop_time),
                                                   ('recog', recog_start_time, recog_stop_time) ]:
                
                print(f'\nprocessing {task_ii} for {session_ii}...')
            
                time_inds = np.logical_and(lfp_time>=task_start, lfp_time<=task_stop)
                
                lfp_time_use = lfp_time[time_inds] - task_start
                lfp_data_use = lfp_data[:,time_inds]
                
                info = mne.create_info(channel_names, sfreq=sfreq, ch_types='seeg')
                raw = mne.io.RawArray(lfp_data_use, info)
                assert abs(raw.times[0] - lfp_time_use[0]) < 0.001 # raw.times = lfp_time_use is not allowed. 
            
                # ----- notch filter -----
                notches = np.arange(60, info['lowpass']+1, 60)
                raw.notch_filter(notches, phase='zero-double', n_jobs=-1)
            
                # ----- high-pass filter 0.1 HZ -----
                raw.filter(0.1, None, n_jobs=-1, verbose=False)
        
                # ----- Apply common average reference to remove common noise and trends -----
                raw, _ = mne.set_eeg_reference(raw.copy(), 'average')
            
                for bp, bp_freqs in bandfreq.items():
            
                    print(f'\ncomputing {bp}...')
                    
                    fname_bp = os.path.join(lfp_process_dir,
                                    f'{os.path.splitext(session_ii)[0]}_{task_ii}_{etype}_{bp}.fif')
            
                    # apply Morlet filters
                    power = mne.time_frequency.tfr_array_morlet(np.expand_dims(raw.copy()._data, 0), # (n_epochs, n_channels, n_times)
                                                           sfreq=raw.info['sfreq'],
                                                           freqs=np.array(bp_freqs),
                                                           n_cycles=5., 
                                                           output='power', # alternative: output='complex', then use np.abs(temp)**2 below,
                                                           verbose=False, n_jobs=-1).squeeze() # remove n_epoches back: (n_epochs=1, n_channels, n_freqs, n_times)
                
                    power = zscore(power, axis=-1) # normalize across time-dimension
                    power = np.mean(power, 1) # average across freq bands
                    
                    # put data back into raw data format of MNE to use MNE tools for saving preprocessed signals.
                    raw_new = raw.copy();
                    raw_new._data = power
                    
                    raw_new.resample(srate, npad='auto')
                    raw_new.save(fname_bp, overwrite=True)    
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to extract and preprocess iEGG signals, both micros and macros.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    parser.add_argument('--lfp_process_dir', type=str, required=True, help='Directory for saving preprocessed HFB signals.')
    
    args = parser.parse_args()
    main(args.nwb_input_dir)
    
    
'''
python prep_filterLFP.py --nwb_input_dir /path/to/nwb_files/ --lfp_process_dir /path/to/lfp_prep_dir

e.g.:
python prep_filterLFP.py --nwb_input_dir /media/umit/easystore/bmovie_NWBfiles --lfp_process_dir /media/umit/easystore/lfp_prep

'''



