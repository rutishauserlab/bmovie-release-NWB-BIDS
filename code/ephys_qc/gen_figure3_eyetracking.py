#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Assess and plot eye tracking data quality recorded during the EMU sessions

"""

import os
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from tqdm import tqdm
from glob import glob

from ephys_utills import get_et_timebins, compute_pixelperDVA, et_heatmap

import matplotlib.pyplot as plt
import argparse

# Set Matplotlib SVG font type
plt.rcParams['svg.fonttype'] = 'none'

SEC2MSEC = 1000.

def main(nwb_input_dir):

    # ----- Basic metadata info about the video stimulus -----
    frame_width, frame_height, vid_fps, nframes = 640, 480, 25.0, 11971
    
    frame_duration_msec = SEC2MSEC / vid_fps
    framesize = [frame_width, frame_height]
    
    # define a duration to timebin the ET data
    timebin_sec = 1.0 # define here as sec. used for saving results
    timebin_msec = timebin_sec * SEC2MSEC # millisec.
    
    nbins = np.round(frame_duration_msec * nframes / timebin_msec).astype(int)

    nwb_session_files = sorted(glob(os.path.join(nwb_input_dir, 'sub-*/*.nwb')))
    
    # ----- Read ET (gaze) data from the NWB files -----
    etdata_ses = []
    pixel_dva_ses = []
    missingdata_ratio = []
    session_ids_et = []
    for session_ii in nwb_session_files:
        print(f'processing {os.path.basename(session_ii)}...')

        # Open the NWB file and read its content
        with NWBHDF5IO(session_ii,'r') as nwb_io: 
            nwbfile = nwb_io.read()
            
            session_ids_et.append(nwbfile.identifier)
            
            trials_df = nwbfile.trials.to_dataframe()
            enc_start_time = trials_df[trials_df['stim_phase']=='encoding']['start_time'].values[0]
            enc_stop_time = trials_df[trials_df['stim_phase']=='encoding']['stop_time'].values[0]
            
            gaze_data = nwbfile.processing['behavior']['EyeTracking']['SpatialSeries']
            gaze_xy = np.asarray(gaze_data.data)
            
            if gaze_data.rate is None:
                gaze_time = gaze_data.timestamps
            else:
                gaze_time = np.arange(0,len(gaze_xy))/(gaze_data.rate) + gaze_data.starting_time
            gaze_encoding = np.logical_and(gaze_time >= enc_start_time, 
                                           gaze_time <= enc_stop_time) 
        
            gaze_df = pd.DataFrame(data=np.c_[gaze_time[gaze_encoding],gaze_xy[gaze_encoding,:]],
                                   columns=['RecTime','GazeX','GazeY'])    
        
            # get video display info and scale gaze to stimulus size
            display_info_raw = gaze_data.comments
            screen_wh, display_wh, display_area_i = display_info_raw.split('::')
        
            screen_w, screen_h = list(map(float,screen_wh.split('=')[1].split(',')))
            display_w, display_h = list(map(float,display_wh.split('=')[1].split(',')))
            display_area = list(map(float,display_area_i.split('=')[1].split(',')))
        
            _, pixel_dva_mean = compute_pixelperDVA([screen_w,screen_h])
            pixel_dva_ses.append(pixel_dva_mean)
        
            scale_dx = frame_width / display_w
            scale_dy = frame_height / display_h
        
            gaze_df['GazeX'] = (gaze_df['GazeX'] - display_area[0])*scale_dx 
            gaze_df['GazeY'] = (gaze_df['GazeY'] - display_area[1])*scale_dy 
        
            problem_inds_x = np.logical_not(gaze_df['GazeX'].astype(float).between(0,frame_width,
                                                                                   inclusive='left'))
            
            problem_inds_y = np.logical_not(gaze_df['GazeY'].astype(float).between(0,frame_height,
                                                                                   inclusive='left'))
            problem_inds = np.logical_or(problem_inds_x, problem_inds_y)
            gaze_df.loc[problem_inds,['GazeX','GazeY']] = np.NaN
        
        
            # Downsample gaze data to the video frame rate
            et_xy_binned = get_et_timebins(gaze_df, timebin_msec, do_op=None,
                                           fix_length=True, nbins=nbins, keep_timebin_index=False,
                                           )
            
            et_xy_binned_v = np.vstack(et_xy_binned)
            missingdata_ratio.append(np.isnan(et_xy_binned_v).any(axis=1).sum() / et_xy_binned_v.shape[0])
            
            et_xy_binned = np.asarray(et_xy_binned, dtype=object)
            etdata_ses.append(et_xy_binned)
    
    
    # --- data to be used for ET analysis ---
    session_ids_et = np.asarray(session_ids_et)
    re_idx = np.argsort(session_ids_et) # resort subject IDs, all in _r1, _r2 order
    session_ids_et = session_ids_et[re_idx]
    etdata_ses = np.asarray(etdata_ses, dtype=object)[re_idx,:]
    pixel_dva_ses = np.asarray(pixel_dva_ses)[re_idx]
    missingdata_ratio = np.asarray(missingdata_ratio)[re_idx]
    
    # ----- Compute heatmap correlations between subjects -----
    sigma = np.mean(pixel_dva_ses) # Standard deviation for Gaussian kernel. Equal for both axes.
    hp_down_factor = float(5.0) # should be float. Downsample heatmaps for a quick analysis
    nan_ratio = 0.50
    
    # remove subjects with missing data more than half of all recording time
    remove_subjs_indx = missingdata_ratio>=nan_ratio
    print(session_ids_et[remove_subjs_indx])
    
    session_ids_et = session_ids_et[~remove_subjs_indx]
    etdata_ses = etdata_ses[~remove_subjs_indx]
    pixel_dva_ses = pixel_dva_ses[~remove_subjs_indx]
    missingdata_ratio = missingdata_ratio[~remove_subjs_indx]
    
    unique_subjs = np.unique([ sii.split('_')[0] for sii in session_ids_et ])
    
    subj2agg_corrs = np.zeros((len(session_ids_et), nbins))
    
    for tii in tqdm(range(nbins)):
        
        etdata_tbin = etdata_ses[:,tii,...]
        
        for sub_ii in unique_subjs:
            
            subj_this_bool = [ True if sub_ii in sii else False for sii in session_ids_et ]
            subj_other_bool = np.logical_not(subj_this_bool)
            
            # reference heatmap from all other subjects:
            agg_data = np.vstack(etdata_tbin[subj_other_bool,...])
            agg_hmap = et_heatmap(agg_data, framesize, sigma, hp_down_factor, get_full=False)
            
            subj_this = session_ids_et[subj_this_bool]
            
            for sub_jj in subj_this:
                
                sub_idx = session_ids_et.tolist().index(sub_jj)
                subj_hmap = et_heatmap(etdata_tbin[sub_idx], framesize, sigma, #pixel_dva_ses[sub_idx], 
                                       hp_down_factor, get_full=False, nan_ratio=nan_ratio)
                
                if subj_hmap is not None:
                    subj2agg_corrs[sub_idx, tii] = np.corrcoef(agg_hmap.ravel(), subj_hmap.ravel())[0,1]
                    
                else:
                    subj2agg_corrs[sub_idx, tii] = np.NaN
                    
    
    # report overall ET correlations
    subj2agg_corrs_z = np.arctanh(subj2agg_corrs)
    
    mean_hmap_corrs = np.tanh( np.nanmean(subj2agg_corrs_z , 1) )
    mean_hmap_corrs_low = np.tanh( np.nanpercentile(subj2agg_corrs_z, 16, 1) )
    mean_hmap_corrs_up = np.tanh( np.nanpercentile(subj2agg_corrs_z, 84, 1) )
    print(np.tanh(np.mean(np.nanmean(subj2agg_corrs_z, 1))), 
          np.tanh(np.std(np.nanmean(subj2agg_corrs_z, 1))) )
    
    # ----- Plot obtained gaze metrics -----
    fcol = '#009ccc'
    fcol_txt = '#008db8'
    
    session_ids_et_txt = [ sii.lower() for sii in session_ids_et ]
    pos = np.arange(len(session_ids_et))
    
    error_config =dict(ecolor ='#006f91', linewidth=1, capsize=2, capthick=1.)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.35))
    ax.bar(pos, mean_hmap_corrs, 
           yerr=[ mean_hmap_corrs-mean_hmap_corrs_low, mean_hmap_corrs_up-mean_hmap_corrs ] , 
           color=fcol, error_kw=error_config, label='')
    ax.set_ylim([0,1])
    
    
    ax.set_ylabel('Average gaze heatmap\n correlation (r)', fontsize=8, color=fcol_txt)
    ax.tick_params(axis='y', which='major', labelsize=7, length=3, pad=1,
                   labelcolor=fcol_txt, color=fcol_txt
                   )
    
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.margins(0.013)
    
    
    fcol2 = '#787878'
    
    ax2 = ax.twinx()
    ax2.bar(pos, missingdata_ratio*100, color=fcol2, width=0.4)
    ax2.set_ylim([0,100])
    
    ax2.set_ylabel('Percentage of missing\n gaze data (%)', fontsize=8, color=fcol2)
    ax2.tick_params(axis='y', which='major', labelsize=7, length=3, pad=1,
                   labelcolor=fcol2, color=fcol2,
                   )
    
    ax2.spines['top'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.margins(0.013)
    
    ax2.spines['right'].set_color(fcol2)
    ax2.spines['left'].set_color(fcol_txt)
    
    ax.set_xticks(pos)
    ax.set_xticklabels(session_ids_et_txt, fontsize=7, rotation=80, ha='center')
    ax.tick_params(axis='x', which='major', length=3, pad=0)
    
    plt.tight_layout()
    fig.savefig("gaze_group_mean-hmap.png", dpi=600, bbox_inches='tight')
    fig.savefig("gaze_group_mean-hmap.svg", format='svg', dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to plot eye tracking data quality.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    
    args = parser.parse_args()
    main(args.nwb_input_dir)
    
    
'''
python gen_figure3_eyetracking.py --nwb_input_dir /path/to/nwb_files/

'''
