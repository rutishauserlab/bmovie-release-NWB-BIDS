#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plot macro-electrode locations for Fig. 1d

"""

# Required Libraries
import os
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
from glob import glob
import mne

from ephys_utills import get_color
import argparse

# Set Matplotlib SVG font type
plt.rcParams['svg.fonttype'] = 'none'

# Dictionary mapping brain region abbreviations to their full names.
brain_abbv_map = {'RHIP':'Right hippocampus', 'LHIP':'Left hippocampus', 
                  'RAMY':'Right amygdala', 'LAMY':'Left amygdala',
                  'RACC':'Right ACC', 
                  'LACC':'Left ACC',
                  'RpreSMA':'Right preSMA',
                  'LpreSMA':'Left preSMA',
                  'RvmPFC':'Right vmPFC',
                  'LvmPFC':'Left vmPFC',
                  } 

# Reverse dictionary mapping (for looking up abbreviations by their full names)
brain_abbv_map_r = {v: k for k, v in brain_abbv_map.items()}


def main(nwb_input_dir):
    
    nwb_session_files = sorted(glob(os.path.join(nwb_input_dir, 'sub-*/*.nwb')))
    
    # Loop over all session files to extract electrode data
    macros_brainlocs = []
    macros_mnilocs_xyz = []
    keep_ids = []
    for session_ii in nwb_session_files:
        print(f'processing {os.path.basename(session_ii)}...')
        
        # Reading NWB file
        with NWBHDF5IO(session_ii,'r') as nwb_io: 
            nwbfile = nwb_io.read()
        
            # Extract electrode data from the NWB file
            electrodes_df = nwbfile.electrodes.to_dataframe()
            use_chans_mac = [ 'macro' in gii for gii in electrodes_df.group_name ]
            use_chans_nan = [ np.isnan(gii).any() for gii in electrodes_df[['x','y','z']].values ]
            use_chans = np.logical_and(use_chans_mac, np.logical_not(use_chans_nan))
            
            # Filter and copy the electrodes dataframe based on the conditions
            electrodes_df_use = electrodes_df[use_chans].copy()
            brainarea = electrodes_df_use['location'].values
            xyz_coords = electrodes_df_use[['x','y','z']].values
        
            # Extract subject ID from the file
            subj_id = nwbfile.identifier.split('_')[0]
            
            if subj_id == 'P53CS':
                if subj_id+'A' in keep_ids:
                    subj_id = subj_id+'B'
                else:
                    subj_id = subj_id+'A'
            keep_ids.append(subj_id)
        
            # Create abbreviated brain area names with additional details
            brainarea_abbv = [ f'{subj_id}_{brain_abbv_map_r[bii]}_{cii}' for cii, bii in enumerate(brainarea) ]
        
            macros_brainlocs.extend(brainarea_abbv)
            macros_mnilocs_xyz.extend(xyz_coords)
    
        
    macros_brainlocs = np.array(macros_brainlocs)
    macros_mnilocs_xyz = np.array(macros_mnilocs_xyz)
    
    # Combine locations and coordinates and remove duplicates
    all_locs = np.column_stack((macros_brainlocs,macros_mnilocs_xyz))
    all_locs_uq, indx_uq = np.unique(all_locs, axis=0, return_index=True)
    macros_brainlocs_uq = macros_brainlocs[indx_uq]
    macros_mnilocs_xyz_uq = macros_mnilocs_xyz[indx_uq]
    
    
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / 'subjects'
    
    ch_names = macros_brainlocs_uq
    ch_colors = [ get_color(chii) for chii in ch_names]
    
    # Convert electrode positions to the right units and set them up for visualization
    elec = macros_mnilocs_xyz_uq*(1e-3) # electrode positions given in meters
    dig_ch_pos = dict(zip(ch_names, elec))
    
    # Create a montage of electrode positions
    mont = mne.channels.make_dig_montage(ch_pos=dig_ch_pos, coord_frame='unknown') 
    print('Created %s channel positions' % len(ch_names))
    
    info = mne.create_info(ch_names.tolist(), 1000., 'seeg')
    info.set_montage(mont)
    
    alpha = 0.2
    trans = mne.transforms.Transform(fro='head', to='mri',
                                     trans=np.eye(4)) #  # not used so just use identity
    
    
    # Visualize the electrode positions on a brain model
    viz_views = ['rostral', 'dorsal', 'frontal', 'medial' ]
    for vii in viz_views:
        brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir,
                              cortex='classic', 
                              alpha=alpha, background='white', 
                              views=vii
                              )
        
        brain.add_sensors(info, trans=trans, 
                          colors=ch_colors
                          )
        
        brain.save_image(f'elect_locs_{vii}.png')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to extract and visualize electrode locations.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    
    args = parser.parse_args()
    main(args.nwb_input_dir)
    
    
'''
python gen_figure1d_electrodelocs.py --nwb_input_dir /path/to/nwb_files/

e.g.:
python gen_figure1d_electrodelocs.py --nwb_input_dir /media/umit/easystore/bmovie_dandi/000623

'''
