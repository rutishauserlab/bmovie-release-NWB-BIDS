#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plot recording locations in a 2D figure for Fig. 2i

"""

import os
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
from glob import glob

import nibabel as nib
import templateflow.api as tflow
import nilearn.image as nimage
from nilearn import plotting

from ephys_utills import get_color
import argparse

# Set Matplotlib SVG font type
plt.rcParams['svg.fonttype'] = 'none'


def main(nwb_input_dir):

    nwb_session_files = sorted(glob(os.path.join(nwb_input_dir, 'sub-*/*.nwb')))
    
    # Initialize empty lists to store brain locations and MNI coordinates
    micros_brainlocs = []
    micros_mnilocs_xyz = []
    # keep track of unique locations
    micros_brainlocs_uq = []
    micros_mnilocs_xyz_uq = []
    
    # Loop over all session files to extract electrode data
    for session_ii in nwb_session_files:
        print(f'processing {os.path.basename(session_ii)}...')

        # Open the NWB file and read its content
        with NWBHDF5IO(session_ii,'r') as nwb_io: 
            nwbfile = nwb_io.read()
            
            # get information about electrodes
            electrodes_df = nwbfile.electrodes.to_dataframe()
    
            # get information about single units
            units_df = nwbfile.units.to_dataframe()  # see units_df.colnames
        
            # Extract the list of unique electrode IDs and their counts
            electrodes_ii = units_df["electrode_id"]
            chans, n_per_wire = np.unique(electrodes_ii, return_counts=True)
            electrodes_labels = units_df["electrodegroup_label"]
            
            electrodes_df_use = electrodes_df[electrodes_df.index.isin(electrodes_ii)]
            electrodes_df_use_altv = electrodes_df[electrodes_df.group_name.isin(electrodes_labels)]
            assert electrodes_df_use.equals(electrodes_df_use_altv)
    
            brainarea = electrodes_df_use['location'].values
            group_names = electrodes_df_use['group_name'].values
            xyz_coords = electrodes_df_use[['x', 'y', 'z']].values
            assert all([ 'microwire' in uqii for uqii in group_names ])
    
            micros_brainlocs.extend(brainarea)
            micros_mnilocs_xyz.extend(xyz_coords)
    
            xyz_coords_uq, indx_uq = np.unique(xyz_coords, axis=0, return_index=True)
            brainarea_uq = brainarea[indx_uq]
            
            micros_brainlocs_uq.extend(brainarea_uq)
            micros_mnilocs_xyz_uq.extend(xyz_coords_uq)
    
    
    # ----- Plot electrode locations -----
    # plot only the unique locations per subject -----
    blocs = np.unique(micros_brainlocs)
    
    blocs_uq = []
    for bii in blocs:
        if bii.startswith('Left '):
            blocs_uq.append( bii.replace('Left ', '')  )
        elif bii.startswith('Right '):
            blocs_uq.append( bii.replace('Right ', '')  )
        else:
            raise ValueError(f'Brain area: {bii} is not implemented here!')
         
    blocs_uq = np.unique(blocs_uq).tolist()
    micros_brainlocs_uq = np.asarray(micros_brainlocs_uq)
    micros_mnilocs_xyz_uq = np.asarray(micros_mnilocs_xyz_uq)
    nonzero_xzy_locs = abs(micros_mnilocs_xyz_uq).sum(1) != 0
    
    # MNI152NLin2009cAsym
    mask_ref_file = tflow.get("MNI152NLin2009cAsym", desc="brain", resolution=1,
                             suffix="mask")
    
    t1w_ref_file = tflow.get('MNI152NLin2009cAsym', desc=None, resolution=1,
                             suffix='T1w', extension='nii.gz')
    
    base_img = nib.load(t1w_ref_file)
    
    # mask brain image to remove skull
    img_data = nib.load(t1w_ref_file).get_fdata()
    mask_data = nib.load(mask_ref_file).get_fdata().astype(bool)
    
    img_data[~mask_data] = 0
    masked_img = nimage.new_img_like(base_img, img_data)
    
    # x_slice, z_slice pairs for areas 
    slice_dict = {'amy':(20,-15),
                  'acc':(5,25),
                  'hip':(25,-15),
                  'vmp':(5,-15),
                  'pre':(5,45)
                  }
    
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(6.5,3), constrained_layout=False )
    
    for cnt_ii, area_ii in enumerate(blocs_uq):
    
        units_uq = [ True if area_ii in mii else False for mii in micros_brainlocs_uq ]
        units_uq = np.logical_and(units_uq, nonzero_xzy_locs)
        units_xyz = micros_mnilocs_xyz_uq[units_uq]
    
        x_slice=slice_dict[area_ii[:3].lower()][0]
        z_slice=slice_dict[area_ii[:3].lower()][1]
        # -------------------------------
        display = plotting.plot_anat(masked_img, display_mode='x', cut_coords=[x_slice], 
                                     figure=fig, axes=axs[1,cnt_ii])
        
        units_xyz_xslice = units_xyz.copy()
        units_xyz_xslice[:,0] = x_slice
        
        display.add_markers(units_xyz_xslice, marker_color=get_color(area_ii, xrgb=False), 
                            marker_size=1)
        
        # -------------------------------
        display = plotting.plot_anat(masked_img, display_mode='z', cut_coords=[z_slice], 
                                     figure=fig, axes=axs[0,cnt_ii])
        
        units_xyz_zslice = units_xyz.copy()
        units_xyz_zslice[:,2] = z_slice
        
        display.add_markers(units_xyz_zslice, marker_color=get_color(area_ii, xrgb=False), 
                            marker_size=1)
        
        if len(area_ii) > 6:
            title_ii = area_ii.capitalize()
        else:
            title_ii = area_ii
            
        axs[0,cnt_ii].set_title(title_ii, fontsize=7)
    
    
    # save the outout figure
    plt.subplots_adjust(hspace=0.005, wspace=0.02)
    plt.savefig('recording_locs.png', dpi=900, bbox_inches='tight', pad_inches=0.05)
    plt.savefig('recording_locs.svg', dpi=900, format='svg',
                bbox_inches='tight', pad_inches=0.05)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to visualize recording locations in 2D.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    
    args = parser.parse_args()
    main(args.nwb_input_dir)
    
    
'''
python gen_figure2i_recordinglocs.py --nwb_input_dir /path/to/nwb_files/

e.g.:
python gen_figure2i_recordinglocs.py --nwb_input_dir /media/umit/easystore/bmovie_dandi/000623

'''
