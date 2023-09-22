#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generate Table 1 by reading NWB and BIDS files

"""

# Required Libraries
import os
import numpy as np
import pandas as pd
from glob import glob
from pynwb import NWBHDF5IO
import argparse


# Function to extract subject IDs from MRI directory
def get_subjects(main_datadir):
    fns = sorted(glob(os.path.join(main_datadir, 'sub-*/')))
    fns = [fn.split('/')[-2] for fn in fns]
    return fns


def main(nwb_input_dir, bids_datadir):
    
    nwb_session_files = sorted(glob(os.path.join(nwb_input_dir, 'sub-*/*.nwb')))

    # Create an empty DataFrame to store session details
    session_df = pd.DataFrame(columns=['# of SU\nruns', '# of fMRI\nruns', 
                                       'Age', 'Sex', 'Epilepsy Diagnosis' ])
    
    # Process each NWB file and extract relevant information
    for session_ii in nwb_session_files:
        
        print(f'processing {os.path.basename(session_ii)}...')

        with NWBHDF5IO(session_ii,'r') as nwb_io: 
            # Read the NWB file
            nwbfile = nwb_io.read()
            
            # Extract session ID
            ses_id = nwbfile.identifier.split('_')[0]
            
            # Check if session ID is already in the DataFrame
            # If it exists, increment the SU runs count        
            if ses_id not in session_df.index:
                session_df.loc[ses_id, '# of SU\nruns'] = 1
            else:
                session_df.loc[ses_id, '# of SU\nruns'] += 1
                continue
            
            # Extract and store age, sex, and epilepsy diagnosis information
            session_df.loc[ses_id, 'Age'] = nwbfile.subject.age.removeprefix('P').removesuffix('Y')
            session_df.loc[ses_id, 'Sex'] = nwbfile.subject.sex
            session_df.loc[ses_id, 'Epilepsy Diagnosis'] = nwbfile.subject.description.removeprefix('epilepsy diagnosis: ')
            
    
    # Get list of subjects from MRI data directory
    subjects = get_subjects(bids_datadir)
    subjects = [ sii.removeprefix('sub-').upper() for sii in subjects ]
    
    # For each subject from MRI directory, check if they exist in the session DataFrame
    # If not, add placeholder values for them
    for sii in subjects:
        if sii not in session_df.index:
            session_df.loc[sii, 'Age'] = '??'
            session_df.loc[sii, 'Sex'] = '??'
            session_df.loc[sii, 'Epilepsy Diagnosis'] = 'NA'
            session_df.loc[sii, '# of SU\nruns'] = np.NaN
    
        # Set the fMRI runs count to 2 for each subject
        session_df.loc[sii, '# of fMRI\nruns'] = 2
        
        
    # Manual data entries for specific subjects (fMRI only subjects):
    session_df.loc['P45CS', 'Age'] = '29'
    session_df.loc['P45CS', 'Sex'] = 'F'
    session_df.loc['P45CS', 'Epilepsy Diagnosis'] = 'Bitemporal'
    
    session_df.loc['P46CS', 'Age'] = '41'
    session_df.loc['P46CS', 'Sex'] = 'M'
    
    session_df.loc['P50CS', 'Age'] = '25'
    session_df.loc['P50CS', 'Sex'] = 'M'
    session_df.loc['P50CS', 'Epilepsy Diagnosis'] = 'Right Temporal Neocortical'
    
    session_df.loc['P59CS', 'Age'] = '34'
    session_df.loc['P59CS', 'Sex'] = 'M'
    session_df.loc['P59CS', 'Epilepsy Diagnosis'] = 'Left Mesial Temporal'
    
    
    # Sort the DataFrame by session ID and compute some summary statistics
    session_df.sort_index(inplace=True)
    session_df.index.name = 'ID'
    tot_su = np.nansum(session_df['# of SU\nruns'].to_numpy(dtype=float))
    tot_fmri = np.nansum(session_df['# of fMRI\nruns'].to_numpy(dtype=float))
    age_mean = np.mean(session_df['Age'].to_numpy(dtype=float)).round(2)
    age_std = np.std(session_df['Age'].to_numpy(dtype=float)).round(2)
    
    # Add a row with summary statistics to the DataFrame
    add_row = {'# of SU\nruns':[f'Total # of\nSU runs: {int(tot_su)}'], 
               '# of fMRI\nruns':[f'Total # of\nfMRI runs: {int(tot_fmri)}'], 
               'Age':[f'Mean (SD) age:\n{age_mean} ({age_std})'], 
               'Sex':[f"{sum(session_df['Sex']=='F')} Female"],
               'Epilepsy Diagnosis':['']
               }
    
    df_add = pd.DataFrame(data=add_row)
    df_add.index = [f'Total # of\nsubjects: {len(session_df)}']
    df_add.index.name = 'ID'
    df = pd.concat([session_df, df_add])
    
    # Save the final DataFrame to a CSV file
    df.to_csv('Table1_subjs_info.csv', na_rep='NA')
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Read NWB files and MRI BIDS file names to generate Table 1')
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    parser.add_argument('--bids_datadir', type=str, required=True, help='Directory containing BIDS files.')

    args = parser.parse_args()

    main(args.nwb_input_dir, args.bids_datadir)
    
    
'''
run using:
python gen_table1_subj_info.py --nwb_input_dir /path/to/nwb_files/ --bids_datadir /path/to/bids_files/

'''
