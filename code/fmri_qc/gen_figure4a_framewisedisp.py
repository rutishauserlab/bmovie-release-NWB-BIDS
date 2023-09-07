#!/usr/bin/env python3

'''

Framewise displacement in mm for each participant across all runs as shown in Fig. 4a.

---------------------------------------------------------------------------------
This code is adapted from:
https://github.com/mvdoc/budapest-fmri-data/tree/master/scripts/quality-assurance

Copyright 2020 Matteo Visconti di Oleggio Castello and Jiahui Guo

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------------------------------
'''


import os
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import argparse

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def get_subjects(fmriprep_dir):
    fns = sorted(glob(os.path.join(fmriprep_dir, 'sub-*/')))
    fns = [fn.split('/')[-2] for fn in fns]
    return fns


def load_add_idx(sub_fns, sub_idx):
    dfs = []
    for i, fn in enumerate(sub_fns):
        df = pd.read_csv(fn, sep='\t', skiprows=[1])
        df['subject'] = sub_idx
        df['run'] = i+1
        dfs.append(df)
    return dfs


def extract_columns(dfs):
    keep_columns = [
    'framewise_displacement',
    'rot_x', 'rot_y', 'rot_z',
    'trans_x', 'trans_y', 'trans_z'
    ,'subject', 'run']
    
    dfs = [df[keep_columns] for df in dfs]
    return pd.concat(dfs)


def all_sub_cols(fns, subjects):
    all_df = []
    for s in subjects:
        sub_files = [fn for fn in fns if s in fn]
        dfs = load_add_idx(sub_files, s)
        dfs_cols = extract_columns(dfs)
        all_df.append(dfs_cols)
        
    return all_df


def plot_all_sub_med(df, sids, col):
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
    pos = np.arange(len(sids))

    parts = ax.violinplot(df, positions=pos, showmedians=True);
    for pc in parts['bodies']:
        pc.set_facecolor('C0')
        pc.set_edgecolor('C0')
        pc.set_alpha(0.3)

    for p in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        parts[p].set_edgecolor('C0')

    ax.set_xticks(pos)
    ax.set_xticklabels(sids, fontsize=8, rotation=45, ha='center')
    ax.tick_params(axis='x', which='major', length=3, pad=0)

    if col == 'framewise_displacement':
        ylabel = 'Framewise displacement (mm)'
        ax.axhline(0.5, color='gray', linestyle=':', zorder=100)
        ax.set_yticks([0, 0.5, 1, 1.5, 2., 2.5, 3., 3.5])
    else:
        ylabel = col
        
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=8, length=3, pad=1)

    sns.despine()
    plt.tight_layout()
    outfile = 'group_median-{}.png'.format(col)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    fig.savefig(outfile.replace('png', 'svg'), dpi=300, format='svg', bbox_inches='tight')
    


def main(fmriprep_dir):
    
    fns = sorted(glob(os.path.join(fmriprep_dir,'*/ses-001/func/*tsv')))
    
    subjects = get_subjects(fmriprep_dir)
    
    dfs = all_sub_cols(fns, subjects)
    
    for col in ['framewise_displacement']:
        print("Working on {}".format(col))
        df_plot = [df[col].values for df in dfs]
        print("Plotting {} for the group".format(col))
        plot_all_sub_med(df_plot, subjects, col)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Loads fMRIprep-processed data to compute framewise displacement for each participant.")
    parser.add_argument('--fmriprep_dir', type=str, required=True, help='Directory containing fMRIprep-processed functional data.')
    
    args = parser.parse_args()
    main(args.fmriprep_dir)

    
'''
python gen_figure4a_framewisedisp.py --fmriprep_dir /path/to/fmriprep_directory/

'''

