#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The tSNR values in native volumetric space are plotted in a violin plot across subjects. 

The tSNR values were computed in compute-tsnr-volume.py script.

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
"""

import os
import nibabel as nib
from glob import glob

import numpy as np
import seaborn as sns
import argparse

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def get_subjects(datadir):
    fns = sorted(glob(os.path.join(datadir, 'sub-*/')))
    fns = [fn.split('/')[-2] for fn in fns]
    return fns


def make_conjuction_mask(subject, fmriprep_dir):
    mask_fns = glob(os.path.join(fmriprep_dir,
                 f'{subject}/ses-001/func/*space-T1w_desc-brain_mask.nii.gz'))
    brainmask = 1.
    for mask_fn in mask_fns:
        bm = nib.load(mask_fn).get_fdata()
        brainmask *= bm
    return brainmask.astype(bool)


def main(fmriprep_dir, tsnr_datadir):
    
    subjects = get_subjects(fmriprep_dir)
    
    tsnr_template_filename = os.path.join(tsnr_datadir,
                      '{subject}/{subject}_task-movie_run-mean_space-T1w_desc-tsnr.nii.gz')
    
    tsnr_subject = []
    for subject in subjects:
        tsnr = nib.load(tsnr_template_filename.format(subject=subject)).get_fdata()
        mask_subject = make_conjuction_mask(subject, fmriprep_dir)
        tsnr_subject.append(tsnr[mask_subject])
    
    subject_ordering = np.arange(len(tsnr_subject))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
    pos = np.arange(len(tsnr_subject))
    
    tsnr_subject_ordered = [tsnr_subject[i] for i in subject_ordering]
    parts = ax.violinplot(tsnr_subject_ordered, positions=pos, showmedians=True);
    for pc in parts['bodies']:
        pc.set_facecolor('C0')
        pc.set_edgecolor('C0')
        pc.set_alpha(0.3)
    
    for p in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        parts[p].set_edgecolor('C0')
    
    ax.set_xticks(pos)
    ax.set_xticklabels([subjects[i] for i in subject_ordering], fontsize=8, 
                       rotation=45, ha='center')
    ax.tick_params(axis='x', which='major', length=3, pad=0)
    
    
    ax.set_ylabel('tSNR', fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=8, length=3, pad=1)
    
    sns.despine()
    plt.tight_layout()
    
    fig.savefig("group_mean-tsnr.png", dpi=600, bbox_inches='tight')
    fig.savefig("group_mean-tsn.svg", format='svg', dpi=600, bbox_inches='tight')
    
    # report values to use in the manuscript
    mean_tsnr_for_each_subject = [np.mean(t) for t in tsnr_subject]
    mean_across_subjects = np.mean(mean_tsnr_for_each_subject)
    std_across_subjects = np.std(mean_tsnr_for_each_subject)
    
    print(f"Mean tSNR is {mean_across_subjects:.2f} ± {std_across_subjects:.2f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load precomputed volumetric tSNR values to show in a violin plot.")
    parser.add_argument('--fmriprep_dir', type=str, required=True, help='Directory containing fMRIprep-processed functional data.')
    parser.add_argument('--tsnr_datadir', type=str, required=True, help='Directory containing precomputed tSNR values.')
    
    args = parser.parse_args()
    main(args.fmriprep_dir, args.tsnr_datadir)

    
'''
python gen_figure4b_tsnr-nativeviolin.py --fmriprep_dir /path/to/fmriprep_directory/ --tsnr_datadir /path/to/tsnr_prep_dir

e.g.:
python gen_figure4b_tsnr-nativeviolin.py --fmriprep_dir /media/umit/easystore/BangU01/fmriprep_21p0p2 --tsnr_datadir /media/umit/easystore/BangU01/tsnr_prep

'''
