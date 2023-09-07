#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Estimate tSNR in fsaverage for each subject after denoising data
(as implemented in budapestcode.utils.clean_data)

The mean tSNR across subjects are plotted on fsaverage template using pycortex in plot-tsnr-fsaverage.py

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
import numpy as np
import pandas as pd

import nibabel as nib
from glob import glob
import argparse

from budapestcode.utils import compute_tsnr

def get_subjects(datadir):
    fns = sorted(glob(os.path.join(datadir, 'sub-*/')))
    fns = [fn.split('/')[-2] for fn in fns]
    return fns



def main(fmriprep_dir, output_dir):
        
    os.makedirs(output_dir, exist_ok=True)
    
    subjects = get_subjects(fmriprep_dir)
    
    for subject in subjects:
    
        assert subject.startswith('sub-')
        
        print(f'\nprocessing {subject}...')
    
        func_fns = sorted(glob(f'{fmriprep_dir}/{subject}/ses-001/func/*_hemi-*_space-fsaverage_bold.func.gii'))
        conf_fns = sorted(glob(f'{fmriprep_dir}/{subject}/ses-001/func/*tsv'))
        conf_fns = sorted(conf_fns * 2)  # we have both L,R hemispheres
    
        # compute tSNR for every run
        tsnr_runs = []
        print("Computing tSNR")
        for f, c in zip(func_fns, conf_fns):
            print(f"  {f.split('/')[-1]}")
            data = nib.load(f)
            data = np.vstack([d.data for d in data.darrays])
            conf = pd.read_csv(c, sep='\t')
            # mask data removing the medial wall
            mask_medial_wall = data.std(0) != 0.
            data = data[:, mask_medial_wall].T  # transpose because of compute_tsnr
            tsnr = compute_tsnr(data, conf)
            tsnr_ = np.zeros_like(mask_medial_wall, dtype=float)
            tsnr_[mask_medial_wall] = tsnr
            tsnr_runs.append(tsnr_)
            
        # compute mean tsnr
        tsnr_mean_l = np.mean(tsnr_runs[::2], 0)
        tsnr_mean_r = np.mean(tsnr_runs[1::2], 0)
    
        # save the tSNR data for a group analysis
        tsnr_tosave = tsnr_runs + [tsnr_mean_l, tsnr_mean_r]
        run_types = [f'{i:02d}' for i in range(1, 3)] + ['mean']
    
        fnouts = []
        for run_type in run_types:
            for hemi in ['L', 'R']:
                fnout = f'{subject}_task-movie_run-{run_type}_space-fsaverage_hemi-{hemi}_desc-tsnr.npy'
                fnouts.append(fnout)
    
        tsnr_outdir = f'{output_dir}/{subject}'
        os.makedirs(tsnr_outdir, exist_ok=True)    
        
        for fnout, t in zip(fnouts, tsnr_tosave):
            print(fnout)
            fnout = f"{tsnr_outdir}/{fnout}"
            np.save(fnout, t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Loads fMRIprep-processed data to compute tSNR in fsaverage template.")
    parser.add_argument('--fmriprep_dir', type=str, required=True, help='Directory containing fMRIprep-processed functional data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save computed tSNR values.')
    
    args = parser.parse_args()
    main(args.fmriprep_dir, args.output_dir)

    
'''
python compute-tsnr-fsaverage.py --fmriprep_dir /path/to/fmriprep_directory/ --output_dir /path/to/tsnr_prep_dir

'''
