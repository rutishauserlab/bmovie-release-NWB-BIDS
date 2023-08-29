#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Estimate tSNR in the subject's native space (volume) for each subject after denoising data
(as implemented in budapestcode.utils.clean_data)

The tSNR values in native space are plotted in a violin plot across subjects in plot-tsnr-group.py. 

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
import nilearn.image as nimage
from glob import glob
import argparse

from budapestcode.utils import compute_tsnr

def get_subjects(fmriprep_dir):
    fns = sorted(glob(os.path.join(fmriprep_dir, 'sub-*/')))
    fns = [fn.split('/')[-2] for fn in fns]
    return fns

def main(fmriprep_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    subjects = get_subjects(fmriprep_dir)
    
    for subject in subjects:
    
        assert subject.startswith('sub-')
        
        print(f'\nprocessing {subject}...')
        
        func_fns = sorted(glob(f'{fmriprep_dir}/{subject}/ses-001/func/*space-T1w_desc-preproc_bold.nii.gz'))
        conf_fns = sorted(glob(f'{fmriprep_dir}/{subject}/ses-001/func/*tsv'))
    
        # compute tSNR for every run
        tsnr_runs = []
        print("Computing tSNR")
        for f, c in zip(func_fns, conf_fns):
            print(f"  {f.split('/')[-1]}")
            data = nib.load(f).get_fdata()
            print(data.shape)
            conf = pd.read_csv(c, sep='\t')
            tsnr_runs.append(compute_tsnr(data, conf))
        
        # compute mean tsnr
        tsnr_mean = np.mean(tsnr_runs, 0)
        
        # save the tSNR data for a group analysis
        tsnr_tosave = tsnr_runs + [tsnr_mean]
        run_types = [f'{i:02d}' for i in range(1, 3)] + ['mean']
        
        tsnr_outdir = f'{output_dir}/{subject}'
        os.makedirs(tsnr_outdir, exist_ok=True)
        
        for run, t in zip(run_types, tsnr_tosave):
            t_img = nimage.new_img_like(func_fns[0], t)
            fnout = f'{subject}_task-movie_run-{run}_space-T1w_desc-tsnr.nii.gz'
            print(fnout)
            fnout = f"{tsnr_outdir}/{fnout}"
            t_img.to_filename(fnout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Loads fMRIprep-processed data to compute tSNR in the subject's native space.")
    parser.add_argument('--fmriprep_dir', type=str, required=True, help='Directory containing fMRIprep-processed functional data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save computed tSNR values.')
    
    args = parser.parse_args()
    main(args.fmriprep_dir, args.output_dir)

    
'''
python compute-tsnr-volume.py --fmriprep_dir /path/to/fmriprep_directory/ --output_dir /path/to/tsnr_prep_dir

e.g.:
python compute-tsnr-volume.py --fmriprep_dir /media/umit/easystore/BangU01/fmriprep_21p0p2 --output_dir /media/umit/easystore/BangU01/tsnr_prep

'''
