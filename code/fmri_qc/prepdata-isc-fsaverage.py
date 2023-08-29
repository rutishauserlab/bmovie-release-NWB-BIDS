#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The scripts prepdata-isc-fsaverage.py and compute-isc-fsaverage.py compute
inter-subject correlation on data projected to fsaverage, after denoising data 
(as implemented in budapestcode.utils.clean_data). 

The mean ISC across subjects is plotted with pycortex in gen_figure5_isc-fsaverage.py.

Note that the original script at:
https://github.com/mvdoc/budapest-fmri-data/blob/master/scripts/quality-assurance/compute-isc-fsaverage.py
is split into two parts: prepdata-isc-fsaverage.py and compute-isc-fsaverage.py here
to improve memory efficieny during computation.


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
from glob import glob
import numpy as np
import nibabel as nib
import pandas as pd
import resampy
import argparse

from budapestcode.utils import clean_data

# Process is
# 1. Load runs
# 2. Clean up signal
# 3. zscore runs


def load_gifti(fn):
    data = nib.load(fn)
    data = np.vstack([d.data for d in data.darrays])
    return data

def get_subjects(fmriprep_dir):
    fns = sorted(glob(os.path.join(fmriprep_dir, 'sub-*/')))
    fns = [fn.split('/')[-2] for fn in fns]
    return fns

def load_data(subject, fmriprep_dir):
    print(f"\nLoading data for subject {subject}")
    datadir = f"{fmriprep_dir}/{subject}/ses-001/func"
    data = []
    for irun in range(1, 3):
        data_ = []
        for hemi in ['L', 'R']:
            print(f"  run{irun:d}, hemi-{hemi}")
            fn = f"{subject}_ses-001_task-movie_run-{irun:02d}_hemi-{hemi}_space-fsaverage_bold.func.gii"
            data_.append(load_gifti(os.path.join(datadir, fn)))
            print( load_gifti(os.path.join(datadir, fn)).shape )
        data.append(np.hstack(data_))
    return data

def load_confounds(subject, fmriprep_dir):
    confounds = []
    confounds_fn = sorted(glob(os.path.join(fmriprep_dir,
                                    f'{subject}/ses-001/func/*tsv')))
    for conf in confounds_fn:
        print(conf.split('/')[-1])
        confounds.append(pd.read_csv(conf, sep='\t'))
    return confounds

def zscore(array):
    array -= array.mean(0)
    array /= (array.std(0) + 1e-8)
    return array


def main(fmriprep_dir, isc_outdir):
    
    subjects = get_subjects(fmriprep_dir)
    print(len(subjects))
    
    os.makedirs(isc_outdir, exist_ok=True)
    
    nsample_init = None
    for subject in subjects:
        data_subject = load_data(subject, fmriprep_dir)
        confounds = load_confounds(subject, fmriprep_dir)
        data_subject = np.vstack([zscore(clean_data(dt, conf)) \
                                  for dt, conf in zip(data_subject, confounds)])
        print(data_subject.shape)
    
        if nsample_init is None:
            nsample_init = data_subject.shape[0]
            
        if nsample_init != data_subject.shape[0]:
            print(f'Apply resampling from {data_subject.shape[0]} to {nsample_init}...')
            data_subject = resampy.resample(data_subject, data_subject.shape[0], nsample_init, axis=0, parallel=True)
            print(data_subject.shape)
    
        np.save(os.path.join(isc_outdir, f'{subject}_data.npy'), data_subject)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load fMRIprep-processed data to denoise and z-score functional data.")
    parser.add_argument('--fmriprep_dir', type=str, required=True, help='Directory containing fMRIprep-processed functional data.')
    parser.add_argument('--isc_outdir', type=str, required=True, help='Directory to save denoised and z-scored functional data.')
    
    args = parser.parse_args()
    main(args.fmriprep_dir, args.isc_outdir)


'''
python prepdata-isc-fsaverage.py --fmriprep_dir /path/to/fmriprep_directory/ --isc_outdir /path/to/isc_prepdata_directory/

e.g.:
python prepdata-isc-fsaverage.py --fmriprep_dir /media/umit/easystore/BangU01/fmriprep_21p0p2 --isc_outdir /media/umit/easystore/BangU01/iscdata

'''
