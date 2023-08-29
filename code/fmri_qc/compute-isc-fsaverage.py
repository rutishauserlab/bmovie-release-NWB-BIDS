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
import argparse


def get_subjects_isc(isc_datadir):
    fns = sorted(glob(os.path.join(isc_datadir, 'sub-*')))
    fns = [ os.path.basename(fn).split('_')[0] for fn in fns ]
    return fns

def zscore(array):
    array -= array.mean(0)
    array /= (array.std(0) + 1e-8)
    return array

def fast_mean_data(subjs_list,isc_datadir):
    mean = None
    for sii in subjs_list:
        print(f'loading {sii}')
        data_sii = np.load( os.path.join(isc_datadir, f'{sii}_data.npy'))
        
        if mean is None:
            mean = data_sii.copy()
        else:
            mean += data_sii
            
    mean /= len(subjs_list)
    return mean


def main(isc_datadir):

    subjects = np.array(get_subjects_isc(isc_datadir))
    print(len(subjects))
    
    n_subjects = len(subjects)
    
    correlations = []
    for isubject, subject in enumerate(subjects):
    
        mask_group = np.ones(n_subjects, dtype=bool)
        mask_group[isubject] = False
        mask_group = np.where(mask_group)[0]
    
        print(f"{subject}: {isubject}, group: {mask_group}")
    
        data_subject = np.load( os.path.join(isc_datadir, f'{subject}_data.npy') )
        data_group = fast_mean_data(subjects[mask_group], isc_datadir)
    
        # compute columnwise correlation
        n_samples = data_subject.shape[0]
        corr = (zscore(data_subject) * zscore(data_group)).sum(0) / n_samples
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    np.save(os.path.join(isc_datadir, 'isc-correlations-all-subjects-fsaverage.npy'), correlations)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load denoised and z-scored functional data to compute the ISC.")
    parser.add_argument('--isc_datadir', type=str, required=True, help='Directory containing data preprocessed by prepdata-isc-faverage.py')
    
    args = parser.parse_args()
    main(args.isc_datadir)


'''
python compute-isc-fsaverage.py --isc_datadir /path/to/isc_prepdata_directory/

e.g.:
python compute-isc-fsaverage.py --isc_datadir /media/umit/easystore/BangU01/iscdata

'''
