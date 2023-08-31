#!/usr/bin/env python
# coding: utf-8

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

import numpy as np
import cortex
import argparse

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def main(corrs_data_file):
    
    # Run this code to download fsaverage surface for pycortex, labeled by Mark Lescroart
    cortex.utils.download_subject(subject_id='fsaverage', download_again=False)
    
    data = np.load(corrs_data_file)
    
    # --- Fisher z-transformation ---
    data_z = np.arctanh(data)
    isc_across_subjects = np.tanh(np.mean(data_z, 0) )
    
    # --- plot data using pycortex and fsaverage template ---
    surface = cortex.Vertex(isc_across_subjects, 'fsaverage', cmap='hot', 
                            vmin=0, 
                            vmax=np.percentile(np.nan_to_num(isc_across_subjects), 99).round(1)
                            )
    
    params = cortex.export.params_inflatedless_lateral_medial_ventral
    windowsize = (1600*2, 900*2)
    viewer_params = dict(labels_visible=[],
                         overlays_visible=[]
                         )
    
    fig = cortex.export.plot_panels(surface, windowsize=windowsize, viewer_params=viewer_params, **params)
    fig.savefig('isc-fsaverage-mean-hotcmap.png', dpi=300)
    fig.savefig('isc-fsaverage-mean-hotcmap.svg', dpi=300, format='svg')
    plt.close()
    
    fig = cortex.quickflat.make_figure(surface, with_rois=True, colorbar_location='right', height=1024)
    fig.savefig('isc-fsaverage-flatmap_mean-hotcmap.png', dpi=300)
    fig.savefig('isc-fsaverage-flatmap_mean-hotcmap.svg', dpi=300, format='svg')
    plt.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Loads pre-computed inter-subject correlation values to plot on the fsaverage template.")
    parser.add_argument('--corrs_data_file', type=str, required=True, help='File containing pre-computed correlation values.')
    
    args = parser.parse_args()
    main(args.corrs_data_file)
    
    
'''
python gen_figure5_isc-fsaverage.py --corrs_data_file /path/to/corrs_data_file

e.g.:
python gen_figure5_isc-fsaverage.py --corrs_data_file /media/umit/easystore/BangU01/iscdata/isc-correlations-all-subjects-fsaverage.npy

'''
