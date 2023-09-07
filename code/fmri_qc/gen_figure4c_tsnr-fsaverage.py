#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The tSNR values in fsaverage are plotted using pycortex library. 

The tSNR values were computed in compute-tsnr-fsaverage.py script.

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
from glob import glob
import cortex
import argparse

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def main(tsnr_datadir):
    
    # Run this code to download fsaverage surface for pycortex, labeled by Mark Lescroart
    cortex.utils.download_subject(subject_id='fsaverage', download_again=False)
    
    # We already pre-computed mean tSNR for each subject across all runs, so we need to
    # load those files and then compute the mean across subjects.
    mean_tsnr_left = sorted(glob(os.path.join(tsnr_datadir,'sub-*/*mean*hemi-L*npy')))
    mean_tsnr_right = sorted(glob(os.path.join(tsnr_datadir,'sub-*/*mean*hemi-R*npy')))
    
    mean_tsnr_both = []
    for left, right in zip(mean_tsnr_left, mean_tsnr_right):
        mean_tsnr_both.append(np.hstack((np.load(left), np.load(right))))
    
    tsnr_across_subjects = np.mean(mean_tsnr_both, 0)
    
    # set medial wall values to nan
    medial_wall = tsnr_across_subjects == 0.
    tsnr_across_subjects[medial_wall] = np.nan
    
    # enter explicitly here the pycortex default limits 
    # to get the same colormap with make_figure below
    vmin = np.percentile(np.nan_to_num(tsnr_across_subjects), 1)
    vmax = np.ceil(np.percentile(np.nan_to_num(tsnr_across_subjects), 99))
    
    # Plot on the cortex
    surface = cortex.Vertex(tsnr_across_subjects, 'fsaverage',
                            vmin=vmin,
                            vmax=vmax, 
                            cmap='hot'
                            );
    
    params = cortex.export.params_inflatedless_lateral_medial_ventral
    windowsize = (1600*2, 900*2)
    viewer_params = dict(
        labels_visible=[],
        overlays_visible=[]
        )
    
    fig = cortex.export.plot_panels(surface, windowsize=windowsize, 
                                    viewer_params=viewer_params, **params)
    fig.savefig('tsnr-fsaverage-group_inflated_mean-hotcmap.png', dpi=300)
    fig.savefig('tsnr-fsaverage-group_inflated_mean-hotcmap.svg', format='svg', dpi=300)
    plt.close()

    
    fig = cortex.quickflat.make_figure(surface, with_rois=False, with_curvature=True, 
                                       colorbar_location='right', height=1024)
    fig.savefig('tsnr-fsaverage-group_flatmap_mean-hotcmap.png', dpi=300)
    fig.savefig('tsnr-fsaverage-group_flatmap_mean-hotcmap.svg', format='svg', dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load precomputed tSNR values to show on fsaverage template.")
    parser.add_argument('--tsnr_datadir', type=str, required=True, help='Directory containing precomputed tSNR values.')
    
    args = parser.parse_args()
    main(args.tsnr_datadir)

    
'''
python gen_figure4c_tsnr-fsaverage.py --tsnr_datadir /path/to/tsnr_prep_dir

'''
