#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Assess and plot behavioral ROC curves for recognition task for the EMU sessions

This code and used helper functions are adapted from:
https://github.com/rutishauserlab/recogmem-release-NWB/tree/master/RutishauserLabtoNWB/events/newolddelay/python/analysis

"""

import os
import numpy as np

from pynwb import NWBHDF5IO
from ephys_utills import cal_d_prime, cal_auc, check_inclusion, dynamic_split

import matplotlib.pyplot as plt
import argparse

# Set Matplotlib SVG font type
plt.rcParams['svg.fonttype'] = 'none'

def main(nwb_input_dir):
    
    # Extract session IDs from NWB files
    session_ids = [ f for f in sorted(os.listdir(nwb_input_dir)) if f.endswith('.nwb') ]
    
    aucAll = []
    stats_all_X = np.zeros((len(session_ids), 7))
    stats_all_Y = np.zeros((len(session_ids), 7))
    accuracies_high = []
    accuracies_low = []
    
    index = 0
    for session_ii in session_ids:
        
        print(f'processing {session_ii}...')
        filepath = os.path.join(nwb_input_dir,session_ii)
        
        # Open the NWB file and read its content
        with NWBHDF5IO(filepath,'r') as nwb_io:
            nwbfile = nwb_io.read()
    
            trials_df = nwbfile.trials.to_dataframe()
            recog_new0_old1 = trials_df['stimulus_file'].to_numpy(dtype=str)[1:]
            recog_new0_old1 = np.array([ 0 if 'new' in nii.lower() else 1 for nii in recog_new0_old1 ])
            
            recog_resp_correct = trials_df['response_correct'].to_numpy()[1:]
            assert len(recog_new0_old1) == len(recog_resp_correct)
    
            actual_resp_button = trials_df['actual_response'].to_numpy()[1:] 
            assert np.isin(actual_resp_button, np.arange(1,6+1)).all() # responses should be within [1,6]
    
            actual_resp_new0_old1 = [ 0 if aii in [1,2,3] else 1 for aii in actual_resp_button]
    
            # --- just a control - can be skipped --- 
            actual_resp_new0_old1_altv = []
            for gt_newold, resp_tf in zip(recog_new0_old1.astype(bool),recog_resp_correct):    
                if resp_tf: actual_resp_new0_old1_altv.append(gt_newold)
                else: actual_resp_new0_old1_altv.append(np.logical_not(gt_newold))
            actual_resp_new0_old1_altv = np.array(actual_resp_new0_old1_altv).astype(int) 
            assert np.array_equal(actual_resp_new0_old1,actual_resp_new0_old1_altv)
            # --- o ---
        
        
        response_recog = actual_resp_button
        new_old_labels = recog_new0_old1 # ground_truth
        
        typecounter = []
        for i in range(6, 0, -1):
            new_old_labels_selected = new_old_labels[response_recog == i]
            nTP = np.sum(new_old_labels_selected == 1)
            nFN = 0
            nTN = 0
            nFP = np.sum(new_old_labels_selected == 0)
            typecounter.append([nTP, nFN, nTN, nFP])
        typecounter =  np.asarray(typecounter).T
    
        n_old = np.sum(typecounter[0, :])
        n_new = np.sum(typecounter[3, :])
    
        stats_all = []
        for i in range(typecounter.shape[1]):
            temp = np.sum(typecounter[:, 0:i+1], axis=1)
            d1, zH1, zF1, H, F, se = cal_d_prime(temp, n_old, n_new);
            stats_all.append([d1, zH1, zF1, H, F, se])
    
        stats_all =  np.asarray(stats_all)
        auc = cal_auc(stats_all)
        aucAll.append(auc)
        
        x = stats_all[0:5, 4]
        y = stats_all[0:5, 3]
        
        x = np.insert(x, 0, 0)
        x = np.append(x, 1)
        y = np.insert(y, 0, 0)
        y = np.append(y, 1)
        
        stats_all_X[index] = x
        stats_all_Y[index] = y
        
        index += 1
           
        is_included = check_inclusion(response_recog, auc)
    
        if is_included:
            split_status, split_mode, ind_TP_high, ind_TP_low, ind_FP_high, ind_FP_low, ind_TN_high, \
            ind_TN_low, ind_FN_high, ind_FN_low, n_response = dynamic_split(response_recog, new_old_labels)
            
            nr_TN_high = len(ind_TN_high[0])
            nr_TP_high = len(ind_TP_high[0])
            nr_TN_low = len(ind_TP_high[0]) + len(ind_TP_low[0])
            nr_TP_low = len(ind_TP_low[0])
            nr_TN_low = len(ind_TN_low[0])
    
            nr_high_response = len(ind_TN_high[0]) + len(ind_TP_high[0]) + len(ind_FN_high[0]) + len(ind_FP_high[0])
            nr_low_response = len(ind_TN_low[0]) + len(ind_TP_low[0]) + len(ind_FN_low[0]) + len(ind_FP_low[0])
        
            per_accuracy_high = (nr_TN_high + nr_TP_high) / nr_high_response
            per_accuracy_low = (nr_TN_low + nr_TP_low) / nr_low_response
    
            accuracies_high.append(per_accuracy_high*100)
            accuracies_low.append(per_accuracy_low*100)
            
    
    # ----- Plot computed values -----
    fig, axes = plt.subplots(1,2, figsize = (5.3, 2.2))
    
    mean_X = np.mean(stats_all_X, axis = 0)
    mean_Y = np.mean(stats_all_Y, axis = 0)
    
    from scipy.stats import ttest_rel
    tval, pval = ttest_rel(accuracies_high,accuracies_low)
    print(ttest_rel(accuracies_high,accuracies_low))
    
    fcol = '#006f91'
    axes[0].plot(mean_X, mean_Y, color=fcol, alpha=1, 
                 linewidth=1.3, markersize=4, marker='D')
    
    for ii in range(0,len(stats_all_X)):
        axes[0].plot(stats_all_X[ii], stats_all_Y[ii], 
                     color='gray', alpha=0.15, marker='.')
        
    axes[0].set_ylim(0,1)
    axes[0].set_xlim(0,1)
    axes[0].set_xlabel('False positive rate', fontsize=8)
    axes[0].set_ylabel('True positive rate',  fontsize=8)
    axes[0].plot([0, 1], [0, 1], color='black', alpha=0.90, linewidth=0.95)
    axes[0].tick_params(axis='both', which='major', labelsize=7, length=3, pad=2)
    axes[0].set_aspect('equal', 'box')
    
    
    fcol = '#006f91'
    x_axis = ['High', 'Low']
    axes[1].plot(x_axis, [accuracies_high, accuracies_low], 
                 marker='o', color=fcol, alpha=0.6, markersize=4,
                 markerfacecolor=fcol, markeredgecolor=None)
    
    axes[1].set_xlabel('Confidence', fontsize=8)
    axes[1].set_ylabel('Accuracy (% correct)', fontsize=8)
    axes[1].tick_params(axis='x', which='major', labelsize=8, length=3, pad=2)
    axes[1].tick_params(axis='y', which='major', labelsize=7, length=3, pad=2)
    
    [x1, x2] = axes[1].get_xlim()
    
    axes[1].errorbar([x1-0.15, x2+0.15], 
                     [np.mean(accuracies_high), np.mean(accuracies_low)], 
                     yerr=[np.std(accuracies_high), np.std(accuracies_low)],
                     fmt='s', capsize=3, capthick=1., markersize=4, color=fcol,
                     )
    
    def label_diff(ax,i,j,x_vals,y_vals,pval,yshift=0.0,h=0.01,text='*'):
        if pval < 0.05: 
            if pval < 0.0001:  text = 'p<0.0001'
            else: text = r'p=%1.4f'%pval
            # else: text = r'p$\approx$%1.3f'%pval
        else: 
            return
        
        x1, x2 = x_vals[i], x_vals[j]
        y = max(y_vals[i], y_vals[j]) + yshift
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8,color='k')
    
        if len(text)>4:
            ax.text((x1+x2)*.5+0.1, y+h+0.01, text, ha='center', va='bottom',color='k',
                    fontsize=7, fontweight='normal',rotation=0)
        else:
            ax.text((x1+x2)*.5, y+h-0.025, text, ha='center', va='bottom',
                    color='k',fontweight='normal')   
        
    signf_max = 102
    # order: pval_MZvsDZ, pval_MZvsRd, pval_DZvsRd
    label_diff(axes[1],0,1,[x1-0.15, x2+0.15],[signf_max,signf_max],pval,yshift=-0.008)
    
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].get_xaxis().tick_bottom()
    axes[1].get_yaxis().tick_left()
    axes[1].margins(0.1)
    
    plt.tight_layout()
    plt.savefig('recognition_roc.png', dpi=300)
    plt.savefig('recognition_roc.svg', dpi=300)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NWB files to plot behavioral ROC curves for recognition task for the EMU sessions.")
    parser.add_argument('--nwb_input_dir', type=str, required=True, help='Directory containing NWB files.')
    
    args = parser.parse_args()
    main(args.nwb_input_dir)
    
    
'''
python gen_figure3_recogtask_roc.py --nwb_input_dir /path/to/nwb_files/

e.g.:
python gen_figure3_recogtask_roc.py --nwb_input_dir /media/umit/easystore/bmovie_NWBfiles

'''   
    
