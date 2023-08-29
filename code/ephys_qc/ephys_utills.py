#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np


# Function to assign colors based on channel names (brain areas)
from matplotlib.colors import to_rgb
def get_color(ch_name, xrgb=True):
    ch_name_u = ch_name.lower()
    
    if 'amy' in ch_name_u:
        return to_rgb('C1') if xrgb else 'C1'
    elif 'acc' in ch_name_u:
        return to_rgb('C0') if xrgb else 'C0'
    elif 'hip' in ch_name_u:
        return to_rgb('C3') if xrgb else 'C3'
    elif 'vmp' in ch_name_u:
        return to_rgb('C8') if xrgb else 'C8'
    elif 'pre' in ch_name_u:
        return to_rgb('C9') if xrgb else 'C9'
    else:
        raise SystemExit('problem in assigning colors!')


# ----- helper functions to timebin gaze data and generate gaze heatmaps from them -----
import warnings

def get_et_timebins(ET_data_pd, timebin_msec, do_op=None, data_unit='sec', 
                    fix_length=False, nbins=None, additional_column=None, fixation_info=None,
                    keep_timebin_index=True, etdata_colnames=['GazeX','GazeY']):
    
    #  do_op = 'mean' # None, 'mean' or 'maxrep'
    assert do_op in ['mean', 'maxrep', None], "do_op should be one of these: ['mean', 'maxrep', None]"
    
    if fix_length and nbins is None:
        raise SystemExit("if fix_length=True, then should enter 'nbins'!")
    
    if timebin_msec < 10: raise ValueError('Binsize should be in msecs!')
    if data_unit == 's' or data_unit == 'sec':
        ET_rec_time = ET_data_pd['RecTime'].values*1000. # second to msec.
    elif data_unit == 'ms' or data_unit == 'msec': 
        if ET_data_pd['RecTime'].values[2] < 1.:  
            print('\nBe sure that RecTime in the input is in msecs (first data: %3.3f)!'%(ET_data_pd['RecTime'].values[2])) 
        ET_rec_time = ET_data_pd['RecTime'].values # in msec.
    else:
        raise ValueError(f'Undefined data_unit: {data_unit}')

    # needed to split ET_data into chunks/bins of duration of a frame (in heatmap calculations this was binsize).
    chunks = (ET_rec_time//timebin_msec + 1).astype(int) # same as np.asarray([ np.divmod(ii,binsize)[0]+1 for ii in ET_rec_time ])
    if additional_column is None:
        ET_data4bin = np.hstack(( chunks.reshape(-1,1), ET_data_pd[etdata_colnames].values ))
    else:
        ET_data4bin = np.hstack(( chunks.reshape(-1,1), ET_data_pd[[*etdata_colnames, additional_column]].values ))
        # to take care of pupil diameter | not sure in which dataset we needed this
        if additional_column=='PupilDiameter':
            ET_data4bin[ET_data4bin[:,-1] == -1, -1] = np.NaN
            ET_data4bin[ET_data4bin[:,-1] == 0, -1] = np.NaN


    if fixation_info is not None:
        ET_data4bin[ fixation_info==0 ,1:] = np.NaN


    # use chunk indices to split data into bins/chunks.
    ET_data4hp_bins = np.split(ET_data4bin,np.where(np.diff(ET_data4bin[:,0]))[0]+1)
    if do_op=='mean': 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ET_data4hp_bins = [ np.nanmean(eii,axis=0) for eii in ET_data4hp_bins ]
    elif do_op=='maxrep':
        ET_data4hp_bins_dum = []
        for sil, eii in enumerate(ET_data4hp_bins):
            if np.isnan(np.sum(eii,axis=1)).sum() == len(eii): 
                ET_data4hp_bins_dum.append(eii[0])
            else: 
                unq_vals,unq_reps = np.unique(eii[~np.isnan(np.sum(eii,axis=1))],axis=0,return_counts=True) # sum+isnan to remove NaN data points. 
                if len(unq_reps)>1: # if the unique elements have the same frequency take one of them. It is better than averaging. 
                    ET_data4hp_bins_dum.append(unq_vals[np.argmax(unq_reps)])
                else: ET_data4hp_bins_dum.append(unq_vals[0]) # There is only one unique array. 
        
        ET_data4hp_bins = ET_data4hp_bins_dum


    # add padding to the beginning if there is a problem in ET recording start time. 
    if do_op=='mean' or do_op=='maxrep': 
        if ET_data4hp_bins[0][0] != 1:
            if additional_column is None:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN]) for ad_ii in range(1,int(ET_data4hp_bins[0][0])) ]
            else:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN, np.NaN]) for ad_ii in range(1,int(ET_data4hp_bins[0][0])) ]
            ET_data4hp_bins = add_part + ET_data4hp_bins # Add NaNs to the beginning.
    else: 
        if ET_data4hp_bins[0][0,0] != 1:
            if additional_column is None:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN]).reshape(-1,3) for ad_ii in range(1,int(ET_data4hp_bins[0][0,0])) ]
            else:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN, np.NaN]).reshape(-1,4) for ad_ii in range(1,int(ET_data4hp_bins[0][0,0])) ]
            ET_data4hp_bins = add_part + ET_data4hp_bins # Add NaNs to the beginning.


    # fill missing frames.
    ET_data4hp_bins_filled = []
    if do_op=='mean' or do_op=='maxrep':
        cnt_etd = 1
        for etd_ii in ET_data4hp_bins:
            while cnt_etd < etd_ii[0]:
                if additional_column is None:
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN]) )
                else:
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN, np.NaN]) )
                cnt_etd += 1
            ET_data4hp_bins_filled.append(etd_ii)
            cnt_etd += 1
    else: 
        cnt_etd = 1
        for etd_ii in ET_data4hp_bins:
            while cnt_etd < etd_ii[0,0]:
                if additional_column is None:
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN]).reshape(-1,3) )
                else:    
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN, np.NaN]).reshape(-1,4) )
                cnt_etd += 1
            ET_data4hp_bins_filled.append(etd_ii)
            cnt_etd += 1
            
    
    # Extend or cut to nframes.     
    if fix_length:
        if len(ET_data4hp_bins_filled) < nbins:
            while len(ET_data4hp_bins_filled) < nbins:
                if do_op=='mean' or do_op=='maxrep': 
                    if additional_column is None:
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0]+1, np.NaN, np.NaN]) ) # Extend with NaNs. 
                    else:
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0]+1, np.NaN, np.NaN, np.NaN]) ) # Extend with NaNs. 
                else:
                    if additional_column is None:
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0][0]+1, np.NaN, np.NaN]).reshape(-1,3)) # Extend with NaNs. 
                    else:    
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0][0]+1, np.NaN, np.NaN, np.NaN]).reshape(-1,4)) # Extend with NaNs. 
                        
        
        elif len(ET_data4hp_bins_filled) > nbins:
            ET_data4hp_bins_filled = ET_data4hp_bins_filled[:nbins]

        if not keep_timebin_index:
            if do_op=='mean' or do_op=='maxrep':
                ET_data4hp_bins_filled = [ exii[1:] for exii in ET_data4hp_bins_filled ]
            else:
                ET_data4hp_bins_filled = [ exii[:,1:] for exii in ET_data4hp_bins_filled ]


    return ET_data4hp_bins_filled



def compute_pixelperDVA(resolution_wh):
    """
    Computes the number of pixels per degree of visual angle

    Parameters
    ----------
    resolution_wh : tuple or list 
        screen resolution (width, height)

    Returns
    -------
    pixelperDVA : float
        pixels per degree of visual angle along horizontal and vertical axes.

    """
    screen_size_x = 40. # cm physical measures
    screen_size_y = 30. # cm physical measures
    distance_to_screen = 60. # cm physical measures
    
    # pixel_size_x = screen_size_x / resolution_wh[0]
    # pixel_size_y = screen_size_y / resolution_wh[1]
    
    # pixel_size = np.array([pixel_size_x, pixel_size_y])
    # DVAperpixel= np.rad2deg(2.*np.arctan(pixel_size/(2.*distance_to_screen)))
    # pixelperDVA = 1. / DVAperpixel

    pixelperDVA = np.tan(np.deg2rad(1.)/2.)*2*distance_to_screen* \
        np.divide(resolution_wh, [screen_size_x,screen_size_y])

    return pixelperDVA, pixelperDVA.mean()



from scipy.ndimage.filters import gaussian_filter

# --- Heatmap utilities ---
def get_heatmap_sci(x, y, sigma=None, framesize = [1000,1000] ):
    if sigma is None: 
        raise ValueError('Be sure that sigma value is proper!')
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=framesize, 
                                range = [[0,framesize[0]],[0,framesize[1]]])
    # needed .T due to the settings of histogram2d's output. 
    heatmap = gaussian_filter(heatmap.T, sigma=sigma, 
                              mode='constant',cval=0.0) 

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap, extent


def et_heatmap(et_xy_in, framesize, sigma, hp_down_factor, get_full=False, get_down=True,
               nan_ratio=1.0, cut_pad_down=None, cut_pad_up=None, trivial_return=True):
    # et_xy_in: [n_samples, 2], columns are x and y components
    # framesize = [frame_width,frame_height]
    
    if et_xy_in is not None and et_xy_in.ndim == 1:
        et_xy_in = et_xy_in[np.newaxis]
    
    # here we can eliminate video blocks which contain more than 50% problematic ET points.
    if et_xy_in is None or np.isnan(et_xy_in).any(axis=1).sum() >= nan_ratio*et_xy_in.shape[0]:
        trivial_output = True
        if get_down:
            if trivial_return:
                heatmap_down = None
            else:
                heatmap_down = np.zeros((np.array(framesize[::-1])/hp_down_factor).astype(int))
        if get_full:
            if trivial_return:
                heatmap = None
            else:
                heatmap = np.zeros((np.array(framesize[::-1])).astype(int))
    else:
        trivial_output = False
        if get_down:
            heatmap_down, _ = get_heatmap_sci(et_xy_in[:,0]/hp_down_factor,et_xy_in[:,1]/hp_down_factor,
                             sigma=sigma/float(hp_down_factor),framesize=(np.array(framesize)/hp_down_factor).astype(int))
        if get_full:
            heatmap, _ = get_heatmap_sci(et_xy_in[:,0],et_xy_in[:,1],sigma=sigma,framesize=framesize)


    if (not trivial_output) and (cut_pad_down is not None) and (cut_pad_up is not None):
        
        if get_down and not get_full:
            heatmap_down = heatmap_down[cut_pad_down//int(hp_down_factor):cut_pad_up//int(hp_down_factor)]
            return heatmap_down
        elif get_full and not get_down:
            heatmap = heatmap[cut_pad_down:cut_pad_up]
            return heatmap
        else:
            heatmap_down = heatmap_down[cut_pad_down//int(hp_down_factor):cut_pad_up//int(hp_down_factor)] 
            heatmap = heatmap[cut_pad_down:cut_pad_up]
            return heatmap_down, heatmap
            

    if get_down and not get_full:
        return heatmap_down
    elif get_full and not get_down:
        return heatmap
    else:
        return heatmap_down, heatmap


# ----- helper functions for assessing behavioral ROC curves for recognition task for the EMU sessions -----
from scipy.stats import norm
import math

def adjustHF(H, F, n_new, n_old):
    """
    adjust hit/false alarm rate for ceiling effects
    according to Macmillan&Creelman, pp8
    urut/nov06
    """

    if H == 1:
        H = 1 - 1/(2*n_old)
    if F == 1:
        F = 1 - 1/(2*n_new)
    if H == 0:
        H = 1/(2*n_old)
    if F == 0:
        F = 1/(2*n_new)

    return H, F

def cal_d_prime(typecounters, n_old, n_new):
    """
    calc D' (d) as well as z-transformed hit rate and false positive rate (zH, zF respectively).

    typeCounters is TP/FN/TN/FP, where positive=OLD

    according to Macmillan&Creelman, Eq 1.1, 1.2, 1.5;
    error estimation of d' is: Eq 13.4, 13.5

    """
    H = 0
    F = 0
    if n_old > 0:
        H = typecounters[0]/n_old
    if n_new > 0:
        F = typecounters[3]/n_new

    # % adjust
    # for perfect(1) and 0(all misses)
    # % acc
    # to
    # pp8
    H, F = adjustHF(H, F, n_new, n_old)

    zH = norm.ppf(H)
    zF = norm.ppf(F)

    d = zH - zF

    # % Eq
    # 13.4, pp325

    phi = lambda p: 1/np.sqrt(2*math.pi)*np.exp(-1/2*np.square(norm.ppf(p)))

    # % Eq
    # 13.4, pp325
    Hterm = H * (1 - H) / (n_old * np.square(phi(H)))
    Fterm = F * (1 - F) / (n_new * np.square(phi(F)))

    stdErr = Hterm + Fterm
    se = np.sqrt(stdErr)

    return d, zH, zF, H, F, se

def cal_auc(stats_all):
    """
    area under the curve AUC of an ROC
    Following eq 3.9, pp 64 of Macmillan book.
    partly copied from novelty/ROC/calcAUC.m (thus overlaps)
    TP/FP are expected to be ordered (ascending), but are not automatically resorted as this
    might introduce artifacts in case of non-monotonic ROCs.

    if reverseOrder=1, TP and FP are expected in descending order
    automatically adds the (0,0) and (1,1) point if it does not exist yet
    """
    TP = stats_all[:, 3]
    FP = stats_all[:, 4]

    if (TP[0] != 0) | (FP[0] != 0):
        TP = np.insert(TP, 0, 0)
        FP = np.insert(FP, 0, 0)

    if (TP[len(TP)-1] != 1) | (FP[len(FP)-1] != 1):
        TP = np.insert(TP, len(TP), 1)
        FP = np.insert(FP, len(FP), 1)

    auc = 0
    for i in range(1, len(TP)):
        auc = auc + (FP[i] - FP[i-1]) * TP[i-1]
        auc = auc + 1/2 * (TP[i]-TP[i-1]) * (FP[i]-FP[i-1])

    return auc


def dynamic_split(recog_response, ground_truth):
    """
    Split low/high confidence dynamically
    """
    n_response = len(recog_response)

    rep_counts = np.zeros(6)
    for i in range(1, 7):
        sum_up = np.sum(recog_response == i)
        rep_counts[i-1] = sum_up


    nr_conf1 = rep_counts[0] + rep_counts[5]
    nr_conf2 = rep_counts[1] + rep_counts[4]
    nr_conf3 = rep_counts[2] + rep_counts[3]

    split1 = nr_conf1 - (nr_conf2 + nr_conf3)
    split2 = (nr_conf1 + nr_conf2) - nr_conf3

    split_status = [split1, split2]

    split_mode = np.where(np.abs(split_status) == np.min(np.abs(split_status)))[0][0]+1

    # If there are no 1 and 6 response, use mode 2 to combine 1,2 and 5,6 for high confidence
    if (rep_counts[0] == 0) | (rep_counts[5] == 0):
        split_mode = 2
    if (rep_counts[1] == 0) | (rep_counts[4] == 0):
        split_mode = 1

    if split_mode == 1:
        # 1, (2,3), (4,5), 6
        # TP & FP
        ind_TP_high = np.where((ground_truth == 1) & (recog_response >= 6))
        ind_TP_low = np.where((ground_truth == 1) & (recog_response == 4))
        ind_FP_high = np.where((ground_truth == 0) & (recog_response >= 6))
        ind_FP_low = np.where((ground_truth == 0) & (recog_response == 4))

        # TN & FN
        ind_TN_high = np.where((ground_truth == 0) & (recog_response <= 1))
        ind_TN_low = np.where((ground_truth == 0) & ((recog_response == 3) | (recog_response == 2)))
        ind_FN_high = np.where((ground_truth == 1) & (recog_response <= 1))
        ind_FN_low = np.where((ground_truth == 1) & ((recog_response == 3) | (recog_response == 2)))

    else:
        # (1,2), 3, 4, (5,6)
        # TP & FP
        ind_TP_high = np.where((ground_truth == 1) & (recog_response >= 5))
        ind_TP_low = np.where((ground_truth == 1) & (recog_response == 4))
        ind_FP_high = np.where((ground_truth == 0) & (recog_response >= 5))
        ind_FP_low = np.where((ground_truth == 0) & (recog_response == 4))

        # TN & FN
        ind_TN_high = np.where((ground_truth == 0) & (recog_response <= 2))
        ind_TN_low = np.where((ground_truth == 0) & (recog_response == 3))
        ind_FN_high = np.where((ground_truth == 1) & (recog_response <= 2))
        ind_FN_low = np.where((ground_truth == 1) & (recog_response == 3))

    return split_status, split_mode, ind_TP_high, ind_TP_low, ind_FP_high, ind_FP_low, \
           ind_TN_high, ind_TN_low, ind_FN_high, ind_FN_low, n_response


def check_inclusion(recog_response, auc):
    """
    Check whether to include the session or not for confidence vs correctness
    """
    resp_count = []
    is_included = 1

    if auc < 0.6:
        is_included = 0

    for i in range(1, 7):
        resp_count.append(np.sum(recog_response == i))

    resp_count = np.asarray(resp_count)

    if (np.sum(resp_count[0:3] == 0) > 1) | (np.sum(resp_count[3:6] == 0) > 1):
        is_included = 0

    return is_included


