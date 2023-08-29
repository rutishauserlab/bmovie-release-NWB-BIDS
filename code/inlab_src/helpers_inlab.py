#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import numpy as np
import pandas as pd
from itertools import groupby


# Load QC metrics from matlab outputs
datafname_map_bdm = {'P41CS_R1':'p41cs_092316_fMRI',
                 'P41CS_R2':'p41cs_092316_fMRI',
                 'P42CS_R1':'p42cs_081016_movie1',
                 'P42CS_R2':'p42cs_081116_movie2',
                 'P43CS_R1':'p43cs_110516_movie',
                 'P43CS_R2':'p43cs_110516_movie',
                 'P44CS_R1':'p44cs_083116_movie1',
                 'P47CS_R1':'p47cs_021717_movie1',
                 'P47CS_R2':'p47cs_021917_movie2-fLoc-movieFull',
                 'P48CS_R1':'p48cs_030517_LOI-movie1-fLoc',
                 'P48CS_R2':'p48cs_030917_movie2-fLoc-movieFull',
                 'P49CS_R1':'p49cs_051917_LOI-movie1',
                 'P49CS_R2':'p49cs_052317_LOI-movie2-movieFull',
                 'P51CS_R1':'p51cs_062517_socnsLOI-movie1',
                 'P51CS_R2':'p51cs_062717_movie2-contNO-movieFull',
                 'P53CS_R1':'p53csA_110417_socnsLOI-movie1',
                 'P53CS_R2':'p53csB_110817_movie2-LOI-sac-movieFull',
                 'P54CS_R1':'p54cs_011918_socnsLOI-movie1-sac',
                 'P54CS_R2':'p54cs_012018_movie2-sac-movieFull',
                 'P55CS_R1':'p55cs_031018_movie1-LOI-sacc',
                 'P55CS_R2':'p55cs_031318_movie2',
                 'P56CS_R1':'p56cs_042118_LOI-movie1',
                 'P56CS_R2':'p56cs_042218_movie2',
                 'P57CS_R1':'p57cs_062718_socnsLOI-movie1',
                 'P57CS_R2':'p57cs_062818_movie2-movieFull',
                 'P58CS_R1':'p58cs_071618_scnsLOI-movie1-pixar-sac',
                 'P60CS_R1':'p60cs_100918_movie1',
                 'P62CS_R1':'p62cs_042419_movie12Full-fLoc',
                 'P62CS_R2':'p62cs_042419_movie12Full-fLoc'
                 }

# inverted datafname_map_bdm dictionary with hyphen to underscore replacement 
# - [removing keys corresponding to movie12]
datafname_map_ur = {}
for k,v in datafname_map_bdm.items():
    if v.replace('-','_') not in datafname_map_ur:
        datafname_map_ur[v.replace('-','_')] = k


# from: https://stackoverflow.com/a/8832212
import scipy.io as spio
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], spio.matlab.mat_struct):
            d[key] = _todict(d[key])
    return d        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dd = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dd[strg] = _todict(elem)
        else:
            dd[strg] = elem
    return dd



def load_qcdata(qcdata_dir,phase):
    
    if phase=='encoding':
        clusterstats = loadmat(os.path.join(qcdata_dir,'clusterStats_all_phase0.mat'))['clusterStats_all']
        # columns = [ sessionID channel clNr1 clNr2 d Rsquare1 Rsquare2 mode] 
        projall = loadmat(os.path.join(qcdata_dir,'statsProjAll_phase0.mat'))['statsProjAll']
    elif phase =='recognition':
        clusterstats = loadmat(os.path.join(qcdata_dir,'clusterStats_all_phase1.mat'))['clusterStats_all']
        # columns = [ sessionID channel clNr1 clNr2 d Rsquare1 Rsquare2 mode] 
        projall = loadmat(os.path.join(qcdata_dir,'statsProjAll_phase1.mat'))['statsProjAll']
    else:
        raise ValueError("phase should be 'encoding' or 'recognition'...")
    
    # ---- Get relevant QC metrics for each cell and separation distances for each channel ----
    qc_data = {}
    d_proj = {}
    for cnt_ii, cii in enumerate(clusterstats):
        cell_ii = cii.__dict__
        
        sessionid_mat = cell_ii['sessionID'].replace(' ', '_')
        sessionid = datafname_map_ur[sessionid_mat.replace('-', '_')]

        # required for combined sessions: p41cs, p43cs, p62cs.
        if sessionid[-1] == '1' and str(cell_ii['indRun']) == '2':
            sessionid = sessionid[:-1] + str(cell_ii['indRun']) 
    
        cell_id = f"{sessionid}_{cell_ii['channel']}_{cell_ii['cellNr']}_{cell_ii['origClusterID']}_{cell_ii['brainAreaOfCell']}"

    
        # ----- get corresponding projall values -----
        ch_id = f"{sessionid}_{cell_ii['channel']}"
        bool1 = projall[:,0] == cell_ii['indSess']
        bool2 = projall[:,1] == cell_ii['channel']
        use_rows = bool1 & bool2
        
        true_blocks = [ np.array(list(g))[:,0] for k, g in groupby(enumerate(use_rows), key=lambda x: x[-1]) if k]
        assert len(true_blocks) in [1, 2]
    
        projall_use = projall[true_blocks[cell_ii['indRun']-1]].copy()
        if len(projall_use)==1:
            assert np.isnan(projall_use[0,4]) or projall_use[0,4]==0
    
        # remove nans
        projall_use = projall_use[~np.isnan(projall_use[:,4])]
        if len(projall_use)>0:
        
            projall_use_uq = np.unique(projall_use, axis=0)
            assert len(projall_use_uq) == len(np.unique(projall_use[:,:4], axis=0))
        
            # check whether projall_use_uq looks like the upper triangular matrix of pair distances
            uniq_cells = np.unique(np.r_[projall_use_uq[:,2],projall_use_uq[:,3]])
            assert len(projall_use_uq) == (len(uniq_cells)**2 - len(uniq_cells)) /2
            
            if ch_id not in d_proj:
                d_proj[ch_id] = projall_use_uq[:,2:5]
            else:
                assert np.array_equal(d_proj[ch_id], projall_use_uq[:,2:5])
        # ----- o -----

        
        rate = cell_ii['rate']
        if rate==0: # no spike during the task.
            # print(cell_id, cell_ii['rate'], cell_ii['rate_full'])
            cell_qcdata = {'meansnr':np.nan, 'isibelow':np.nan, 
                           'cv2':np.nan, 'peaksnr':np.nan, 
                           'isoldist':np.nan, 'm_waveform':np.nan,
                           'rate':cell_ii['rate'], 'rate_full':cell_ii['rate_full']}
        else:
            cell_qcdata = {'meansnr':cell_ii['unitStats'][5], 'isibelow':cell_ii['unitStats'][7], 
                           'cv2':cell_ii['unitStats'][10], 'peaksnr':cell_ii['unitStats'][13], 
                           'isoldist':cell_ii['IsolDist'], 'm_waveform':cell_ii['m_waveforms'],
                           'rate':cell_ii['rate'], 'rate_full':cell_ii['rate_full']}
        
        qc_data[cell_id] = cell_qcdata
    
    assert len(clusterstats) == len(qc_data)
    
    return qc_data, d_proj
    



# ----- load eye tracking data from Eyelink asc files -----
def load_et_from_asc(asc_file):

    gaze = []
    saccs = []
    fixes = []
    blinks = []
    ttlmsg = []
    gaze_coords = None
    with open(asc_file) as f:
        for line in f:
            if line[0].isdigit():
                gaze_line = [ii for ii in line.split() if not (ii=='...' or ii=='I..')]
                gaze.append(gaze_line)
            elif line.startswith('ESACC'):
                if len(line.split()) == 17:
                    line_use = line.split()[:5] + line.split()[-6:]
                elif len(line.split()) == 11:
                    line_use = line.split()
                else: raise ValueError('Problem in decoding EFIX!')
                saccs.append(line_use)
            elif line.startswith('EFIX'):
                if len(line.split()) == 11:
                    line_use = line.split()[:5] + line.split()[-3:]
                elif len(line.split()) == 8:
                    line_use = line.split()
                else: raise ValueError('Problem in decoding EFIX!')
                fixes.append(line_use)
            elif line.startswith('EBLINK'):
                blinks.append(line.split())
            elif line.startswith('START'):
                start_time = int(line.split()[1])
            elif line.startswith('END'):
                end_time = int(line.split()[1])
            elif 'GAZE_COORDS' in line:
                gaze_coords = [ float(gii) for gii in line.split()[-4:] ]
            elif line.startswith('MSG') and 'TTL' in line:
                ttlmsg.append(line.split()[1:])


    ttlmsg_df = None
    if len(ttlmsg) > 0:
        ttlmsg = [ [mii[0], mii[1].replace('TTL=', '')] for mii in ttlmsg ]
        ttlmsg_cols = ['time', 'ttl']
        ttlmsg_df = pd.DataFrame(ttlmsg, columns=ttlmsg_cols)
        ttlmsg_df = ttlmsg_df.apply(pd.to_numeric, errors='coerce')


    gaze_cols = ['RecTime', 'GazeX', 'GazeY', 'Pupil_size']
    # gaze_cols_dtype = [int, float, float, int]
    gaze_df = pd.DataFrame(gaze, columns=gaze_cols)
    gaze_df = gaze_df.apply(pd.to_numeric, errors='coerce')
    gaze_df[['RecTime','Pupil_size']] = gaze_df[['RecTime','Pupil_size']].astype(int) 
    gaze_df = gaze_df.round({'GazeX': 0, 'GazeY': 0}) # can't be integer type due to np.NaNs

    
    fixes_cols = ['Event', 'Eye', 'Start_time', 'End_time', 'Duration', 'FixationX', 'FixationY', 'Pupil_size_avg']
    fixes_cols_dtype = [int, str, int, int, int, float, float, int]
    fixes_df = pd.DataFrame(fixes, columns=fixes_cols)
    fixes_df['Event'] = 1
    for col_ii, col_type in zip(fixes_cols,fixes_cols_dtype):
        fixes_df[col_ii] = fixes_df[col_ii].astype(col_type)
        
    round_cols = [ 'FixationX', 'FixationY' ]
    fixes_df[round_cols] = fixes_df[round_cols].round(0)
    fixes_df[round_cols] = fixes_df[round_cols].astype(int) 
    fixes_df.drop(columns=['Event'],inplace=True)
    
    
    saccs_cols = ['Event', 'Eye', 'Start_time', 'End_time', 'Duration', 'StartX', 'StartY', 'EndX', 'EndY', 'Ampl', 'Pupil_vel']
    saccs_cols_dtype = [int, str, int, int, int, float, float, float, float, float, int]
    saccs_df = pd.DataFrame(saccs, columns=saccs_cols)
    saccs_df['Event'] = 2
    saccs_df = saccs_df[ np.logical_and(saccs_df['StartX'] != '.' , saccs_df['EndX'] != '.' )].copy()
    
    for col_ii, col_type in zip(saccs_cols,saccs_cols_dtype):
        saccs_df[col_ii] = saccs_df[col_ii].astype(col_type)
    
    round_cols = [ 'StartX', 'StartY', 'EndX', 'EndY' ]
    saccs_df[round_cols] = saccs_df[round_cols].round(0)
    saccs_df[round_cols] = saccs_df[round_cols].astype(int) 
    saccs_df.drop(columns=['Event'],inplace=True)
    
    
    blinks_cols = ['Event', 'Eye', 'Start_time', 'End_time', 'Duration']
    blinks_cols_dtype = [int, str, int, int, int ]
    blinks_df = pd.DataFrame(blinks, columns=blinks_cols)
    blinks_df['Event'] = 0
    for col_ii, col_type in zip(blinks_cols,blinks_cols_dtype):
        blinks_df[col_ii] = blinks_df[col_ii].astype(col_type)
    blinks_df.drop(columns=['Event'],inplace=True)


    return start_time, end_time, gaze_df, fixes_df, saccs_df, blinks_df, ttlmsg_df, gaze_coords

