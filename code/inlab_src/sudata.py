#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

import configparser

from datetime import datetime
from zoneinfo import ZoneInfo

from helpers_inlab import load_et_from_asc, loadmat, datafname_map_bdm
from pymatreader import read_mat
from mne.io.eeglab import read_raw_eeglab


def get_epilepsy_dx(diag_code):
    '''
    Get the Epilepsy Dx from the diagnostic Code. See defineNOsessions.ini 
    to see a list of diagnostic codes
    
    Inputs:
        diag_code (int): Diagnostsic Code for the Epilepsy Dx.
    Returns:
       Epilepsy Dx (str): The epilepsy diagnosis
    '''
    dx_map = {0: 'Not Localized', 1: 'Right Mesial Temporal', 2: 'Left Mesial Temporal', 
              3: 'Right Neocortical Temporal', 4: 'Left Neocortical Temporal', 
              5: 'Right Lateral Frontal', 6: 'Left Lateral Frontal',
              7: 'Bilateral Independent Temporal', 8: 'Bilateral Independent Frontal', 
              9: 'Right Other', 10: 'Left Other',
              11: 'Right Occipital Cortex', 12: 'Left Occipital Cortex', 
              13: 'Bilateral Occipital Cortex', 14: 'Right Insula',
              15: 'Left Insula', 16: 'Independent Bilateral Insula'}
    
    return dx_map[diag_code]


def _reformat_sessions(sessions_in):
    
    keys_init = list(sessions_in.keys())
    # As the lab convention each key starts with an experiment name: e.g., NOsessions.session
    # remove that common word from all keys.
    rm_word = False
    if '.' in keys_init[0]: 
        rm_word = True
        rm_word_txt = keys_init[0].split('.')[0] +'.'

    sessions_out = {}
    for kii,vii in sessions_in.items():
        if rm_word:
            assert rm_word_txt in kii, 'formating error in .ini file'
            kii = kii.replace(rm_word_txt,'')
            
        sessions_out[kii] = vii.strip("'")
        try:
            sessions_out[kii] = int(vii)
        except ValueError:
            pass

    # for nwb format.     
    if 'date' in sessions_out.keys():    
        rawdate = sessions_out['date']
        rawdate = datetime.strptime(rawdate, '%Y-%m-%d')
        # rawdate = rawdate.replace(hour = 0, minute = 0)
        expdate = rawdate.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        
        sessions_out['date'] = expdate

    for mii in ['rhip', 'lhip', 'ramy', 'lamy', 'racc', 'lacc', 'rsma', 'lsma', 'rofc', 'lofc',
                'la', 'lh', 'ra', 'rh', 'lph', 'rph' ]:
        
        if mii in sessions_out.keys():
            raw_coords = sessions_out[mii].split(',')
            if raw_coords[0] == 'NaN':
                sessions_out[mii] = [np.NaN]*3
            else:    
                sessions_out[mii] = list(map(float,raw_coords))
                
    return sessions_out



MICRO2SEC = 1000000.
MILLI2SEC = 1000.

LFP_PAD_us =  10. * MICRO2SEC # in micro second


class SUData:
    """SUData class provides an interface for loading single-unit data from standard 
    osort output folders: sorted and events  
    """
    
    def __init__(self, data_dir, file_ini, experiment_type='all', patient_dir_first=False):
        """Initializes the SUData class with the session info [name]
        """
        self.data_dir = data_dir
        self.session_ids, self.sessions = self.load_config(file_ini)
        # whether the order of directories are in the format of:
        # events/P42CS_... and sorted/P42CS_...  that is patient_dir_first=False (NO, CB datasets)
        # or  P42CS_.../events and P42CS_.../sorted that is patient_dir_first=True (BDM dataset)
        self.patient_dir_first = patient_dir_first

        if all(['taskdescr' in self.sessions[sii].keys() for sii in self.session_ids ]):
            task_pre = [ self.sessions[sii]['taskdescr'] for sii in self.session_ids ]
            if np.unique(task_pre).size == 1:
                print(f"\nTask is taken as '{np.unique(task_pre)[0]}' automatically!"+
                  "\nsu_data.taskname = 'some other name' can be used to overwrite this "+
                  "[optional].\n")
                self.taskname = np.unique(task_pre)[0]
                if self.taskname =='NO':
                    self.brain_areas2include = [1, 2, 3, 4, 13, 18] # include cells from these areas only.
                elif self.taskname =='BDM':
                    self.brain_areas2include = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # include cells from these areas only.

        self.experiment_type = experiment_type
        if experiment_type.lower() == 'all':
            print(f"\nexperiment_type is taken as '{experiment_type}'"+
              "\nsu_data.experiment_type = 'some other name' can be used to change this,"+
              " e.g., 'encoding', 'recognition'.")

            
    # ---- taskname -----
    def _get_taskname(self): # getter
        try:
            return self._taskname
        except AttributeError:
            return None
    
    def _set_taskname(self, taskname): # setter
        self._taskname = taskname
    
    taskname = property(_get_taskname, _set_taskname)  
    # ----- o -----

    @staticmethod
    def load_config(file_ini):
        #  Check config file path
        if not os.path.exists(file_ini):
            raise SystemExit(f'This file does not exist:\n{file_ini}'+\
                             '\nCheck filename and/or directory!')
        # Read the config file
        try:
            config = configparser.ConfigParser() # initialize the ConfigParser() class
            config.read(file_ini) # read .ini file
        except:
            raise SystemExit(f'Failed to read the config file:\n{file_ini}\nPlease check its format!')
        
        # Read Meta-data from INI file and keep as a dictionary of sessions.
        # Walk through config.sections
        section_info = { sii:dict(config[sii]) for sii in config.sections() }
        section_ids = list(section_info.keys())
        assert len(section_ids)>0
        # double check
        assert section_ids == config.sections()
        
        for sii in section_ids:
            section_info[sii] = _reformat_sessions(section_info[sii])

        return section_ids, section_info


    def _construct_data_path(self, session_name, target_folder, experiment_type=None):

        if experiment_type is None:
            experiment_type = self.experiment_type
            
        taskdescr_key = self._taskdescr_key(experiment_type)
        task_dir = self.sessions[session_name][taskdescr_key]

        if self.patient_dir_first:
            if self.taskname=='BDM':
                if self.sessions[session_name]['patientsession'] == 11 or \
                    self.sessions[session_name]['patientsession'] == 12:
                        task_dir_mv = "movie12"
                else:
                    task_dir_mv = f"movie{self.sessions[session_name]['patientsession']}"
                
                try: 
                    subj_folder = self.sessions[session_name]['sessionid']
                except KeyError:
                    subj_folder = session_name.split('_')[0]
                    
                # depending on the Filesystem type, font case might matter
                subj_folder = subj_folder.lower()
                path = os.path.join(self.data_dir, subj_folder, task_dir_mv, target_folder)
                
            else:
                try: 
                    subj_folder = self.sessions[session_name]['session']
                except KeyError:
                    subj_folder = session_name
                    
                subj_folder = subj_folder.lower()
                path = os.path.join(self.data_dir, subj_folder, task_dir, target_folder)
        else:
            try: 
                subj_folder = self.sessions[session_name]['session']
            except KeyError:
                subj_folder = session_name
                
            subj_folder = subj_folder.lower()
            path = os.path.join(self.data_dir, target_folder, subj_folder, task_dir)

        return path


    def _taskdescr_key(self,experiment_type=None):
        
        if experiment_type is None:
            experiment_type = self.experiment_type
        
        # in a general use we might have/want other options.
        if experiment_type == 'some_other':
            return 'taskdescr_some'
        else:
            # raise ValueError(f"Undefined experiment_type='{experiment_type}' in _taskdescr_key()!")
            assert all(['taskdescr' in self.sessions[sii].keys() for sii in self.session_ids ])
            return 'taskdescr'
    
    
    def _construct_brain_area_path(self, session_name, experiment_type):
        
        path_pre = self._construct_data_path(session_name, 'events', experiment_type)
        
        path = os.path.join(path_pre, 'brainAreaNEW.mat')
        if os.path.isfile(path):
            brain_area_file_path = path
        else:
            brain_area_file_path = os.path.join(path_pre, 'brainArea.mat')
        
        return brain_area_file_path    


    def get_chanlocs(self, session_name, etype='micro', experiment_type=None):

        if experiment_type is None:
            experiment_type = self.experiment_type

        taskdescr_key = self._taskdescr_key(experiment_type)
        task_dir = self.sessions[session_name][taskdescr_key]

        if self.patient_dir_first:
            if self.taskname=='BDM':
                try: 
                    subj_folder = self.sessions[session_name]['sessionid']
                except KeyError:
                    subj_folder = session_name.split('_')[0]
                # depending on the Filesystem type, font case might matter
                subj_folder = subj_folder.lower()

                chanlocs_file = os.path.join(self.data_dir, subj_folder, 'channels', f'{etype}_MNI.xyz')
                
                if not os.path.isfile(chanlocs_file): # A specific case for p53cs
                    chanlocs_file = os.path.join(self.data_dir, subj_folder, 
                                                 f"channels_movie{self.sessions[session_name]['patientsession']}", 
                                                 f'{etype}_MNI.xyz')
                    
                if not os.path.isfile(chanlocs_file): # still couldn't find it. 
                    print(f"\n\t Couldn't find chanlocs_file for {etype}s! Skipped!")
                    return None
                    
            else:
                try: 
                    subj_folder = self.sessions[session_name]['session']
                except KeyError:
                    subj_folder = session_name
                subj_folder = subj_folder.lower()

                chanlocs_file = os.path.join(self.data_dir, subj_folder, 'channels', f'{etype}_MNI.xyz')
        else:
            try: 
                subj_folder = self.sessions[session_name]['session']
            except KeyError:
                subj_folder = session_name
            subj_folder = subj_folder.lower()

            raise NotImplementedError('Need to test this case! Not used so far!')
            chanlocs_file = os.path.join(self.data_dir, 'some_target_folder', subj_folder, task_dir)
        
        # channel numbers are indicies.
        chanlocs_info = pd.read_table(chanlocs_file, names=['chan_num', 'x', 'y', 'z', 'brain_area'],index_col=0)
        
        return chanlocs_info


    def get_event_data(self, session_name, experiment_type=None):
        '''
        returns: contains a list of TTLs. Columns: timestamp, TTL, experimentID.        

        '''
        if experiment_type is None:
            experiment_type = self.experiment_type
        
        event_file = os.path.join(self._construct_data_path(session_name, 'events', experiment_type), 'eventsRaw.mat')
        # contains a list of TTLs. Columns: timestamp, TTL, experimentID. 
        events = pd.DataFrame(loadmat(event_file)['events'])
        events.rename(columns={0:'time', 1:'ttl', 2:'task_id'}, inplace=True)

        if experiment_type.lower() == 'all' or experiment_type.endswith('_all'):
            if self.taskname == 'BDM':
                experiment_id = self.sessions[session_name]['experimentidmovie']
                experiment_id_no = self.sessions[session_name]['experimentidmovieno']
                # some times several different tasks were performed in the same session. Get movie and movieNO 
                events = events.loc[ (events['task_id']==experiment_id) | (events['task_id']==experiment_id_no)]
                
                if self.sessions[session_name]['patientsession'] == 11 or \
                    self.sessions[session_name]['patientsession'] == 12:
                        
                        event_ids = events.loc[:,'task_id'].to_numpy()
                        event_id_blocks = np.where(np.diff(event_ids)!=0)[0]+1
                        split_df = np.split(events, event_id_blocks)

                        if self.sessions[session_name]['patientsession'] == 11:             
                            events_out = pd.concat([ split_df[0], split_df[1] ])
                        elif self.sessions[session_name]['patientsession'] == 12:                
                            events_out = pd.concat([ split_df[2], split_df[3] ])
                        
                        return events_out
                    
                return events
            else:
                return events
        else:
            exp_id_key = self._exp_id_key(experiment_type)
            experiment_id = self.sessions[session_name][exp_id_key]
            events = events.loc[events['task_id'] ==  experiment_id] 
            return events


    def get_logfile(self, session_name, experiment_type=None):

        if self.taskname == 'BDM':
            experiment_id = self.sessions[session_name]['experimentidmovie']
            # experiment_id_no = self.sessions[session_name]['experimentidmovieno']
        else:
            raise SystemExit("get_logfile is implemented yet only for the BDM task!")

        if experiment_type is None:
            experiment_type = self.experiment_type

        log_filebase = self._construct_data_path(session_name, 'events', experiment_type)
        
        if self.sessions[session_name]['patientsession'] == 12:
            log_file = os.path.join(log_filebase,f'log{experiment_id}-2.txt')
            # log_file_no = os.path.join(log_filebase,f'log{experiment_id_no}-2.txt')
        else:
            log_file = os.path.join(log_filebase,f'log{experiment_id}-1.txt')
            # log_file_no = os.path.join(log_filebase,f'log{experiment_id_no}-1.txt')
            
        col_names = ['time', 'ttl', 'f1', 't1', 'resp_stim', 'a1', 'a2', 'nullcol'] # otherwise the next line has an error.
        log_df = pd.read_csv(log_file, sep=r';|,', names=col_names, engine='python', header=None)
        log_df.dropna(axis='columns', how='all', inplace=True)

        # clean "C:\stimuli\JuliensTasks\BangYoureDead\newold\" part. 
        log_df['resp_stim'] = [ eii.split('\\')[-1] if isinstance(eii,str) else eii \
                               for eii in log_df['resp_stim'] ]

        return log_df


    def get_eyetracking_data(self, session_name, experiment_type=None):

        if self.taskname == 'BDM':
            experiment_id = self.sessions[session_name]['experimentidmovie']
            # experiment_id_no = self.sessions[session_name]['experimentidmovieno']
        else:
            raise SystemExit("get_eyetracking_data is implemented yet only for the BDM task!")

        if experiment_type is None:
            experiment_type = self.experiment_type

        et_filebase = self._construct_data_path(session_name, 'events', experiment_type)
        
        if self.sessions[session_name]['patientsession'] == 12:
            asc_file = os.path.join(et_filebase,f'log{experiment_id}-2.asc')
            mat_file = os.path.join(et_filebase,f'log{experiment_id}-2.mat')
        else: # includes patientsession 1, 2, or 11.
            asc_file = os.path.join(et_filebase,f'log{experiment_id}-1.asc')
            mat_file = os.path.join(et_filebase,f'log{experiment_id}-1.mat')
            

        # --- Load info about the session from ET mat file ---
        expinfo_et = read_mat(mat_file)['h']
        screen_w, screen_h = expinfo_et['w'], expinfo_et['h']
        display_w, display_h = expinfo_et['displayWidth'], expinfo_et['displayHeight']
        display_area = expinfo_et['rect']
        
        actualFrameTime = expinfo_et['actualFrameTime'] - expinfo_et['actualFrameTime'][0]
        # --- o ---
        
        _, _, gaze_df, fixes_df, saccs_df, blinks_df, ttls_df, gaze_coords = load_et_from_asc(asc_file)
        
        event_markers = self.event_markers()

        movie_etstart_time = ttls_df.loc[ttls_df['ttl'] == event_markers['startvideo'], 'time'].values
        assert len(movie_etstart_time) == 1
        movie_etstart_time = movie_etstart_time[0]

        # reset all eye tracking time instances to the start time of movie watching 
        gaze_df = gaze_df.loc[gaze_df['RecTime'] >= movie_etstart_time].copy()
        gaze_df['RecTime'] = (gaze_df['RecTime'] - movie_etstart_time) / MILLI2SEC

        ttls_df = ttls_df.loc[ttls_df['time'] >= movie_etstart_time].copy()
        ttls_df['time'] = (ttls_df['time'] - movie_etstart_time) / MILLI2SEC

        fixes_df = fixes_df.loc[fixes_df['Start_time'] >= movie_etstart_time].copy()
        fixes_df['Start_time'] = (fixes_df['Start_time'] - movie_etstart_time) / MILLI2SEC
        fixes_df['End_time'] = (fixes_df['End_time'] - movie_etstart_time) / MILLI2SEC
        fixes_df['Duration'] = fixes_df['Duration'] / MILLI2SEC

        blinks_df = blinks_df.loc[blinks_df['Start_time'] >= movie_etstart_time].copy()
        blinks_df['Start_time'] = (blinks_df['Start_time'] - movie_etstart_time) / MILLI2SEC
        blinks_df['End_time'] = (blinks_df['End_time'] - movie_etstart_time) / MILLI2SEC
        blinks_df['Duration'] = blinks_df['Duration'] / MILLI2SEC

        saccs_df = saccs_df.loc[saccs_df['Start_time'] >= movie_etstart_time].copy()
        saccs_df['Start_time'] = (saccs_df['Start_time'] - movie_etstart_time) / MILLI2SEC
        saccs_df['End_time'] = (saccs_df['End_time'] - movie_etstart_time) / MILLI2SEC
        saccs_df['Duration'] = saccs_df['Duration'] / MILLI2SEC


        assert fixes_df['Eye'].unique().size == 1
        assert saccs_df['Eye'].unique().size == 1
        assert blinks_df['Eye'].unique().size == 1

        fix_eye = fixes_df['Eye'].unique()[0]
        sac_eye = saccs_df['Eye'].unique()[0]
        blk_eye = blinks_df['Eye'].unique()[0]

        assert fix_eye == sac_eye and sac_eye==blk_eye

        fixes_df.drop(columns=['Eye'],inplace=True)
        saccs_df.drop(columns=['Eye'],inplace=True)
        blinks_df.drop(columns=['Eye'],inplace=True)

        return gaze_df, fixes_df, saccs_df, blinks_df, ttls_df, fix_eye, \
            actualFrameTime, screen_w, screen_h, display_w, display_h, display_area



    def _exp_id_key(self,experiment_type=None):
        
        if experiment_type is None:
            experiment_type = self.experiment_type
        
        if (self.taskname=='NO' and experiment_type=='encoding') or \
            experiment_type == 'no_learn':
            return 'experimentidlearn'
        elif (self.taskname=='NO' and experiment_type=='recognition') or \
            experiment_type == 'no_recog':
            return 'experimentidrecog'

        elif (self.taskname=='BDM' and experiment_type=='encoding') or \
            experiment_type == 'movie':
            return 'experimentidmovie'
        elif (self.taskname=='BDM' and experiment_type=='recognition') or \
            experiment_type == 'movie_no':
            return 'experimentidmovieno'
        # can enter info about other dataset here
        else:
            raise ValueError(f"Undefined experiment_type='{experiment_type}' in _exp_id_key()!")



    # Specific for movie watching (BDM) task
    def trial_info(self, session_name, experiment_type=None):

        if self.taskname!='BDM':
            raise SystemExit("self.trial_info() is implemented for taskname='BDM' only!")

        events = self.get_event_data(session_name, experiment_type)
        event_markers = self.event_markers()

        log_df = self.get_logfile(session_name, experiment_type)

        experiment_id = self.sessions[session_name]['experimentidmovie']
        experiment_id_no = self.sessions[session_name]['experimentidmovieno']
        
        # stim_data = self.get_stim_data(session_name, experiment_type)

        # ------ encoding (video watching) block from the events file ------
        encoding_events = events.loc[events['task_id'] == experiment_id ]
        session_start_us = encoding_events.loc[encoding_events['ttl'] == event_markers['startvideo'],'time'].values
        assert len(session_start_us) == 1
        session_start_us = session_start_us[0]

        if 10 in encoding_events['ttl'].tolist():
            valid_videottls = [4, 10] + np.arange(40,49+1).tolist() # [4] + [40, ..., 49]
        else:
            valid_videottls = [4] + np.arange(40,49+1).tolist() # [4] + [40, ..., 49]

        encoding_block = encoding_events.loc[encoding_events['ttl'].isin(valid_videottls)].copy()
        
        if 10 in encoding_events['ttl'].tolist():
            assert encoding_block['ttl'].values[-1] == 10
            assert all(np.diff(encoding_block.index.values[-5:])==1)
        encoding_block.reset_index(drop=True, inplace=True)

        encoding_end_us = encoding_block['time'].values[-1]
        assert session_start_us ==  encoding_block['time'].values[0]
        # encoding_block_time = (encoding_block['time'].values - session_start_us) / 1000.
        # ----- o -----

        # --- encoding (video watching) block from the log file ---
        log_start_us = log_df.loc[log_df['ttl'] == event_markers['startvideo'], 'time' ].values
        assert len(log_start_us) == 1
        log_start_us = log_start_us[0]
        
        log_block = log_df.loc[log_df['ttl'].isin(valid_videottls) ].copy()
        # log_block['time'] -= log_start_us
        log_block.drop(columns=['time', 't1', 'resp_stim', 'a1', 'a2'], inplace=True) # not avail. for encoding.
        log_block.reset_index(drop=True, inplace=True)
        log_block.rename(columns={'ttl':'ttl_log', 'f1':'frames'},inplace=True)


        if len(encoding_block) != len(log_block):
            print('Time instances in the events file and the log file do not match.\n'+
                  'Removes one problematic data point in the events file!')
            diff_vals = np.diff(encoding_block['time'].values)
            rm_inds = np.where(diff_vals < 200)[0][0]
            encoding_block.drop(index=rm_inds, axis=0, inplace=True)
            diff_vals_post = np.diff(encoding_block['time'].values)
            assert all(diff_vals_post>200)
            encoding_block.reset_index(drop=True, inplace=True)
            
        assert len(encoding_block) == len(log_block)
        encoding_block = pd.concat([encoding_block,log_block], axis=1)
        assert len(encoding_block) == len(log_block) # second check whether the indicies matched. 
        
        encoding_frametimes = encoding_block[['time', 'frames']].to_numpy(dtype=float)
        encoding_frametimes[:,1] -= 1
        # ----- o -----


        # ------ recognition (video new-old) block from the event file ------
        recog_events = events.loc[events['task_id'] == experiment_id_no ]

        recog_probes = recog_events.loc[recog_events['ttl'] == event_markers['startprobe']].copy()
        recog_probes.reset_index(drop=True, inplace=True)

        recog_responses = recog_events.loc[np.logical_and(recog_events['ttl'] == event_markers['keypress'],
                              recog_events['time'] > recog_probes['time'].values[0]) ].copy() # remove initial keypress
        recog_responses.reset_index(drop=True, inplace=True)

        recog_iti = recog_events.loc[np.logical_and(recog_events['ttl'] == event_markers['start_iti'],
                                                recog_events['time'] > recog_probes['time'].values[0]) ] 
        recog_iti.reset_index(drop=True, inplace=True)
        
        assert all(recog_probes['task_id'] == experiment_id_no)
        assert len(recog_probes) == len(recog_responses)
        # ----- o -----

        # --- recognition (video new-old) block from the log file ---
        exp_note = log_df['f1'].astype(str).to_list()
        exp_txt = [ eii for eii in exp_note if len(eii)>7 ]
        if len(exp_txt) > 0:
            print(f"---> Note that the recognition part is : {exp_txt}.\n"+
                  "There are missing new/old trials!")
            
        log_probes = log_df.loc[log_df['ttl'] == event_markers['startprobe']].copy()
        
        log_responses = log_df.loc[np.logical_and(log_df['ttl'] == event_markers['keypress'],
                                                  log_df['time'] > log_probes['time'].values[0]) ].copy() # remove initial keypress

        # log_iti = log_df.loc[np.logical_and(log_df['ttl'] == event_markers['start_iti'],
                                              # log_df['time'] > log_probes['time'].values[0]) ] 

        log_probes.drop(columns=['time', 'ttl', 'a1', 'a2'], inplace=True)
        log_probes.reset_index(drop=True, inplace=True)
        log_probes.rename(columns={'f1':'trial', 't1':'new0_old1' }, inplace=True)

        log_responses.dropna(axis=0,inplace=True)
        log_responses['resp_stim'] = log_responses['resp_stim'].astype(float)
        log_responses.drop(columns=['time', 'ttl'], inplace=True)
        log_responses.reset_index(drop=True, inplace=True)
        log_responses.rename(columns={'f1':'trial', 't1':'response', 
                                      'resp_stim':'response_time',
                                      'a1':'correct', 'a2':'confidence' }, inplace=True)

        assert len(recog_probes) == len(log_probes)
        recog_probes = pd.concat([recog_probes,log_probes], axis=1)

        # if the experiment 'aborted by user' (session nb: 16, P53CS_BDM_R2) 
        # then len(recog_responses) != len(log_responses) is possible and okay.
        recog_responses = pd.concat([recog_responses,log_responses], axis=1)
        
        assert len(recog_probes) == len(recog_responses) # second check against additional NaN entries. 
        # ----- o -----


        # ----- reset all time instances to the video start time (i.e., ttl=4) -----
        encoding_frametimes[:,0] = ( encoding_frametimes[:,0] - session_start_us) / MICRO2SEC 

        # --- generate trial objects ---
        trials = []
        trials.append( Trial((session_start_us - session_start_us)/MICRO2SEC, 
                             (encoding_end_us - session_start_us)/MICRO2SEC, 
                             trial_type='encoding', trial_stim='bd_movie' ))

        # recog_iti is safer to use than recog_responses for tii in case of the session was aborted by user.
        for tii in range(len(recog_iti)): 
            
            trial_starttime = recog_probes['time'].values[tii] 
            trial_responsetime = recog_responses['time'].values[tii] # which one should be trial_stop time ASK!!!
            trial_iti_start = recog_iti['time'].values[tii]
            assert trial_starttime < trial_responsetime < trial_iti_start

            trials.append( Trial(start_time = (trial_starttime - session_start_us)/MICRO2SEC, 
                                 # trial_response_time is taken as trial_stoptime
                                 stop_time = (trial_responsetime - session_start_us)/MICRO2SEC, 
                                 trial_type ='recognition', 
                                 trial_stim = recog_probes['resp_stim'].values[tii],
                                 response_correct = recog_responses['correct'].values[tii],
                                 response_confidence= recog_responses['confidence'].values[tii], 
                                 actual_response= recog_responses['response'].values[tii]-1, # original range was 2-7 rather than 1-6.
                                 response_time_actual = (trial_responsetime - session_start_us)/ MICRO2SEC,
                                 response_time_diff = (trial_responsetime - trial_starttime) / MICRO2SEC,
                                 # response_time_diff = recog_responses['response_time'].values[tii] # value saved in the log file. ~0.04 sec different.
                                 ))

            # print( tii, (trial_responsetime - trial_starttime) / MICRO2SEC - recog_responses['response_time'].values[tii] )
            
        return trials, encoding_frametimes



    def ls_cells(self, session_name, experiment_type=None):
        """
        The ls_cells function list all the available cells for a particular session number
        output:
            cell_list = a list of tuples (channel_nr, ch_cell_nr, osort_cluster_id, brainArea)
            
            brainArea mapping coded in brain_area[:,3]
            1=RH, 2=LH, 3=RA, 4=LA, 5=RAC, 6=LAC, 7=RSMA, 8=LSMA, 9=ROFC, 10=LOFC, 
            50=RFFA, 51=REC
        """
        
        brain_area_file = self._construct_brain_area_path(session_name,experiment_type)
        # brain_area has 4 columns:
        # [ channel_nr, cell_nr in each channel, OSort cluster_id in each channel, brain area code ]
        brain_area = loadmat(brain_area_file)['brainArea']
        
        cell_list = []
        for ii in range(brain_area.shape[0]):
            
            if (brain_area[ii,0] != 0) and (brain_area[ii,1] != 0):
                if self.taskname is not None and self.taskname=='NO':
                    if brain_area[ii,3] in self.brain_areas2include:
                        cell_list.append( brain_area[ii,:] )
                elif self.taskname is not None and self.taskname=='BDM':
                    if brain_area[ii,3] in self.brain_areas2include:
                        cell_list.append( brain_area[ii,:] )
                else:
                    cell_list.append( brain_area[ii,:] )

        return np.asarray(cell_list, dtype=int)



    def load_channel_data(self, session_name, channel_nr, experiment_type=None):
        """
        load the raw channel data
        """

        channel_path = os.path.join(self._construct_data_path(session_name, 'sorted', experiment_type),
                                 f'A{channel_nr}_cells.mat')

        if not os.path.isfile(channel_path): # another naming convention was used for recent subjects.
            channel_path = os.path.join(self._construct_data_path(session_name, 'sorted', experiment_type),
                                     f'A{channel_nr:03}_cells.mat')
        
        if os.path.isfile(channel_path):
            raw_ch_spikes = loadmat(channel_path)['spikes']
        else:
            raw_ch_spikes = None

        return raw_ch_spikes



    def load_lfpdata(self, session_name, main_rdatadir, etype, 
                     experiment_type=None):
        # load lfp data from eeglab .set and .fdt files
        
        if self.taskname == 'BDM':
            experiment_id = self.sessions[session_name]['experimentidmovie']
            # experiment_id_no = self.sessions[session_name]['experimentidmovieno']
        else:
            raise SystemExit("load_lfpdata was tested only for the BDM task!")

        if experiment_type is None:
            experiment_type = self.experiment_type

        if self.sessions[session_name]['patientsession'] == 12:
            lbase = f'{etype}_{experiment_id}-2.set'
            # lbase_no = f'micro_{experiment_id_no}-2.set'
        else:
            lbase = f'{etype}_{experiment_id}-1.set'
            # lbase_no = f'micro_{experiment_id_no}-1.set'
        
        rdatadir_init = datafname_map_bdm[session_name]
        fname_set = os.path.join(main_rdatadir, *rdatadir_init.split('_',1), 'lfpdata_n', lbase)
        
        lfp_info = read_mat(fname_set)
        assert lfp_info['session_info'] == rdatadir_init 
        
        lfp_mne = read_raw_eeglab(fname_set, preload=True)
        lfp_data = lfp_mne._data.T # time should be the first dimension for NWB format.
        lfp_datatime = lfp_mne.times
        # sfreq = lfp_mne.info['sfreq']
        
        # lfp_orgtime_init = lfp_info['orgtime_init']
        lfp_orgtime_dw = lfp_info['orgtime_down']
        # assert lfp_orgtime_dw[0] == lfp_orgtime_init
        
        assert np.allclose(np.diff(lfp_orgtime_dw)/MICRO2SEC, np.diff(lfp_datatime))
        
        # return lfp_orgtime_init, lfp_datatime, lfp_data, lfp_orgtime_dw, lfp_events
        return lfp_orgtime_dw, lfp_data, lfp_datatime


    def event_markers(self):
        """
        This static method is used to define the useful markers to index the event data.
        :return: marker dictionary
        """
        if self.taskname == 'NO':
            markers = {'stimulus_on': 1,
                       'stimulus_off': 2,
                       'delay1_off': 3,
                       'delay2_off': 6,
                       'response_1': 31,
                       'response_2': 32,
                       'response_3': 33,
                       'response_4': 34,
                       'response_5': 35,
                       'response_6': 36,
                       'response_learning_animal': 20,
                       'response_learning_non_animal': 21,
                       'experiment_on': 55,
                       'experiment_off': 66}

        elif self.taskname == 'BDM':
            markers = {'endofmovie': 10,
                       'startvideo': 4,
                       'startprobe': 7,
                       'start_iti': 9,
                       'keypress': 33,
                       'endexp': 66}

        else:
            raise ValueError(f"Undefined taksname='{self.taksname}' in self.event_markers()!")
        
        return markers


class Trial:
    def __init__(self, start_time, stop_time, 
                 trial_type, trial_stim='NA', 
                 response_correct=None, response_confidence=None, actual_response=None,
                 response_time_actual=None, response_time_diff=None):
        
        self.start_time = start_time
        self.stop_time = stop_time
        self.trial_type = trial_type
        self.stim = trial_stim
        self.response_correct = response_correct
        self.response_confidence = response_confidence
        self.actual_response = actual_response
        self.response_time_actual = response_time_actual
        self.response_time_diff = response_time_diff
        

