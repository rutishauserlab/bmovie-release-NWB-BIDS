#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from pynwb import NWBFile, NWBHDF5IO
from pynwb import TimeSeries
from pynwb.file import Subject
from pynwb.misc import AnnotationSeries
from pynwb.behavior import SpatialSeries, EyeTracking, PupilTracking, BehavioralTimeSeries
from pynwb.ecephys import ElectricalSeries, LFP
from hdmf.backends.hdf5.h5_utils import H5DataIO

from helpers_inlab import load_qcdata
from sudata import SUData, get_epilepsy_dx, MICRO2SEC, LFP_PAD_us


# Directories to load the single-unit and electrode data
main_datadir = '/media/umit/easystore/bmovie_SUready' # sorted SU data # patient_dir_first=True
main_raw_datadir = '/media/umit/easystore/bmovie_ephysdata' # data from microwires and macro electrodes

# create a directory to save output NWB files
output_dir = '/media/umit/easystore/bmovie_NWBfiles'
os.makedirs(output_dir, exist_ok=True)


# load QC data metrics from matlab outputs
qcdata_dir = os.path.join(main_datadir,'qc_data')
qc_data_encod, d_proj_encod = load_qcdata(qcdata_dir, phase='encoding')
qc_data_recog, d_proj_recog = load_qcdata(qcdata_dir, phase='recognition')

# Get .ini file path
subjects_ini = 'BDmovie_sessions.ini'

# load general info about all sessions and runs for all available subjects. 
su_data = SUData(main_datadir, subjects_ini, patient_dir_first=True)
session_ids = su_data.session_ids

brain_abbv_map = {'RHIP':'Right hippocampus', 'LHIP':'Left hippocampus', 
                  'RAMY':'Right amygdala', 'LAMY':'Left amygdala',
                  'RACC':'Right ACC', 
                  'LACC':'Left ACC',
                  'RSMA':'Right preSMA',
                  'LSMA':'Left preSMA',
                  'ROFC':'Right vmPFC',
                  'LOFC':'Left vmPFC',
                  'RSPE':'RSPE',
                  'LSPE':'LSPE'} 


# ----- scan sessions and runs to generate NWB files -----
for ses_ii, session_ii in enumerate(session_ids):
    
    # if ses_ii < 10: continue
    
    print(f'\nprocessing session: {ses_ii+1} - {session_ii}')

    # su_data.experiment_type = 'all' # [default]
    # use: 'all', 'encoding', 'recognition', or specific types such as 'no_learn'. 
    # See sudata._exp_id_key for details.  
    
    # Get infomation about this session
    session = su_data.sessions[session_ii]
    
    # --------- Generate an NWB file and add information about the session and subject ---------
    nwbfile = NWBFile(
        session_description = f'Movie watching and new/old recognition task for session: {session_ii}',
        identifier = session_ii,
        session_start_time = session['date'], 
        experiment_description = 'The data contained within this file describe a movie watching and new/old recognition task performed in '+
                                 'patients with intractable epilepsy implanted with depth electrodes and Behnke-Fried '+
                                 'microwires in the human Medical Temporal Lobe (MTL).',
        experimenter = 'Julien Dubois, PhD',
        institution = session['institution'],
        keywords = ['Intracranial Recordings', 'Intractable Epilepsy', 'Single-Unit Recordings', 'Cognitive Neuroscience', 
                    'Local field potentials', 'Movie', 'Neurosurgery'],
        lab = 'The Rutishauser Laboratory at Cedars-Sinai Medical Center',
        file_create_date = datetime.now().replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        )

    nwbfile.subject = Subject(subject_id = session['id'], # required by DANDI 
                              age = f"P{session['age']}Y", 
                              description = f"epilepsy diagnosis: {get_epilepsy_dx(session['diagnosiscode'])}",
                              sex = session['sex'], species = 'Homo sapiens' )


    # --------- Add information about event-TTLs ---------
    event_markers = su_data.event_markers()
    ttl_videostart = event_markers['startvideo']
    ttl_endexp = event_markers['endexp']
    events = su_data.get_event_data(session_ii) # returns movie and movieNO events together
    session_start_us = events.loc[events['ttl']==ttl_videostart,'time'].values # _us denotes microsecond
    session_end_us = events.loc[events['ttl']==ttl_endexp,'time'].values
    assert len(session_start_us) == 1
    session_start_us = session_start_us[0]
    
    if len(session_end_us) == 1: 
        session_end_us = session_end_us[0]
        if session_end_us != events['time'].values[-1]:
            assert session_end_us == events['time'].values[-2] # in some cases the last ttl is 0. that is fine.
    elif len(session_end_us) == 0: # for P57CS_R1 the endexp ttl is missing. that is fine.
        # print('--->', session_ii)
        session_end_us = events['time'].values[-1]
    else:
        raise SystemExit('Not expected to have more than one endexp TTLs!')

    # continue
    events = events[events['time'] >= session_start_us].copy()
    events['time'] = events['time'] - session_start_us

    # print(np.unique(events.values[:,1],return_counts=True))

    events_description = ("""The events correspond to the TTL markers for each session. 
                          The TTL markers are the following: 
                          4 = start of the movie, 40-49 = frame number mapping to a log file, 
                          33 = keypress, 7 = start probe for recognition, 9 = startITI, 1 = start fixation, 
                          10 = end fixation, 52 = screen of instruction, 66 = end of experiment""")

    event_ttl = TimeSeries(name='events_ttl', data=events['ttl'].to_numpy(dtype=int), timestamps=events['time'].to_numpy()/MICRO2SEC,
                          description=events_description, unit='NA')

    nwbfile.add_acquisition(event_ttl)


    # --------- Add information about experimentid --------- 
    experimentids_description = (f"""The experimentid (or task_id) corresponds to the encoding 
                                 (i.e., movie watching as one trial) or recognition trials. 
                                 The encoding is indexed by: {session['experimentidmovie']}. 
                                 The recognition trials are indexed by: {session['experimentidmovieno']}.""")

    experiment_ids = TimeSeries(name='experiment_ids', unit='NA', data=events['task_id'].to_numpy(dtype=int),
                               timestamps=events['time'].to_numpy()/MICRO2SEC, description = experimentids_description)

    nwbfile.add_acquisition(experiment_ids)
    
    
    # # --------- Add information about trials to the NWB file ---------
    trials, time2frame = su_data.trial_info(session_ii)
    
    # # --------- Add some extra columns to the trial table ---------
    nwbfile.add_trial_column('stim_phase', 'Encoding (movie watching) or recognition phase during the session')
    nwbfile.add_trial_column('stimulus_file', 'The file name for the stimulus')
    nwbfile.add_trial_column('response_correct', 'Whether the response was correct=1 or incorrect=0 for each new/old trial')
    nwbfile.add_trial_column('response_confidence', '''The confidence level of response 
                              (1=guessing, 2=unsure, 3=sure) for each each new/old trial''')
    nwbfile.add_trial_column('actual_response', '''The button pressed during experiment, 1-6, where
                             1:new-sure, 2:new-unsure, 3:new-guessing, 4:old-guessing, 5:old-unsure, 6:old-sure'''
                              )
                              
    nwbfile.add_trial_column('response_time', 'The response time for each new/old trial')

    
    for trial_ii in trials:
        # trial_ii.trial_type=='encoding' is movie watching
        # trial_ii.trial_type=='recognition' is movie new/old recognition task
        nwbfile.add_trial(start_time=trial_ii.start_time,
                          stop_time=trial_ii.stop_time,
                          stim_phase=trial_ii.trial_type,
                          stimulus_file=trial_ii.stim,
                          response_correct = trial_ii.response_correct if trial_ii.trial_type=='recognition' else np.nan,
                          response_confidence = trial_ii.response_confidence if trial_ii.trial_type=='recognition' else np.nan,
                          actual_response = trial_ii.actual_response if trial_ii.trial_type=='recognition' else np.nan,
                          response_time = trial_ii.response_time_actual if trial_ii.trial_type=='recognition' else trial_ii.stop_time,
                          )

    # Step 4: add eye-tracking (x,y) and pupil diameter info to the NWB file. 
    # time instances are already aligned to video start time and converted to seconds. 
    gaze_df, fixes_df, saccs_df, blinks_df, ttls_df, used_eye, \
        actualFrameTime, screen_w, screen_h, display_w, display_h, display_area \
            = su_data.get_eyetracking_data(session_ii)

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="processed behavioral data" )

    # ------ add gaze data ------    
    gaze_timestamps_sec = gaze_df['RecTime'].to_numpy()
    gaze_tdiff = np.unique(np.diff(gaze_timestamps_sec).round(6))
    if len(gaze_tdiff) == 1:
        gaze_timestamps = None
        gaze_starting_time = gaze_timestamps_sec[0]
        gaze_rate = 1. / gaze_tdiff[0]
    else:
        gaze_timestamps = gaze_timestamps_sec
        gaze_starting_time = None
        gaze_rate = None


    gaze_spatialseries = SpatialSeries(
        name="SpatialSeries",
        description="(x,y) of gaze points",
        data=gaze_df[['GazeX', 'GazeY']].to_numpy(), # cannot be integers due to NaNs.
        timestamps=gaze_timestamps,
        starting_time=gaze_starting_time,
        rate=gaze_rate,
        reference_frame="(0,0) is top left corner",
        unit='pixels', 
        comments=f'screen_width,screen_height={screen_w},{screen_h}::'+\
        f"display_w,display_h={display_w},{display_h}::display_area={','.join(map(str,display_area))}"
        )
    
    gaze_obj = EyeTracking(spatial_series=gaze_spatialseries)
    behavior_module.add(gaze_obj)
    # to access use: print(nwbfile.processing['behavior']['EyeTracking']['SpatialSeries'])


    # ------ add pupil diameter data ------    
    pupil_timeseries = TimeSeries(
        name="TimeSeries",
        description="pupil size (number of pixels inside the pupil contour)",
        data=gaze_df['Pupil_size'].to_numpy(), # cannot be integers due to NaNs.
        timestamps=gaze_timestamps,
        starting_time=gaze_starting_time,
        rate=gaze_rate,
        unit='NA (number of pixels inside the pupil contour)' )

    pupil_obj = PupilTracking(time_series=pupil_timeseries)
    behavior_module.add(pupil_obj)
    # to access use: print(nwbfile.processing['behavior']['PupilTracking']['TimeSeries'])


    # ------ add fixation information ------
    fixation_timeseries = TimeSeries(
        name="TimeSeries",
        description=f'''fixation information for {used_eye} eye; timestamps are start times; columns are 
            ['Duration', 'FixationX', 'FixationY', 'Pupil_size_avg']''',
        data=fixes_df[['Duration', 'FixationX', 'FixationY', 'Pupil_size_avg']].to_numpy(), 
        timestamps=fixes_df['Start_time'].to_numpy(),
        unit='NA',
        comments=f'screen_width,screen_height={screen_w},{screen_h}::'+\
        f"display_w,display_h={display_w},{display_h}::display_area={','.join(map(str,display_area))}"
        )

    fixation_obj = BehavioralTimeSeries(time_series=fixation_timeseries,name='Fixation')
    behavior_module.add(fixation_obj)
    # to access use: print(nwbfile.processing['behavior']['Fixation']['TimeSeries'])


    # ------ add saccade information ------
    saccade_timeseries = TimeSeries(
        name="TimeSeries",
        description=f'''saccade information for {used_eye} eye; timestamps are start times; columns are 
        ['Duration', 'StartX', 'StartY', 'EndX', 'EndY', 'Ampl', 'Pupil_vel']''',
        data=saccs_df[['Duration', 'StartX', 'StartY', 'EndX', 'EndY', 'Ampl', 'Pupil_vel']].to_numpy(), 
        timestamps=saccs_df['Start_time'].to_numpy(),
        unit='NA',
        comments=f'screen_width,screen_height={screen_w},{screen_h}::'+\
        f"display_w,display_h={display_w},{display_h}::display_area={','.join(map(str,display_area))}"
        )

    saccade_obj = BehavioralTimeSeries(time_series=saccade_timeseries,name='Saccade')
    behavior_module.add(saccade_obj)
    # to access use: print(nwbfile.processing['behavior']['Saccade']['TimeSeries'])


    # ------ add blink information ------
    blink_timeseries = TimeSeries(
        name="TimeSeries",
        description=f"blink information for {used_eye} eye; timestamps are start times; data are 'Duration'",
        data=blinks_df['Duration'].to_numpy(), 
        timestamps=blinks_df['Start_time'].to_numpy(),
        unit='seconds' )

    blink_obj = BehavioralTimeSeries(time_series=blink_timeseries,name='Blink')
    behavior_module.add(blink_obj)
    # to access use: print(nwbfile.processing['behavior']['Blink']['TimeSeries'])

    
    # --------- Add information to the NWB file about movie frame numbers associated with time instances in events file. 
    movieframe_nrs = AnnotationSeries(name = 'movieframe_time', 
                                      data=np.arange(actualFrameTime.size,dtype=int), timestamps=actualFrameTime,
                                      # data=time2frame[:,1].astype(int), timestamps=time2frame[:,0],
                                      description = '''Movie frame numbers associated with time instances during 
                                      the encoding (movie watching) phase [starts counting from 0--Python way]''')

    nwbfile.add_stimulus(movieframe_nrs)

    # Step 5: add electrode and units info to the NWB file. 
    # --- get info about all micro channels ---
    chanlocs_info_df = su_data.get_chanlocs(session_ii,etype='micro')
    channums_all = chanlocs_info_df.index.tolist()
    assert all(np.diff(channums_all)==1)
    assert np.array_equal(channums_all, np.arange(len(channums_all))+1 )

    chan_brainarea_init = chanlocs_info_df['brain_area'].tolist()
    chan_brainarea = [ctii[:4] for ctii in chan_brainarea_init ]
    channel_area_map = dict(zip(channums_all,chan_brainarea))
    channel_area_map_org = dict(zip(channums_all,chan_brainarea_init))

    # load id info about all cells.
    cell_list = su_data.ls_cells(session_ii) # cols: [channel_nr, ch_cell_nr, osort_cluster_id, brainArea]
    channel_ids_uniq, dum_indx = np.unique(cell_list[:,0], return_index=True)

    # Scan through all units and store relevant information to add to the NWB file. 
    allunits_unitids = []
    allunits_unitids_full = []
    allunits_clusterids = []
    allunits_spike_timestamps = []
    allunits_channel_id = []
    allunits_origclusterids = []
    
    allunits_isoldist = []
    allunits_meansnr = []
    allunits_peaksnr = []
    allunits_isibelow = []
    allunits_cv2 = []
    
    allunits_mean_waveform_encod = []
    allunits_mean_waveform_recog = []

    for channel_ii in channel_ids_uniq:

        # Load full data here. Below we isolate the time window for the movie sessions.
        ch_spikes = su_data.load_channel_data(session_ii, channel_ii)
        
        ch_cells = cell_list[cell_list[:,0] == channel_ii]

        for cell_ii in ch_cells:
            
            cell_id = '_'.join(map(str,cell_ii))
            cell_qc_id = f"{session_ii}_{cell_id}"
            
            assert cell_qc_id in qc_data_encod, f"Couldn't find QC data for cell: {cell_qc_id}..."

            cell_qc_encod = qc_data_encod[cell_qc_id]
            cell_qc_recog = qc_data_recog[cell_qc_id]

            cell_spiketime_raw = ch_spikes[ ch_spikes[:,0] == cell_ii[1], 2 ]
            cell_spike_timestamps = cell_spiketime_raw[(session_start_us <= cell_spiketime_raw) & \
                                                       (cell_spiketime_raw <= session_end_us)]
                
            if len(cell_spike_timestamps) == 0:
                continue # no spikes during movie watching.
            
            # reset time to video start time and convert to seconds.
            cell_spike_timestamps = (cell_spike_timestamps - session_start_us) / MICRO2SEC

            wm_encod = cell_qc_encod['m_waveform']
            wm_recog = cell_qc_recog['m_waveform']
            
            # assert not isinstance(wm_encod,float)
            # assert len(wm_encod) == 256
            
            if isinstance(wm_encod,float) or len(wm_encod) != 256:
                wm_encod = np.zeros(256)
            
            if isinstance(wm_recog,float) or len(wm_recog) != 256:
                wm_recog = np.zeros(256)

            cell_clusterid = cell_ii[2] # OSort cluster ids
            cell_brainarea = channel_area_map[cell_ii[0]]
            
            # Keep unit data to store in a NWB file
            allunits_unitids.append(cell_id) 
            allunits_unitids_full.append(cell_qc_id) 
            allunits_clusterids.append(cell_clusterid) # keep the same as the original cluster ids
            allunits_spike_timestamps.append(cell_spike_timestamps)
            allunits_channel_id.append(channel_ii)
            allunits_origclusterids.append(cell_clusterid)

            allunits_isoldist.append(cell_qc_encod['isoldist'])
            allunits_meansnr.append(cell_qc_encod['meansnr'])
            allunits_peaksnr.append(cell_qc_encod['peaksnr'])
            allunits_isibelow.append(cell_qc_encod['isibelow'])
            allunits_cv2.append(cell_qc_encod['cv2'])
            
            allunits_mean_waveform_encod.append(wm_encod)
            allunits_mean_waveform_recog.append(wm_recog)
            
            

    # the total number of units
    n_units = len(allunits_clusterids)

    # Add waveeform sampling rate
    waveform_mean_sampling_rate = [10**5] # [98.4*10**3] in Nand's codes.
    allunits_waveform_mean_sampling_rate = [waveform_mean_sampling_rate] * n_units

    # --------- Add an electrodes table to the NWB file ---------
    # Add an extra column to the electrodes table to store original channel id.
    nwbfile.add_electrode_column(name='origchannel', description='The original channel ID for the channel')
    nwbfile.add_electrode_column('pairwise_distances', 'Pairwise distance between all possible pairs of units on the channel')
    nwbfile.add_electrode_column(name='origchannel_name', 
                                 description="The original lab specific channel name for the channel")

    # Add a device and its info
    device = nwbfile.create_device(name=session['system'], description=f"CS - {session['system']}")

    # Add electrodes (brain area locations, MNI coordinates for microwires)
    electrode2group_map = {}
    electrode_counter = 0
    for elec_cnt, electrode_ii in enumerate(channums_all): # one channel per electrode
        
        brainarea_abbv = channel_area_map[electrode_ii]
        brainarea_fullname = brain_abbv_map[brainarea_abbv]

        electrode_name = f"{session['system']}-microwire-{electrode_ii}"
        description = "Behnke Fried/Micro Inner Wire Bundle (Behnke-Fried BF08R-SP05X-000 and WB09R-SP00X-0B6; Ad-Tech Medical)"
        electrode_coords = chanlocs_info_df.loc[electrode_ii,['x', 'y', 'z']].tolist()
        
        # Add an electrode group
        electrode_group = nwbfile.create_electrode_group(name=electrode_name,
                                                         description=description,
                                                         location=brainarea_fullname,
                                                         device=device)
        
        electrode2group_map[electrode_ii] = electrode_group
        
        pairwise_distance_init = d_proj_encod.get(f'{session_ii}_{electrode_ii}')
        if pairwise_distance_init is None:
            pairwise_distance_val = 'NA'
        elif len(pairwise_distance_init) == 1 and pairwise_distance_init[0,1] == 0:
            pairwise_distance_val = 'NA'
        else:
            pairwise_distance_val = '_'.join(map(str,pairwise_distance_init[:,2].round(5)))            
            
        # Add an electrode
        nwbfile.add_electrode(id=elec_cnt,
                              x=electrode_coords[0], y=electrode_coords[1], z=electrode_coords[2],
                              location=brainarea_fullname,
                              group=electrode_group, # this part yields an error in nwbinspector. Not clear how to address.
                              origchannel=f'micro-{electrode_ii}',
                              origchannel_name=channel_area_map_org[electrode_ii],
                              pairwise_distances=pairwise_distance_val,
                              )
        electrode_counter += 1
    # to examine, use: nwbfile.electrodes.to_dataframe()

    electrode_ids = { elecii:elii for elii, elecii in enumerate(channums_all) }


    # ----- Add units to the file -----
    # Add some extra columns to the units table.
    nwbfile.add_unit_column(name='unit_id', description='The unit id in lab specific format')
    nwbfile.add_unit_column('unit_id_session', 'The unit id with session info in lab specific format')
    nwbfile.add_unit_column('electrode_id', '''The id number of corresponding electrode 
                            (used to replace 'electrodes' entry, usage: nwbfile.electrodes[electrode_id])''')
    nwbfile.add_unit_column('electrodegroup_label', '''The label for corresponding electrode group 
                            (used to replace 'electrode_group' entry, 
                            usage: nwbfile.electrode_groups[electrodegroup_label])''')
    nwbfile.add_unit_column('origcluster_id', 'The original OSort cluster id')
    nwbfile.add_unit_column('waveform_mean_encoding', 'The mean waveform for encoding phase')
    nwbfile.add_unit_column('waveform_mean_recognition', 'The mean waveform for the recognition phase')
    nwbfile.add_unit_column('isolationdist', 'The isolation distance of the unit')
    nwbfile.add_unit_column('meanSNR', 'The SNR of the entire waveform of the unit')
    nwbfile.add_unit_column('peakSNR', 'The SNR of the mean waveform peak of the unit')
    nwbfile.add_unit_column('isibelow', 'The proportion of inter-spike intervals (ISIs) which are shorter than 3 ms')
    nwbfile.add_unit_column('cv2', 'The mean modified coefficient of variation of variability in the ISI (CV2)')
    nwbfile.add_unit_column('waveform_mean_sampling_rate', 'The sampling rate of waveform')


    # Add units to the NWB file
    for index_id in range(n_units):
        nwbfile.add_unit(id=index_id,
                         spike_times=allunits_spike_timestamps[index_id], 
                         unit_id=allunits_unitids[index_id], 
                         unit_id_session=allunits_unitids_full[index_id],
                         origcluster_id=allunits_origclusterids[index_id],
                         electrode_id=electrode_ids[allunits_channel_id[index_id]],
                         electrodegroup_label=electrode2group_map[allunits_channel_id[index_id]].name,
                         electrodes= [ electrode_ids[allunits_channel_id[index_id]] ], # should be in this format. 
                         # electrode_group=electrode2group_map[allunits_channel_id[index_id]],
                         isolationdist=allunits_isoldist[index_id], 
                         meanSNR=allunits_meansnr[index_id], 
                         peakSNR=allunits_peaksnr[index_id],
                         isibelow=allunits_isibelow[index_id],
                         cv2=allunits_cv2[index_id],
                         waveform_mean_encoding=allunits_mean_waveform_encod[index_id],
                         waveform_mean_recognition=allunits_mean_waveform_recog[index_id], 
                         waveform_mean_sampling_rate=allunits_waveform_mean_sampling_rate[index_id]
                         )
    # to examine, use: nwbfile.units.to_dataframe()


    # --------- Load lfp data from eeglab .set and .fdt files and add to the nwbfile ---------
    ecephys_module = nwbfile.create_processing_module(
        name='ecephys', 
        description='processed extracellular electrophysiology data')

    table_region_micros = nwbfile.create_electrode_table_region(
        region=list(range(electrode_counter)),  # reference row indices 0 to N-1
        description='electrodes for micros')

    # ---------- Micro-wires ----------
    lfp_orgtime_dw, lfp_data, _ = su_data.load_lfpdata(session_ii, main_raw_datadir, 'micro')
    lfp_start_us = session_start_us - LFP_PAD_us
    lfp_end_us = session_end_us + LFP_PAD_us
    
    lfp_use_inds = np.logical_and( lfp_orgtime_dw >= lfp_start_us, lfp_orgtime_dw <= lfp_end_us )
    lfp_time_use_init = lfp_orgtime_dw[lfp_use_inds]
    lfp_data_micros = lfp_data[lfp_use_inds,:]

    lfp_time_micros = lfp_time_use_init - lfp_start_us
    assert lfp_time_micros[0] >= 0

    timestamps_micros = lfp_time_micros/MICRO2SEC
    micros_tdiff = np.unique(np.diff(timestamps_micros).round(6))
    if len(micros_tdiff) == 1:
        micros_timestamps = None
        micros_starting_time = timestamps_micros[0]
        micros_rate = 1. / micros_tdiff[0]
    else:
        micros_timestamps = timestamps_micros
        micros_starting_time = None
        micros_rate = None


    wrapped_data = H5DataIO(data=lfp_data_micros, compression=True) 

    lfp_electrical_series = ElectricalSeries(
        name = 'ElectricalSeries',
        data = wrapped_data,
        electrodes = table_region_micros,
        timestamps = micros_timestamps,
        starting_time = micros_starting_time,
        rate = micros_rate,
        description = 'Local field potentials from micro-wires. '+\
        f'Note that the session starts at {LFP_PAD_us/MICRO2SEC} seconds.')

    lfp_obj = LFP(name='LFP_micro', electrical_series=lfp_electrical_series)
    
    ecephys_module.add(lfp_obj)
    # to access use: print(nwbfile.processing['ecephys']['LFP_micro']['ElectricalSeries'])


    # ---------- Macro electrodes ---------- 
    lfp_orgtime_dw, lfp_data, _ = su_data.load_lfpdata(session_ii,main_raw_datadir, 'macro')
    lfp_start_us = session_start_us - LFP_PAD_us
    lfp_end_us = session_end_us + LFP_PAD_us
    
    lfp_use_inds = np.logical_and( lfp_orgtime_dw >= lfp_start_us, lfp_orgtime_dw <= lfp_end_us )
    lfp_time_use_init = lfp_orgtime_dw[lfp_use_inds]
    lfp_data_macros = lfp_data[lfp_use_inds,:]

    lfp_time_macros = lfp_time_use_init - lfp_start_us
    assert lfp_time_macros[0] >= 0

    # --- add electrodes for macros ---
    macro_locs_info_df = su_data.get_chanlocs(session_ii,etype='macro')
    
    # special case for P62: use only the first 40 macro channels:
    if 'P62' in session_ii:
        macro_locs_info_df = macro_locs_info_df[:40]
        lfp_data_macros = lfp_data_macros[:,:40]
    
    
    if (macro_locs_info_df is not None) and (len(macro_locs_info_df) == lfp_data_macros.shape[1]):

        macros_brainarea = [ctii[:4] for ctii in macro_locs_info_df['brain_area'] ]
        macros_brainarea_org = macro_locs_info_df['brain_area'].to_list()

        macro_counter_init = electrode_counter
        for macro_cnt, macro_ii in enumerate(macro_locs_info_df.index): 
    
            brainarea_abbv = macros_brainarea[macro_cnt]
            brainarea_fullname = brain_abbv_map[brainarea_abbv]
    
            electrode_name = f"{session['system']}-macros-{macro_ii}"
            description = "macros"
            electrode_coords = macro_locs_info_df.loc[macro_ii,['x', 'y', 'z']].tolist()
            
            # Add an electrode group
            electrode_group = nwbfile.create_electrode_group(electrode_name,
                                                             description=description,
                                                             location=brainarea_fullname,
                                                             device=device)
            
            # Add an electrode
            nwbfile.add_electrode(id=macro_counter_init+macro_cnt,
                                  x=electrode_coords[0], y=electrode_coords[1], z=electrode_coords[2],
                                  location=brainarea_fullname,
                                  group=electrode_group,
                                  origchannel=f'macro-{macro_ii}',
                                  origchannel_name=macros_brainarea_org[macro_cnt],
                                  pairwise_distances = 'NA',
                                  )
            electrode_counter += 1
        # to examine, use: nwbfile.electrodes.to_dataframe()
        
        table_region_macros = nwbfile.create_electrode_table_region(
            region=list(range(macro_counter_init,electrode_counter)),  # reference row indices 0 to N-1
            description='electrodes for macros')

    
        timestamps_macros = lfp_time_macros/MICRO2SEC
        macros_tdiff = np.unique(np.diff(timestamps_macros).round(6))
        if len(macros_tdiff) == 1:
            macros_timestamps = None
            macros_starting_time = timestamps_macros[0]
            macros_rate = 1. / macros_tdiff[0]
        else:
            macros_timestamps = timestamps_macros
            macros_starting_time = None
            macros_rate = None    
    
    
        wrapped_data = H5DataIO(data=lfp_data_macros, compression=True) 
        lfp_electrical_series = ElectricalSeries(
            name = 'ElectricalSeries',
            data = wrapped_data,
            electrodes = table_region_macros,
            timestamps = macros_timestamps,
            starting_time = macros_starting_time,
            rate = macros_rate,
            description = 'Local field potentials from macro electrodes. '+\
            f'Note that the session starts at {LFP_PAD_us/MICRO2SEC} seconds.')
    
        lfp_obj = LFP(name='LFP_macro', electrical_series=lfp_electrical_series)
        
        ecephys_module.add(lfp_obj)
        # to access use: print(nwbfile.processing['ecephys']['LFP_macro']['ElectricalSeries'])

    else:
        print('\n---> Problem in macro electrodes data! Skipped!\n')


    # --------- Save the NWB file ---------
    filename2save = session.get('filename') if session.get('filename') is not None else f'{session_ii}.nwb' 

    with NWBHDF5IO(os.path.join(output_dir,filename2save),'w') as io:
        io.write(nwbfile)

