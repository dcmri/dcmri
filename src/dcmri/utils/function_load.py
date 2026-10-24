import numpy as np
import pydmr

from dcmri.utils.function_fetch import fetch
from dcmri.utils.const import T1 as T1_lit
from dcmri.utils.const import r1 as r1_lit



def load_minipig():
    data_file = fetch('minipig_renal_fibrosis')
    data = pydmr.read(data_file, 'nest')
    rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']

    time = pars['TS'] * np.arange(len(rois['Aorta']))

    # --- Phase lines
    Nph = int(np.round(pars['TS'] / pars['TR'])) 
    Nk0 = int(np.round(Nph / 2))

    return {
        'agent': "gadoterate",
        'dose': pars['dose'],
        'rate': pars['rate'],
        'field_strength': pars['B0'],
        'weight': pars['weight'],


        'FA': pars['FA'],
        'TR': pars['TR'],
        'R1_ki': 1 / T1_lit(pars['B0'], 'kidney'),
        'R1_b':  1 / T1_lit(pars['B0'], 'blood'),
        'nb': pars['n0'],
        'Nph': Nph,
        'Nk0': Nk0,

        'tS_lk': time,
        'S_lk': rois['LeftKidney'],

        'tS_rk': time,
        'S_rk': rois['RightKidney'],

        'tS_ao': time,
        'S_ao': rois['Aorta'],

        'tS_ar': time,
        'S_ar': rois['Aorta'],
    }


def load_kruk(subjects=None, visit=None, kidney=None):

    datafile = fetch('KRUK')
    data = pydmr.read(datafile, 'nest')

    if subjects is None:
        load_subjects = set(data['rois'].keys())
    elif isinstance(subjects, str):
        load_subjects = [subjects]
    else:
        load_subjects = subjects

    if kidney is None:
        data = {subj: _load_kruk_both_kidneys(data, subj, visit) for subj in load_subjects}

    if kidney in ['LK', 'RK']:
        data = {subj: _load_kruk_kidney(data, subj, visit, kidney) for subj in load_subjects}

    if isinstance(subjects, str):
        return data[subjects]
    else:
        return data


def _load_kruk_both_kidneys(data, subject, visit):

    if visit not in data['rois'][subject]:
        return None

    # --- Fetch the data
    roi = data['rois'][subject][visit] 
    par = data['pars'][subject][visit] 

    # --- B0 and precontrast T1
    B0 = par['field_strength']
    T1 = {
        kidney: par[f'{kidney} T1'] if f'{kidney} T1' in par else T1_lit(B0, 'kidney') 
        for kidney in ['LK', 'RK']
    }

    # --- Phase lines
    ts = roi['time'][1] - roi['time'][0]
    Nph = int(np.round(ts / par['TR'])) 
    Nk0 = int(np.round(Nph / 2))

    # --- Define kidney model
    return {

        # Electromagnetic quantities
        'R1_b': 1 / T1_lit(B0, 'blood'),
        'R1_lk': 1 / T1['LK'],
        'R1_rk': 1 / T1['RK'],
        
        # Indicator quantities
        'agent': par['agent'],
        'r1': r1_lit(B0, 'blood', par['agent']),
        'dose': 2 * par['dose'], # convert gadoterate dose from mmmol/kg to mL/kg
        'rate': par['rate'],

        # Physiological
        'nmGFR_lk': par[f'LK iso-SK-GFR'],
        'nmGFR_rk': par[f'RK iso-SK-GFR'],
        'H': 0.45,

        # Sequence
        'FA': par['FA'],
        'field_strength': B0,
        'Nph': Nph,
        'Nk0': Nk0,        
        'TR': par['TR'],

        # Signal
        'nb': par['n0'],
        'S_ao': roi['aorta'], 
        'S_lk': roi['LK'],
        'S_rk': roi['RK'],
        'tS_ao': roi['time'],
        'tS_lk': roi['time'],
        'tS_rk': roi['time'],
        
        # Whole-body
        'vol_lk': par[f'LK vol'],
        'vol_rk': par[f'RK vol'],
        'weight': par['weight']
    }

def _load_kruk_kidney(data, subject, visit, kidney):

    if visit not in data['rois'][subject]:
        return None

    # --- Fetch the data
    roi = data['rois'][subject][visit] 
    par = data['pars'][subject][visit] 

    # --- B0 and precontrast T1
    B0 = par['field_strength']
    T1 = par[f'{kidney} T1'] if f'{kidney} T1' in par else T1_lit(B0, 'kidney')

    # --- Phase lines
    ts = roi['time'][1] - roi['time'][0]
    Nph = int(np.round(ts / par['TR'])) 
    Nk0 = int(np.round(Nph / 2))

    # --- Define kidney model
    return {

        # Indicator quantities
        'agent': par['agent'],
        'r1': r1_lit(B0, 'blood', par['agent']),

        # Signal quantities
        'field_strength': B0,
        'FA': par['FA'],
        'TR': par['TR'],
        'Nph': Nph,
        'Nk0': Nk0,

        # Electromagnetic quantities
        'R1_ki': 1 / T1,
        'R1_b': 1 / T1_lit(B0, 'blood'),

        'tS_ki': roi['time'],
        'S_ki': roi[kidney],

        'tS_ar': roi['time'],
        'S_ar': roi['aorta'],        

        'nb': par['n0'],

        'nmGFR': par[f'{kidney} iso-SK-GFR'],
        'vol_ki': par[f'{kidney} vol'],
    }


def load_tristan_rats(study_name, subject, visit):

    # --- Get the data of the study
    data_file = fetch(study_name)
    all_data = pydmr.read(data_file, 'nest')

    # --- Get the data of this subject visit
    roi = all_data['rois'][subject][visit]
    par = all_data['pars'][subject][visit]

    # --- Number of phase lines
    # Data are acquired with a triggered sequence - meaning k-space 
    # lines are acquired for a given duration (57 sec) along with a 
    # breathing signal, and only lines measured in expiration are 
    # used to reconstruct the image.

    ts = roi['time'][1] - roi['time'][0]
    Nph = int(np.round(ts / par['TR'])) 

    # The sequence is linearly encoded so the central line falls 
    # approximately in the middle:

    Nk0 = int(np.round(Nph / 2))

    # --- Build the data dictionary 
    return {

        # --- Data
        'tS_li': roi['time'],
        'S_li': roi['liver'],

        # --- Contrast agent
        'agent': 'gadoxetate',
        'BAT': par['BAT'] + roi['time'][1] / 2,
        'duration': par['duration'],

        # --- Sequence parameters
        'field_strength': par['field_strength'],
        'TR': par['TR'],
        'FA': par['FA'],
        'Nph': Nph, 
        'Nk0': Nk0,

        # --- Hyperparameters
        'nb': par['n0'],
    }


def load_tristan_humans(study, subjects=None, visits=['control', 'drug'], scans=[1,2], rois=['ao', 'li']):

    file = fetch(study)
    data = pydmr.read(file, 'nest')

    if subjects is None:
        load_subjects = set(data['rois'].keys())
    elif isinstance(subjects, str):
        load_subjects = [subjects]
    else:
        load_subjects = subjects

    if len(visits) == 2 and len(scans) == 2 and len(rois) == 2:
        data = {subj: _load_tristan_humans_all(data, subj) for subj in load_subjects}

    if len(visits) == 1 and len(scans) == 2 and len(rois) == 2:
        data = {subj: _load_tristan_humans_visit(data, subj, visits[0]) for subj in load_subjects}

    if len(visits) == 2 and len(scans) == 1 and len(rois) == 2:
        data = {subj: _load_tristan_humans_scan(data, subj, scans[0]) for subj in load_subjects}

    if len(visits) == 1 and len(scans) == 1 and len(rois) == 2:
        data = {subj: _load_tristan_humans_visit_scan(data, subj, visits[0], scans[0]) for subj in load_subjects}

    if len(visits) == 1 and len(scans) == 1 and len(rois) == 1:
        data = {subj: _load_tristan_humans_visit_scan_roi(data, subj, visits[0], scans[0], rois[0]) for subj in load_subjects}

    if isinstance(subjects, str):
        return data[subjects]
    else:
        return data


def _load_tristan_humans_visit_scan_roi(study_name, subject, visit, roi):
    pass


def _load_tristan_humans_visit_scan(data, participant, visit, scan): 

    # --- Get the subject-level data
    roi = data['rois'][participant][visit]
    par = data['pars'][participant][visit]

    # --- Get GFR from screening visit if available
    if 'eGFR_abs' in data['pars'][participant]['screening']:
        eGFR = data['pars'][participant]['screening']['eGFR_abs']
    else:
        eGFR = data['pars'][participant][visit]['eGFR_abs']

  # --- Time arrays
    time = [
        roi[f'time_{scan}'] - roi['time_1'][0],
    ]

    data = {     

        # Electromagnetic
        'R1_b': 1 / par['T1_aorta_1'], 
        'R1_li': 1 / par['T1_liver_1'],

        # Hyperparameter
        'dt': 0.5,

        # Indicator
        'agent': 'gadoxetate',
        
        # Physiological
        'GFR': eGFR / 60,

        # Sequence
        'FA': par['FA_1'],
        'field_strength': 3,
        'Nk0': int(np.round(time[0][1] / par['TR'] / 2)),
        'Nph': int(np.round(time[0][1] / par['TR'])),
        'TR': par['TR'],

        # Whole body
        'vol_li': par['liver_volume'],
        'weight': par['weight'], 
    }

    # Indicator group
    data[f'rate'] = 1
    data[f'dose'] = par[f'dose_{scan}']

    # Signal group
    data[f'nb'] = int(par['t0'] / (time[0][1] - time[0][0]))
    for r in ['aorta', 'liver']:
        data[f'tS_{r[:2]}'] = time[0]
        data[f'S_{r[:2]}'] = roi[f'{r}_{scan}']
        data[f'iStrig_{r[:2]}'] = roi[f'{r}_{scan}']

    return data


def _load_tristan_humans_scan(data, participant, scan): 

    # --- Get the subject-level data
    roi = data['rois'][participant]
    par = data['pars'][participant]

    # --- Get GFR from screening visit if available
    if 'eGFR_abs' in par['screening']:
        eGFR = par['screening']['eGFR_abs']
    else:
        eGFR = (par['control']['eGFR_abs'] + par['drug']['eGFR_abs']) / 2

    # --- Time arrays
    time = [
        roi['control']['time_1'] - roi['control']['time_1'][0],
        roi['drug']['time_1'] - roi['drug']['time_1'][0],
    ]

    data = {     
        # Electromagnetic
        'R1_b': (1 / par['control']['T1_aorta_1'] + 1 / par['drug']['T1_aorta_1']) / 2, 
        'R1_li':(1 / par['control']['T1_liver_1'] + 1 / par['drug']['T1_liver_1']) / 2,

        # Hyperparameter
        'dt': 0.5,

        # Indicator
        'agent': 'gadoxetate',

        # Physiological
        'GFR': eGFR / 60,

        # Sequence
        'FA': par['control']['FA_1'],
        'field_strength': 3,
        'Nk0': int(np.round(time[0][1] / par['control']['TR'] / 2)),
        'Nph': int(np.round(time[0][1] / par['control']['TR'])),
        'TR': par['control']['TR'],

        # Whole body
        'vol_1_li': par['control']['liver_volume'],
        'vol_2_li': par['drug']['liver_volume'],
        'weight':(par['control']['weight'] + par['drug']['weight']) / 2, 
    }

    idx = 0
    for visit in ['control', 'drug']:
        idx += 1

        # Indicator group
        data[f'rate_{idx}'] = 1
        data[f'dose_{idx}'] = par[visit][f'dose_{scan}']

        # Signal group
        data[f'nb_{idx}'] = int(par[visit]['t0'] / (time[idx - 1][1] - time[idx - 1][0]))
        for r in ['aorta', 'liver']:
            data[f'tS_{idx}_{r[:2]}'] = time[idx - 1]
            data[f'S_{idx}_{r[:2]}'] = roi[visit][f'{r}_{scan}']
            data[f'iStrig_{idx}_{r[:2]}'] = roi[visit][f'{r}_{scan}_accept']
                     
    return data
    
    
def _load_tristan_humans_visit(data, participant, visit): 

    # --- Get the subject-level data
    roi = data['rois'][participant][visit]
    par = data['pars'][participant][visit]
    
    # --- Get GFR from screening visit if available
    par_screen = data['pars'][participant]['screening']
    eGFR = par_screen['eGFR_abs'] if 'eGFR_abs' in par_screen else par['eGFR_abs']

    # --- Time arrays
    time_1 = roi['time_1'] - roi['time_1'][0]
    time_2 = roi['time_2'] - roi['time_1'][0]

    return {     

        # Electromagnetic
        'R1_b': 1 / par['T1_aorta_1'], 
        'R1_li': 1 / par['T1_liver_1'], 

        # Hyperparameter
        'dt': 0.5,

        # Indicator
        'agent': 'gadoxetate',
        'dose_1': par['dose_1'],
        'dose_2': par['dose_2'],
        'rate_1': 1,
        'rate_2': 1,

        # Physiological
        'GFR': eGFR / 60,

        # Sequence
        'FA': par['FA_1'],
        'field_strength': 3,
        'Nk0': int(np.round(time_1[1] / par['TR'] / 2)),
        'Nph': int(np.round(time_1[1] / par['TR'])),
        'TR': par['TR'],

        # Signal
        'iStrig_1_ao': roi['aorta_1_accept'],
        'iStrig_1_li': roi['liver_1_accept'],
        'iStrig_2_ao': roi['aorta_2_accept'],
        'iStrig_2_li': roi['liver_2_accept'],

        'nb_1': int(par['t0'] / (time_1[1] - time_1[0])),
        'nb_2': int(par['t0'] / (time_2[1] - time_2[0])),

        'S_1_ao': roi['aorta_1'] , 
        'S_2_ao': roi['aorta_2'],
        'S_1_li': roi['liver_1'], 
        'S_2_li': roi['liver_2'],

        'tS_1_ao': time_1,
        'tS_2_ao': time_2,
        'tS_1_li': time_1,
        'tS_2_li': time_2,

        # Whole body
        'vol_li': par['liver_volume'],
        'weight': par['weight'], 
    }


def _load_tristan_humans_all(data, participant): 

    # --- Get the subject-level data
    roi = data['rois'][participant]
    par = data['pars'][participant]
    
    # --- Get GFR from screening visit if available
    par_screen = par['screening']
    if 'eGFR_abs' in par_screen:
        eGFR = par_screen['eGFR_abs'] 
    else: 
        eGFR = (par['control']['eGFR_abs'] + par['drug']['eGFR_abs']) / 2

    # --- Time arrays
    time = [
        roi['control']['time_1'] - roi['control']['time_1'][0],
        roi['control']['time_2'] - roi['control']['time_1'][0],
        roi['drug']['time_1'] - roi['drug']['time_1'][0],
        roi['drug']['time_2'] - roi['drug']['time_1'][0],
    ]

    data = {     

        # Electromagnetic
        'R1_b': (1 / par['control']['T1_aorta_1'] + 1 / par['drug']['T1_aorta_1']) / 2, 
        'R1_li':(1 / par['control']['T1_liver_1'] + 1 / par['drug']['T1_liver_1']) / 2,

        # Hyperparameter
        'dt': 0.5,

        # Indicator
        'agent': 'gadoxetate',

        # Physiological
        'GFR': eGFR / 60,

        # Sequence
        'FA': par['control']['FA_1'],
        'field_strength': 3,
        'Nk0': int(np.round(time[0][1] / par['control']['TR'] / 2)),
        'Nph': int(np.round(time[0][1] / par['control']['TR'])),
        'TR': par['control']['TR'],

        # Whole body
        'vol_1_li': par['control']['liver_volume'],
        'vol_2_li': par['drug']['liver_volume'],
        'weight': par['control']['weight'], 
    }

    idx = 0
    for visit in ['control', 'drug']:
        for scan in [1, 2]:
            idx += 1

            # Indicator group
            data[f'rate_{idx}'] = 1
            data[f'dose_{idx}'] = par[visit][f'dose_{scan}']

            # Signal group
            data[f'nb_{idx}'] = int(par[visit]['t0'] / (time[idx - 1][1] - time[idx - 1][0]))
            for r in ['aorta', 'liver']:
                data[f'tS_{idx}_{r[:2]}'] = time[idx - 1]
                data[f'S_{idx}_{r[:2]}'] = roi[visit][f'{r}_{scan}']
                data[f'iStrig_{idx}_{r[:2]}'] = roi[visit][f'{r}_{scan}_accept']
                     
    return data