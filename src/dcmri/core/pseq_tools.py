import numpy as np

from dcmri.core.dictionaries import AGENTS, SEQUENCES

def all_sequences():
    return dict(SEQUENCES)

def all_contrast_agents():
    return set(AGENTS)

def get_sequence(prop, sequence=None):
    if prop=='name':
        return set(SEQUENCES.keys()) - {'3D-SPGR-SSI'}

    if prop=='steady-state':
        if sequence is None:
            return {s for s in SEQUENCES if SEQUENCES[s]['steady-state'] and s != '3D-SPGR-SSI'}
        return SEQUENCES[sequence]['steady-state']
                                        
    if prop=='mz_prep_inflow':
        return SEQUENCES[sequence]['mz_prep_inflow']

    if prop=='mz_prep_tissue':
        return SEQUENCES[sequence]['mz_prep_tissue']

    if prop=='prep_params':
        return SEQUENCES[sequence]['parameters']['prep']

    if prop=='read_params':
        return SEQUENCES[sequence]['parameters']['read']

    if prop=='tissue_params':
        return SEQUENCES[sequence]['parameters']['tissue']

    raise ValueError(f'{prop} is an unknown sequence property')


def sequence_channels(sequence):
    return 2 if sequence in ['Eq-DE-EPI', '2D-DE-EPI', '3D-DE-EPI'] else 1

def sequence_readout_pulses(sequence, p):
    if sequence in [
            'ZTE-3D-SPGR-SS',
            'ZTE-3D-SPGR-SSI',
            '3D-SPGR-SS',
            '2D-SPGR-SS',
            '3D-SPGR',
            '2D-SPGR',
            '3D-SPGR-SSI',
        ]:
        return p['Nk0']
    
    elif sequence in [
            'ZTE-3D-IR-SPGR-SS',
            '3D-IR-SPGR-SS',
            '3D-SR-SPGR-SS',
            '3D-PR-SPGR-SS',
            '3D-IR-SPGR',
            '3D-SR-SPGR',
            '3D-PR-SPGR',
            '2D-SR-SPGR',
        ]:
        return 1 + p['Nk0']

    else:
        return 0

def sequence_readout_time(sequence, p):
    if sequence in [
            'ZTE-3D-SPGR-SS',
            'ZTE-3D-SPGR-SSI',
            '3D-SPGR-SS',
            '2D-SPGR-SS',
            '3D-SPGR',
            '2D-SPGR',
            '3D-SPGR-SSI',
        ]:
        return p['Nk0'] * p['TR']
    
    elif sequence in [
            'ZTE-3D-IR-SPGR-SS',
            '3D-IR-SPGR-SS',
            '3D-SR-SPGR-SS',
            '3D-PR-SPGR-SS',
            '3D-IR-SPGR',
            '3D-SR-SPGR',
            '3D-PR-SPGR',
            '2D-SR-SPGR',
        ]:
        return p['TP'] + p['Nk0'] * p['TR']

    else:
        return 0

def sequence_repetition_time(sequence, p):
    if sequence in [
        'ZTE-3D-SPGR-SS',
        '3D-SPGR-SS',
        '2D-SPGR-SS',
        '3D-SPGR',
        '2D-SPGR',
        'ZTE-3D-SPGR-SSI',
        '3D-SPGR-SSI',
    ]:
        return p['Nph'] * p['TR']

    elif sequence in [
        'ZTE-3D-IR-SPGR-SS',
        '3D-IR-SPGR-SS',
        '3D-SR-SPGR-SS',
        '3D-PR-SPGR-SS',
        '3D-IR-SPGR',
        '3D-SR-SPGR',
        '3D-PR-SPGR',
        '2D-SR-SPGR',
    ]:
        return p['TP'] + p['Nph'] * p['TR'] + p['TD']

    elif sequence in [
        '2D-GE-EPI',
        '2D-SE-EPI',
        '2D-DE-EPI',
        '3D-GE-EPI',
        '3D-SE-EPI',
        '3D-DE-EPI',
    ]:
        return p['TR']
    
    return p['TA']


def sequence_acquisition_times(sequence, p, tacq):
    TR = sequence_repetition_time(sequence, p)
    t0 = sequence_readout_time(sequence, p)
    nt = np.floor(tacq / TR)
    return t0 + TR * np.arange(nt)