import numpy as np
from scipy.linalg import expm

def channels(sequence):
    return 2 if sequence in ['Eq-DE-EPI', '2D-DE-EPI', '3D-DE-EPI'] else 1

def pulse_readout(sequence, p):
    if sequence in [
            'ZTE-3D-SPGR-SS',
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

def readout_time(sequence, p):
    if sequence in [
            'ZTE-3D-SPGR-SS',
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

def repetition_time(sequence, p):
    if sequence in [
        'ZTE-3D-SPGR-SS',
        '3D-SPGR-SS',
        '2D-SPGR-SS',
        '3D-SPGR',
        '2D-SPGR',
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


def acquisition_times(sequence, p, tacq):
    TR = repetition_time(sequence, p)
    t0 = readout_time(sequence, p)
    nt = np.floor(tacq / TR)
    return t0 + TR * np.arange(nt)


def mz_readout(Mz: np.ndarray, R2: np.ndarray, FA, TE):
    # Shapes for Mz, R2: (nc, nt)
    # Other parameters are scalar
    # returns shape (nc, nt,)
    sFA = np.sin(np.radians(FA))
    decay = np.exp(-TE * R2)
    Mxy = decay * sFA * Mz
    return Mxy

    # Mxy = np.sum(Mxy, axis=0) # sum over compartments
    # signal = S0 * np.abs(Mxy)
    # return signal_rice(signal, noise_sdev)
    

def mz_readout_wrapper(Mz, p, seq, R2=None, R2s=None):
    nc, nt = Mz.shape
    FA = p['FA'] * p['B1corr']
    
    if seq in ['2D-SE-EPI', '3D-SE-EPI']:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float) # (channels, components, compartments, times)
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2[c, :], FA, p['TE'])
    
    elif seq in ['2D-DE-EPI', '3D-DE-EPI']:
        Mxy = np.zeros((2, 2, nc, nt), dtype=float)
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, p['TE1'])
            Mxy[1, 0, c, :] = mz_readout(Mz[c, :], R2[c, :], FA, p['TE2'])

    elif seq in ['ZTE-3D-SPGR-SS', 'ZTE-3D-IR-SPGR-SS']:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float)
        R2s = np.zeros(nt)
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, 0)
    
    else:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float) 
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, p['TE'])  

    return Mxy