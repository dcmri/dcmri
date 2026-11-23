import numpy as np

from dcmri.core import pseq_tools

# OBSELETE FUNCTIONS
def get_sequence(*args, **kwargs):
    return pseq_tools.get_sequence(*args, **kwargs)

def channels(*args, **kwargs):
    return pseq_tools.sequence_channels(*args, **kwargs)

def pulse_readout(*args, **kwargs):
    return pseq_tools.sequence_readout_pulses(*args, **kwargs)

def readout_time(*args, **kwargs):
    return pseq_tools.sequence_readout_time(*args, **kwargs)

def repetition_time(*args, **kwargs):
    return pseq_tools.sequence_repetition_time(*args, **kwargs)

def acquisition_times(*args, **kwargs):
    return pseq_tools.sequence_acquisition_times(*args, **kwargs)



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

    elif seq in ['ZTE-3D-SPGR-SS', 'ZTE-3D-SPGR-SSI', 'ZTE-3D-IR-SPGR-SS']:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float)
        R2s = np.zeros(nt)
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, 0)
    
    else:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float) 
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, p['TE'])  

    return Mxy