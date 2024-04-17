import numpy as np

def signal_spgress(TR, FA, R1, S0):

    E = np.exp(-TR*R1)
    cFA = np.cos(FA*np.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def signal_srspgre(R1, S0, TR, FA, Tsat, TI):
    """Selective FLASH readout with non-selective saturation preparation"""
    if Tsat > TI:
        msg = 'Incorrect sequence parameters.'
        msg += 'Tsat must be smaller than TI.'
        raise ValueError(msg)
    cFA = np.cos(np.pi*FA/180)
    T1_app = TR/(TR*R1-np.log(cFA))

    ER = np.exp(-TR*R1)
    E_sat = np.exp(-Tsat*R1)
    E_center = np.exp(-(TI-Tsat)/T1_app)

    S_sat = S0 * (1-E_sat)
    S_ss = S0 * (1-ER)/(1-cFA*ER)

    return S_ss*(1-E_center) + S_sat*E_center

def signal_eqspgre(R1, S0, TR, FA, TI):
    """Selective FLASH readout from equilibrium (inflow model)"""
    #TI is the residence time in the slab
    cFA = np.cos(np.pi*FA/180)
    R1_app = R1 - np.log(cFA)/TR

    ER = np.exp(-TR*R1)
    EI = np.exp(-TI*R1_app)

    S_ss = S0 * (1-ER)/(1-cFA*ER)

    return S_ss*(1-EI) + S0*EI


def signal_sr(R1, S0, TI):
    """Saturation-recovery sequence ignoring the effect of the readout (for fast flowing magnetization)"""
    E = np.exp(-TI*R1)
    return S0 * (1-E)


def sample(t, S, ts, dts): 
    """Sample the signal assuming sample times are at the start of the acquisition"""

    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t >= tk) & (t < tk+dts)
        data = S[np.nonzero(tacq)[0]]
        Ss[k] = np.average(data)
    return Ss 