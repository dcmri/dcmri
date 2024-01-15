import numpy as np

def signal_spgress(TR, FA, R1, S0):

    E = np.exp(-TR*R1)
    cFA = np.cos(FA*np.pi/180)
    return S0 * (1-E) / (1-cFA*E)


def sample(t, S, ts, dts): 
    """Sample the signal assuming sample times are at the start of the acquisition"""

    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t >= tk) & (t < tk+dts)
        data = S[np.nonzero(tacq)[0]]
        Ss[k] = np.average(data)
    return Ss 