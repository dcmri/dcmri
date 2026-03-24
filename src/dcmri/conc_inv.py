import numpy as np

from dcmri import sig, mz

def params_conc(sequence):
    return {
        'T2w': ['TE', 'r2', 'n0'],
        'SS': ['TR', 'FA', 'r1', 'n0'],
        'SR': ['TC', 'TR', 'FA', 'TP', 'r1', 'n0'],
        'IR': ['TC', 'TR', 'FA', 'TP', 'r1', 'n0'],
        'SPGR': ['TC', 'TR', 'FA', 'TP', 'n_init', 'r1', 'n0'],
        'free': ['TC', 'FA', 'r1', 'n0'],
        'lin': ['r1', 'n0'],
    }[sequence]
    

def conc(
    # model
    sequence, 
    # signal
    S, 
    # parameters
    R10=None, S0=None, r1=0.005, r2=0.5, n0=1, TE=None, TR=None, FA=None, 
    TC=None, TP=None, n_init=1
):
    # Possible shapes for S:
    # 1D (nt, ) or 2D (n_samples, nt)
    # Shapes for R10:
    # 1D (n_samples)

    # Keep input shape for return values
    input_shape = np.shape(S)

    # Reshape S to standard shape (n_samples, nt)
    S = np.atleast_1d(S)
    if S.ndim==1:
        n_samples=1
    else:
        n_samples=S.shape[0]
    S = S.reshape(n_samples, -1)
    nt = S.shape[1]

    # Exclude nt=1: conc requires baseline
    if nt==1:
        raise ValueError("Signal needs more than 1 time point for conc calculation")

    # Reshape R10 to standard dims (n_samples)
    if R10 is not None:
        R10 = np.atleast_1d(R10)
        if R10.size==1:
            R10 = np.full(n_samples, R10[0])
        elif R10.size != n_samples:
            raise ValueError("R10 must have the same number of samples as the signal")
        
    # Reshape S0 to standard dims (n_samples)
    if S0 is not None:
        S0 = np.atleast_1d(S0)
        if S0.size==1:
            S0 = np.full(n_samples, S0[0])
        elif S0.size != n_samples:
            raise ValueError("R10 must have the same number of samples as the signal")
    
    if sequence == 'T2w':
        conc = _conc_t2w(S, TE, r2, n0)
    elif sequence == 'SS':
        conc = _conc_ss(S, TR, FA, R10, r1, n0, S0) 
    elif sequence == 'SPGR':
        conc = _conc_spgr(S, TC, TR, FA, TP, R10, n_init, r1, n0, S0)
    elif sequence == 'SR':
        conc = _conc_spgr(S, TC, TR, FA, TP, R10, 0, r1, n0, S0)
    elif sequence == 'IR':
        conc = _conc_spgr(S, TC, TR, FA, TP, R10, -1, r1, n0, S0)
    elif sequence == 'free':
        conc = _conc_free(S, TC, FA, R10, r1, n0, S0)
    elif sequence == 'lin':
        conc = _conc_lin(S, R10, r1, n0, S0)
    else:
        raise ValueError(f'Sequence {sequence} is not recognised.')
    
    # Return result in original shape
    return conc.reshape(input_shape)
    

def _conc_t2w(S, TE, r2=0.5, n0=1) -> np.ndarray:
    # S/Sb = exp(-TE(R2-R2b))
    #   ln(S/Sb) = -TE(R2-R2b)
    #   R2-R2b = -ln(S/Sb)/TE
    # R2 = R2b + r2C
    #   C = (R2-R2b)/r2
    #   C = -ln(S/Sb)/TE/r2

    # 1. Calculate Sb (the baseline)
    Sb = np.mean(S[:, :n0], axis=1)[:, np.newaxis]
    # Reshape Sb to (n_samples, 1) to divide S (n_samples, n_times)
    S_normalized = np.divide(S, Sb, out=np.zeros_like(S, dtype=float), where=Sb != 0)

    # 2. Calculate Concentration
    # Since S_normalized is now the same shape as S, 
    # the rest of the math follows naturally.
    C = -np.log(S_normalized) / (TE * r2)
    return C


def _conc_spgr(S, T, TR, FA, TP, R10, n_init, r1=0.005, n0=1, S0=None):
    # S = signal_spgr(S0, R1, T, TR, FA, TP) -> R1?
    # Compute
    # S0 = Sb / signal_spgr(1, R10, T, TR, FA, TP)
    # Sn = S / S0
    # New question:
    # Sn = signal_spgr(1, R1, T, TR, FA, TP) -> R1?
    # Lookup R1
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        Mb = mz.Mz('SPGR', R10, TC=T, TR=TR, FA=FA, TP=TP, n_init=n_init)
        Sb0 = sig.signal(Mb, S0=1, FA=FA)
        S0 = np.divide(Sb, Sb0, out=np.zeros_like(Sb, dtype=float), where=Sb0 > 0)

    S0 = S0[:, np.newaxis]
    Sn = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Estimate maximum realistic R1 using maximum realistic relaxivity
    # and concentration
    r1_max = 10.0 # Hz/mM 
    c_max = 10 # mM
    R1_max = r1_max * c_max # 10 Hz/mM * 10 mM = 100 Hz corresponds to T1=10ms
    R1_step = 0.01 # 10000 steps

    # Create lookup table
    R1_lookup = np.arange(0, R1_max, R1_step)
    Mz_lookup = mz.Mz('SPGR', R1_lookup, TC=T, TR=TR, FA=FA, TP=TP, n_init=n_init)
    Sn_lookup = sig.signal(Mz_lookup, S0=1, FA=FA)

    # Look up R1 values
    R1 = np.interp(Sn, Sn_lookup, R1_lookup)

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    # Convert to concentrations
    R10 = R10[:, np.newaxis]
    return (R1 - R10)/r1


def _conc_ss(S, TR: float, FA: float, R10: float, r1=0.005, n0=1, S0=None) -> np.ndarray:
    # S = Sinf * (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sb = Sinf * (1-exp(-TR*R10)) / (1-cFA*exp(-TR*R10))
    # Sn = (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sn * (1-cFA*exp(-TR*R1)) = 1-exp(-TR*R1)
    # exp(-TR*R1) - Sn *cFA*exp(-TR*R1) = 1-Sn
    # (1-Sn*cFA) * exp(-TR*R1) = 1-Sn

    cFA = np.cos(np.radians(FA))
    sFA = np.sin(np.radians(FA))

    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        Mb0 = mz.Mz('SS', R10, TR=TR, FA=FA)
        Sb0 = sig.signal(Mb0, S0=1, FA=FA)
        S0 = np.divide(Sb, Sb0, out=np.zeros_like(Sb, dtype=float), where=Sb0 > 0)

    S0 = S0[:, np.newaxis] 
    Sn = np.divide(S, sFA * S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    Sn = (1 - Sn) / (1 - cFA * Sn)
    with np.errstate(divide='ignore', invalid='ignore'):
        R1 = np.where(Sn <= 0, 0, -np.log(Sn)/TR)  

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10 )/ r1


def _conc_free(S, TC: float, FA, R10: float, r1=0.005, n0=1, S0=None) -> np.ndarray:
    # S = S0*(1-exp(-TC*R1))
    # S/Sb = (1-exp(-TC*R1))/(1-exp(-TC*R10))
    # (1-exp(-TC*R10))*S/Sb = 1-exp(-TC*R1)
    # 1-(1-exp(-TC*R10))*S/Sb = exp(-TC*R1)
    # ln(1-(1-exp(-TC*R10))*S/Sb) = -TC*R1
    # -ln(1-(1-exp(-TC*R10))*S/Sb)/TC = R1
    # Sn = 1 - exp(-TC*R1)

    sFA = np.sin(np.radians(FA))
    
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        Mb = mz.Mz('free', R10, TC=TC)
        Sb0 = sig.signal(Mb, S0=1, FA=FA)
        S0 = np.divide(Sb, Sb0, out=np.zeros_like(Sb, dtype=float), where=Sb0 > 0)

    S0 = S0[:, np.newaxis] 
    Sn = np.divide(S, sFA * S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    Sn = 1 - Sn
    with np.errstate(divide='ignore', invalid='ignore'):
        R1 = np.where(Sn <= 0, 0, -np.log(Sn)/TC) 

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis] 
    return (R1 - R10 )/ r1


def _conc_lin(S, R10, r1=0.005, n0=1, S0=None):
    # S = S0 * R1
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        S0 = Sb / R10
        S0 = np.divide(Sb, R10, out=np.zeros_like(Sb, dtype=float), where=R10 > 0)

    S0 = S0[:, np.newaxis] 
    R1 = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10) / r1