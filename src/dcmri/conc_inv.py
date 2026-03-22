import numpy as np

from dcmri import sig

def params_conc(sequence):
    return {
        'T2w': ['TE', 'r2', 'n0'],
        'SS': ['TR', 'FA', 'R10', 'r1', 'n0'],
        'SR': ['TC', 'TR', 'FA', 'TP', 'R10', 'r1', 'n0'],
        'SRC': ['TC', 'R10', 'r1', 'n0'],
        'lin': ['R10', 'r1', 'n0'],
    }[sequence]
    

def conc(
    # model
    sequence, 
    # signal
    S, 
    # parameters
    R10=None, r1=0.005, r2=0.5, n0=1, TE=None, TR=None, FA=None, 
    TC=None, TP=None,
):

    if sequence == 'T2w':
        return _conc_t2w(S, TE, r2, n0)
    if sequence == 'SS':
        return _conc_ss(S, TR, FA, R10, r1, n0) 
    elif sequence == 'SR':
        return _conc_spgr(S, TC, TR, FA, TP, R10, r1, n0)
    elif sequence == 'SRC':
        return _conc_src(S, TC, R10, r1, n0)
    elif sequence == 'lin':
        return _conc_lin(S, R10, r1, n0)
    else:
        raise ValueError(f'Sequence {sequence} is not recognised.')
    


def _conc_t2w(S, TE: float, r2=0.5, n0=1) -> np.ndarray:
    """Concentration for a DSC scan with T2-weighting.

    Args:
        S (array-like): Signal in arbitrary units.
        TE (float): Echo time in sec.
        r2 (float, optional): Transverse relaxivity in Hz/M. Defaults to 0.5.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration in M, same length as S.
    """
    # S/Sb = exp(-TE(R2-R2b))
    #   ln(S/Sb) = -TE(R2-R2b)
    #   R2-R2b = -ln(S/Sb)/TE
    # R2 = R2b + r2C
    #   C = (R2-R2b)/r2
    #   C = -ln(S/Sb)/TE/r2

    # 1. Calculate Sb (the baseline)
    # If S is 2D (samples, times), we mean across the second axis (time)
    if np.ndim(S) > 1:
        # Mean of first n0 columns for every row
        Sb = np.mean(S[:, :n0], axis=1)
        # Reshape Sb to (n_samples, 1) to divide S (n_samples, n_times)
        S_normalized = S / Sb[:, np.newaxis]
    else:
        # Standard 1D array case
        Sb = np.mean(S[:n0])
        S_normalized = S / Sb

    # 2. Calculate Concentration
    # Since S_normalized is now the same shape as S, 
    # the rest of the math follows naturally.
    C = -np.log(S_normalized) / (TE * r2)
    return C



def _conc_ss(S, TR: float, FA: float, R10: float, r1=0.005, n0=1, S0=None) -> np.ndarray:
    """Concentration of a spoiled gradient echo sequence applied in steady state.

    Args:
        S (array-like): Signal in arbitrary units.
        TR (float): Repetition time, or time between successive selective excitations, in sec.
        FA (float): Flip angle in degrees.
        T10 (array-like): baseline T1 value in sec.
        r1 (float, optional): Longitudinal relaxivity in Hz/M. Defaults to 0.005.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration inM , same length as S.
    """
    # S = Sinf * (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sb = Sinf * (1-exp(-TR*R10)) / (1-cFA*exp(-TR*R10))
    # Sn = (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sn * (1-cFA*exp(-TR*R1)) = 1-exp(-TR*R1)
    # exp(-TR*R1) - Sn *cFA*exp(-TR*R1) = 1-Sn
    # (1-Sn*cFA) * exp(-TR*R1) = 1-Sn
    S = np.array(S)
    shape = S.shape
    S = S.reshape(-1, shape[-1])
    if np.isscalar(R10):
        R10 = np.full(S.shape[0], R10)
    else:
        R10 = R10.reshape(S.shape[0])

    cFA = np.cos(FA*np.pi/180)
    if S0 is None:
        # If S0 is not provided, estimate from the data
        Sb = np.mean(S[:, :n0], axis=-1)
        Sb = np.broadcast_to(Sb[..., None], S.shape)
        R10 = np.broadcast_to(R10[..., None], S.shape)
        Sn = np.zeros(S.shape)
        nozero = Sb > 0
        E0 = np.exp(-TR*R10)
        E0 = E0[nozero]
        Sn[nozero] = (S[nozero]/Sb[nozero])*(1-E0)/(1-cFA*E0)	     
        # Replace any Nan values by interpolating between nearest neighbours
        outrange = Sn >= 1
        if np.sum(outrange) > 0:
            inrange = Sn < 1
            x = np.arange(Sn.size).reshape(Sn.shape)
            Sn[outrange] = np.interp(x[outrange], x[inrange], Sn[inrange])
    else:
        S0 = np.array(S0).reshape(-1)
        S0 = np.broadcast_to(S0[..., None], S.shape)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Sn = np.where(S0>0, S/S0/np.sin(np.deg2rad(FA)), 0)
        R10 = np.broadcast_to(R10[..., None], S.shape)
    Sn = (1-Sn)/(1-cFA*Sn)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        R1 = np.where(Sn==0, 0, -np.log(Sn)/TR)  
    C = (R1 - R10)/r1
    return C.reshape(shape)


def _conc_spgr(S, T, TR, FA, TP, R10, r1=0.005, n0=1):
    """Concentration of a general spoiled gradient-echo sequence.

    Args:
        S (np.ndarray): Signal (arbitrary units).
        T (float): time since the first readout rf-pulse.
        TR (float): Repetition time, or time between successive selective 
          excitations, in sec. 
        FA (array-like): Flip angle in degrees.
        TP (float): Time (sec) between the preparation pre-pulse and 
          the first readout pulse. Defaults to 0.
        T10 (float): baseline T1 in sec.
        r1 (float, optional): Longitudinal relaxivity in Hz/M. Defaults to 0.005.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
    """
    # S = signal_spgr(S0, R1, T, TR, FA, TP) -> R1?
    # Compute
    # Sinf = signal_spgr(1, 1/T10, T, TR, FA, TP)
    # Sn = S / Sinf
    # New question:
    # Sn = signal_spgr(1, R1, T, TR, FA, TP) -> R1?
    # Lookup R1
    Sb = np.sum(S[:n0]) / n0
    Sinf = Sb / sig.signal('SR', R10, S0=1, TC=T, TR=TR, FA=FA, TP=TP)
    Sn = S / Sinf if Sinf > 0 else S * 0

    # Estimate maximum realistic R1 using maximum realistic relaxivity
    # and concentration
    r1_max = 10.0 # Hz/mM 
    c_max = 10 # mM
    R1_max = r1_max * c_max # 10 Hz/mM * 10 mM = 100 Hz corresponds to T1=10ms
    R1_step = 0.01 # 10000 steps

    # Create lookup table
    R1_lookup = np.arange(0, R1_max, R1_step)
    Sn_lookup = sig.signal('SR', R1_lookup, S0=1, TC=T, TR=TR, FA=FA, TP=TP)

    # Look up R1 values
    R1 = np.interp(Sn, Sn_lookup, R1_lookup)

    # Convert to concentrations
    return (R1 - R10)/r1


def _conc_src(S, TC: float, R10: float, r1=0.005, n0=1, S0=None) -> np.ndarray:
    """Concentration of a saturation-recovery sequence with a center-encoded readout.

    Args:
        S (array-like): Signal in arbitrary units.
        TC (float): Time (sec) between the saturation pulse and the acquisition of the k-space center.
        T10 (float): baseline T1 value in sec.
        r1 (float, optional): Longitudinal relaxivity in Hz/M. Defaults to 0.005.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration in M, same length as S.

    Example:

        We generate some signals from ground-truth concentrations, then reconstruct the concentrations and check against the ground truth:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        First define some constants:

        >>> T10 = 1         # sec
        >>> TC = 0.2        # sec
        >>> r1 = 0.005      # Hz/M
        >>> FA = 15         # deg

        Generate ground truth concentrations and signal data:

        >>> t = np.arange(0, 5*60, 0.1)     # sec
        >>> C = 0.003*(1-np.exp(-t/60))     # M
        >>> R1 = 1/T10 + r1*C               # Hz
        >>> S = dc.signal_free(100, R1, TC, FA)  # au

        Reconstruct the concentrations from the signal data:

        >>> Crec = dc._conc_src(S, TC, T10, r1)

        Check results by plotting ground truth against reconstruction:

        >>> plt.plot(t/60, 1000*C, 'ro', label='Ground truth')
        >>> plt.plot(t/60, 1000*Crec, 'b-', label='Reconstructed')
        >>> plt.title('SRC signal inverse')
        >>> plt.xlabel('Time (min)')
        >>> plt.ylabel('Concentration (mM)')
        >>> plt.legend()
        >>> plt.show()

    """
    # S = S0*(1-exp(-TC*R1))
    # S/Sb = (1-exp(-TC*R1))/(1-exp(-TC*R10))
    # (1-exp(-TC*R10))*S/Sb = 1-exp(-TC*R1)
    # 1-(1-exp(-TC*R10))*S/Sb = exp(-TC*R1)
    # ln(1-(1-exp(-TC*R10))*S/Sb) = -TC*R1
    # -ln(1-(1-exp(-TC*R10))*S/Sb)/TC = R1
    S = np.array(S)
    shape = S.shape
    S = S.reshape(-1, shape[-1])
    if np.isscalar(R10):
        R10 = np.full(S.shape[0], R10)
    else:
        R10 = R10.reshape(S.shape[0])

    R1 = np.zeros(S.shape)
    if S0 is None:
        Sb = np.mean(S[:, :n0], axis=-1)
        Sb = np.broadcast_to(Sb[..., None], S.shape)
        R10 = np.broadcast_to(R10[..., None], S.shape)
        E = np.exp(-TC*R10)
        nozero = Sb > 0
        R1[nozero] = -np.log(1-(1-E[nozero])*S[nozero]/Sb[nozero])/TC
    else:
        nozero = S0 > 0
        R1[nozero] = - np.log(1 - S[nozero] / S0[nozero]) / TC
        R10 = np.broadcast_to(R10[..., None], S.shape)
    C = (R1 - R10)/r1
    return C.reshape(shape)


def _conc_lin(S, R10, r1=0.005, n0=1, S0=None):
    """Concentration for any sequence operating in the linear regime.

    Args:
        S (array-like): Signal in arbitrary units.
        T10 (float): baseline T1 value in sec.
        r1 (float, optional): Longitudinal relaxivity in Hz/M. Defaults to 0.005.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration in M, same length as S.
    """
    # S = S0 * R10
    S = np.array(S)
    shape = S.shape
    S = S.reshape(-1, shape[-1])
    if np.isscalar(R10):
        R10 = np.full(S.shape[0], R10)
    else:
        R10 = R10.reshape(S.shape[0])
    R10 = np.broadcast_to(R10[..., None], S.shape)

    R1 = np.zeros(S.shape)
    if S0 is None:
        Sb = np.mean(S[:, :n0], axis=-1)
        Sb = np.broadcast_to(Sb[..., None], S.shape)
        nozero = Sb > 0
        R1[nozero] = R10[nozero]*S[nozero]/Sb[nozero]  # relaxation rate in 1/msec
    else:
        S0 = S0.reshape(S.shape[0])
        S0 = np.broadcast_to(S0[..., None], S.shape)
        nozero = S0 > 0
        R1[nozero] = S[nozero] / S0[nozero]
    C = (R1 - R10)/r1
    return C.reshape(shape)