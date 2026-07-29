import numpy as np
from scipy.interpolate import interp1d

from dcmri.bloch import functions_sequences



def _interpolate_2d(R, tR, TR, t0=0, kind='linear'):
    """
    Interpolate a 2D array R along its second dimension (axis=1) 
    """
    
    # Original time points along axis 1
    num_time_points = R.shape[1]
    if num_time_points == 1:
        return np.array(t0), R
    
    # Target time points
    tR = np.array(tR)
    t_new = np.arange(t0, np.max(tR), TR)
    
    # Create interpolation function operating along axis=1
    f = interp1d(tR, R, axis=1, kind=kind)
    
    return t_new, f(t_new)



def Mz_se(tR1, R1, v, Fw, j, me, TE, TR, FA, t0=0): 
    """
    Calculate steady-state longitudinal magnetization (Mz) for a single slice in a SE sequence.

    Constructs a two-pulse sequence model consisting of an excitation pulse 
    (flip angle `FA`) followed by a refocusing pulse (180°). Interpolates 
    time-varying relaxation rates (`R1`) and flux parameters (`j`) over a single TR 
    period, evaluating steady-state magnetization immediately prior to each RF pulse.

    Parameters
    ----------
    tR1 : array-like
        Time points of R1 [s].
    R1 : 2D array-like
        Longitudinal relaxation rate(s) over time with shape `(N_compartments, N_time)`.
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : 2D array-like or None
        Time-varying exchange flux or flow parameters matching `R1` shape. 
        If `None`, defaults to an array of zeros with shape matching `R1`.
    me : float or array-like
        Equilibrium magnetization value(s).
    TE : float
        Echo time [s].
    TR : float
        Repetition time [s].
    FA : float
        Excitation flip angle [degrees].
    t0 : float
        Time of the first pulse [s].  

    Returns
    -------
    t_Mz : ndarray
        1D array of time points corresponding to the interpolated grid over `TR`.
    Mz : ndarray
        Steady-state longitudinal magnetization evaluated before each pulse in the sequence, 
        with shape `(N_compartments, len(t_Mz), 2)`. Index 0 corresponds to magnetization 
        prior to the excitation pulse (`FA`), and index 1 corresponds to magnetization 
        prior to the 180° refocusing pulse.
    """
    if j is None:
        j = np.zeros_like(R1)

    pulse_sequence = [
        [FA, TE / 2], 
        [180, TR - TE/2]
    ]
    duration = TR

    t_Mz, R1_interp = _interpolate_2d(R1, tR1, duration, t0)
    _, j_interp = _interpolate_2d(j, tR1, duration, t0)

    Mz = np.zeros(R1_interp.shape + (len(pulse_sequence), )) # Mz before each pulse
    for k in range(Mz.shape[1]):
        R1k, jk = R1_interp[:, k], j_interp[:, k]
        Mz[:, k, 0] = functions_sequences.Mz_ss(R1k, v, Fw, jk, me, pulse_sequence)
        Mz[:, k, 1] = functions_sequences.Mz_prop(Mz[:, k, 0], R1k, v, Fw, jk, me, [[FA, TE / 2]])

    return t_Mz, Mz


def Mz_spgr_in_ss(tR1, R1, v, Fw, j, me, TR, FA, Nph, t0=0) -> tuple:
    """
    Calculate the steady-state longitudinal magnetization (Mz) for an SPGR sequence.

    Interpolates time-varying relaxation rates (`R1`) and flux parameters (`j`) over 
    a total acquisition duration determined by the number of phase encoding steps 
    (`Nph`). Computes the steady-state magnetization at the start of each frame and 
    records `Mz` prior to each RF pulse across all `Nph` steps.

    Parameters
    ----------
    tR1 : array-like
        Time points of R1 [s].
    R1 : 2D array-like
        Longitudinal relaxation rate(s) over time with shape `(N_compartments, N_time)`.
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : 2D array-like or None
        Time-varying exchange flux or flow parameters matching `R1` shape.
        If `None`, defaults to an array of zeros with shape matching `R1`.
    me : float or array-like
        Equilibrium magnetization value(s).
    TR : float
        Repetition time [s].
    FA : float
        Flip angle [degrees].
    Nph : int
        Number of phase encoding steps / readout pulses in the sequence frame.
    t0 : float
        Time of the first pulse [s].    

    Returns
    -------
    t_Mz : ndarray
        1D array of time points corresponding to the interpolated time vector.
    Mz : ndarray
        Calculated longitudinal magnetization before each readout pulse, with shape 
        `(N_compartments, len(t_Mz), Nph)`.

    Notes
    -----
    - Total duration evaluated per frame is `TR * Nph` starting from `t0 = 0`.
    - Solves initial steady-state magnetization using `functions_sequences.Mz_ss_spgr`.
    - Assumes $R_1$ remains constant during the fast readout train (`Nph` pulses).
    """
    if j is None:
        j = np.zeros_like(R1)
        
    duration = TR * Nph

    t_Mz, R1_interp = _interpolate_2d(R1, tR1, duration, t0)
    _, j_interp = _interpolate_2d(j, tR1, duration, t0)

    Mz = np.zeros(R1_interp.shape + (Nph, )) # Mz before each pulse
    for k in range(Mz.shape[1]):
        R1k, jk = R1_interp[:, k], j_interp[:, k]
        Mz[:, k, 0] = functions_sequences.Mz_ss_spgr(R1k, v, Fw, jk, me, TR, FA)

        # Assume R1 is const during the dynamic 
        # can be generalised easily at the cost of more interpolations
        for n in range(Nph-1):
            Mz[:, k, 1 + n] = Mz[:, k, 0]

    return t_Mz, Mz


def Mz_pr_spgr(tR1, R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA, t0=0): 
    """
    Model longitudinal magnetization for a prep-recovery SPGR sequence with linear k-space ordering.

    Simulates dynamic magnetization propagation over time across interpolated intervals. 
    At each temporal step $k$, a preparation pulse (`PA`) with delay (`TP`) is applied, 
    followed by an SPGR readout train of `Nph` pulses separated by `TR`, and a delay 
    `TD` before propagating state $M_0$ into step $k+1$.

    Parameters
    ----------
    tR1 : array-like
        Time points of R1 [s].
    R1 : 2D array-like
        Longitudinal relaxation rate(s) over time with shape `(N_compartments, N_time)`.
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : 2D array-like or None
        Time-varying exchange flux or flow parameters matching `R1` shape.
        If `None`, defaults to an array of zeros with shape matching `R1`.
    me : float or array-like
        Equilibrium magnetization value(s).
    TR : float
        Repetition time of individual readout pulses during SPGR acquisition [s].
    FA : float
        Flip angle of SPGR readout pulses [degrees].
    Nph : int
        Number of phase encoding steps / readout pulses per frame.
    TP : float
        Preparation delay / time from preparation pulse to readout train start [s].
    TD : float
        Delay time following the readout train before the next frame [s].
    PA : float
        Preparation pulse flip angle [degrees] (e.g., inversion or saturation pulse).
    t0 : float
        Time of the first pulse [s].    

    Returns
    -------
    t_Mz : ndarray
        1D array of interpolated time points across total duration `TP + Nph * TR + TD`.
    Mz : ndarray
        Longitudinal magnetization state recorded before each pulse, with shape 
        `(N_compartments, len(t_Mz), 1 + Nph)`. Index 0 represents magnetization prior 
        to the prep pulse (`PA`), while indices `1` to `Nph` represent magnetization 
        prior to each readout pulse (`FA`).

    Notes
    -----
    - Frame duration is calculated as `duration = TP + Nph * TR + TD`.
    - End-of-frame state after `TD` is iteratively propagated as $M_0$ for step $k+1$.
    """
    if j is None:
        j = np.zeros_like(R1)
        
    duration = TP + Nph * TR + TD
    M0 = v * me

    t_Mz, R1_interp = _interpolate_2d(R1, tR1, duration, t0)
    _, j_interp = _interpolate_2d(j, tR1, duration, t0)

    Mz = np.zeros(R1_interp.shape + (1 + Nph, )) # Mz before each pulse
    Mz[:, 0, 0] = M0
    for k in range(Mz.shape[1]):
        R1k, jk = R1_interp[:, k], j_interp[:, k]
        Mz[:, k, 1] = functions_sequences.Mz_prop(Mz[:, k, 0], R1k, v, Fw, jk, me, [[PA, TP]])
        for n in range(Nph - 1):
            Mz[:, k, 2 + n] = functions_sequences.Mz_prop(Mz[:, k, 1 + n], R1k, v, Fw, jk, me, [[FA, TR]])
        if k < Mz.shape[1] - 1:
            Mz[:, k + 1, 0] = functions_sequences.Mz_prop(Mz[:, k, Nph], R1k, v, Fw, jk, me, [[FA, TR + TD]])

    return t_Mz, Mz


def Mz_pr_spgr_in_ss(tR1, R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA, t0=0):
    """
    Model steady-state longitudinal magnetization for a prep-recovery SPGR sequence.

    Simulates a Preparation-Recovery Spoiled Gradient Recalled Echo (PR-SPGR) sequence 
    operating in steady state across interpolated time steps. At each step $k$, it solves 
    for the steady-state initial magnetization (`Mz[:, k, 0]`) prior to the preparation 
    pulse using `functions_sequences.Mz_ss_pr_spgr`, then propagates this state through 
    the prep delay (`TP`) and each of the `Nph` readout pulses (`TR`).

    Parameters
    ----------
    tR1 : array-like
        Time points of R1 [s].
    R1 : 2D array-like
        Longitudinal relaxation rate(s) over time with shape `(N_compartments, N_time)`.
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : 2D array-like or None
        Time-varying exchange flux or flow parameters matching `R1` shape.
        If `None`, defaults to an array of zeros with shape matching `R1`.
    me : float or array-like
        Equilibrium magnetization value(s).
    TR : float
        Repetition time of individual readout pulses during SPGR acquisition [s].
    FA : float
        Flip angle of readout pulses [degrees].
    Nph : int
        Number of phase encoding steps / readout pulses per frame.
    TP : float
        Preparation delay / time from preparation pulse to readout train start [s].
    TD : float
        Delay time following readout train before the next frame [s].
    PA : float
        Preparation pulse flip angle [degrees] (e.g., inversion or saturation pulse).
    t0 : float
        Time of the first pulse [s].

    Returns
    -------
    t_Mz : ndarray
        1D array of interpolated time points across total duration `TP + Nph * TR + TD`.
    Mz : ndarray
        Longitudinal magnetization recorded before each pulse, with shape 
        `(N_compartments, len(t_Mz), 1 + Nph)`. Index 0 represents magnetization 
        prior to the prep pulse (`PA`), while indices `1` to `Nph` represent 
        magnetization prior to each readout pulse (`FA`).

    Notes
    -----
    - Frame duration is calculated as `duration = TP + Nph * TR + TD`.
    - Solves steady-state initial conditions independently at each time step $k$.
    - Unlike `Mz_pr_spgr`, this function assumes inter-frame steady state rather than 
      dynamically carrying forward the end-of-frame residual magnetization.
    """
    if j is None:
        j = np.zeros_like(R1)
        
    duration = TP + Nph * TR + TD
    
    t_Mz, R1_interp = _interpolate_2d(R1, tR1, duration, t0)
    _, j_interp = _interpolate_2d(j, tR1, duration, t0)

    Mz = np.zeros(R1_interp.shape + (1 + Nph, )) # Mz before each pulse
    for k in range(Mz.shape[1]):
        R1k, jk = R1_interp[:, k], j_interp[:, k]
        Mz[:, k, 0] = functions_sequences.Mz_ss_pr_spgr(R1k, v, Fw, jk, me, TR, FA, Nph, TP, TD, PA)
        Mz[:, k, 1] = functions_sequences.Mz_prop(Mz[:, k, 0], R1k, v, Fw, jk, me, [[PA, TP]])
        for n in range(Nph - 1):
            Mz[:, k, 2 + n] = functions_sequences.Mz_prop(Mz[:, k, 1 + n], R1k, v, Fw, jk, me, [[FA, TR]])

    return t_Mz, Mz


def Mz_spgr_in_ssi(tR1, R1, v, Fw, j, me, TR, FA, Nph, TF, SA): 
    """
    Model steady-state imaging (SSI) longitudinal magnetization with inflow effects.

    Simulates a steady-state acquisition with steady-state inflow (SSI) over an 
    interpolated time window determined by the number of phase encodings (`Nph`). 
    At each time step $k$, computes the inflow-affected steady-state longitudinal 
    magnetization using `functions_sequences.Mz_prop_ssi` and replicates it across 
    all `Nph` readout steps in the frame.

    Parameters
    ----------
    tR1 : array-like
        Time points of R1 [s].
    R1 : 2D array-like
        Longitudinal relaxation rate(s) over time with shape `(N_compartments, N_time)`.
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : 2D array-like or None
        Time-varying exchange flux or flow parameters matching `R1` shape.
        If `None`, defaults to an array of zeros with shape matching `R1`.
    me : float or array-like
        Equilibrium magnetization value(s).
    TR : float
        Repetition time during SPGR readout [s].
    FA : float
        Readout flip angle [degrees].
    Nph : int
        Number of phase encoding steps / readout pulses per frame.
    TF : float
        Inflow time / duration of pre-readout inflow period [s].
    SA : float
        Saturation flip angle applied to inflowing spins outside the slab [degrees].

    Returns
    -------
    t_Mz : ndarray
        1D array of interpolated time points across the frame duration (`TR * Nph`).
    Mz : ndarray
        Calculated longitudinal magnetization recorded prior to each readout pulse, 
        with shape `(N_compartments, len(t_Mz), Nph)`.

    Notes
    -----
    - Total evaluation duration per dynamic step is `TR * Nph` starting from `t0 = 0`.
    - Assumes relaxation rate $R_1$ and flow $j$ remain constant during the fast 
      `Nph` readout train within each dynamic step $k$.
    """
    if j is None:
        j = np.zeros_like(R1)
        
    duration = TR * Nph
    t0 = 0

    t_Mz, R1_interp = _interpolate_2d(R1, tR1, duration, t0)
    _, j_interp = _interpolate_2d(j, tR1, duration, t0)

    Mz = np.zeros(R1_interp.shape + (Nph, )) # Mz before each pulse
    for k in range(Mz.shape[1]):
        R1k, jk = R1_interp[:, k], j_interp[:, k]
        Mz[:, k, 0] = functions_sequences.Mz_ssi(R1k, v, Fw, jk, me, TR, FA, TF, SA)

        # Assume R1 is const during the dynamic 
        # can be generalised easily at the cost of more interpolations
        for n in range(Nph-1):
            Mz[:, k, 1 + n] = Mz[:, k, 0]

    return t_Mz, Mz