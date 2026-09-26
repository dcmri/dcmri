import numpy as np
from scipy.interpolate import interp1d

from dcmri.bloch import functions_sequences


def Mz_wrapper_k0(sequence, mz_prep_sequence, tR1, R1, p, v=None, Kw=None, tj=None, j=None, tstart=0, t_end=None):

    # Catch the scalar case
    if R1.ndim == 1:
        R1 = R1.reshape(1, -1)
        v = np.full(1, v,)
        Kw = np.full((1, 1), Kw)
        if j is not None:
            j = j.reshape(1, -1)

    if mz_prep_sequence == 'Eq':
        Mz = np.full(R1.shape, v * p['me'])
        tMz = tR1.reshape(tR1.size)
        return tMz, Mz

    if mz_prep_sequence == 'IR-SS':
        TA = functions_sequences.repetition_time(sequence, p)
        return Mz_dyn_ss_k0_spgr(tR1, R1, v, Kw, j, p['me'], 180, TA, 1, 0, tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'SR-SS':
        TA = functions_sequences.repetition_time(sequence, p)
        return Mz_dyn_ss_k0_spgr(tR1, R1, v, Kw, j, p['me'], 90, TA, 1, 0, tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'PR-SS':
        TA = functions_sequences.repetition_time(sequence, p)
        return Mz_dyn_ss_k0_spgr(tR1, R1, v, Kw, j, p['me'], p['PA'], TA, 1, 0, tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'SPGR':
        return Mz_dyn_k0_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], tj=tj, tstart=tstart, t_end=t_end) 

    if mz_prep_sequence == 'SR-SPGR':
        _check_TP(p['TP'])
        t0 = p['iz'] * (p['TP'] + p['Nph'] * p['TR'] + p['TD']) if sequence == '2D-SR-SPGR' else 0
        tstart += t0
        return Mz_dyn_k0_pr_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], p['TP'], p['TD'], 90, tj=tj, tstart=tstart, t_end=t_end) 

    if mz_prep_sequence == 'IR-SPGR':
        _check_TP(p['TP'])
        return Mz_dyn_k0_pr_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], p['TP'], p['TD'], 180, tj=tj, tstart=tstart, t_end=t_end) 

    if mz_prep_sequence == 'PR-SPGR':
        _check_TP(p['TP'])
        return Mz_dyn_k0_pr_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], p['TP'], p['TD'], p['PA'], tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'SPGR-SS':
        return Mz_dyn_ss_k0_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'SR-SPGR-SS':
        _check_TP(p['TP'])
        return Mz_dyn_ss_k0_pr_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], p['TP'], p['TD'], 90, tj=tj, tstart=tstart, t_end=t_end) 

    if mz_prep_sequence == 'IR-SPGR-SS':
        _check_TP(p['TP'])
        return Mz_dyn_ss_k0_pr_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], p['TP'], p['TD'], 180, tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'PR-SPGR-SS':
        _check_TP(p['TP'])
        return Mz_dyn_ss_k0_pr_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['Nk0'], p['TP'], p['TD'], p['PA'], tj=tj, tstart=tstart, t_end=t_end) 

    if mz_prep_sequence == 'SSI':
        return Mz_dyn_ss_k0_spgri(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], p['Nph'], p['TF'], p['SA'], tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'GE-SS':
        t0 = p['iz'] * p['TR'] / p['Nz'] if sequence == '2D-GE-EPI' else 0
        tstart += t0
        return Mz_dyn_ss_k0_spgr(tR1, R1, v, Kw, j, p['me'], p['FA'] * p['B1corr'], p['TR'], 1, 0, tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'SE-SS':
        t0 = p['iz'] * p['TR'] / p['Nz'] if sequence == '2D-SE-EPI' else 0
        tstart += t0
        return Mz_dyn_k0_se(tR1, R1, v, Kw, j, p['me'], p['TE'], p['FA'] * p['B1corr'], p['TR'], tj=tj, tstart=tstart, t_end=t_end)

    if mz_prep_sequence == 'DE-SS':
        t0 = p['iz'] * p['TR'] / p['Nz'] if sequence == '2D-DE-EPI' else 0
        tstart += t0
        return Mz_dyn_k0_se(tR1, R1, v, Kw, j, p['me'], p['TE2'], p['FA'] * p['B1corr'], p['TR'], tj=tj, tstart=tstart, t_end=t_end)
    


def _check_TP(TP):
    if TP==0:
        raise ValueError("The delay time (TP) after a preparation pulse must be greater than 0.")


def _interpolate_2d_var(t_new, tR, R):
    """
    Interpolate a 2D array R along its second dimension (axis=1) 
    """
    # Original time points along axis 1
    if len(tR) == 1:
        return np.repeat(R, len(t_new), axis=1)
    
    # Create interpolation function operating along axis=1
    f = interp1d(tR, R, axis=1, kind='linear', bounds_error=False,
             fill_value=(R[:, 0], R[:, -1]))

    # Interpolate
    return f(t_new)


def _interpolate_inputs(t_pulses, tR1, R1, tj, j):
    n_comps = R1.shape[0]
    n_pulses = t_pulses.size

    # Interpolate properties at pulse times
    R1_pulses = _interpolate_2d_var(t_pulses, tR1, R1)
    if j is None:
        j_pulses = np.zeros((n_comps, n_pulses))
    elif tj is None:
        j_pulses = j.reshape((n_comps, n_pulses))
    else:
        tj = tj.reshape(tj.size)
        j = j.reshape((n_comps, tj.size))
        j_pulses = _interpolate_2d_var(t_pulses, tj, j)

    return R1_pulses, j_pulses



def Mz_dyn_k0_spgr(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, FA, TR, Nph, Nk0=0, tj:np.ndarray=None, tstart=0, t_end:float=None): 
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
    Kw : float or array-like
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
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times.
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

    Returns
    -------
    t_Mz : ndarray
        1D array of interpolated time points across total duration `TP + Nph * TR + TD`.
    Mz : ndarray
        Longitudinal magnetization state recorded before each pulse, with shape 
        `(N_compartments, len(t_Mz), 1 + Nph)`. Index 0 represents magnetization prior 
        to the prep pulse (`PA`), while indices `1` to `Nph` represent magnetization 
        prior to each readout pulse (`FA`).
    """
    if t_end is None:
        t_end = tR1.max()
    if t_end > tR1.max():
        raise ValueError("Mz end time must be less or equal to the maximum time of R1.")

    # Dimensions
    pulses_per_period = Nph * [[FA, TR]]
    period = np.sum([p[1] for p in pulses_per_period])
    n_periods = int((t_end - tstart) // period) if tR1.size > 1 else 1
    n_comps = R1.shape[0]

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse timings
    t0 = np.sum([p[1] for p in pulses_per_period[:Nk0]])
    t_periods = tstart + t0 + period * np.arange(n_periods)

    # Interpolate properties at pulse times
    R1_periods, j_periods = _interpolate_inputs(t_periods, tR1, R1, tj, j)

    # Compute
    Mz = np.zeros((n_comps, n_periods)) 
    Mz[:, 0] = v * me
    if Nk0 > 0:
        Mz[:, 0] = functions_sequences.Mz_prop_spgr_vectorized(Mz[:, 0], R1_periods[:, 0], v, Kw, j_periods[:, 0], me, FA, TR, Nk0)[:, 0]

    Mz[:, 1:] = functions_sequences.Mz_prop_spgr_vectorized(Mz[:, 0], R1_periods[:, 1:], v, Kw, j_periods[:, 1:], me, FA, TR, Nph)

    return t_periods, Mz



def Mz_dyn_k0_pr_spgr(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, FA, TR, Nph, Nk0, TP, TD, PA, tj:np.ndarray=None, tstart=0, t_end:float=None): 
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
    Kw : float or array-like
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
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times. 
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

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
    if t_end is None:
        t_end = tR1.max()
    if t_end > tR1.max():
        raise ValueError("Mz end time must be less or equal to the maximum time of R1.")

    # Dimensions
    pulses_per_period = [[PA, TP]] + (Nph - 1) * [[FA, TR]] + [[FA, TR + TD]]
    period = np.sum([p[1] for p in pulses_per_period])
    n_periods = int((t_end - tstart) // period) if tR1.size > 1 else 1
    n_comps = R1.shape[0]

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse timings
    t0 = np.sum([p[1] for p in pulses_per_period[:Nk0]])
    t_periods = tstart + t0 + period * np.arange(n_periods)

    # Interpolate properties at pulse times
    R1_periods, j_periods = _interpolate_inputs(t_periods, tR1, R1, tj, j)

    # Compute
    
    Mz = np.zeros((n_comps, n_periods)) 
    Mz0 = v * me
    Mz[:, 0] = functions_sequences.Mz_prop_vectorized(Mz0, R1_periods[:, 0], v, Kw, j_periods[:, 0], me, pulses_per_period[:Nk0 + 1])[:, 0]
    Mz[:, 1:] = functions_sequences.Mz_prop_pr_spgr_vectorized(Mz[:, 0], R1_periods[:, 1:], v, Kw, j_periods[:, 1:], me, FA, TR, Nph, TP, TD, PA, Nk0)

    return t_periods, Mz


def Mz_dyn_ss_k0_pr_spgr(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, FA, TR, Nph, Nk0, TP, TD, PA, tj:np.ndarray=None, tstart=0, t_end:float=None):
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
    Kw : float or array-like
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
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times. 
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

    Returns
    -------
    t_pulses : ndarray
        2D array of interpolated time points across total duration `TP + Nph * TR + TD`.
    Mz : ndarray
        Longitudinal magnetization recorded before each pulse, with shape 
        `(N_compartments, len(t_pulses), 1 + Nph)`. Index 0 represents magnetization 
        prior to the prep pulse (`PA`), while indices `1` to `Nph` represent 
        magnetization prior to each readout pulse (`FA`).

    Notes
    -----
    - Frame duration is calculated as `duration = TP + Nph * TR + TD`.
    - Solves steady-state initial conditions independently at each time step $k$.
    - Unlike `Mz_dyn_pr_spgr`, this function assumes inter-frame steady state rather than 
      dynamically carrying forward the end-of-frame residual magnetization.
    """
    if t_end is None:
        t_end = tR1.max()
    if t_end > tR1.max():
        raise ValueError("Mz end time must be less or equal to the maximum time of R1.")
    
    # Dimensions
    n_comps = R1.shape[0]
    period = TP + Nph * TR + TD
    n_periods = int((t_end - tstart) // period) if tR1.size > 1 else 1
   
    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulses
    pulses_per_period = [[PA, TP]] + (Nph - 1) * [[FA, TR]] + [[FA, TR + TD]]

    # Pulse timings
    tk0 = np.sum([p[1] for p in pulses_per_period[:1 + Nk0]])
    t_periods = tstart + tk0 + period * np.arange(n_periods)

    # Interpolate properties at pulse times
    R1_periods, j_periods = _interpolate_inputs(t_periods, tR1, R1, tj, j)

    R1_periods = R1_periods.reshape((n_comps, n_periods))
    j_periods = j_periods.reshape((n_comps, n_periods))

    # Compute Mz before each pulse
    Mz = functions_sequences.Mz_ss_k0_pr_spgr_vectorized(R1_periods, v, Kw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0)

    return t_periods, Mz



def Mz_dyn_ss_k0_spgr(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, FA, TR, Nph, Nk0=0, tj:np.ndarray=None, tstart=0, t_end:float=None) -> tuple:
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
    Kw : float or array-like
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
    tj : array-like, optional
        Time points for interpolation. If not provided, uses `tR1`.
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

    Returns
    -------
    t_Mz : ndarray
        1D array of time points corresponding to the interpolated time vector.
    Mz : ndarray
        Calculated longitudinal magnetization before each readout pulse, with shape 
        `(N_compartments, len(t_Mz), Nph)`.

    Notes
    -----
    - Total duration evaluated per frame is `TR * Nph` starting from `tstart = 0`.
    - Solves initial steady-state magnetization using `functions_sequences.Mz_ss_spgr`.
    - Assumes $R_1$ remains constant during the fast readout train (`Nph` pulses).
    """
    if t_end is None:
        t_end = tR1.max()
    if t_end > tR1.max():
        raise ValueError("Mz end time must be less or equal to the maximum time of R1.")
    
    # Dimensions
    n_comps = R1.shape[0]
    n_pulses_per_period = Nph
    period = n_pulses_per_period * TR
    n_periods = int((t_end - tstart) // period) if tR1.size > 1 else 1

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 ({tR1.max()}) is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Interpolate properties at pulse times
    t_periods = tstart + TR * Nk0 + period * np.arange(n_periods)
    R1_periods, j_periods = _interpolate_inputs(t_periods, tR1, R1, tj, j)

    # Compute
    Mz = functions_sequences.Mz_ss_spgr_vectorized(R1_periods, v, Kw, j_periods, me, FA, TR)

    # Reshape (probably not necessary)
    Mz = Mz.reshape((n_comps, n_periods))

    return t_periods, Mz



def Mz_dyn_k0_se(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, TE, FA, TR, tj:np.ndarray=None, tstart=0, t_end:float=None) -> tuple: 
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
    Kw : float or array-like
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
    tj : array-like, optional
        Time points for interpolation. If not provided, uses `tR1`. 
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

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
    # Pulses
    pulses_per_period = [[FA, TE / 2], [180, TR - TE/2]]
    Nk0 = 1
    return Mz_dyn_ss_k0(tR1, R1, v, Kw, j, me, pulses_per_period, Nk0, tj, tstart, t_end)


def Mz_dyn_ss_k0_spgri(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, FA, TR, Nph, TF, SA, Nk0=0, tj:np.ndarray=None, tstart=0, t_end:float=None): 
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
    Kw : float or array-like
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
    tj : array-like, optional
        Time points for interpolation. If not provided, uses `tR1`. 
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

    Returns
    -------
    t_Mz : ndarray
        1D array of interpolated time points across the frame duration (`TR * Nph`).
    Mz : ndarray
        Calculated longitudinal magnetization recorded prior to each readout pulse, 
        with shape `(N_compartments, len(t_Mz), Nph)`.

    Notes
    -----
    - Total evaluation duration per dynamic step is `TR * Nph` starting from `tstart = 0`.
    - Assumes relaxation rate $R_1$ and flow $j$ remain constant during the fast 
      `Nph` readout train within each dynamic step $k$.
    """
    if t_end is None:
        t_end = tR1.max()
    if t_end > tR1.max():
        raise ValueError("Mz end time must be less or equal to the maximum time of R1.")
    
    # Dimensions
    n_pulses_per_period = Nph
    period = n_pulses_per_period * TR
    n_periods = int((t_end - tstart) // period) if tR1.size > 1 else 1

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse locations
    t_periods = tstart + TR * Nk0 + period * np.arange(n_periods)

    # Interpolate properties at pulse times
    R1_periods, j_periods = _interpolate_inputs(t_periods, tR1, R1, tj, j)

    # Compute
    Mz = functions_sequences.Mz_ss_spgri_vectorized(R1_periods, v, Kw, j_periods, me, FA, TR, TF, SA)

    return t_periods, Mz


def Mz_dyn_k0(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, pulses_per_period, Nk0=0, tj:np.ndarray=None, tstart=0, t_end:float=None): 
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
    Kw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : 2D array-like or None
        Time-varying exchange flux or flow parameters matching `R1` shape.
        If `None`, defaults to an array of zeros with shape matching `R1`.
    me : float or array-like
        Equilibrium magnetization value(s).
    pulses_per_period : list of lists
        List of pulses in the sequence, where each pulse is defined as `[flip_angle, delay]`.
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times.
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

    Returns
    -------
    t_Mz : ndarray
        1D array of interpolated time points across total duration `TP + Nph * TR + TD`.
    Mz : ndarray
        Longitudinal magnetization state recorded before each pulse, with shape 
        `(N_compartments, len(t_Mz), 1 + Nph)`. Index 0 represents magnetization prior 
        to the prep pulse (`PA`), while indices `1` to `Nph` represent magnetization 
        prior to each readout pulse (`FA`).
    """
    if t_end is None:
        t_end = tR1.max()
    if t_end > tR1.max():
        raise ValueError("Mz end time must be less or equal to the maximum time of R1.")

    # Dimensions
    period = np.sum([p[1] for p in pulses_per_period])
    n_periods = int((t_end - tstart) // period) if tR1.size > 1 else 1
    n_comps = R1.shape[0]

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse timings
    t0 = np.sum([p[1] for p in pulses_per_period[:Nk0]])
    t_periods = tstart + t0 + period * np.arange(n_periods)

    # Interpolate properties at pulse times
    R1_periods, j_periods = _interpolate_inputs(t_periods, tR1, R1, tj, j)

    # Compute
    Mz = np.zeros((n_comps, n_periods)) 
    Mz[:, 0] = v * me
    if Nk0 > 0:
        Mz[:, 0] = functions_sequences.Mz_prop_vectorized(Mz[:, 0], R1_periods[:, 0], v, Kw, j_periods[:, 0], me, pulses_per_period[:Nk0])[:, 0]

    pulses_per_period_from_k0 = pulses_per_period[Nk0:] + pulses_per_period[:Nk0]
    Mz[:, 1:] = functions_sequences.Mz_prop_vectorized(Mz[:, 0], R1_periods[:, 1:], v, Kw, j_periods[:, 1:], me, pulses_per_period_from_k0)

    return t_periods, Mz



def Mz_dyn_ss_k0(tR1:np.ndarray, R1:np.ndarray, v, Kw, j:np.ndarray, me, pulses_per_period, Nk0=0, tj:np.ndarray=None, tstart=0, t_end:float=None) -> tuple: 
    """
    Calculate steady-state longitudinal magnetization (Mz) for an arbitrary pulse sequence applied dynamically.

    Parameters
    ----------
    tR1 : array-like
        Time points of R1 [s].
    R1 : 2D array-like
        Longitudinal relaxation rate(s) over time with shape `(N_compartments, N_time)`.
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Kw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : 2D array-like or None
        Time-varying exchange flux or flow parameters matching `R1` shape. 
        If `None`, defaults to an array of zeros with shape matching `R1`.
    me : float or array-like
        Equilibrium magnetization value(s).
    pulses_per_period : list of lists
        List of pulses in the sequence, where each pulse is defined as `[flip_angle, delay]`.
    tj : array-like, optional
        Time points for interpolation. If not provided, uses `tR1`. 
    tstart : float
        Time [s] of the first pulse in the sequence. Default is 0.
    t_end : float
        Time [s] of the last acquisition in the sequence. Defaults to max(tR1).

    Returns
    -------
    t_Mz : ndarray
        2D array of time points corresponding to the interpolated grid over `TR`.
    Mz : ndarray
        Steady-state longitudinal magnetization evaluated before each pulse in the sequence, 
        with shape `(N_compartments, N_periods, N_pulses_per_period)`. 
    """
    if t_end is None:
        t_end = tR1.max()
    if t_end > tR1.max():
        raise ValueError("Mz end time must be less or equal to the maximum time of R1.")

    # Dimensions
    period = np.sum([p[1] for p in pulses_per_period])
    n_periods = int((t_end - tstart) // period) if tR1.size > 1 else 1
    n_comps = R1.shape[0]

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse timings
    tk0 = np.sum([p[1] for p in pulses_per_period[:Nk0]])
    t_periods = tstart + tk0 + period * np.arange(n_periods)

    # Interpolate properties at pulse times
    R1_periods, j_periods = _interpolate_inputs(t_periods, tR1, R1, tj, j)

    # Compute Mz before each pulse
    R1_periods = R1_periods.reshape((n_comps, n_periods))
    j_periods = j_periods.reshape((n_comps, n_periods))

    pulses_per_period_from_k0 = pulses_per_period[Nk0:] + pulses_per_period[:Nk0]
    Mz = functions_sequences.Mz_ss_vectorized(R1_periods, v, Kw, j_periods, me, pulses_per_period_from_k0)

    return t_periods, Mz