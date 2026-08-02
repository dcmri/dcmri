import numpy as np
from scipy.interpolate import interp1d

from dcmri.bloch import functions_sequences


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


def Mz_spgr(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, TR, FA, Nph, tj:np.ndarray=None, t0=0): 
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
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times.
    t0 : float
        Time [s] of the first pulse in the sequence. Default is 0.

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
    pulses_per_period = Nph * [[FA, TR]]
    return Mz_dyn(tR1, R1, v, Fw, j, me, pulses_per_period, tj, t0)



def Mz_pr_spgr(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, TR, FA, Nph, TP, TD, PA, tj:np.ndarray=None, t0=0): 
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
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times. 
    t0 : float
        Time [s] of the first pulse in the sequence. Default is 0.

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
    pulses_per_period = [[PA, TP]] + (Nph - 1) * [[FA, TR]] + [[FA, TR + TD]]
    return Mz_dyn(tR1, R1, v, Fw, j, me, pulses_per_period, tj, t0)


def Mz_pr_spgr_in_ss(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, TR, FA, Nph, TP, TD, PA, tj:np.ndarray=None, t0=0):
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
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times. 
    t0 : float
        Time [s] of the first pulse in the sequence. Default is 0.

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
    - Unlike `Mz_pr_spgr`, this function assumes inter-frame steady state rather than 
      dynamically carrying forward the end-of-frame residual magnetization.
    """
    # Dimensions
    n_comps = R1.shape[0]
    n_pulses_per_period = int(Nph + 1)
    period = TP + Nph * TR + TD
    n_periods = int((tR1.max() - t0) // period) if tR1.size > 1 else 1
    n_pulses = n_pulses_per_period * n_periods

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulses
    pulses_per_period = [[PA, TP]] + (Nph - 1) * [[FA, TR]] + [[FA, TR + TD]]

    # Pulse timings
    t_pulses_per_period = np.zeros(n_pulses_per_period)
    t_pulses_per_period[1:] = TP + TR * np.arange(Nph)
    t_pulses = np.zeros(n_pulses)
    for i in range(n_periods):
        t_pulses[i * n_pulses_per_period: (i + 1) * n_pulses_per_period] = t0 + i * period + t_pulses_per_period

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
        
    # Compute Mz before each pulse
    Mz = np.zeros((n_comps, n_periods, n_pulses_per_period))
    R1_pulses = R1_pulses.reshape((n_comps, n_periods, n_pulses_per_period))
    j_pulses = j_pulses.reshape((n_comps, n_periods, n_pulses_per_period))
    for i in range(n_periods):
        Mz[:, i, 0] = functions_sequences.Mz_ss_pr_spgr(R1_pulses[:, i, 0], v, Fw, j_pulses[:, i, 0], me, TR, FA, Nph, TP, TD, PA)
        Mz[:, i, 1:] = functions_sequences.Mz_prop(Mz[:, i, 0], R1_pulses[:, i, :-1], v, Fw, j_pulses[:, i, :-1], me, pulses_per_period[:-1])

    # Return
    t_pulses = t_pulses.reshape((n_periods, n_pulses_per_period))
    return t_pulses, Mz


def Mz_spgr_in_ss(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, TR, FA, Nph, tj:np.ndarray=None, t0=0) -> tuple:
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
    tj : array-like, optional
        Time points for interpolation. If not provided, uses `tR1`.
    to : float
        Time [s] of the first pulse in the sequence. Default is 0.

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
    # Dimensions
    n_comps = R1.shape[0]
    n_pulses_per_period = Nph
    period = n_pulses_per_period * TR
    n_periods = int((tR1.max() - t0) // period) if tR1.size > 1 else 1
    n_pulses = int(n_pulses_per_period * n_periods)

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse locations
    t_pulses = t0 + TR * np.arange(n_pulses)

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

    # Compute
    Mz = np.zeros((n_comps, n_pulses)) 
    for i in range(n_pulses): # TODO vectorize Mz_ssi and reduce to a single call
        Mz[:, i] = functions_sequences.Mz_ss_spgr(R1_pulses[:, i], v, Fw, j_pulses[:, i], me, TR, FA)

    # Reshape
    t_pulses = t_pulses.reshape((n_periods, n_pulses_per_period))
    Mz = Mz.reshape((n_comps, n_periods, n_pulses_per_period))

    return t_pulses, Mz


def Mz_se(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, TE, TR, FA, tj:np.ndarray=None, t0=0) -> tuple: 
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
    tj : array-like, optional
        Time points for interpolation. If not provided, uses `tR1`. 
    t0 : float
        Time [s] of the first pulse in the sequence. Default is 0.

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
    return Mz_dyn_ss(tR1, R1, v, Fw, j, me, pulses_per_period, tj, t0)


def Mz_spgr_in_ssi(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, TR, FA, Nph, TF, SA, tj:np.ndarray=None): 
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
    tj : array-like, optional
        Time points for interpolation. If not provided, uses `tR1`. 

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

    # Dimensions
    n_comps = R1.shape[0]
    n_pulses_per_period = Nph
    period = n_pulses_per_period * TR
    n_periods = int(tR1.max() // period) if tR1.size > 1 else 1
    n_pulses = int(n_pulses_per_period * n_periods)

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse locations
    t_pulses = TR * np.arange(n_pulses)

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

    # Compute
    Mz = np.zeros((n_comps, n_pulses)) 
    for i in range(n_pulses): # TODO vectorize Mz_ssi and reduce to a single call
        Mz[:, i] = functions_sequences.Mz_ssi(R1_pulses[:, i], v, Fw, j_pulses[:, i], me, TR, FA, TF, SA)

    # Reshape
    t_pulses = t_pulses.reshape((n_periods, n_pulses_per_period))
    Mz = Mz.reshape((n_comps, n_periods, n_pulses_per_period))

    return t_pulses, Mz


def Mz_dyn(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, pulses_per_period, tj:np.ndarray=None, t0=0): 
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
    pulses_per_period : list of lists
        List of pulses in the sequence, where each pulse is defined as `[flip_angle, delay]`.
    tj : array-like or None
        Time points of j [s]. If None, it is assumed j is already defined at the correct times.
    t0 : float
        Time [s] of the first pulse in the sequence. Default is 0.

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

    # Dimensions
    n_pulses_per_period = len(pulses_per_period)
    period = np.sum([p[1] for p in pulses_per_period])
    n_periods = int((tR1.max() - t0) // period) if tR1.size > 1 else 1
    n_pulses = int(n_pulses_per_period * n_periods)
    n_comps = R1.shape[0]

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse timings
    t_pulses_per_period = np.concatenate(([0], np.cumsum([p[1] for p in pulses_per_period[:-1]])))
    t_pulses = np.zeros(n_pulses)
    for i in range(n_periods):
        t_pulses[i * n_pulses_per_period: (i + 1) * n_pulses_per_period] = t0 + i * period + t_pulses_per_period

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

    # Compute
    pulses = n_periods * pulses_per_period
    Mz = np.zeros((n_comps, n_pulses)) # Mz before each pulse
    Mz[:, 0] = v * me
    Mz[:, 1:] = functions_sequences.Mz_prop(Mz[:, 0], R1_pulses, v, Fw, j_pulses, me, pulses[:-1])

    # Reshape
    t_pulses = t_pulses.reshape((n_periods, n_pulses_per_period))
    Mz = Mz.reshape((n_comps, n_periods, n_pulses_per_period))

    return t_pulses, Mz

def Mz_dyn_ss(tR1:np.ndarray, R1:np.ndarray, v, Fw, j:np.ndarray, me, pulses_per_period, tj:np.ndarray=None, t0=0) -> tuple: 
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
    Fw : float or array-like
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
    t0 : float
        Time [s] of the first pulse in the sequence. Default is 0.

    Returns
    -------
    t_Mz : ndarray
        2D array of time points corresponding to the interpolated grid over `TR`.
    Mz : ndarray
        Steady-state longitudinal magnetization evaluated before each pulse in the sequence, 
        with shape `(N_compartments, N_periods, N_pulses_per_period)`. 
    """

    # Dimensions
    n_pulses_per_period = len(pulses_per_period)
    period = np.sum([p[1] for p in pulses_per_period])
    n_periods = int((tR1.max() - t0) // period) if tR1.size > 1 else 1
    n_pulses = n_pulses_per_period * n_periods
    n_comps = R1.shape[0]

    if n_periods==0:
        raise ValueError(f"Maximum time for R1 {tR1.max()} is less than the duration {period} of a single pulse cycle. Extend R1-range and try again.")

    # Pulse timings
    t_pulses_per_period = np.concatenate(([0], np.cumsum([p[1] for p in pulses_per_period[:-1]])))
    t_pulses = np.zeros(n_pulses)
    for i in range(n_periods):
        t_pulses[i * n_pulses_per_period: (i + 1) * n_pulses_per_period] = t0 + i * period + t_pulses_per_period

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
        
    # Compute Mz before each pulse
    Mz = np.zeros((n_comps, n_periods, n_pulses_per_period))
    R1_pulses = R1_pulses.reshape((n_comps, n_periods, n_pulses_per_period))
    j_pulses = j_pulses.reshape((n_comps, n_periods, n_pulses_per_period))
    for i in range(n_periods):
        Mz[:, i, 0] = functions_sequences.Mz_ss(R1_pulses[:, i, 0], v, Fw, j_pulses[:, i, 0], me, pulses_per_period)
        Mz[:, i, 1:] = functions_sequences.Mz_prop(Mz[:, i, 0], R1_pulses[:, i, :-1], v, Fw, j_pulses[:, i, :-1], me, pulses_per_period[:-1])

    # Return
    t_pulses = t_pulses.reshape((n_periods, n_pulses_per_period))
    return t_pulses, Mz
