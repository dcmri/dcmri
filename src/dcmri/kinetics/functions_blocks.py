import math
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import lambertw

from dcmri.utils import convolution, misc
from dcmri.kinetics import functions_utils


CONC_PARAMETERS = {
    'trap': ['J', 'dt'],
    'pass': ['J', 'T'],
    'comp': ['J', 'dt', 'T'],
    'plug': ['J', 'dt', 'T'],
    'bicomp': ['J', 'dt', 'T'],
    'chain': ['J', 'dt', 'T', 'D'],
    'step': ['J', 'dt', 'T', 'D'],
    'free': ['J', 'dt', 'h', 'TT'],
    'ncomp': ['J', 'dt', 'T', 'E'],
    'nscomp': ['J', 'dt', 'T'],
    'mmcomp': ['J', 'dt', 'Vmax', 'Km'],
    '2cxm': ['J', 'dt', 'T', 'E'],
}

FLUX_PARAMETERS = {
    'trap': ['J', ],
    'pass': ['J', ],
    'comp': ['J', 'dt', 'T'],
    'plug': ['J', 'dt', 'T'],
    'bicomp': ['J', 'dt', 'T'],
    'plucom': ['J', 'dt', 'T', 'fp'],
    'chain': ['J', 'dt', 'T', 'D'],
    'step': ['J', 'dt', 'T', 'D'],
    'pfcomp': ['J', 'dt', 'T', 'D'],
    'free': ['J', 'dt', 'h', 'TT'],
    'ncomp': ['J', 'dt', 'T', 'E'],
    'nscomp': ['J', 'dt', 'T'],
    'mmcomp': ['J', 'dt', 'Vmax', 'Km'],
    '2cxm': ['J', 'dt', 'T', 'E'],
}
        

# Functions


def flux(model, *args, **kwargs) -> np.ndarray:
    """
    Wrapper function to compute the flux for a specified model.

    This function dynamically dispatches the flux calculation to a model-specific
    function named `flux_<model>`. 

    Parameters
    ----------
    model : str
        The name of the flux model to evaluate (e.g., 'pass', 'trap'). This 
        determines which underlying function (`flux_<model>`) is called.
    *args : tuple
        Positional arguments passed directly to the underlying model function.
    **kwargs : dict, optional
        Keyword arguments passed to the underlying model function. If `model` 
        is 'pass' or 'trap', the keys 't' and 'dt' are stripped out.

    Returns
    -------
    numpy.ndarray
        The calculated flux values from the designated model function.

    Raises
    ------
    KeyError
        If the corresponding function `flux_<model>` does not exist in the 
        global namespace.

    See Also
    --------
    conc : Corresponding wrapper function for concentration models.

    Examples
    --------
    >>> import dcmri as dc
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux('chain', J, dt=2.0, T=5, D=0.5)  
    array([0.        , 0.14378527, 0.70435485, 1.46665593, 2.0385268 ])
    """
    if model in ['pass','trap']:
        kwargs = {k: v for k, v in kwargs.items() if k not in ['t', 'dt']}
    return globals()[f"flux_{model}"](*args, **kwargs)


def conc(model, *args, **kwargs) -> np.ndarray:
    """
    Wrapper function to compute the concentration for a specified model.

    This function dynamically dispatches the concentration calculation to a 
    model-specific function named `conc_<model>`. 

    Parameters
    ----------
    model : str
        The name of the concentration model to evaluate (e.g., 'pass'). This 
        determines which underlying function (`conc_<model>`) is called.
    *args : tuple
        Positional arguments passed directly to the underlying model function.
    **kwargs : dict, optional
        Keyword arguments passed to the underlying model function. If `model` 
        is 'pass', the keys 't' and 'dt' are stripped out.

    Returns
    -------
    numpy.ndarray
        The calculated concentration values from the designated model function.

    Raises
    ------
    KeyError
        If the corresponding function `conc_<model>` does not exist in the 
        global namespace.

    See Also
    --------
    flux : Corresponding wrapper function for flux models.

    Examples
    --------
    >>> import dcmri as dc
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc('chain', J, dt=2.0, T=5, D=0.5)  
    array([ 0.        ,  2.85621473,  7.00807462, 10.83706384, 12.33188111])
    """
    if model in ['pass']:
        kwargs = {k: v for k, v in kwargs.items() if k not in ['t', 'dt']}
    return globals()[f"conc_{model}"](*args, **kwargs)


# 0 Parameters

# Trap

def res_trap(t=None):
    """
    Residue function of a trap.

    See section :ref:`define-trap` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the residue function is calculated.

    Returns
    -------
    np.ndarray
        Residue function as a 1D array.

    See Also
    --------
    prop_trap : Propagator function of a trap.
    conc_trap : Concentration in a trap.
    flux_trap : Outflux from a trap.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 1, 2, 3, 4]
    >>> dc.res_trap(t)
    array([1., 1., 1., 1., 1.])
    """
    return np.ones(len(t))


def prop_trap(t=None):
    """
    Propagator of a trap.

    See section :ref:`define-trap` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the propagator is calculated.

    Returns
    -------
    np.ndarray
        Propagator as a 1D array.

    See Also
    --------
    res_trap : Residue function of a trap.
    conc_trap : Concentration in a trap.
    flux_trap : Outflux from a trap.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 1, 2, 3, 4]
    >>> dc.prop_trap(t)
    array([0., 0., 0., 0., 0.])
    """
    return np.zeros(len(t))


def conc_trap(J=None, t=None, dt=1.0):
    """
    Tissue concentration in a trap.

    See section :ref:`define-trap` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the trap.
    t : array_like, optional
        The time points of the indicator flux `J`. If None, the time points 
        are assumed to be uniformly spaced with spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data. This parameter 
        is ignored if `t` is explicitly provided. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    res_trap : Residue function of a trap.
    prop_trap : Propagator function of a trap.
    flux_trap : Outflux from a trap.

    Examples
    --------
    >>> import dcmri as dc
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc_trap(J, dt=2.0)
    array([ 0.,  3.,  8., 14., 19.])
    """
    return misc.trapz(J, t=t, dt=dt)


def flux_trap(J=None):
    """
    Flux out of a trap.

    See section :ref:`define-trap` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the trap.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    res_trap : Residue function of a trap.
    prop_trap : Propagator function of a trap.
    conc_trap : Concentration in a trap.

    Examples
    --------
    >>> import dcmri as dc
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_trap(J)
    array([0., 0., 0., 0., 0.])
    """
    return np.zeros(len(J))


# 1 Parameter

# Pass (no dispersion)

def res_pass(t, T=None):
    """
    Residue function of a pass.

    See section :ref:`define-pass` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the residue function is calculated.
    T : float
        Transit time of the pass.

    Returns
    -------
    np.ndarray
        Residue function of the pass as a 1D array.

    See Also
    --------
    prop_pass : Propagator function of a pass.
    conc_pass : Concentration in a pass.
    flux_pass : Outflux from a pass.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.res_pass(t, 5)
    array([3.33333333, 0.        , 0.        , 0.        ])
    """
    return T * functions_utils.ddelta(0, t)


def prop_pass(t):
    """
    Propagator of a pass.

    See section :ref:`define-pass` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the propagator is calculated.

    Returns
    -------
    np.ndarray
        Propagator as a 1D array.

    See Also
    --------
    res_pass : Residue function of a pass.
    conc_pass : Concentration in a pass.
    flux_pass : Outflux from a pass.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.prop_pass(t)
    array([0.66666667, 0.        , 0.        , 0.        ])
    """
    return functions_utils.ddelta(0, t)


def conc_pass(J=None, T=None):
    """
    Tissue concentration in a pass.

    See section :ref:`define-pass` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the pass.
    T : float
        Transit time of the pass.

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    res_pass : Residue function of a pass.
    prop_pass : Propagator function of a pass.
    flux_pass : Outflux from a pass.

    Examples
    --------
    >>> import dcmri as dc
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc_pass(J, 5)
    array([ 5, 10, 15, 15, 10])
    """
    return T * np.array(J)


def flux_pass(J=None):
    """
    Flux out of a pass.

    See section :ref:`define-pass` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the pass.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    res_pass : Residue function of a pass.
    prop_pass : Propagator function of a pass.
    conc_pass : Concentration in a pass.

    Examples
    --------
    >>> import dcmri as dc
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_pass(J)
    array([1, 2, 3, 3, 2])
    """
    return np.array(J)


# Compartment

def res_comp(t, T=None):
    """
    Residue function of a compartment.

    See section :ref:`define-compartment` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the residue function is calculated, in the same 
        units as `T`.
    T : float
        Mean transit time of the compartment. Any non-negative value is 
        allowed, including T = 0 and T = inf (in which case the compartment 
        acts as a trap).

    Returns
    -------
    np.ndarray
        Residue function of the compartment as a 1D array.

    See Also
    --------
    prop_comp : Propagator function of a compartment.
    conc_comp : Concentration in a compartment.
    flux_comp : Outflux from a compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.res_comp(t, 5)
    array([1.        , 0.54881164, 0.44932896, 0.30119421])
    """
    if T == np.inf:
        return res_trap(t)
    if T == 0:
        r = np.zeros(len(t))
        r[0] = 1
        return r
    return np.exp(-np.array(t)/T)


def prop_comp(t, T=None):
    """
    Propagator of a compartment.

    See section :ref:`define-compartment` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the propagator is calculated, in the same units 
        as `T`.
    T : float
        Mean transit time of the compartment. Any non-negative value is 
        allowed, including T = 0 and T = inf (in which case the compartment 
        acts as a trap).

    Returns
    -------
    np.ndarray
        Propagator as a 1D array.

    See Also
    --------
    res_comp : Residue function of a compartment.
    conc_comp : Concentration in a compartment.
    flux_comp : Outflux from a compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.prop_comp(t, 5)
    array([0.2       , 0.10976233, 0.08986579, 0.06023884])
    """
    if T == np.inf:
        return prop_trap(t)
    if T == 0:
        return functions_utils.ddelta(T, t)
    return np.exp(-np.array(t)/T)/T


def conc_comp(J=None, t=None, dt=1.0, T=None):
    """
    Tissue concentration in a compartment.

    See section :ref:`define-compartment` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the compartment. Any non-negative value is 
        allowed, including T = 0 and T = inf (in which case the compartment 
        acts as a trap).

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    res_comp : Residue function of a compartment.
    prop_comp : Propagator function of a compartment.
    flux_comp : Outflux from a compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc_comp(J, t, T=5)
    array([ 0.        ,  5.        , 12.16166179, 14.85868746, 10.83091743])
    """
    if T == np.inf:
        return conc_trap(J, t=t, dt=dt)
    convexp = convolution.expconv(J, T, t=t, dt=dt, tol=1e-6)
    convexp[convexp < 0] = 0 # may have small negative values
    return T * convexp


def flux_comp(J=None, t=None, dt=1.0, T=None):
    """
    Flux out of a compartment.

    See section :ref:`define-compartment` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the compartment. Any non-negative value is 
        allowed, including T = 0 and T = inf (in which case the compartment 
        acts as a trap).

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    res_comp : Residue function of a compartment.
    prop_comp : Propagator function of a compartment.
    conc_comp : Concentration in a compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_comp(J, t, T=5)
    array([0.        , 1.        , 2.43233236, 2.97173749, 2.16618349])
    """
    if T == np.inf:
        return flux_trap(J)
    return convolution.expconv(J, T, t=t, dt=dt)


# Bicomp

def conc_bicomp(J=None, t=None, dt=1.0, T=None):
    """
    Tissue concentration in a chain of 2 compartments.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the first compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : list of float
        Mean transit times of the two compartments. Any non-negative value is 
        allowed, including T = 0 and T = inf (in which case the compartment 
        acts as a trap).


    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    conc_comp : Tissue concentration in a single compartment.
    flux_bicomp : Flux out of a two-compartment system.
    conc_chain : Tissue concentration in an arbitrary N-compartment chain.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc_bicomp(J, t, T=[5, 10])
    array([ 0.        ,  7.13061319, 24.53593245, 39.11621776, 34.77241541])
    """
    Tc = T[0]
    if np.isscalar(Tc):
        C0 = conc_comp(J, t=t, dt=dt, T=Tc)
        J = flux_comp(J, t=t, dt=dt, T=Tc) # unnecessary conv here
    else:
        C0 = conc_nscomp(J, t=t, dt=dt, T=Tc)
        J = flux_nscomp(J, t=t, dt=dt, T=Tc) # unnecessary conv here
        
    Tc = T[1]
    if np.isscalar(Tc):
        C1 = conc_comp(J, t=t, dt=dt, T=Tc)
    else:
        C1 = conc_nscomp(J, t=t, dt=dt, T=Tc)
        
    return C0 + C1

def flux_bicomp(J=None, t=None, dt=1.0, T=None):
    """
    Flux out of a chain of 2 compartments.

    See section :ref:`define-compartment` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the first compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : list of float
        Mean transit times of the two compartments. Any non-negative value is 
        allowed, including T = 0 and T = inf (in which case the compartment 
        acts as a trap).


    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    flux_comp : Flux out of a single compartment.
    conc_bicomp : Tissue concentration in a two-compartment system.
    flux_chain : Flux out of an arbitrary N-compartment chain.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_bicomp(J, t, T=[5, 10])
    array([0.        , 0.21306132, 1.23742707, 2.42575303, 2.3941498 ])
    """
    for Tc in T:
        if np.isscalar(Tc):
            J = flux_comp(J, t=t, dt=dt, T=Tc)
        else:
            J = flux_nscomp(J, t=t, dt=dt, T=Tc)
    return J


def flux_plucom(J=None, t=None, dt=1.0, T=None, fp=None):
    """
    Flux out of a parallel arrangement of a plug-flow system and a compartment. 

    See section :ref:`define-compartment` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the first compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : list of float
        Mean transit times of two systems, in the following order: 
        [compartment, plug-flow]. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the compartment acts as a trap).
    fp : float
        Plug-flow fraction, or the fraction of the flux that is going through 
        the plug-flow route. This is a number in the range [0, 1].

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    flux_comp : Flux out of a single compartment.
    conc_bicomp : Tissue concentration in a two-compartment system.
    flux_chain : Flux out of an arbitrary N-compartment chain.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_plucom(J, t, T=[5, 10], fp=0.2)
    (array([0.        , 1.        , 2.43233236, 2.97173749, 2.16618349]), array([0.        , 0.        , 2.        , 3.        , 2.33333333]), array([0.        , 0.8       , 2.34586589, 2.97738999, 2.19961346]))
    """
    Jc = flux_comp(J, t=t, dt=dt, T=T[0])
    Jp = flux_plug(J, t=t, dt=dt, T=T[1])
    return Jc, Jp, fp * Jp + (1 - fp) * Jc

# Plug flow

def prop_plug(t, T=None):
    """
    Propagator of a plug flow system.

    See section :ref:`define-plug-flow` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the propagator is calculated, in the same units 
        as `T`.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).


    Returns
    -------
    np.ndarray
        Propagator as a 1D array.

    See Also
    --------
    res_plug : Residue function of a plug flow system.
    conc_plug : Concentration in a plug flow system.
    flux_plug : Outflux from a plug flow system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.prop_plug(t, 5)
    array([0.        , 0.        , 0.33333333, 0.5       ])
    """
    return functions_utils.ddelta(T, t)


def res_plug(t, T=None):
    """
    Residue function of a plug flow system.

    See section :ref:`define-plug-flow` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the residue function is calculated, in the same 
        units as `T`.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).

    Returns
    -------
    np.ndarray
        Residue function as a 1D array.

    See Also
    --------
    prop_plug : Propagator function of a plug flow system.
    conc_plug : Concentration in a plug flow system.
    flux_plug : Outflux from a plug flow system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.res_plug(t, 5)
    array([1.00000000e+00, 1.00000000e+00, 8.33333333e-01, 1.11022302e-16])
    """
    h = prop_plug(t, T)
    return 1 - misc.trapz(h, t)


def conc_plug(J=None, t=None, dt=1.0, T=None, solver='interp'):
    """
    Tissue concentration in a plug flow system.

    See section :ref:`define-plug-flow` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).
    solver : str, optional
        Solver for the system, either 'conv' for explicit convolution with a 
        discrete impulse response (slow) or 'interp' for interpolation 
        (fast). Defaults to 'interp'.

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    res_plug : Residue function of a plug flow system.
    prop_plug : Propagator function of a plug flow system.
    flux_plug : Outflux from a plug flow system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc_plug(J, t, T=5)
    array([ 0.  ,  5.  , 12.5 , 16.25, 13.75])
    """
    if T == np.inf:
        return conc_trap(J)
    if T == 0:
        return 0*J
    t = misc.tarray(len(J), t=t, dt=dt)
    if solver == 'conv':
        r = res_plug(t, T)
        return convolution.conv(r, J, t=t, dt=dt)
    elif solver == 'interp':
        Jo = np.interp(t-T, t, J, left=0)
        return misc.trapz(J-Jo, t)


def flux_plug(J=None, t=None, dt=1.0, T=None, solver='interp'):
    """
    Flux out of a plug flow system.

    See section :ref:`define-plug-flow` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).
    solver : str, optional
        Solver for the system, either 'conv' for explicit convolution with a 
        discrete impulse response (slow) or 'interp' for interpolation 
        (fast). Defaults to 'interp'.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    res_plug : Residue function of a plug flow system.
    prop_plug : Propagator function of a plug flow system.
    conc_plug : Concentration in a plug flow system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_plug(J, t, T=5)
    array([0.        , 1.        , 2.5       , 3.        , 2.16666667])
    """
    if T == np.inf:
        return flux_trap(J)
    if T == 0:
        return J
    t = misc.tarray(len(J), t=t, dt=dt)
    if solver == 'conv':
        h = prop_plug(t, T)
        return convolution.conv(h, J, t=t, dt=dt)
    elif solver == 'interp':
        return np.interp(t-T, t, J, left=0)
    else:
        raise ValueError('Solver ' + solver + ' does not exist.')


# 2 Parameters

# Chain

def prop_chain(t, T=None, D=None):
    """
    Propagator of a chain.

    See section :ref:`define-chain` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the propagator is calculated, in the same units 
        as `T`.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).
    D : float
        Dispersion of the system. Values must be between 0 (no dispersion) 
        and 1 (maximal dispersion).

    Returns
    -------
    np.ndarray
        Propagator as a 1D array.

    Raises
    ------
    ValueError
        If one of the parameters is out of bounds.

    See Also
    --------
    res_chain : Residue function of a chain.
    conc_chain : Concentration in a chain.
    flux_chain : Outflux from a chain.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.prop_chain(t, T=5, D=0.5)
    array([0.        , 0.14457322, 0.12921377, 0.08708924])
    """
    if T < 0:
        raise ValueError('T must be non-negative')
    if D < 0:
        raise ValueError('D cannot be negative')
    if D > 1:
        raise ValueError('D cannot be larger than 1')
    if D == 0:
        return prop_plug(t, T)
    if D == 1:
        return prop_comp(t, T)
    n = 1/D
    g = convolution.nexpconv(n, T/n, t)
    return g


def res_chain(t, T=None, D=None):
    """
    Residue function of a chain.

    See section :ref:`define-chain` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the residue function is calculated, in the same 
        units as `T`.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).
    D : float
        Dispersion of the system. Values must be between 0 (no dispersion) 
        and 1 (maximal dispersion).


    Returns
    -------
    np.ndarray
        Residue function as a 1D array.

    See Also
    --------
    prop_chain : Propagator function of a chain.
    conc_chain : Concentration in a chain.
    flux_chain : Outflux from a chain.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.res_chain(t, 5, 0.5)
    array([1.        , 0.78314017, 0.64624667, 0.42994366])
    """
    if D == 0:
        return res_plug(t, T)
    if D == 1:
        return res_comp(t, T)
    h = prop_chain(t, T, D)
    return 1-misc.trapz(h, t)


def conc_chain(J=None, t=None, dt=1.0, T=None, D=None, solver='step'):
    """
    Tissue concentration in a chain.

    See section :ref:`define-chain` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the compartment acts as 
        a trap).
    D : float
        Dispersion of the system. Values must be between 0 (no dispersion) 
        and 1 (maximal dispersion).
    solver : str, optional
        Solver used for the chain system calculation. Defaults to 'step'.

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    res_chain : Residue function of a chain.
    prop_chain : Propagator function of a chain.
    flux_chain : Outflux from a chain.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc_chain(J, t, T=5, D=0.5)
    array([ 0.        ,  6.82332358, 21.45008965, 30.86598366, 33.12222937])
    """
    if D == 0:
        return conc_plug(J, t=t, dt=dt, T=T)
    if D == 1:
        return conc_comp(J, t=t, dt=dt, T=T)

    # TODO: THIS NEEDS DEBUGGING
    # if solver=='diag':
    #     n0 = np.floor(1/D)
    #     Tc, Ec = _chain_ncomp(n0, T)
    #     Ji = np.zeros((n0,len(J)))
    #     Ji[0,:] = J
    #     C = conc_ncomp(Ji, Tc, Ec, t=t, dt=dt).sum(axis=0)
    #     if n0==1/D:
    #         return C
    #     Tc, Ec = _chain_ncomp(n0+1, T)
    #     C += conc_ncomp(Ji, Tc, Ec, t=t, dt=dt).sum(axis=0)
    #     return C/2

    tr = misc.tarray(len(J), t=t, dt=dt)
    r = res_chain(tr, T, D)
    return convolution.conv(r, J, t=t, dt=dt, solver=solver)


def flux_chain(J=None, t=None, dt=1.0, T=None, D=None, solver='step'):
    """
    Flux out of a chain.

    See section :ref:`define-chain` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the compartment acts as 
        a trap).
    D : float
        Dispersion of the system. Values must be between 0 (no dispersion) 
        and 1 (maximal dispersion).
    solver : str, optional
        Solver used for the chain system calculation. Defaults to 'step'.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    res_chain : Residue function of a chain.
    prop_chain : Propagator function of a chain.
    conc_chain : Concentration in a chain.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_chain(J, t, T=5, D=0.5)
    array([0.        , 0.27067057, 1.9393115 , 2.64461893, 1.94721897])
    """
    if D == 0:
        return flux_plug(J, t=t, dt=dt, T=T)
    if D == 1:
        return flux_comp(J, t=t, dt=dt, T=T)

    # TODO: This needs debugging
    # if solver=='diag':
    #     n0 = int(np.floor(1/D))
    #     Tc, Ec = _chain_ncomp(n0, T)
    #     Ji = np.zeros((n0,len(J)))
    #     Ji[0,:] = J
    #     Jo = flux_ncomp(Ji, Tc, Ec, t=t, dt=dt)[n0-1,n0-1,:]
    #     if n0==1/D:
    #         return Jo
    #     Tc, Ec = _chain_ncomp(n0+1, T)
    #     Jo += flux_ncomp(Ji, Tc, Ec, t=t, dt=dt)[n0,n0,:]
    #     return Jo/2

    th = misc.tarray(len(J), t=t, dt=dt)
    h = prop_chain(th, T, D)
    return convolution.conv(h, J, t=t, dt=dt, solver=solver)

# Helper function in diag solver for chain model
# def _chain_ncomp(n, T):
#     # Helper function
#     Tarr = np.full(n, T/n)
#     E = np.zeros((n,n))
#     for i in range(n-1):
#         E[i+1,i] = 1
#     E[n-1,n-1] = 1
#     return Tarr, E


# Step

def prop_step(t, T=None, D=None):
    """
    Propagator of a step.

    See section :ref:`define-step` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the propagator is calculated, in the same units 
        as `T`.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).
    D : float
        Dispersion of the system, or half-width of the step given as a 
        fraction of `T`. Values must be between 0 (no dispersion) and 1 
        (maximal dispersion).

    Returns
    -------
    np.ndarray
        Propagator as a 1D array.

    Raises
    ------
    ValueError
        If one of the parameters is out of bounds.

    See Also
    --------
    res_step : Residue function of a step.
    conc_step : Concentration in a step.
    flux_step : Outflux from a step.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.prop_step(t, 5, 0.5)
    array([0.03508772, 0.21052632, 0.21052632, 0.21052632])
    """
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    if T < 0:
        raise ValueError('T must be non-negative')
    if D < 0:
        raise ValueError('D cannot be negative')
    if D > 1:
        raise ValueError('D cannot be larger than 1')
    if T == np.inf:
        return prop_trap(t)
    if D == 0:
        return prop_plug(t, T)
    return functions_utils.dstep(T-D*T, T+D*T, t)


def res_step(t, T=None, D=None):
    """
    Residue function of a step.

    See section :ref:`define-step` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the residue function is calculated, in the same 
        units as `T`.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the system acts as a trap).
    D : float
        Dispersion of the system, or half-width of the step given as a 
        fraction of `T`. Values must be between 0 (no dispersion) and 1 
        (maximal dispersion).


    Returns
    -------
    np.ndarray
        Residue function as a 1D array.

    See Also
    --------
    prop_step : Propagator function of a step.
    conc_step : Concentration in a step.
    flux_step : Outflux from a step.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 3, 4, 6]
    >>> dc.res_step(t, 5, 0.5)
    array([1.        , 0.63157895, 0.42105263, 0.        ])
    """
    h = prop_step(t, T, D)
    return 1 - misc.trapz(h, t)


def conc_step(J=None, t=None, dt=1.0, T=None, D=None):
    """
    Tissue concentration inside a step.

    See section :ref:`define-step` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the compartment acts as 
        a trap).
    D : float
        Dispersion of the system, or half-width of the step given as a 
        fraction of `T`. Values must be between 0 (no dispersion) and 1 
        (maximal dispersion).

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    res_step : Residue function of a step.
    prop_step : Propagator function of a step.
    flux_step : Outflux from a step.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.conc_step(J, t, T=5, D=0.5)
    array([ 0.        ,  6.71052632, 20.65789474, 28.42105263, 21.05263158])
    """
    if D == 0:
        return conc_plug(J, t=t, dt=dt, T=T)
    t = misc.tarray(len(J), t=t, dt=dt)
    r = res_step(t, T, D)
    return convolution.conv(r, J, t)


def flux_step(J=None, t=None, dt=1.0, T=None, D=None):
    """
    Flux out of a step.

    See section :ref:`define-step` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : float
        Mean transit time of the system. Any non-negative value is allowed, 
        including T = 0 and T = inf (in which case the compartment acts as 
        a trap).
    D : float
        Dispersion of the system, or half-width of the step given as a 
        fraction of `T`. Values must be between 0 (no dispersion) and 1 
        (maximal dispersion).

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    res_step : Residue function of a step.
    prop_step : Propagator function of a step.
    conc_step : Concentration inside a step.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_step(J, t, T=5, D=0.5)
    array([0.        , 0.42105263, 1.94736842, 2.94736842, 2.23684211])
    """
    if D == 0:
        return flux_plug(J, t=t, dt=dt, T=T)
    t = misc.tarray(len(J), t=t, dt=dt)
    h = prop_step(t, T, D)
    return convolution.conv(h, J, t)


def flux_pfcomp(J=None, t=None, dt=1.0, T=None, D=None, solver='interp'):
    """
    Flux out of a plug-flow compartment.

    See section :ref:`define-pfcomp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment (mmol/sec).
    t : array_like, optional
        The time points of the indicator flux `J` (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    T : float
        Mean transit time of the compartment (sec). Any non-negative value is 
        allowed, including T = 0 and T = inf (in which case the compartment 
        acts as a trap).
    D : float
        Dispersion of the system defined as the ratio of the compartmental 
        mean transit time versus the total mean transit time.
    solver : str, optional
        Solver for the system, either 'conv' for explicit convolution with a 
        discrete impulse response (slow) or 'interp' for interpolation 
        (fast). Defaults to 'interp'.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array (mmol/sec).

    See Also
    --------
    res_pfcomp : Residue function of a plug-flow compartment.
    prop_pfcomp : Propagator function of a plug-flow compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> dc.flux_pfcomp(J, t, T=5, D=0.2)
    array([0.        , 0.35892193, 2.45784099, 2.97333203, 2.16222222])
    """
    if D < 0 or D > 1:
        raise ValueError('Dispersion must be in the range [0,1]')
    if D == 0:
        return flux_plug(J, t=t, dt=dt, T=T, solver=solver)
    if D == 1:
        return flux_comp(J, t=t, dt=dt, T=T)
    Tc = D * T
    Tp = (1 - D) * T
    J = flux_comp(J, t=t, dt=dt, T=Tc)
    J = flux_plug(J, t=t, dt=dt, T=Tp, solver=solver)
    return J



# N parameters

# Free


def prop_free(t, h=None, TT=None, TTmin=0, TTmax=None):
    """
    Propagator of a free system.

    See section :ref:`define-free` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the propagator is calculated, in the same units 
        as `TT`.
    H : array_like
        Frequencies of the transit time histogram in each transit time bin. 
        These do not have to be normalized - the function normalizes to unit 
        area by default.
    TT : array_like, optional
        Boundaries of the transit time histogram bins. The number of elements 
        in this array must be one more than the number of elements in `H`. 
        If `TT` is not provided, the boundaries are equally distributed between 
        `TTmin` and `TTmax`. Defaults to None.
    TTmin : float, optional
        Minimal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to 0.
    TTmax : float, optional
        Maximal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to the maximum of `t`.

    Returns
    -------
    np.ndarray
        Propagator as a 1D array.

    Raises
    ------
    ValueError
        If the array of transit times `TT` has an incorrect length relative 
        to `H`.

    See Also
    --------
    res_free : Residue function of a free system.
    conc_free : Concentration in a free system.
    flux_free : Outflux from a free system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 1, 2, 3]

    Assume the transit time histogram is provided by two equally sized bins 
    covering the entire time interval, with frequencies 2 and 1, 
    respectively:

    >>> dc.prop_free(t, [2, 1])
    array([0.33333333, 0.41666667, 0.33333333, 0.16666667])

    Assume the transit time has two equally sized bins, but between the 
    values [0.5, 2.5]:

    >>> dc.prop_free(t, [2, 1], TTmin=0.5, TTmax=2.5)
    array([0.19047619, 0.47619048, 0.38095238, 0.0952381 ])

    Assume the transit time histogram is provided by two bins in the same 
    range, but with different sizes: one from 0.5 to 1 and the other from 
    1 to 2.5. The frequencies in the bins are the same as in the previous 
    example:

    >>> dc.prop_free(t, [2, 1], TT=[0.5, 1.0, 2.5])
    array([0.33333333, 0.64814815, 0.14814815, 0.07407407])
    """
    nTT = len(h)
    if TT is None:
        if TTmax is None:
            TTmax = np.amax(t)
        TT = np.linspace(TTmin, TTmax, nTT+1)
    else:
        if len(TT) != nTT+1:
            msg = 'The array of transit time boundaries needs to have length N+1, '
            msg += '\n with N the size of the transit time distribution H.'
            raise ValueError(msg)
    h = functions_utils.ddist(h, TT, t)
    return h/trapezoid(h, t)


def res_free(t, h=None, TT=None, TTmin=0, TTmax=None):
    """
    Residue function of a free system.

    See section :ref:`define-free` for more detail.

    Parameters
    ----------
    t : array_like
        Time points where the residue function is calculated, in the same units 
        as `TT`.
    H : array_like
        Frequencies of the transit time histogram in each transit time bin. 
        These do not have to be normalized - the function normalizes to unit 
        area by default.
    TT : array_like, optional
        Boundaries of the transit time histogram bins. The number of elements 
        in this array must be one more than the number of elements in `H`. 
        If `TT` is not provided, the boundaries are equally distributed between 
        `TTmin` and `TTmax`. Defaults to None.
    TTmin : float, optional
        Minimal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to 0.
    TTmax : float, optional
        Maximal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to the maximum of `t`.

    Returns
    -------
    np.ndarray
        Residue function as a 1D array.

    Raises
    ------
    ValueError
        If the array of transit times `TT` has an incorrect length relative 
        to `H`.

    See Also
    --------
    prop_free : Propagator function of a free system.
    conc_free : Concentration in a free system.
    flux_free : Outflux from a free system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 1, 2, 3]

    Assume the transit time histogram is provided by two equally sized bins 
    covering the entire time interval, with frequencies 2 and 1, 
    respectively:

    >>> dc.res_free(t, [2, 1])
    array([1.   , 0.625, 0.25 , 0.   ])

    Assume the transit time has two equally sized bins, but between the 
    values [0.5, 2.5]:

    >>> dc.res_free(t, [2, 1], TTmin=0.5, TTmax=2.5)
    array([1.00000000e+00, 6.66666667e-01, 2.38095238e-01, 2.22044605e-16])

    Assume the transit time histogram is provided by two bins in the same 
    range, but with different sizes: one from 0.5 to 1 and the other from 
    1 to 2.5. The frequencies in the bins are the same as in the previous 
    example:

    >>> dc.res_free(t, [2, 1], TT=[0.5, 1.0, 2.5])
    array([1.00000000e+00, 5.09259259e-01, 1.11111111e-01, 2.22044605e-16])
    """
    h = prop_free(t, h, TT=TT, TTmin=TTmin, TTmax=TTmax)
    r = 1 - misc.trapz(h, t)
    r[r < 0] = 0
    return r


def conc_free(J=None, t=None, dt=1.0, h=None, TT=None, TTmin=0, TTmax=None, solver='trap'):
    """
    Tissue concentration in a free system.

    See section :ref:`define-free` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `TT`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `TT`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    H : array_like
        Frequencies of the transit time histogram in each transit time bin. 
        These do not have to be normalized - the function normalizes to unit 
        area by default.
    TT : array_like, optional
        Boundaries of the transit time histogram bins. The number of elements 
        in this array must be one more than the number of elements in `H`. 
        If `TT` is not provided, the boundaries are equally distributed between 
        `TTmin` and `TTmax`. Defaults to None.
    TTmin : float, optional
        Minimal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to 0.
    TTmax : float, optional
        Maximal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to the maximum of `t`.
    solver : str, optional
        Numerical solver used for the integration. Defaults to 'trap'.

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    See Also
    --------
    res_free : Residue function of a free system.
    prop_free : Propagator function of a free system.
    flux_free : Outflux from a free system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]

    Assume the transit time histogram is provided by two equally sized bins 
    covering the entire time interval, with frequencies 2 and 1, 
    respectively:

    >>> dc.conc_free(J, t, h=[2, 1])
    array([ 0.        ,  7.25308642, 29.41358025, 61.41975309, 77.56944444])

    Assume the transit time has two equally sized bins, but between the 
    values [0.5, 2.5]:

    >>> dc.conc_free(J, t, h=[2, 1], TTmin=0.5, TTmax=2.5)
    array([ 0.        ,  4.75925926, 10.15740741, 11.5       ,  8.10185185])

    Assume the transit time histogram is provided by two bins in the same 
    range, but with different sizes: one from 0.5 to 1 and the other from 
    1 to 2.5. The frequencies in the bins are the same as in the previous 
    example:

    >>> dc.conc_free(J, t, h=[2, 1], TT=[0.5, 1.0, 2.5])
    array([ 0.        ,  4.64814815,  9.58101852, 10.75      ,  7.5462963 ])

    If the time array is not provided, the function assumes uniform time 
    resolution with a time step of 1:

    >>> dc.conc_free(J, h=[2, 1], TT=[0.5, 1.0, 2.5])
    array([0.        , 1.17777778, 2.45555556, 3.25277778, 3.075     ])

    If the time step is different from 1, it needs to be provided 
    explicitly:

    >>> dc.conc_free(J, dt=2.0, h=[2, 1], TT=[0.5, 1.0, 2.5])
    array([0.        , 2.05555556, 3.87037037, 4.76388889, 4.14351852])
    """
    u = misc.tarray(len(J), t=t, dt=dt)
    r = res_free(u, h, TT=TT, TTmin=TTmin, TTmax=TTmax)
    return convolution.conv(r, J, t=t, dt=dt, solver=solver)


def flux_free(J=None, t=None, dt=1.0, h=None, TT=None, TTmin=0, TTmax=None):
    """
    Flux out of a free system.

    See section :ref:`define-free` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `TT`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `TT`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    H : array_like
        Frequencies of the transit time histogram in each transit time bin. 
        These do not have to be normalized - the function normalizes to unit 
        area by default.
    TT : array_like, optional
        Boundaries of the transit time histogram bins. The number of elements 
        in this array must be one more than the number of elements in `H`. 
        If `TT` is not provided, the boundaries are equally distributed between 
        `TTmin` and `TTmax`. Defaults to None.
    TTmin : float, optional
        Minimal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to 0.
    TTmax : float, optional
        Maximal transit time to be considered. If `TT` is provided, this 
        argument is ignored. Defaults to the maximum of `t`.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    res_free : Residue function of a free system.
    prop_free : Propagator function of a free system.
    conc_free : Concentration in a free system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]

    Assume the transit time histogram is provided by two equally sized bins 
    covering the entire time interval, with frequencies 2 and 1, 
    respectively:

    >>> dc.flux_free(J, t, h=[2, 1])
    array([0.        , 0.11111111, 0.48148148, 1.24074074, 2.625     ])

    Assume the transit time has two equally sized bins, but between the 
    values [0.5, 2.5]:

    >>> dc.flux_free(J, t, h=[2, 1], TTmin=0.5, TTmax=2.5)
    array([0.        , 1.55555556, 2.82222222, 3.        , 2.04444444])

    Assume the transit time histogram is provided by two bins in the same 
    range, but with different sizes: one from 0.5 to 1 and the other from 
    1 to 2.5. The frequencies in the bins are the same as in the previous 
    example:

    >>> dc.flux_free(J, t, h=[2, 1], TT=[0.5, 1.0, 2.5])
    array([0.        , 1.63888889, 2.85555556, 3.        , 2.03611111])

    If the time array is not provided, the function assumes uniform time 
    resolution with a time step of 1:

    >>> dc.flux_free(J, h=[2, 1], TT=[0.5, 1.0, 2.5])
    array([0.        , 0.65      , 1.83333333, 2.7       , 2.76666667])

    If the time step is different from 1, it needs to be provided 
    explicitly:

    >>> dc.flux_free(J, dt=2.0, h=[2, 1], TT=[0.5, 1.0, 2.5])
    array([0.        , 1.18055556, 2.38888889, 2.94444444, 2.52777778])
    """
    u = misc.tarray(len(J), t=t, dt=dt)
    h = prop_free(u, h, TT=TT, TTmin=TTmin, TTmax=TTmax)
    return convolution.conv(h, J, t=t, dt=dt)


# N compartments

# TODO: check that the sum of E's for a compartment = 1. Is it true that it can be something else? No sure..
# Maybe trapping and creation needs to be modelled with extra constants?
# The amounts trapped or created are not proportional to the amount inside.
def _K_ncomp(T, E):
    # dC/dt = J - KC
    if not isinstance(T, np.ndarray):
        T = np.array(T)
    if not isinstance(E, np.ndarray):
        E = np.array(E)
    # Helper function
    if np.amin(E) < 0:
        raise ValueError('Extraction fractions cannot be negative.')
    n = T.size
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == i:
                # Diagonal elements
                # sum of column i
                Ei = np.sum(E[:, i])
                if Ei == 0:
                    K[i, i] = 0
                else:
                    K[i, i] = Ei/T[i]
            else:
                # Off-diagonal elements
                if E[j, i] == 0:
                    K[j, i] = 0
                else:
                    K[j, i] = -E[j, i]/T[i]
    return K


def _J_ncomp(C, T, E):
    K = _K_ncomp(T, E)
    nc, nt = C.shape[0], C.shape[1]
    J = np.zeros((nc, nc, nt))
    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            if i == j:
                # Flux to outside
                Kii = np.sum(K[:, i])
                J[i, i, :] = Kii*C[i, :]
            else:
                # Flux to other compartments
                J[j, i, :] = -K[j, i]*C[i, :]
    return J

# Helper function


def _conc_ncomp_prop(J, T, E, t=None, dt=1.0, dt_prop=None):
    t = misc.tarray(len(J[0, :]), t=t, dt=dt)
    K = _K_ncomp(T, E)
    nt, nc = len(t), len(T)
    C = np.zeros((nc, nt))
    Kmax = K.diagonal().max()
    for k in range(nt-1):
        # Dk/nk <= 1/Kmax
        # Dk*Kmax <= nk
        Dk = t[k+1]-t[k]
        SJk = (J[:, k+1]-J[:, k])/Dk
        nk = int(np.ceil(Dk*Kmax))
        if dt_prop is not None:
            nk = np.amax([int(np.ceil(Dk/dt_prop)), nk])
        dk = Dk/nk
        Jk = J[:, k]
        Ck = C[:, k]
        for _ in range(nk):
            Jk_next = Jk + dk*SJk
            Ck_in = dk*(Jk+Jk_next)/2
            Ck = Ck + Ck_in - dk*np.matmul(K, Ck)
            Jk = Jk_next
        C[:, k+1] = Ck
    return C

# Helper function


def _conc_ncomp_diag(J, T, E, t=None, dt=1.0):
    t = misc.tarray(J.shape[1], t=t, dt=dt)
    # Calculate system matrix, eigenvalues and eigenvectors
    K = _K_ncomp(T, E)
    # From here, create generic function that solves n-comp system
    K, Q = np.linalg.eig(K)
    Qi = np.linalg.inv(Q)
    # Initialize concentration-time array
    nc, nt = len(T), len(t)
    C = np.zeros((nc, nt), dtype=K.dtype)
    Ei = np.zeros((nc, nt), dtype=K.dtype)
    # Loop over the inlets
    for i in range(nc):
        # Loop over the eigenvalues
        for d in range(nc):
            # Calculate elements of diagonal matrix
            Ei[d, :] = conc_comp(J[i, :], t, T=1/K[d])
            # Right-multiply with inverse eigenvector matrix
            Ei[d, :] *= Qi[d, i]
        # Left-multiply with eigenvector matrix
        C += np.matmul(Q, Ei)
    # Absolute value because K can be complex
    return np.absolute(C)


def conc_ncomp(J=None, t=None, dt=1.0, T=None, E=None, solver='diag', dt_prop=None):
    """
    Concentration in an n-compartment system.

    See section :ref:`define-ncomp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system, as a rectangular 2D array with 
        dimensions `(n, k)`, where `n` is the number of compartments and `k` 
        is the number of time points in `J`.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If `t` is not provided, the time points are assumed to be uniformly 
        spaced with spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced time points, in the 
        same units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : array_like
        An `n`-element array with mean transit times of each compartment.
    E : array_like
        Dimensionless and square `n x n` matrix. An off-diagonal element 
        `E[j, i]` is the extraction fraction from compartment `i` to 
        compartment `j`. A diagonal element `E[i, i]` is the extraction 
        fraction from compartment `i` to the outside.
    solver : str, optional
        A string specifying the numerical method for solving the system. Two 
        options are available:
        
        * 'diag' : Solves the system by diagonalizing the system matrix.
        * 'prop' : Solves the system by forward propagation.
        
        Defaults to 'diag'.
    dt_prop : float, optional
        Internal time resolution for the forward propagation when 
        `solver = 'prop'`. This must be in the same units as `T`. If not 
        provided, it defaults to the sampling interval, or the smallest time 
        step needed for stable results (whichever is smaller). This argument 
        is ignored when `solver = 'diag'`. Defaults to None.

    Returns
    -------
    np.ndarray
        Concentration in each compartment, and at each time point, as a 2D 
        array with dimensions `(n, k)`, where `n` is the number of 
        compartments and `k` is the number of time points in `J`.

    See Also
    --------
    res_ncomp : Residue function of an n-compartment system.
    prop_ncomp : Propagator function of an n-compartment system.
    flux_ncomp : Outfluxes from an n-compartment system.

    Notes
    -----
    The default solver 'diag' should be most accurate and fastest, but 
    currently does not allow for compartments that trap the tracer. It relies 
    on matrix diagonalization which may be more problematic in very large 
    systems, such as spatiotemporal models. 
    
    The alternative solver 'prop' is simple and robust and is a suitable 
    alternative in such cases. It is slower and less accurate, though the 
    accuracy can be improved at the cost of larger computation times by 
    setting a smaller `dt_prop`.

    Examples
    --------
    >>> import numpy as np
    >>> import dcmri as dc

    Consider a measurement with 10 time points from 0 to 20s, and a 
    2-compartment system with a constant influx in each compartment. The 
    influx in compartment 1 is twice as large as in compartment 0:

    >>> t = np.linspace(0, 20, 10)
    >>> J = np.zeros((2, t.size))
    >>> J[0, :] = 1
    >>> J[1, :] = 2

    The transit times are 6s for compartment 0 and 12s for compartment 1.

    >>> T = [6, 12]

    The extraction fraction from compartment 0 to compartment 1 is 0.3 and 
    the extraction fraction from 1 to 0 is 0.8. These are the off-diagonal 
    elements of `E`. No indicator is trapped or created inside the system 
    so the extraction fractions for each compartment must add up to 1. The 
    extraction fractions to the outside are therefore 0.7 and 0.2 for 
    compartment 0 and 1, respectively. These are the diagonal elements of `E`:

    >>> E = [
    ...   [0.7, 0.8],
    ...   [0.3, 0.2]]

    Calculate the concentrations in both compartments of the system:

    >>> C = dc.conc_ncomp(J, t, T=T, E=E)

    The concentrations in compartment 0 are:

    >>> C[0, :]
    array([ 0.        ,  2.13668993,  4.09491578,  5.87276879,  7.47633644,
            8.91605167, 10.20442515, 11.3546615 , 12.37983769, 13.29243667])

    The concentrations in compartment 1 are:

    >>> C[1, :]
    array([ 0.        ,  4.170364  ,  7.84318653, 11.0842876 , 13.94862323,
           16.48272778, 18.72645063, 20.71421877, 22.47597717, 24.03790679])

    Solving by forward propagation produces a different result because of the 
    relatively low time resolution:

    >>> C = dc.conc_ncomp(J, t, T=T, E=E, solver='prop')
    >>> C[1, :]
    array([ 0.        ,  4.44444444,  8.3127572 , 11.69333943, 14.65551209,
           17.25550803, 19.54012974, 21.54905722, 23.31636527, 24.87156916])

    But the difference can be made arbitrarily small by choosing a smaller 
    `dt_prop` (at the cost of some computation time). In this case the 
    results become very close with `dt_prop = 0.01`:

    >>> C = dc.conc_ncomp(J, t, T=T, E=E, solver='prop', dt_prop=0.01)
    >>> C[1, :]
    array([ 0.        ,  4.17147736,  7.84511918, 11.08681805, 13.95158088,
           16.48597905, 18.72988986, 20.71776196, 22.47955758, 24.04147164])
    """
    if solver == 'prop':
        return _conc_ncomp_prop(J, T, E, t=t, dt=dt, dt_prop=dt_prop)
    if solver == 'diag':
        if len(T) == 2:
            return _conc_2comp(J, T, E, t=t, dt=dt)
        return _conc_ncomp_diag(J, T, E, t=t, dt=dt)


def flux_ncomp(J=None, t=None, dt=1.0, T=None, E=None, solver='diag', dt_prop=None):
    """
    Flux out of an n-compartment system.

    See section :ref:`define-ncomp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the system, as a rectangular 2D array with 
        dimensions `(n, k)`, where `n` is the number of compartments and `k` 
        is the number of time points in `J`.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If `t` is not provided, the time points are assumed to be uniformly 
        spaced with spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced time points, in the 
        same units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : array_like
        An `n`-element array with mean transit times of each compartment.
    E : array_like
        Dimensionless and square `n x n` matrix. An off-diagonal element 
        `E[j, i]` is the extraction fraction from compartment `i` to 
        compartment `j`. A diagonal element `E[i, i]` is the extraction 
        fraction from compartment `i` to the outside.
    solver : str, optional
        A string specifying the numerical method for solving the system. Two 
        options are available:
        
        * 'diag' : Solves the system by diagonalizing the system matrix.
        * 'prop' : Solves the system by forward propagation.
        
        Defaults to 'diag'.
    dt_prop : float, optional
        Internal time resolution for the forward propagation when 
        `solver = 'prop'`. This must be in the same units as `T`. If not 
        provided, it defaults to the sampling interval, or the smallest time 
        step needed for stable results (whichever is smaller). This argument 
        is ignored when `solver = 'diag'`. Defaults to None.

    Returns
    -------
    np.ndarray
        Outflux out of each compartment, and at each time point, as a 3D 
        array with dimensions `(n, n, k)`, where `n` is the number of 
        compartments and `k` is the number of time points in `J`. 
        
        Encoding of the first two indices is the same as for `E`: `flux[j, i, :]` 
        is the flux from compartment `i` to `j`, and `flux[i, i, :]` is the 
        flux from `i` directly to the outside.

    See Also
    --------
    res_ncomp : Residue function of an n-compartment system.
    prop_ncomp : Propagator function of an n-compartment system.
    conc_ncomp : Concentration in an n-compartment system.

    Examples
    --------
    >>> import numpy as np
    >>> import dcmri as dc

    Consider a measurement with 10 time points from 0 to 20s, and a 
    2-compartment system with a constant influx in each compartment. The 
    influx in compartment 1 is twice as large as in compartment 0:

    >>> t = np.linspace(0, 20, 10)
    >>> J_in = np.zeros((2, t.size))
    >>> J_in[0, :] = 1
    >>> J_in[1, :] = 2

    The transit times are 6s for compartment 0 and 12s for compartment 1.

    >>> T = [6, 12]

    The extraction fraction from compartment 0 to compartment 1 is 0.3 and 
    the extraction fraction from 1 to 0 is 0.8. These are the off-diagonal 
    elements of `E`. No indicator is trapped or created inside the system 
    so the extraction fractions for each compartment must add up to 1. The 
    extraction fractions to the outside are therefore 0.7 and 0.2 for 
    compartment 0 and 1, respectively. These are the diagonal elements of `E`:

    >>> E = [
    ...   [0.7, 0.8],
    ...   [0.3, 0.2]]

    Calculate the outflux out of both compartments:

    >>> J_out = dc.flux_ncomp(J_in, t, T=T, E=E)

    The indicator flux out of compartment 0 to the outside is:

    >>> J_out[0, 0, :]
    array([0.        , 0.24928049, 0.47774017, 0.68515636, 0.87223925,
           1.04020603, 1.19051627, 1.32471051, 1.4443144 , 1.55078428])

    The indicator flux from compartment 1 to 0 is:

    >>> J_out[1, 0, :]
    array([0.        , 0.1068345 , 0.20474579, 0.29363844, 0.37381682,
           0.44580258, 0.51022126, 0.56773308, 0.61899188, 0.66462183])
    """
    C = conc_ncomp(J, t=t, dt=dt, T=T, E=E, solver=solver, dt_prop=dt_prop)
    return _J_ncomp(C, T, E)


def res_ncomp(t, T=None, E=None):
    """
    Residue function of an n-compartment system.

    See section :ref:`define-ncomp` for more detail.

    Parameters
    ----------
    t : array_like
        The time points where the residue function is calculated, in the same 
        units as `T`.
    T : array_like
        An `n`-element array with mean transit times of each compartment.
    E : array_like
        Dimensionless and square `n x n` matrix. An off-diagonal element 
        `E[j, i]` is the extraction fraction from compartment `i` to 
        compartment `j`. A diagonal element `E[i, i]` is the extraction 
        fraction from compartment `i` to the outside.

    Returns
    -------
    np.ndarray
        Residue in each compartment, and at each time point, as a 3D array 
        with dimensions `(n, n, k)`, where `n` is the number of compartments 
        and `k` is the number of time points in `t`. 
        
        Encoding of the first two indices is as follows: `R[j, i, :]` is the 
        residue in compartment `i` from an impulse injected into 
        compartment `j`.

    See Also
    --------
    flux_ncomp : Outfluxes from an n-compartment system.
    prop_ncomp : Propagator function of an n-compartment system.
    conc_ncomp : Concentration in an n-compartment system.

    Examples
    --------
    >>> import numpy as np
    >>> import dcmri as dc

    Consider a measurement with 10 time points from 0 to 20s, and a 
    2-compartment system defined by `T` and `E` as follows:

    >>> t = np.linspace(0, 20, 10)
    >>> T = [20, 2]
    >>> E = [[0.7, 0.9], [0.3, 0.1]]

    Calculate the residue in both compartments:

    >>> R = dc.res_ncomp(t, T, E)

    Given an impulse in compartment 1 at time `t = 0`, the residue in 
    compartment 1 is strictly decreasing:

    >>> R[1, 1, :]
    array([1.        , 0.337098  , 0.12441734, 0.05534255, 0.03213718,
           0.02364203, 0.01991879, 0.01779349, 0.01624864, 0.01495455])

    Given an impulse in compartment 1 at time `t = 0`, the residue in 
    compartment 0 is zero initially and peaks at a later time:

    >>> R[1, 0, :]
    array([-2.12339668e-17,  5.68742770e-01,  7.06912434e-01,  7.11111562e-01,
            6.75629514e-01,  6.30290545e-01,  5.84389102e-01,  5.40692137e-01,
            4.99900895e-01,  4.62071906e-01])
    """
    if len(T) == 2:
        return _res_2comp(T, E, t)
    # Calculate system matrix, eigenvalues and eigenvectors
    K = _K_ncomp(T, E)
    K, Q = np.linalg.eig(K)
    Qi = np.linalg.inv(Q)
    # Initialize concentration-time array
    nc, nt = len(T), len(t)
    R = np.zeros((nc, nc, nt), dtype=K.dtype)
    Ei = np.zeros((nc, nt), dtype=K.dtype)
    # Loop over the inlets
    for i in range(nc):
        # Loop over the eigenvalues
        for d in range(nc):
            # Calculate elements of diagonal matrix
            Ei[d, :] = np.exp(-t*K[d])
            # Right-multiply with inverse eigenvector matrix
            Ei[d, :] *= Qi[d, i]
        # Left-multiply with eigenvector matrix
        R[i, :, :] = np.matmul(Q, Ei)
    # Absolute because K can be complex
    return np.absolute(R)


def prop_ncomp(t, T=None, E=None):
    """
    Propagator of an n-compartment system.

    See section :ref:`define-ncomp` for more detail.

    Parameters
    ----------
    t : array_like
        The time points where the propagator is calculated, in the same units 
        as `T`.
    T : array_like
        An `n`-element array with mean transit times of each compartment.
    E : array_like
        Dimensionless and square `n x n` matrix. An off-diagonal element 
        `E[j, i]` is the extraction fraction from compartment `i` to 
        compartment `j`. A diagonal element `E[i, i]` is the extraction 
        fraction from compartment `i` to the outside.

    Returns
    -------
    np.ndarray
        Propagator for each arrow as a 4D array with dimensions 
        `(n, n, n, k)`, where `n` is the number of compartments and `k` is 
        the number of time points in `t`. 
        
        Encoding of the indices is as follows: `H[i, k, j, :]` is the 
        propagator from the inlet at compartment `i` to the outlet from 
        `j` to `k`. The diagonal element `H[i, j, j, :]` is the propagator 
        from the inlet at `i` to the outlet of `j` directly to the environment.

    See Also
    --------
    flux_ncomp : Outfluxes from an n-compartment system.
    res_ncomp : Residue function of an n-compartment system.
    conc_ncomp : Concentration in an n-compartment system.

    Examples
    --------
    >>> import numpy as np
    >>> import dcmri as dc

    Consider a measurement with 10 time points from 0 to 20s, and a 
    2-compartment system defined by `T` and `E` as follows:

    >>> t = np.linspace(0, 20, 10)
    >>> T = [20, 2]
    >>> E = [[0.7, 0.9], [0.3, 0.1]]

    Calculate the propagator for the system:

    >>> H = dc.prop_ncomp(t, T, E)

    The propagator from the inlet at 1 (first index = 1) to the outlet of 
    compartment 0 to the environment (diagonal case) is:

    >>> H[1, 0, 0, :]
    array([-7.43188837e-19,  1.99059970e-02,  2.47419352e-02,  2.48889047e-02,
            2.36470330e-02,  2.20601691e-02,  2.04536186e-02,  1.89242248e-02,
            1.74965313e-02,  1.61725167e-02])

    The propagator from the inlet at 1 (first index = 1) to the outlet from 
    compartment 0 to 1 is:

    >>> H[1, 1, 0, :]
    array([-3.18509502e-19,  8.53114156e-03,  1.06036865e-02,  1.06666734e-02,
            1.01344427e-02,  9.45435818e-03,  8.76583652e-03,  8.11038205e-03,
            7.49851343e-03,  6.93107859e-03])
    """
    R = res_ncomp(t, T, E)
    nc, nt = len(T), len(t)
    H = np.zeros((nc, nc, nc, nt))
    for i in range(nc):
        H[i, :, :, :] = _J_ncomp(R[i, :, :], T, E)
    return H


# 2 compartments (analytical)

def _K_2comp(T, E):
    K = _K_ncomp(T, E)
    if np.array_equal(K, np.identity(2)):
        return K, np.ones(2), K
    # Calculate the eigenvalues Ke
    D = math.sqrt((K[0, 0]-K[1, 1])**2 + 4*K[0, 1]*K[1, 0])
    Ke = [0.5*(K[0, 0]+K[1, 1]+D),
          0.5*(K[0, 0]+K[1, 1]-D)]
    # Build the matrix of eigenvectors (one per column)
    Q = np.array([
        [K[1, 1]-Ke[0], -K[0, 1]],
        [-K[1, 0], K[0, 0]-Ke[1]],
    ])
    # Build the inverse of the eigenvector matrix
    Qi = np.array([
        [K[0, 0]-Ke[1], K[0, 1]],
        [K[1, 0], K[1, 1]-Ke[0]]
    ])
    N = (K[0, 0]-Ke[1])*(K[1, 1]-Ke[0]) - K[0, 1]*K[1, 0]
    Qi /= N
    return Q, Ke, Qi


def _conc_2comp(J, T, E, t=None, dt=1.0):
    # Check input parameters
    if np.amin(T) <= 0:
        raise ValueError('T must be strictly positive.')
    if not isinstance(J, np.ndarray):
        J = np.array(J)
    if not isinstance(T, np.ndarray):
        T = np.array(T)
    if not isinstance(E, np.ndarray):
        E = np.array(E)
    # Build the system matrix K
    Q, K, Qi = _K_2comp(T, E)
    # Initialize concentration-time array
    t = misc.tarray(len(J[0, :]), t=t, dt=dt)
    C = np.zeros((2, len(t)))
    Ei = np.empty((2, len(t)))
    # Loop over the inlets
    for i in [0, 1]:
        # Loop over th eigenvalues
        for d in [0, 1]:
            # Calculate elements of diagonal matrix
            Ei[d, :] = conc_comp(J[i, :], t, T=1/K[d])
            # Right-multiply with inverse eigenvector matrix
            Ei[d, :] *= Qi[d, i]
        # Left-multiply with eigenvector matrix
        C += np.matmul(Q, Ei)
    return C


def _res_2comp(T, E, t):
    # Calculate system matrix, eigenvalues and eigenvectors
    Q, K, Qi = _K_2comp(T, E)
    # Initialize concentration-time array
    nc, nt = len(T), len(t)
    R = np.zeros((nc, nc, nt))
    Ei = np.empty((nc, nt))
    # Loop over the inlets
    for i in range(nc):
        # Loop over the eigenvalues
        for d in range(nc):
            # Calculate elements of diagonal matrix
            Ei[d, :] = np.exp(-t*K[d])
            # Right-multiply with inverse eigenvector matrix
            Ei[d, :] *= Qi[d, i]
        # Left-multiply with eigenvector matrix
        R[i, :, :] = np.matmul(Q, Ei)
    return R


# Non-stationary compartment


def conc_nscomp(J=None, t=None, dt=1.0, T=None):
    """
    Tissue concentration in a non-stationary compartment.

    See section :ref:`define-nscomp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced time points, in the 
        same units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : array_like
        Array with the mean transit time as a function of time, with the same 
        length as `J`. Only finite and strictly positive values are allowed.

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    Raises
    ------
    ValueError
        If one of the parameters is out of bounds (e.g., non-positive or 
        infinite values in `T`).

    See Also
    --------
    flux_nscomp : Outflux from a non-stationary compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> T = [1, 2, 3, 4, 5]
    >>> dc.conc_nscomp(J, t, T=T)
    array([ 0.        ,  3.09885687,  7.96130923, 11.53123615, 10.28639254])
    """
    if np.isscalar(T):
        raise ValueError('T must be an array of the same length as J.')
    if len(T) != len(J):
        raise ValueError('T and J must have the same length.')
    if np.amin(T) <= 0:
        raise ValueError('T must be strictly positive.')
    t = misc.tarray(len(J), t=t, dt=dt)
    n = len(t)
    C = np.zeros(n)
    for k in range(n-1):
        Dk = t[k+1]-t[k]
        Tk = (T[k]+T[k+1])/2
        Fk = Dk/Tk
        if Fk <= 1:
            Jk = (J[k]+J[k+1])/2
            C[k+1] = C[k] + Dk*Jk - Fk*C[k]
        else:
            nk = int(np.ceil(Dk/np.min(T[k:k+2])))
            STk = (T[k+1]-T[k])/Dk
            SJk = (J[k+1]-J[k])/Dk
            Jk = J[k]
            Tk = T[k]
            Ck = C[k]
            dk = Dk/nk
            for _ in range(nk):
                Jk_next = Jk + dk*SJk
                Tk_next = Tk + dk*STk
                Jk_curr = (Jk+Jk_next)/2
                Tk_curr = (Tk+Tk_next)/2
                Ck = Ck + dk*Jk_curr - dk*Ck/Tk_curr
                Jk = Jk_next
                Tk = Tk_next
            C[k+1] = Ck
    return C


def flux_nscomp(J=None, t=None, dt=1.0, T=None):
    """
    Flux out of a non-stationary compartment.

    See section :ref:`define-nscomp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as `T`. 
        If None, the time points are assumed to be uniformly spaced with 
        spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `T`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    T : array_like
        Array with the mean transit time as a function of time, with the same 
        length as `J`. Only finite and strictly positive values are allowed.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    See Also
    --------
    conc_nscomp : Tissue concentration in a non-stationary compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> T = [1, 2, 3, 4, 5]
    >>> dc.flux_nscomp(J, t, T=T)
    array([0.        , 1.54942844, 2.65376974, 2.88280904, 2.05727851])
    """
    C = conc_nscomp(J, t=t, dt=dt, T=T)
    return C/T

# TODO: Defaults for solver to None - everywhere




# Michaelis-Menten compartment

def _mmcomp_solve(J, Vmax, Km, t):
    # Schnell-Mendoza
    n = len(t)
    C = np.zeros(n)
    for k in range(n-1):
        Dk = t[k+1]-t[k]
        Jk = (J[k]+J[k+1])/2
        u = (C[k]/Km) * np.exp((C[k]-Vmax*Dk)/Km)
        C[k+1] = Jk*Dk + Km*np.real(lambertw(u))
    return C


def _mmcomp_prop(J, Vmax, Km, t):
    n = len(t)
    C = np.zeros(n)
    for k in range(n-1):
        Dk = t[k+1]-t[k]
        SJk = (J[k+1]-J[k])/Dk
        Jk = J[k]
        Ck = C[k]
        Tk = (Km+Ck)/Vmax
        nk = int(np.ceil(Dk/Tk))
        dk = Dk/nk
        for _ in range(nk):
            Jk_next = Jk + dk*SJk
            Jk_curr = (Jk+Jk_next)/2
            Ck = Ck + dk*Jk_curr - dk*Ck*Vmax/(Km+Ck)
            Jk = Jk_next
            Ck = np.amax([Ck, 0])
        C[k+1] = Ck
    return C


def conc_mmcomp(J=None, t=None, dt=1.0, Vmax=None, Km=None, solver='SM'):
    """
    Tissue concentration in a Michaelis-Menten compartment.

    See section :ref:`define-mmcomp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as 
        `Km / Vmax`. If None, the time points are assumed to be uniformly 
        spaced with spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `Km / Vmax`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    Vmax : float
        Limiting rate in the same units as `J`. Must be non-negative.
    Km : float
        Michaelis-Menten constant in units of concentration (or flux x time). 
        Must be non-negative.
    solver : str, optional
        Choose which solver to use. The options are:
        
        * 'SM' : Schnell and Mendoza analytical solution.
        * 'prop' : Numerical solution by forward propagation.
        
        Defaults to 'SM'.

    Returns
    -------
    np.ndarray
        Concentration as a 1D array.

    Raises
    ------
    ValueError
        If one of the parameters is out of bounds (e.g., negative `Vmax` or 
        `Km`).

    See Also
    --------
    flux_mmcomp : Outflux from a Michaelis-Menten compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> Vmax, Km = 1, 12
    >>> dc.conc_mmcomp(J, t, Vmax=Vmax, Km=Km)
    array([  0.        ,   7.5       ,  29.26723718,  64.27756059,
           114.97656637])
    """
    if Vmax < 0:
        raise ValueError('Vmax must be non-negative.')
    if Km < 0:
        raise ValueError('Km must be non-negative.')
    t = misc.tarray(len(J), t=t, dt=dt)
    if solver == 'SM':
        return _mmcomp_solve(J, Vmax, Km, t)
    if solver == 'prop':
        return _mmcomp_prop(J, Vmax, Km, t)


def flux_mmcomp(J=None, t=None, dt=1.0, Vmax=None, Km=None, solver='SM'):
    """
    Flux out of a Michaelis-Menten compartment.

    See section :ref:`define-mmcomp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment.
    t : array_like, optional
        The time points of the indicator flux `J`, in the same units as 
        `Km / Vmax`. If None, the time points are assumed to be uniformly 
        spaced with spacing `dt`. Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data, in the same 
        units as `Km / Vmax`. This parameter is ignored if `t` is explicitly 
        provided. Defaults to 1.0.
    Vmax : float
        Limiting rate in the same units as `J`. Must be non-negative.
    Km : float
        Michaelis-Menten constant in units of concentration (or flux x time). 
        Must be non-negative.

    solver : str, optional
        Choose which solver to use. The options are:
        
        * 'SM' : Schnell and Mendoza analytical solution.
        * 'prop' : Numerical solution by forward propagation.
        
        Defaults to 'SM'.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array.

    Raises
    ------
    ValueError
        If one of the parameters is out of bounds (e.g., negative `Vmax` or 
        `Km`).

    See Also
    --------
    conc_mmcomp : Tissue concentration in a Michaelis-Menten compartment.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> Vmax, Km = 1, 12
    >>> dc.flux_mmcomp(J, t, Vmax=Vmax, Km=Km)
    array([0.        , 0.38461538, 0.70921242, 0.84267981, 0.90549437])
    """
    C = conc_mmcomp(J, t, dt, Vmax, Km, solver=solver)
    return C*Vmax/(Km+C)


# Two-compartment exchange

def conc_2cxm(J=None, t=None, dt=1.0, T=None, E=None) -> np.ndarray:
    """
    Tissue concentration in a 2-compartment exchange system.

    See section :ref:`define-2comp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment (mmol/sec).
    t : array_like, optional
        The time points of the indicator flux `J` (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    T : array_like
        A 2-element array containing the mean transit times of the plasma 
        and extravascular compartments, respectively. Mean transit times 
        can take any value, including 0 and inf. Negative values are 
        unphysical but will only trigger an error if no solution exists.
    E : float
        Extraction fraction out of the plasma compartment. Value must be 
        between 0 and 1; boundary values E = 0 and E = 1 are correctly 
        handled. Values outside this range are unphysical but will only 
        trigger an error if no solution exists.

    Returns
    -------
    np.ndarray
        Concentration in each compartment, and at each time point, as a 2D 
        array with dimensions `(2, k)`, where 2 is the number of compartments 
        and `k` is the number of time points in `J`.

    Raises
    ------
    ValueError
        If no real solution exists because of unphysical parameter values 
        (usually E < 0).

    See Also
    --------
    flux_2cxm : Outflux from a 2-compartment exchange system.
    res_2cxm : Residue function of a 2-compartment exchange system.
    prop_2cxm : Propagator function of a 2-compartment exchange system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> T = [2.0, 10.0]
    >>> E = 0.4
    >>> dc.conc_2cxm(J, t, T=T, E=E)
    array([[ 0.        ,  3.26322857,  6.84087835,  8.75116763,  7.34480955],
           [ 0.        ,  1.52522203,  7.44971676, 14.44375764, 15.85129012]])
    """
    # T = [ Tp, Te]

    # Definition
    # ----------
    # vp cp' = Fpca - Fpcp + PSci - PScp
    # vi ci' = PScp - PSci
    #
    # vp, vi, Fp, PS


    # C-form
    # ------
    # Cp' = Fpca - (Fp/vp)Cp + (PS/vi)Ci - (PS/vp)Cp
    # Ci' = (PS/vp)Cp - (PS/vi)Ci
    #
    # Cp' = Ja - Cp/Tp + Ci/Ti
    # Ci' = Cp*E/Tp - Ci/Ti
    #
    # E, Tp, Ti
    #
    # Cp        ca       1/Tp  -1/Ti      Cp
    #     = Fp      - 
    # Ci         0      -E/Tp   1/Ti      Ci


    # c-form
    # ------
    # cp' = (Fp/vp)ca - (Fp/vp)cp + (PS/vp)ci - (PS/vp)cp
    # ci' = (PS/vi)cp - (PS/vi)ci   
    # 
    # cp' = Ja - cp/Tp + ci*E/Tp 
    # ci' = cp/Ti - ci/Ti 
    #
    # E, Tp, Ti

    #
    # cp           ca       1/Tp  -E/Tp      cp
    #     = Fp/vp      - 
    # ci            0      -1/Ti   1/Ti      ci

    if E == 0:
        Cp = conc_comp(J, t=t, dt=dt, T=T[0])
        Ce = np.zeros(len(J))
        return np.stack((Cp, Ce))

    if E == 1:
        # D
        # = sqrt( (Kp-Ke)**2 + 4 E Kp Ke  )
        # = sqrt( (Kp+Ke)**2 -4KeKp + 4 E Kp Ke  )
        # = sqrt( (Kp+Ke)**2 - 4(1-E)KeKp)
        # D(x) = sqrt( (Kp+Ke)**2 - 4 x KeKp)
        # D(0) = Kp+Ke
        # D'(x) = 0.5 ( (Kp+Ke)**2 - 4 x KeKp)**(-0.5) (-4 KeKp)
        # D'(0) = -2 KeKp/(Kp+Ke)
        # D = Kp+Ke -2 (1-E) KeKp/(Kp+Ke)

        # KT = Kp + Ke
        # Kpos = Kp + Ke
        # Kneg = (1-E) KeKp/(Kp+Ke)

        # Jpos
        # = utils.expconv(J, 1/Kpos, t=t, dt=dt)
        # Jneg
        # = Kneg exp(-tKneg) * J

        # Eneg
        # = (Kp+Ke - (1-E)Kp) / (Kp+Ke - (1-E) KeKp/(Kp+Ke))

        # 1 - Eneg
        # = [Kp+Ke - (1-E) KeKp/(Kp+Ke) - (Kp+Ke - (1-E)Kp)] / (Kp+Ke - (1-E) KeKp/(Kp+Ke))
        # = [(1-E)Kp - (1-E) KeKp/(Kp+Ke)] / (Kp+Ke - (1-E) KeKp/(Kp+Ke))
        # = (1-E) [Kp - KeKp/(Kp+Ke)] / (Kp+Ke - (1-E) KeKp/(Kp+Ke))

        # 1- Eneg = Epos
        # approx  [Kp - KeKp/(Kp+Ke)] (1-E) / (Kp+Ke)
        # (1-E) [Kp - KeKp/(Kp+Ke)] / (Kp+Ke)
        # (1-E) Rpos

        # Cp
        # = [ Epos*Jpos + Eneg*Jneg ] (1/Kp) / (1-E)
        # = [ Rpos*Jpos + (KeKp/(Kp+Ke)) exp(-tKneg) * J  ] (1/Kp)
        # = [ Rpos*Jpos + (KeKp/(Kp+Ke)) * J  ] (1/Kp)
        # = [ Jpos [Kp - KeKp/(Kp+Ke)] / (Kp+Ke) + (KeKp/(Kp+Ke)) * J  ] (1/Kp)
        # = [ Jpos [Kp - KeKp/(Kp+Ke)]  + KeKp * J  ] (1/Kp)  / (Kp+Ke)
        # Ce
        # = [(Kneg exp(-tKneg) * J*Kpos - Jpos*Kneg) / (Kpos - Kneg)] (1/Ke) E / (1-E)
        # = [Kpos exp(-tKneg) * J - Jpos) / (Kpos - Kneg)] Kneg (1/Ke) E / (1-E)
        # = [Kpos exp(-tKneg) * J - Jpos) / (Kpos - Kneg)] KeKp/(Kp+Ke) (1/Ke)
        # = [(Ke + Kp) exp(-tKneg) * J - Jpos) / (Ke + Kp)] KeKp/(Kp+Ke) (1/Ke)
        # = [(Ke + Kp) * J - Jpos) / (Ke + Kp)] KeKp/(Kp+Ke) (1/Ke)

        # Kp, Ke = 1/T[0], 1/T[1]
        # Kpos = Kp + Ke
        # Jpos = utils.expconv(J, 1/Kpos, t=t, dt=dt)

        # Jint = conc_trap(J, t=t, dt=dt)

        # K = Ke*Kp/(Kp+Ke)

        # Cp = ( Jpos*(Kp - K)  + Ke*Kp * Jint  ) * (1/Kp)  / (Kp+Ke)
        # Ce = ( (Jint - Jpos/(Ke + Kp))  ) * (1/Ke) * K

        # Cp = ( Jpos*(Kp - Ke*Kp/(Kp+Ke))  + Ke*Kp * Jint  ) * (1/Kp)  / (Kp+Ke)
        # Ce = ( (Jint - Jpos/(Ke + Kp))  ) * (1/Ke) * Ke*Kp/(Kp+Ke)

        # Cp = ( Jpos*(1/Ke - 1/(Kp+Ke))  +  Jint  ) * Ke/(Kp+Ke)
        # Ce = ( (Jint - Jpos/(Ke + Kp))  ) * Kp/(Kp+Ke)

        # Solution In Kp-representation
        # Cp = ( Jpos* Kp/(Kp+Ke)  +  Ke*Jint  ) /(Kp+Ke)
        # Ce = ( (Jint - Jpos/(Ke + Kp))  ) *Kp/(Kp+Ke)

        # Solution In Tp-representation
        # Tp, Te = T[0], T[1]

        # Cp = ( Jpos* Te/(Tp+Te)  +  Jint /Te ) *Tp*Te/(Tp+Te)
        # Ce = ( (Jint - Jpos*Tp*Te/(Tp + Te))  ) *Te/(Tp+Te)

        Jint = conc_trap(J, t=t, dt=dt)

        if T[0] == 0:
            Cp = np.zeros(len(J))
            Ce = Jint
            return np.stack((Cp, Ce))

        if T[1] == 0:
            Cp = Jint
            Ce = np.zeros(len(J))
            return np.stack((Cp, Ce))

        if np.isinf(T[0]):
            Cp = Jint
            Ce = np.zeros(len(J))
            return np.stack((Cp, Ce))

        if np.isinf(T[1]):
            # T[0] is not inf - covered above
            Jpos = convolution.expconv(J, T[0], t=t, dt=dt)
            Cp = Jpos*T[0]
            Ce = Jint - Jpos*T[0]
            return np.stack((Cp, Ce))

        Kpos = 1/T[0] + 1/T[1]
        Jpos = convolution.expconv(J, 1/Kpos, t=t, dt=dt)

        X = T[0]*T[1]/(T[0]+T[1])
        Cp = (Jpos * X/T[0] + Jint / T[1]) * X
        Ce = (Jint - Jpos*X) * X/T[0]

        # Check:
        # Cp + Ce
        # Jint (X/T[1] + X/T[0]) + Jpos(X*X/T[0] - X*X/T[0])
        # = Jint

        return np.stack((Cp, Ce))

    if T[0] == 0:
        # D = sqrt( (1/TP-1/TE)**2 + 4 * E * (1/TP) * (1/TE)  )
        # D = (1/TP) * sqrt( (1-TP/TE)**2 + 4 * E * TP/TE)
        # KT = 1/TP + 1/TE
        # KT = (1/TP) (1 + TP/TE)
        # Kpos = 0.5/TP * [ 1 + TP/TE + sqrt( (1-TP/TE)**2 + 4 * E * TP/TE) ]
        # Kneg = 0.5/TP * [ 1 + TP/TE - sqrt( (1-TP/TE)**2 + 4 * E * TP/TE) ]
        # First order term in Kneg dominates so need power expansion:
        # f(x) = sqrt( (1-x)**2 + 4E*x )
        # f(0) = 1
        # f'(x) = 1/2 * ((1-x)**2 + 4E*x)**(-1/2) * (-2*(1-x)+4E)
        # f'(0) = 1/2 * (-2+4E) = -1+2E
        # f(x) = 1 + (-1+2E)*x
        # TP->0:
        # Kpos = 0.5/TP * 2
        # Kpos = 1/TP
        # Kneg = 0.5/TP * [ TP/TE - (-1+2E) TP/TE]
        # Kneg = 0.5 * [1 - (-1+2E)] / TE
        # Kneg = (1-E)/TE
        # Then:
        # KB = (1-E)/TP
        # Eneg = (1/TP - (1-E)/TP) / (1/TP - (1-E)/TE)
        # Eneg = E / (1 - (1-E) TP/TE)
        # Eneg -> E

        # Jpos = J
        Jneg = convolution.expconv(J, T[1]/(1-E), t=t, dt=dt)

        # Je = (Jneg/TP - Jpos*(1-E)/TE) / (1/TP - (1-E)/TE)
        # Je = (Jneg - TP*Jpos*(1-E)/TE) / (1 - TP(1-E)/TE)
        # Je = Jneg

        # Jp = (1-E)*Jpos + E*Jneg
        # Je = Jneg

        # Cp = 0*Jp
        Cp = np.zeros(len(J))
        Ce = Jneg * T[1] * E / (1-E)

        return np.stack((Cp, Ce))

    if T[1] == 0:
        # D symmetric in TP and TE, so power expansion in TE is:
        # Kpos = 1/TE
        # Kneg = (1-E)/TP

        # Jpos = J
        Jneg = convolution.expconv(J, T[0]/(1-E), t=t, dt=dt)

        # KB = (1-E)/TP
        # Eneg = (1/TE - (1-E)/TP) / (1/TE - (1-E)/TP) = 1

        Jp = Jneg
        # Je
        # = (Jneg/TE - Jpos*(1-E)/TP) / (1/TE - (1-E)/TP)
        # = Jneg
        # Je = Jneg

        Cp = Jp*T[0]/(1-E)
        # Ce = Je*0
        Ce = np.zeros(len(J))

        return np.stack((Cp, Ce))

    if np.isinf(T[0]):
        Cp = conc_trap(J, t=t, dt=dt)
        Ce = np.zeros(len(J))
        return np.stack((Cp, Ce))

    if np.isinf(T[1]):
        Cp = conc_comp(J, t=t, dt=dt, T=T[0])
        Jp = Cp/T[0]
        Ce = conc_trap(E * Jp, t=t, dt=dt)
        return np.stack((Cp, Ce))

    K = np.array([
        [1/T[0], -1/T[1]],
        [-E/T[0], 1/T[1]],
    ])

    Dsq = (K[0, 0]-K[1, 1])**2 + 4*K[0, 1]*K[1, 0]
    if Dsq < 0:
        msg = 'No real solution to the 2CXM exists because of unphysical parameter values.'
        if E < 0:
            msg += '\n-> The extraction fraction is negative (E = '+str(E)+').'
        raise ValueError(msg)
    D = np.sqrt(Dsq)

    KT = K[0, 0] + K[1, 1]
    Kpos = 0.5*(KT + D)
    Kneg = 0.5*(KT - D)

    Tpos = 1/Kpos if Kpos != 0 else np.inf
    Tneg = 1/Kneg if Kneg != 0 else np.inf
    Jpos = convolution.expconv(J, Tpos, t=t, dt=dt)
    Jneg = convolution.expconv(J, Tneg, t=t, dt=dt)

    KB = K[0, 0] + K[1, 0]
    Eneg = (Kpos - KB)/(Kpos - Kneg)

    Jp = (1-Eneg)*Jpos + Eneg*Jneg
    Je = (Jneg*Kpos - Jpos*Kneg) / (Kpos - Kneg) 

    # Jp = Fp*cp, Cp = vp*cp -> Jp/Cp = Fp/vp = (1-E)/TP -> Cp = Jp*TP/(1-E)
    # Je = Fp*ce, Ce = ve*ce -> Je/Ce = Fp/ve = (1-E)/E * 1/TE -> Ce = Je*TE * E/(1-E)

    Cp = Jp * T[0] / (1 - E) 
    Ce = Je * T[1] * E / (1 - E)

    return np.stack((Cp, Ce))


def flux_2cxm(J=None, t=None, dt=1.0, T=None, E=None):
    """
    Flux out of a 2-compartment exchange system.

    See section :ref:`define-2comp` for more detail.

    Parameters
    ----------
    J : array_like
        The indicator flux entering the compartment (mmol/sec).
    t : array_like, optional
        The time points of the indicator flux `J` (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    T : array_like
        A 2-element array containing the mean transit times of the plasma 
        and extravascular compartments, respectively. Mean transit times 
        can take any value, including 0 and inf. Negative values are 
        unphysical but will only trigger an error if no solution exists.
    E : float
        Extraction fraction out of the plasma compartment. Value must be 
        between 0 and 1; boundary values E = 0 and E = 1 are correctly 
        handled. Values outside this range are unphysical but will only 
        trigger an error if no solution exists.
    solver : str, optional
        Solver for the system, either 'conv' for explicit convolution with a 
        discrete impulse response (slow) or 'interp' for interpolation 
        (fast). Defaults to 'interp'.

    Returns
    -------
    np.ndarray
        Outflux as a 1D array (mmol/sec).

    Raises
    ------
    ValueError
        If no real solution exists because of unphysical parameter values 
        (usually E < 0).

    See Also
    --------
    conc_2cxm : Tissue concentration in a 2-compartment exchange system.
    res_2cxm : Residue function of a 2-compartment exchange system.
    prop_2cxm : Propagator function of a 2-compartment exchange system.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> J = [1, 2, 3, 3, 2]
    >>> T = [2.0, 10.0]
    >>> E = 0.4
    >>> dc.flux_2cxm(J, t, T=T, E=E)
    array([0.        , 0.97896857, 2.0522635 , 2.62535029, 2.20344286])
    """
    # T = [ Tp, Te]

    # Jp = Fp*cp, Cp = vp*cp -> Jp/Cp = Fp/vp = (1-E)/TP -> Cp = Jp*TP/(1-E)
    # Jp = Cp*(1-E)/Tp

    if E == 1:
        return np.zeros(len(J))

    if T[0] == 0:
        # D = sqrt( (1/TP-1/TE)**2 + 4 * E * (1/TP) * (1/TE)  )
        # D = (1/TP) * sqrt( (1-TP/TE)**2 + 4 * E * TP/TE)
        # KT = 1/TP + 1/TE
        # KT = (1/TP) (1 + TP/TE)
        # Kpos = 0.5/TP * [ 1 + TP/TE + sqrt( (1-TP/TE)**2 + 4 * E * TP/TE) ]
        # Kneg = 0.5/TP * [ 1 + TP/TE - sqrt( (1-TP/TE)**2 + 4 * E * TP/TE) ]
        # First order term in Kneg dominates so need power expansion:
        # f(x) = sqrt( (1-x)**2 + 4E*x )
        # f(0) = 1
        # f'(x) = 1/2 * ((1-x)**2 + 4E*x)**(-1/2) * (-2*(1-x)+4E)
        # f'(0) = 1/2 * (-2+4E) = -1+2E
        # f(x) = 1 + (-1+2E)*x
        # TP->0:
        # Kpos = 0.5/TP * 2
        # Kpos = 1/TP
        # Kneg = 0.5/TP * [ TP/TE - (-1+2E) TP/TE]
        # Kneg = 0.5 * [1 - (-1+2E)] / TE
        # Kneg = (1-E)/TE
        # Then:
        # KB = (1-E)/TP
        # Eneg = (1/TP - (1-E)/TP) / (1/TP - (1-E)/TE)
        # Eneg = E / (1 - (1-E) TP/TE)
        # Eneg -> E
        # Jpos = J
        # Jneg = utils.expconv(J, T[1]/(1-E), t=t, dt=dt)

        # Je = (Jneg/TP - Jpos*(1-E)/TE) / (1/TP - (1-E)/TE)
        # Je = (Jneg - TP*Jpos*(1-E)/TE) / (1 - TP(1-E)/TE)
        # Je = Jneg

        # Jp = (1-E)*Jpos + E*Jneg
        # Je = Jneg

        # The case E=1 is already handled above

        if np.isinf(T[1]):
            return (1-E)*J

        Jpos = J
        Jneg = convolution.expconv(J, T[1]/(1-E), t=t, dt=dt)
        return (1-E)*Jpos + E*Jneg

    C = conc_2cxm(J, t=t, dt=dt, T=T, E=E)
    Jp = C[0, :]*(1-E)/T[0]

    return Jp

    # K = np.array([
    #     [1/T[0], -1/T[1]],
    #     [-E/T[0], 1/T[1]],
    # ])

    # D = np.sqrt((K[0,0]-K[1,1])**2 + 4*K[0,1]*K[1,0])

    # KT = K[0,0] + K[1,1]
    # Kpos = 0.5*(KT + D)
    # Kneg = 0.5*(KT - D)

    # Jpos = convolution.expconv(J, 1/Kpos, t=t, dt=dt)
    # Jneg = convolution.expconv(J, 1/Kneg, t=t, dt=dt)

    # KB = K[0,0] + K[1,0]
    # Eneg = (Kpos - KB)/(Kpos - Kneg)

    # Jp = (1-Eneg)*Jpos + Eneg*Jneg

    # return Jp
