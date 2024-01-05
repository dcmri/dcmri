import math
import numpy as np
from scipy.special import gamma
from scipy.stats import rv_histogram, norm

import dcmri.tools as tools

# 0 Parameters

# Trap

def res_trap(t):
    """Residue function of a trap.

    A trap is a space where all indicator that enters is trapped forever. In practice it is used to model tissues where the transit times are much longer than the acquisition window. 

    Args:
        t (array_like): Time points where the residue function is calculated.

    Returns:
        numpy.ndarray: residue function as a 1D array.

    See Also:
        `prop_trap`, `conc_trap`, `flux_trap`

    Notes: 
        The residue function of a trap is a function with a constant value of 1 everywhere, and can therefore easily be generated using the standard numpy function `numpy.ones`. The function is nevertheless included in the `dcmri` package for consistency and completeness. 

    Example:
        >>> import dcmri as dc
        >>> t = [0,1,2,3,4]
        >>> dc.res_trap(t)
        array([1., 1., 1., 1., 1.])  
    """
    return np.ones(len(t))


def prop_trap(t):
    """Propagator or transit time distribution of a trap.

    A trap is a space where all indicator that enters is trapped forever. In practice it is used to model tissues where the transit times are much longer than the acquisition window. 

    Args:
        t (array_like): Time points where the residue function is calculated.

    Returns:
        numpy.ndarray: propagator as a 1D array.

    See Also:
        `res_trap`, `conc_trap`, `flux_trap`

    Notes: 
        The propagator of a trap is a function with a constant value of 0 everywhere, and can therefore easily be generated using the standard numpy function `numpy.zeros`. The function is nevertheless included in the `dcmri` package for consistency and completeness. 

    Example:
        >>> import dcmri as dc
        >>> t = [0,1,2,3,4]
        >>> dc.prop_trap(t)
        array([0., 0., 0., 0., 0.])  
    """
    return np.zeros(len(t))

def conc_trap(J, t=None, dt=1.0):
    """Indicator concentration inside a trap.

    A trap is a space where all indicator that enters is trapped forever. In practice it is used to model tissues where the transit times are much longer than the acquisition window. 

    Args:
        J (array_like): the indicator flux entering the trap.
        t (array_like, optional): the time points of the indicator flux J. If t=None, the time points are assumed to be uniformly spaced with spacing dt. Defaults to None.
        dt (float, optional): spacing between time points for uniformly spaced time points. This parameter is ignored if t is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Concentration as a 1D array.

    See Also:
        `res_trap`, `prop_trap`, `flux_trap`

    Notes: 
        The concentration inside a trap is the time-integral of the influx, here calculated using trapezoidal integration.

    Example:
        >>> import dcmri as dc
        >>> J = [1,2,3,3,2]
        >>> dc.conc_trap(J, dt=2.0)
        array([ 0.,  3.,  8., 14., 19.])
    """
    return tools.trapz(J, t=t, dt=dt)

def flux_trap(J):
    """Indicator flux out of a trap.

    A trap is a space where all indicator that enters is trapped forever. In practice it is used to model tissues where the transit times are much longer than the acquisition window. 

    Args:
        J (array_like): the indicator flux entering the trap.

    Returns:
        numpy.ndarray: outflux as a 1D array.

    See Also:
        `res_trap`, `conc_trap`, `prop_trap`

    Notes: 
        The outflux out of a trap is always zero, and can therefore easily be generated using the standard numpy function `numpy.zeros`. The function is nevertheless included in the `dcmri` package for consistency and completeness. 

    Example:
        >>> import dcmri as dc
        >>> J = [1,2,3,3,2]
        >>> dc.flux_trap(J, dt=2.0)
        array([0., 0., 0., 0., 0.])  
    """
    return np.zeros(len(J))


# 1 Parameter

# Pass (no dispersion)

def res_pass(T, t):
    """Residue function of a pass.

    A pass is a space where the concentration is proportional to the input. In practice it is used to model tissues where the transit times are shorter than the temporal sampling interval. Under these conditions any bolus broadening is not detectable. 

    Args:
        T (float): transit time of the pass.
        t (array_like): Time points where the residue function is calculated.

    Returns:
        numpy.ndarray: residue function of the pass as a 1D array.

    See Also:
        `prop_pass`, `conc_pass`, `flux_pass`

    Notes: 
        The residue function of a pass is a delta function and therefore can only be approximated numerically. The numerical approximation becomes accurate only at very short sampling intervals.

    Example:
        >>> import dcmri as dc
        >>> t = [0,3,4,6]
        >>> dc.res_pass(5,t)
        array([3.33333333, 0.        , 0.        , 0.        ])  
    """
    return T*tools.prop_ddelta(t)

def prop_pass(t):
    """Propagator or transit time distribution of a pass.

    A pass is a space where the concentration is proportional to the input. In practice it is used to model tissues where the transit times are shorter than the temporal sampling interval. Under these conditions any bolus broadening is not detectable. 

    Args:
        t (array_like): Time points where the residue function is calculated.

    Returns:
        numpy.ndarray: propagator as a 1D array.

    See Also:
        `res_pass`, `conc_pass`, `flux_pass`

    Notes: 
        The propagator of a pass is a delta function and therefore can only be approximated numerically. The numerical approximation becomes accurate only at very short sampling intervals. 

    Example:
        >>> import dcmri as dc
        >>> t = [0,3,4,6]
        >>> dc.prop_pass(t)
        array([0.66666667, 0.        , 0.        , 0.        ])  
    """
    return tools.prop_ddelta(t)


def conc_pass(J, T):
    """Indicator concentration inside a pass.

    A pass is a space where the concentration is proportional to the input. In practice it is used to model tissues where the transit times are shorter than the temporal sampling interval. Under these conditions any bolus broadening is not detectable. 

    Args:
        J (array_like): the indicator flux entering the pass.
        T (float): transit time of the pass.

    Returns:
        numpy.ndarray: Concentration as a 1D array.

    See Also:
        `res_pass`, `prop_pass`, `flux_pass`

    Example:
        >>> import dcmri as dc
        >>> J = [1,2,3,3,2]
        >>> dc.conc_pass(J, 5)
        array([ 5, 10, 15, 15, 10])
    """
    return T*np.array(J)

def flux_pass(J):
    """Indicator flux out of a pass.

    A pass is a space where the concentration is proportional to the input. In practice it is used to model tissues where the transit times are shorter than the temporal sampling interval. Under these conditions any bolus broadening is not detectable. 

    Args:
        J (array_like): the indicator flux entering the pass.

    Returns:
        numpy.ndarray: outflux as a 1D array.

    See Also:
        `res_pass`, `conc_pass`, `prop_pass`

    Notes: 
        The outflux out of a pass is always the same as the influx, and therefore this function is an identity. It is nevertheless included in the `dcmri` package for consistency with other functionality. 

    Example:
        >>> import dcmri as dc
        >>> J = [1,2,3,3,2]
        >>> dc.flux_pass(J)
        array([1, 2, 3, 3, 2]) 
    """
    return np.array(J)


# Compartment

def res_comp(T, t):
    """Residue function of a compartment.

    A compartment is a space with a uniform concentration everywhere - also known as a well-mixed space. The residue function of a compartment is a mono-exponentially decaying function.

    Args:
        T (float): mean transit time of the compartment. Any non-negative value is allowed, including :math:`T=0` and :math:`T=\\infty`, in which case the compartment is a trap.
        t (array_like): time points where the residue function is calculated, in the same units as T.

    Returns:
        numpy.ndarray: residue function of the compartment as a 1D array.

    See Also:
        `prop_comp`, `conc_comp`, `flux_comp`
        
    Example:
        >>> import dcmri as dc
        >>> t = [0,3,4,6]
        >>> dc.res_comp(5,t)
        array([1.        , 0.54881164, 0.44932896, 0.30119421])  
    """
    if T == np.inf:
        return res_trap(t)
    if T == 0:
        return tools.res_ddelta(t)
    return np.exp(-np.array(t)/T)

def prop_comp(T, t):
    """Propagator or transit time distribution of a compartment.

    A compartment is a space with a uniform concentration everywhere - also known as a well-mixed space. The propagator of a compartment is a mono-exponentially decaying function. 

    Args:
        T (float): mean transit time of the compartment. Any non-negative value is allowed, including :math:`T=0` and :math:`T=\\infty`, in which case the compartment is a trap.
        t (array_like): time points where the residue function is calculated, in the same units as T.

    Returns:
        numpy.ndarray: propagator as a 1D array.

    See Also:
        `res_comp`, `conc_comp`, `flux_comp`

    Example:
        >>> import dcmri as dc
        >>> t = [0,3,4,6]
        >>> dc.prop_comp(5,t)
        array([0.2       , 0.10976233, 0.08986579, 0.06023884])  
    """
    if T == np.inf:
        return prop_trap(t)
    if T == 0:
        return tools.prop_ddelta(t)
    return np.exp(-np.array(t)/T)/T


def conc_comp(J, T, t=None, dt=1.0):
    """Indicator concentration inside a compartment.

    A compartment is a space with a uniform concentration everywhere - also known as a well-mixed space. 

    Args:
        J (array_like): the indicator flux entering the compartment.
        T (float): mean transit time of the compartment. Any non-negative value is allowed, including :math:`T=0` and :math:`T=\\infty`, in which case the compartment is a trap.
        t (array_like, optional): the time points of the indicator flux J, in the same units as T. If t=None, the time points are assumed to be uniformly spaced with spacing dt. Defaults to None.
        dt (float, optional): spacing between time points for uniformly spaced time points, in the same units as T. This parameter is ignored if t is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Concentration as a 1D array.

    See Also:
        `res_comp`, `prop_comp`, `flux_comp`

    Example:
        >>> import dcmri as dc
        >>> t = [0,5,15,30,60]
        >>> J = [1,2,3,3,2]
        >>> dc.conc_comp(J, 5, t)
        array([ 0.        ,  5.        , 12.16166179, 14.85868746, 10.83091743])
    """
    if T == np.inf:
        return conc_trap(J, t=t, dt=dt)
    return T*tools.expconv(J, T, t=t, dt=dt)


def flux_comp(J, T, t=None, dt=1.0):
    """Indicator flux out of a compartment.

    A compartment is a space with a uniform concentration everywhere - also known as a well-mixed space. 

    Args:
        J (array_like): the indicator flux entering the compartment.
        T (float): mean transit time of the compartment. Any non-negative value is allowed, including :math:`T=0` and :math:`T=\\infty`, in which case the compartment is a trap.
        t (array_like, optional): the time points of the indicator flux J, in the same units as T. If t=None, the time points are assumed to be uniformly spaced with spacing dt. Defaults to None.
        dt (float, optional): spacing between time points for uniformly spaced time points, in the same units as T. This parameter is ignored if t is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: outflux as a 1D array.

    See Also:
        `res_comp`, `conc_comp`, `prop_comp`

    Example:
        >>> import dcmri as dc
        >>> import dcmri as dc
        >>> t = [0,5,15,30,60]
        >>> J = [1,2,3,3,2]
        >>> dc.flux_comp(J, 5, t)
        array([0.        , 1.        , 2.43233236, 2.97173749, 2.16618349]) 
    """
    if T == np.inf:
        return flux_trap(J, t=t, dt=dt)
    return tools.expconv(J, T, t=t, dt=dt)






# Plug flow

def conc_plug(J, T, t=None, dt=1.0):
    if T==np.inf:
        return conc_trap(J, t=t, dt=dt)
    if T==0:
        return 0*J
    t = tools.tarray(len(J), t=t, dt=dt)
    Jo = np.interp(t-T, t, J, left=0)
    return tools.trapz(t, J-Jo)

def flux_plug(J, T, t=None, dt=1.0):
    if T==np.inf:
        return flux_trap(J, t=t, dt=dt)
    if T==0:
        return J
    t = tools.tarray(len(J), t=t, dt=dt)
    return np.interp(t-T, t, J, left=0) 

def res_plug(T, t):
    g = np.ones(len(t))
    g[np.where(t>T)] = 0
    return g

def prop_plug(T, t):
    h = tools.prop_ddelta(t)
    return np.interp(t-T, t, h, left=0)

# 2 Parameters

# Chain

def conc_chain(J, T, D, t=None, dt=1.0):
    if D == 0:
        return conc_plug(J, T, t=t, dt=dt)
    if D == 100:
        return conc_comp(J, T, t=t, dt=dt)
    t = tools.tarray(len(J), t=t, dt=dt)
    r = res_chain(T, D, t)
    return tools.conv(r, J, t)

def flux_chain(J, T, D, t=None, dt=1.0):
    if D == 0:
        return prop_plug(J, T, t=t, dt=dt)
    if D == 100:
        return prop_comp(J, T, t=t, dt=dt)
    t = tools.tarray(len(J), t=t, dt=dt)
    h = prop_chain(T, D, t)
    return tools.conv(h, J, t)

def res_chain(T, D, t):
    if D==0: 
        return res_plug(T, t)
    if D==100: 
        return res_comp(T, t)
    n = 100/D
    Tx = T/n
    norm = Tx*gamma(n)
    if norm == np.inf:
        raise ValueError('A chain model is not numerically stable with these parameters. Consider restricting the minimal dispersion allowed.')
    u = t/Tx  
    nt = len(t)
    g = np.ones(nt)
    g[0] = 0
    fnext = u[0]**(n-1)*np.exp(-u[0])/norm
    for i in range(nt-1):
        fi = fnext
        pow = u[i+1]**(n-1)
        if pow == np.inf:
            raise ValueError('A chain model is not numerically stable with these parameters. Consider restricting the minimal dispersion allowed.')
        fnext = pow * np.exp(-u[i+1])/norm
        g[i+1] = g[i] + (t[i+1]-t[i]) * (fnext+fi) / 2
    return 1-g

def prop_chain(T, D, t): 
    if D==0: 
        return prop_plug(T, t)
    if D==100: 
        return prop_comp(T, t)
    n = 100/D
    g = tools.nexpconv(n, T/n, t)
    gfin = np.isfinite(g)
    n_notfin = g.size - np.count_nonzero(gfin)
    if n_notfin > 0:
        raise ValueError('A chain model is not numerically stable with these parameters. Consider restricting the minimal dispersion allowed.')
    return g

# Step

def conc_step(J, T, D, t=None, dt=1.0):
    if D==0 or D>1:
        raise ValueError('D must be strictly positive and no larger than 1.')
    t = tools.tarray(len(J), t=t, dt=dt)
    h = res_step(T, D, t)
    return tools.conv(r, J, t)

def flux_step(J, T, D, t=None, dt=1.0):
    if D==0 or D>1:
        raise ValueError('D must be strictly positive and no larger than 1.')
    t = tools.tarray(len(J), t=t, dt=dt)
    h = prop_step(T, D, t)
    return tools.conv(h, J, t)

def res_step(T, D, t):
    h = prop_step(T, D, t)
    c = np.ones(len(t))
    # Integration is convolution with a constant
    return 1 - tools.conv(c, h, t)

def prop_step(T, D, t): 
    if D==0 or D>1:
        raise ValueError('D must be strictly positive and no larger than 1.')
    T0 = T-D*T
    T1 = T+D*T
    h = np.zeros(len(t))
    h[(t>=T0)*(t<=T1)] = 1/(T1-T0)
    return h

# N parameters

# Free

def conc_free(J, H, t=None, dt=1.0, TT=None, TTmin=0, TTmax=None):
    u = tools.tarray(len(J), t=t, dt=dt)
    r = res_free(H, u, TT=TT, TTmin=TTmin, TTmax=TTmax)
    return tools.conv(r, J, t=t, dt=dt)

def flux_free(J, H, t=None, dt=1.0, TT=None, TTmin=0, TTmax=None):
    u = tools.tarray(len(J), t=t, dt=dt)
    h = prop_free(H, u, TT=TT, TTmin=TTmin, TTmax=TTmax)
    return tools.conv(h, J, t=t, dt=dt)

def res_free(H, t, TT=None, TTmin=0, TTmax=None):
    H = np.array(H)
    nTT = len(H)
    if TT is None:
        if TTmax is None:
            TTmax = np.amax(t)
        TT = np.linspace(TTmin, TTmax, nTT+1)
    else:
        if len(TT) != nTT+1:
            msg = 'The array of transit time boundaries needs to have length N+1, '
            msg += '\n with N the size of the transit time distribution H.'
            raise ValueError(msg)
    dist = rv_histogram((H,TT), density=True)
    return 1 - dist.cdf(t)

def prop_free(H, t, TT=None, TTmin=0, TTmax=None):
    H = np.array(H)
    nTT = len(H)
    if TT is None:
        if TTmax is None:
            TTmax = np.amax(t)
        TT = np.linspace(TTmin, TTmax, nTT+1)
    else:
        if len(TT) != nTT+1:
            msg = 'The array of transit time boundaries needs to have length N+1, '
            msg += '\n with N the size of the transit time distribution H.'
            raise ValueError(msg)
    dist = rv_histogram((H,TT), density=True)
    return dist.pdf(t)


# N compartments

def K_ncomp(T, E):
    if np.amin(E) < 0:
        raise ValueError('Extraction fractions cannot be negative.')
    nc = T.size
    K = np.zeros((nc,nc))
    for i in range(nc):
        Ei = np.sum(E[:,i])
        if Ei==0:
            K[i,i] = 0
        else:
            K[i,i] = Ei/T[i]
        for j in range(nc):
            if j!=i:
                if E[j,i]==0:
                    K[j,i] = 0
                else:
                    K[j,i] = -E[j,i]/T[i]
    return K


def Ko_ncomp(T, E):
    if np.amin(E) < 0:
        raise ValueError('Extraction fractions cannot be negative.')
    nc = T.size
    K = np.zeros(nc)
    for i in range(nc):
        if E[i,i]==0:
            K[i] = 0
        else:
            K[i] = E[i,i]/T[i]
    return K


def conc_ncomp(J, T, E, t=None, dt=1.0):
    """Concentration in a general n-compartment model.

    T is an n-element array with MTTs for each compartment.
    E is the nxn system matrix with E[j,i] = Eji (if j!=i) and E[i,i] = Eoi.
    Note:
    - if sum_j Eji < 1 then compartment i contains a trap.
    - if sum_j Eji > 1 then compartment i produces indicator.
    """
    t = tools.tarray(len(J[:,0]), t=t, dt=dt)
    K = K_ncomp(T, E)
    Kmax = K.diagonal().max()
    nc = len(T)
    nt = len(t)
    C = np.zeros((nt,nc))
    for k in range(nt-1):
        Dk = t[k+1]-t[k]
        Jk = (J[k+1,:]+J[k,:])/2
        if Dk*Kmax <= 1:
            C[k+1,:] = C[k,:] + Dk*Jk - Dk*np.matmul(K, C[k,:])  
        else:
            # Dk/nk <= 1/Kmax
            # Dk*Kmax <= nk
            nk = np.ceil(Dk*Kmax)
            Dk = Dk/nk
            Jk = Jk/nk
            Ck = C[k,:]
            for _ in range(nk):
                Ck = Ck + Dk*Jk - Dk*np.matmul(K, Ck)
            C[k+1,:] = Ck
    return C

def flux_ncomp(J, T, E, t=None, dt=1.0):
    """Flux out of a general n-compartment model.
    """
    C = conc_ncomp(J, T, E, t=t, dt=dt)
    t = tools.tarray(len(J[:,0]), t=t, dt=dt)
    K = Ko_ncomp(T, E)
    Jo = np.zeros(C.shape)
    for k in range(C.shape[0]):
        Jo[k,:] = K*C[k,:]
    return Jo

def res_ncomp(T, E, t):
    nc = len(T)
    nt = len(t)
    J = np.zeros((nt, nc))
    r = np.zeros((nt, nc, nc))
    for c in range(nc):
        J[0,c] = 1
        r[:,:,c] = conc_ncomp(J, T, E, t)
        J[0,c] = 0
    return r

def prop_ncomp(T, E, t):
    nc = len(T)
    nt = len(t)
    J = np.zeros((nt, nc))
    h = np.zeros((nt, nc, nc))
    for c in range(nc):
        J[0,c] = 1
        h[:,:,c] = flux_ncomp(J, T, E, t)
        J[0,c] = 0
    return h



# 2 compartments (analytical)

def conc_2comp(J, T, E, t=None, dt=1.0):
    """Concentration in a general 2-compartment system.
    """
    if np.amin(T) <= 0:
        raise ValueError('T must be strictly positive.')
    t = tools.tarray(len(J[:,0]), t=t, dt=dt)
    K0 = (E[0,0]+E[1,0])/T[0]
    K1 = (E[0,1]+E[1,1])/T[1]
    K10 = E[1,0]/T[0]
    K01 = E[0,1]/T[1]
    Dsq = (K0-K1)**2 + 4*K01*K10
    D = math.sqrt(D)
    Kp = (K0+K1+Dsq)/2
    Kn = (K0+K1-Dsq)/2
    Np = K01*(Kp+K1) + K10*(Kp+K0)
    Nn = K01*(Kn+K1) + K10*(Kn+K0)
    Ap = math.sqrt(K01*(Kp+K1)/Np)
    An = math.sqrt(K01*(Kn+K1)/Nn)
    Bp = math.sqrt(K10*(Kp+K0)/Np)
    Bn = math.sqrt(K10*(Kn+K0)/Nn)
    E0p = conc_comp(J[:,0], 1/Kp, t)
    E0n = conc_comp(J[:,0], 1/Kn, t)
    E1p = conc_comp(J[:,1], 1/Kp, t)
    E1n = conc_comp(J[:,1], 1/Kn, t)
    C0 = Ap*Ap*E0p + An*An*E0n + Ap*Bp*E1p + An*Bn*E1n
    C1 = Ap*Bp*E0p + An*Bn*E0n + Bp*Bp*E1p + Bn*Bn*E1n
    return np.stack((C0, C1), axis=-1)

def flux_2comp(J, T, E, t=None, dt=1.0):
    """Concentration in a general 2-compartment system
    """
    C = conc_2comp(J, T, E, t=t, dt=dt)
    t = tools.tarray(len(J[:,0]), t=t, dt=dt)
    K0 = (E[0,0]+E[1,0])/T[0]
    K1 = (E[0,1]+E[1,1])/T[1]
    J0 = K0*C[:,0]
    J1 = K1*C[:,1]
    return np.stack((J0, J1), axis=-1)

def res_2comp(T, E, t):
    if np.amin(T) <= 0:
        raise ValueError('T must be strictly positive.')
    K0 = (E[0,0]+E[1,0])/T[0]
    K1 = (E[0,1]+E[1,1])/T[1]
    K10 = E[1,0]/T[0]
    K01 = E[0,1]/T[1]
    Dsq = (K0-K1)**2 + 4*K01*K10
    D = math.sqrt(D)
    Kp = (K0+K1+Dsq)/2
    Kn = (K0+K1-Dsq)/2
    Np = K01*(Kp+K1) + K10*(Kp+K0)
    Nn = K01*(Kn+K1) + K10*(Kn+K0)
    Ap = math.sqrt(K01*(Kp+K1)/Np)
    An = math.sqrt(K01*(Kn+K1)/Nn)
    Bp = math.sqrt(K10*(Kp+K0)/Np)
    Bn = math.sqrt(K10*(Kn+K0)/Nn)
    Ep = res_comp(t, 1/Kp)
    En = res_comp(t, 1/Kn)
    # Residue for injection in 0
    r00 = Ap*Ap*Ep + An*An*En
    r10 = Ap*Bp*Ep + An*Bn*En
    r_0 = np.stack((r00, r10), axis=-1)
    # Residue for injection in 1
    r01 = Ap*Bp*Ep + An*Bn*En
    r11 = Bp*Bp*Ep + Bn*Bn*En
    r_1 = np.stack((r01, r11), axis=-1)
    # Residue for the system
    return np.stack((r_0, r_1), axis=-1)

def prop_2comp(T, E, t):
    r = res_2comp(T, E, t)
    K0 = (E[0,0]+E[1,0])/T[0]
    K1 = (E[0,1]+E[1,1])/T[1]
    r[:,0,:] = K0*r[:,0,:]
    r[:,1,:] = K1*r[:,1,:]
    return r


# 2 compartment exchange (analytical)

def conc_2cxm(J, T, E, t=None, dt=1.0):
    """Concentration in a 2-compartment exchange model system.

    E is the scalar extraction fraction E10
    """
    if np.amin(T) <= 0:
        raise ValueError('T must be strictly positive.')
    t = tools.tarray(len(J), t=t, dt=dt)
    K0 = 1/T[0]
    K1 = 1/T[1]
    K10 = E/T[0]
    K01 = 1/T[1]
    Dsq = (K0-K1)**2 + 4*K01*K10
    D = math.sqrt(D)
    Kp = (K0+K1+Dsq)/2
    Kn = (K0+K1-Dsq)/2
    Np = K01*(Kp+K1) + K10*(Kp+K0)
    Nn = K01*(Kn+K1) + K10*(Kn+K0)
    Ap = math.sqrt(K01*(Kp+K1)/Np)
    An = math.sqrt(K01*(Kn+K1)/Nn)
    Bp = math.sqrt(K10*(Kp+K0)/Np)
    Bn = math.sqrt(K10*(Kn+K0)/Nn)
    E0p = conc_comp(J, 1/Kp, t)
    E0n = conc_comp(J, 1/Kn, t)
    C0 = Ap*Ap*E0p + An*An*E0n 
    C1 = Ap*Bp*E0p + An*Bn*E0n 
    return np.stack((C0, C1), axis=-1)

def flux_2cxm(J, T, E, t=None, dt=1.0):
    C = conc_2cxm(J, T, E, t=t, dt=dt)
    t = tools.tarray(len(J), t=t, dt=dt)
    J0 = C[:,0]*(1-E)/T[0]
    return J0

def res_2cxm(T, E, t):
    K0 = 1/T[0]
    K1 = 1/T[1]
    K10 = E/T[0]
    K01 = 1/T[1]
    Dsq = (K0-K1)**2 + 4*K01*K10
    D = math.sqrt(D)
    Kp = (K0+K1+Dsq)/2
    Kn = (K0+K1-Dsq)/2
    Np = K01*(Kp+K1) + K10*(Kp+K0)
    Nn = K01*(Kn+K1) + K10*(Kn+K0)
    Ap = math.sqrt(K01*(Kp+K1)/Np)
    An = math.sqrt(K01*(Kn+K1)/Nn)
    Bp = math.sqrt(K10*(Kp+K0)/Np)
    Bn = math.sqrt(K10*(Kn+K0)/Nn)
    E0p = res_comp(t, 1/Kp)
    E0n = res_comp(t, 1/Kn)
    C0 = Ap*Ap*E0p + An*An*E0n 
    C1 = Ap*Bp*E0p + An*Bn*E0n 
    return np.stack((C0, C1), axis=-1)

def prop_2cxm(T, E, t):
    r = res_2cxm(T, E, t)
    h0 = r[:,0]*(1-E)/T[0]
    return h0


# 2 compartment filtration model


def conc_2cfm(J, T, E, t=None, dt=1.0):
    t = tools.tarray(len(J), t=t, dt=dt)
    C0 = conc_comp(J, T[0], t)
    if E==0:
        C1 = np.zeros(len(t))
    elif T[0]==0:
        C1 = conc_comp(E*J, T[1], t)
    else:
        C1 = conc_comp(C0*E/T[0], T[1], t)
    return np.stack((C0, C1), axis=-1)

def flux_2cfm(J, T, E, t=None, dt=1.0):
    t = tools.tarray(len(J), t=t, dt=dt)
    J0 = flux_comp(J, T[0], t)
    if E==0:
        J1 = np.zeros(len(t))
    else:    
        J1 = flux_comp(E*J0, T[1], t)
    return np.stack(((1-E)*J0, J1), axis=-1)

def res_2cfm(T, E, t):
    C0 = res_comp(T[0], t)
    if E==0:
        C1 = np.zeros(len(t))
    elif T[0]==0:
        C1 = E*res_comp(T[1], t)
    else:
        C1 = conc_comp(C0*E/T[0], T[1], t)
    return np.stack((C0, C1), axis=-1)

def prop_2cfm(T, E, t):
    J0 = prop_comp(T[0], t)
    if E==0:
        J1 = np.zeros(len(t))
    else:    
        J1 = flux_comp(E*J0, T[1], t)
    return np.stack(((1-E)*J0, J1), axis=-1)


# Non-stationary compartment

def conc_nscomp(J, T, t=None, dt=1.0):
    if np.isscalar(T):
        raise ValueError('T must be an array of the same length as J.')
    if len(T) != len(J):
        raise ValueError('T and J must have the same length.')
    if np.amin(T) <= 0:
        raise ValueError('T must be strictly positive.')
    t = tools.tarray(len(J), t=t, dt=dt)
    Dt = t[1:]-t[:-1]
    Tt = (T[1:]+T[:-1])/2
    Jt = (J[1:]+J[:-1])/2
    n = len(t)
    C = np.zeros(n)
    for i in range(n-1):
        # Dt/T <= 1 or Dt <= T
        if Dt[i] <= Tt[i]:
            C[i+1] = C[i] + Dt[i]*Jt[i] - C[i]*Dt[i]/Tt[i]
        else:
            # Dt[i]/nk <= T[i]
            # Dt[i]/T[i] <= nk
            nk = np.ceil(Dt[i]/Tt[i])
            Dk = Dt[i]/nk
            Ck = C[i]
            for _ in range(nk):
                Ck = Ck + Dk*Jt[i]/nk - Ck*Dk/Tt[i]
            C[i+1] = Ck
    return C

def flux_nscomp(J, T, t=None, dt=1.0):
    C = conc_nscomp(J, T, t=t, dt=dt)
    return C/T


if __name__ == "__main__":

    print('All pk tests passed!!')


