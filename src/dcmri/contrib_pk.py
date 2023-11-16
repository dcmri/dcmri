"""
Pharmacokinetics
"""

import math
import numpy as np
from scipy.special import gamma

##### HELPER FUNCTIONS ####
###########################
# These are not exposed to the package user and don't need proper docstrings or documentation
# Just enough information in the docstring so that other developers know what it is and how it works.
# They do need solid tests because they underly much other functionality.


def expconv(T, time, a):
    """Convolve a 1D-array with a normalised exponential.

    expconv() uses an efficient and accurate numerical formula to calculate the convolution,
    as detailed in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006.

    Note (1): by definition, expconv preserves the area under a(time)
    Note (2): if T=0, expconv returns a copy of a

    Arguments
    ---------
    a : numpy array
        the 1D array to be convolved.
    time : numpy array
        the time points where the values of ca are defined
        these do not have to to be equally spaced.
    T : float
        the characteristic time of the the exponential function.
        time and T must be in the same units.

    Returns
    -------
    a numpy array of the same shape as ca.

    Example
    -------
    coming soon..

    """
    if T==0: 
        return a
    n = len(a)  
    f = np.zeros(n)
    dt = time[1:n] - time[0:n-1]
    x = dt/T
    da = (a[1:n] - a[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = a[0:n-1]*E0 + da*E1
    for i in range(0,n-1):
        f[i+1] = E[i]*f[i] + add[i]      
    return f

def trapz(t, f):
    """Trapezoidal integration. 
    
    Check if can be replaced byt numpy function."""
    n = len(f)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + (t[i+1]-t[i]) * (f[i+1]+f[i]) / 2
    return g

def utrapz(dt, f):
    """Helper functiom not to be exposed in the package interface.
    
    Performs trapezoidal integration over an equally space time array.

    Check if can be replaced by numpy function.
    """
    n = len(f)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + dt * (f[i+1]+f[i]) / 2
    return g

def uconv(dt, f, h):
    """Helper functiom not to be exposed in the package interface.

    Performs convolution of two arrays both defined over teh same uniform time grid
    """
    n = len(f) 
    g = np.empty(n)
    h = np.flip(h)
    g[0] = 0
    for i in np.arange(1, n):
        g[i] = np.trapz(f[:i+1]*h[-(i+1):], dx=dt)
    return g

def conc_scomp(t, J, K, C0=0):
    """Build the resiude for a stationary compartment"""
    if K == 0:
        return trapz(t, J)
    T = 1/K
    return C0 + T*expconv(T, t, J)

def conc_nscomp(t, J, K, C0=0):
    """This builds the reside for a non-stationary (ns) compartment where K is not a fixed number by an array of the same length as t and J.

    A test could consist of running it with an array constant K and compare to res_comp
    """
    #dtK must be <=1 everywhere
    #dC(t)/dt = -K(t)C(t) + J(t)
    #C(t+dt)-C(t) = -dtK(t)C(t) + dtJ(t)
    #C(t+dt) = C(t) - dtK(t)C(t) + dtJ(t)
    #C(t+dt) = (1-dtK(t))C(t) + dtJ(t)
    n = len(t)
    if len(J) != n:
        msg = 'The flux must be the same length as the array of time points'
        raise ValueError(msg)
    if len(K) != n:
        msg = 'The K-array must be the same length as the array of time points'
        raise ValueError(msg)
    C = np.zeros(n)
    C[0] = C0
    for i in range(n-1):
        dt = t[i+1]-t[i]
        R = 1-dt*K[i]
        C[i+1] = R*C[i] + dt*J[i] 
    return C




######### PACKAGE INTERFACE ##########
# These functions are exposed to the package user and need to be fully covered by tests, documentation and examples.

def conc_trap(t, J):
    return trapz(t, J)

def flux_trap(t, J):
    return J*0

def conc_comp(t, J, K, C0=0):
    """Concentration in a compartment"""
    if np.isscalar(K):
        return conc_scomp(t, J, K, C0=C0)
    else:
        return conc_nscomp(t, J, K, C0=C0)
    
def flux_comp(t, J, K, C0=0):
    """FLux out of a compartment"""
    return K*conc_comp(t, J, K, C0=C0)

def res_comp(t, K):
    """Residue function of a compartment"""
    return np.exp(-t*K)

def prop_comp(t, K):
    """Propagator of a compartment"""
    return K*np.exp(-t*K)

def conc_2cfm(t, J, K01, K21, K02):
    """Concentration in a 2-compartment filtration model"""
    K1 = K01 + K21
    C1 = conc_comp(t, J, K1)
    C2 = conc_comp(t, K21*C1, K02)
    return C1, C2

def flux_2cfm(t, J, K01, K21, K02):
    """Flux out of a 2-compartment filtration model"""
    C1, C2 = conc_2cfm(t, J, K01, K21, K02)
    return K01*C1, K02*C2

def flux_plug(t, J, T):
    """Flux out of a plug flow system"""
    if T==np.inf:
        return np.zeros(len(t))
    return np.interp(t-T, t, J, left=0) 

def conc_plug(t, J, T):
    """Concentration inside a plug flow system"""
    if T==np.inf:
        return trapz(t, J)
    Jo = flux_plug(t, J, T)
    return trapz(t, J-Jo)

def res_plug(t, T):
    """Residue function of a plug flow system"""
    g = np.ones(len(t))
    g[np.where(t>T)] = 0
    return g

def conc_chain(dt, J, T, D):
    """Concentration inside a chain"""
    t = dt*np.arange(len(J))
    R = res_chain(t, T, D)
    return uconv(dt, R, J)

def flux_chain(dt, J, T, D):
    """Flux out of a chain"""
    C = conc_chain(dt, J, T, D)
    return J - np.gradient(C, dt)

def res_chain(t, MTT, disp):
    """Residue function of a chain"""
    if disp==0: # plug flow
        g = np.ones(len(t))
        return res_plug(t, MTT)
    if disp==100: # compartment
        return res_comp(t, MTT)
    n = 100/disp
    Tx = MTT/n
    norm = Tx*gamma(n)
    if norm == np.inf:
        return res_plug(t, MTT)
    u = t/Tx  
    nt = len(t)
    g = np.ones(nt)
    g[0] = 0
    fnext = u[0]**(n-1)*np.exp(-u[0])/norm
    for i in range(nt-1):
        fi = fnext
        pow = u[i+1]**(n-1)
        if pow == np.inf:
            return 1-g
        fnext = pow * np.exp(-u[i+1])/norm
        g[i+1] = g[i] + (t[i+1]-t[i]) * (fnext+fi) / 2
    return 1-g

def prop_chain(t, MTT, disp): # dispersion in %
    """Propagator through a chain"""
    n = 100/disp
    Tx = MTT/n
    u = t/Tx
    return u**(n-1) * np.exp(-u)/Tx/gamma(n)

# Everything below here is first draft - not in use and needs testing

def conc_2comp(t, J1, J2, K01, K02, K21, K12):
    """Concentration in a general 2-compartment system
    
    Needs testing and debugging
    """
    K1 = K01 + K21
    K2 = K12 + K02
    Dsq = (K1-K2)**2 + 4*K12*K21
    D = math.sqrt(D)
    Kp = (K1+K2+Dsq)/2
    Kn = (K1+K2-Dsq)/2
    Np = K12*(Kp+K2) + K21*(Kp+K1)
    Nn = K12*(Kn+K2) + K21*(Kn+K1)
    Ap = math.sqrt(K12*(Kp+K2)/Np)
    An = math.sqrt(K12*(Kn+K2)/Nn)
    Bp = math.sqrt(K21*(Kp+K1)/Np)
    Bn = math.sqrt(K21*(Kn+K1)/Nn)
    E1p = conc_comp(t, J1, Kp)
    E1n = conc_comp(t, J1, Kn)
    E2p = conc_comp(t, J2, Kp)
    E2n = conc_comp(t, J2, Kn)
    C1 = Ap*Ap*E1p + An*An*E1n + Ap*Bp*E2p + An*Bn*E2n
    C2 = Ap*Bp*E1p + An*Bn*E1n + Bp*Bp*E2p + Bn*Bn*E2n
    return C1, C2

def flux_2comp(t, J1, J2, K01, K02, K21, K12):
    """Concentration in a general 2-compartment system
    
    Needs testing and debugging
    """
    C1, C2 = conc_2comp(t, J1, J2, K01, K02, K21, K12)
    return K01*C1, K02*C2

def conc_2cxm(t, J1, K01, K21, K12):
    """Concentration in a 2-compartment exchange model
    
    Needs testing and debugging
    """
    K1 = K01 + K21
    K2 = K12
    Dsq = (K1-K2)**2 + 4*K12*K21
    D = math.sqrt(D)
    Kp = (K1+K2+Dsq)/2
    Kn = (K1+K2-Dsq)/2
    Np = K12*(Kp+K2) + K21*(Kp+K1)
    Nn = K12*(Kn+K2) + K21*(Kn+K1)
    Ap = math.sqrt(K12*(Kp+K2)/Np)
    An = math.sqrt(K12*(Kn+K2)/Nn)
    Bp = math.sqrt(K21*(Kp+K1)/Np)
    Bn = math.sqrt(K21*(Kn+K1)/Nn)
    E1p = conc_comp(t, J1, Kp)
    E1n = conc_comp(t, J1, Kn)
    C1 = Ap*Ap*E1p + An*An*E1n
    C2 = Ap*Bp*E1p + An*Bn*E1n
    return C1, C2

def flux_2cxm(t, J1, K01, K21, K12):
    """Flux out of a 2-compartment exchange model
    
    Needs testing and debugging
    """
    C1, _ = conc_2cxm(t, J1, K01, K21, K12)
    return K01*C1

def res_2cxm(t, K01, K21, K12):
    """Residue function of a 2-compartment exchange model
    
    Needs testing and debugging
    """
    K1 = K01 + K21
    K2 = K12
    Dsq = (K1-K2)**2 + 4*K12*K21
    D = math.sqrt(D)
    Kp = (K1+K2+Dsq)/2
    Kn = (K1+K2-Dsq)/2
    Np = K12*(Kp+K2) + K21*(Kp+K1)
    Nn = K12*(Kn+K2) + K21*(Kn+K1)
    Ap = math.sqrt(K12*(Kp+K2)/Np)
    An = math.sqrt(K12*(Kn+K2)/Nn)
    Bp = math.sqrt(K21*(Kp+K1)/Np)
    Bn = math.sqrt(K21*(Kn+K1)/Nn)
    E1p = res_comp(t, Kp)
    E1n = res_comp(t, Kn)
    R1 = Ap*Ap*E1p + An*An*E1n
    R2 = Ap*Bp*E1p + An*Bn*E1n
    return R1, R2

def conc_ncomp(t, J, K):
    """Concentration in a general n-compartment model.

    K is the nxn system matrix with K[j,i] = Kji and K[i,i] = K0i
    """
    # dtK must be <= 1 everywhere

    # Build Kmatrix from system matrix
    nc = K.shape[0]
    Kmat = np.zeros((nc,nc))
    for i in range(nc):
        for j in range(nc):
            if j==i:
                Kmat[j,i] = np.sum(K[:,i])
            else:
                Kmat[j,i] = -K[j,i]

    # Propagate concentrations
    nt = len(t)
    C = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        C[k+1,:] = C[k,:] + dt*J[k,:] - dt*np.matmul(Kmat, C[k,:])  
    return C

def flux_ncomp(t, J, K):
    """Flux out of a general n-compartment model.

    K is the nxn system matrix with K[j,i] = Kji and K[i,i] = K0i
    """
    C = conc_ncomp(t, J, K)
    nc = K.shape[0]
    for i in range(nc):
        C[:,i] = K[i,i]*C[:,i]
    return C


