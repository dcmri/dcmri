import math
import numpy as np
from scipy.special import gamma

import matplotlib.pyplot as plt

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

def signal_SPGRESS(TR, FA, R1, S0):
    """Signal aof a spoiled gradient echo sequence in the steady state"""
    E = np.exp(-TR*R1)
    cFA = np.cos(FA*math.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def sample(t, S, ts, dts): 
    """Sample the signal"""
    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t > tk) & (t < tk+dts)
        Ss[k] = np.average(S[np.nonzero(tacq)[0]])
    return Ss 

def step_inject(t, weight=70, conc=0.5, dose=0.2, rate=2, start=0):
    """
    Injected flux as a steo function

    weight (kg)
    conc (mmol/mL)
    dose mL/kg
    rate mL/sec

    returns dose injected per unit time (mM/sec)"""

    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > start) & (t < start+duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    return J

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

def conc_space_1d1c(t, Jp, Jn, Kp, Kn):
    """Concentration in a spatial 1-compartment model in 1D"""
    nt = len(Jp)
    nc = len(Kp)
    K = Kp + Kn
    C = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C[k+1,:] = C[k,:]
        # Add influxes at the boundaries:
        C[k+1,0] += dt*Jp[k]
        C[k+1,-1] += dt*Jn[k]
        # Remove outflux to the neigbours:
        C[k+1,:] -= dt*K*C[k,:]
        # Add influx from the neighbours:
        C[k+1,:-1] += dt*Kn[1:]*C[k,1:]
        C[k+1,1:] += dt*Kp[:-1]*C[k,:-1]
    return C

def conc_space_2d1c(t, Jpx, Jnx, Jpy, Jny, Kpx, Knx, Kpy, Kny):
    """Concentration in a spatial 1-compartment model in 2D"""
    nt = len(Jpx)
    nx, ny = Kpx.shape
    C = np.zeros((nt,nx,ny))
    K = Kpx + Knx + Kpy + Kny
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C[k+1,:,:] = C[k,:,:]
        # Add influxes at boundaries
        C[k+1,0,:] += dt*Jpx[k,:]
        C[k+1,-1,:] += dt*Jnx[k,:]
        C[k+1,:,0] += dt*Jpy[k,:]
        C[k+1,:,-1] += dt*Jny[k,:]
        # Remove outflux to the neigbours
        C[k+1,:,:] -= dt*K*C[k,:,:]
        # Add influx from the neighbours
        C[k+1,:-1,:] += dt*Knx[1:,:]*C[k,1:,:]
        C[k+1,1:,:] += dt*Kpx[:-1,:]*C[k,:-1,:]
        C[k+1,:,:-1] += dt*Kny[:,1:]*C[k,:,1:]
        C[k+1,:,1:] += dt*Kpy[:,:-1]*C[k,:,:-1]
    return C

def conc_space_1d2c(t, Jp1, Jn1, Jp2, Jn2, Kp1, Kn1, Kp2, Kn2, K12, K21):
    """Concentration in a spatial 2-compartment model in 1D"""
    nt = len(Jp1)
    nc = len(Kp1)
    K1 = Kp1 + Kn1 + K21
    K2 = Kp2 + Kn2 + K12
    C1 = np.zeros((nt,nc))
    C2 = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:] = C1[k,:]
        C2[k+1,:] = C2[k,:]
        # Add influxes at the boundaries:
        C1[k+1,0] += dt*Jp1[k]
        C1[k+1,-1] += dt*Jn1[k]
        C2[k+1,0] += dt*Jp2[k]
        C2[k+1,-1] += dt*Jn2[k]
        # Remove outflux to the neigbours:
        C1[k+1,:] -= dt*K1*C1[k,:]
        C2[k+1,:] -= dt*K2*C2[k,:]
        # Add influx from the neighbours:
        C1[k+1,:-1] += dt*Kn1[1:]*C1[k,1:]
        C1[k+1,1:] += dt*Kp1[:-1]*C1[k,:-1]
        C2[k+1,:-1] += dt*Kn2[1:]*C2[k,1:]
        C2[k+1,1:] += dt*Kp2[:-1]*C2[k,:-1]
        # Add influx at the same location
        C1[k+1,:,:] += dt*K21*C1[k,:,:]
        C2[k+1,:,:] += dt*K12*C2[k,:,:]
    return C1, C2

def conc_space_2d2c(t, 
            Jp1x, Jn1x, Jp2x, Jn2x, 
            Jp1y, Jn1y, Jp2y, Jn2y, 
            Kp1x, Kn1x, Kp2x, Kn2x, 
            Kp1y, Kn1y, Kp2y, Kn2y, 
            K12, K21):
    """Concentration in a spatial 2-compartment model in 2D"""
    nt = len(Jp1x)
    nx, ny = Kp1x.shape
    C = np.zeros((nt,nx,ny))
    K1 = Kp1x + Kn1x + Kp1y + Kn1y + K21
    K2 = Kp2x + Kn2x + Kp2y + Kn2y + K12
    C1 = np.zeros((nt,nx,ny))
    C2 = np.zeros((nt,nx,ny))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:,:] = C1[k,:,:]
        C2[k+1,:,:] = C2[k,:,:]
        # Add influxes at the boundaries:
        C1[k+1,0,:] += dt*Jp1x[k,:]
        C1[k+1,-1,:] += dt*Jn1x[k,:]
        C1[k+1,:,0] += dt*Jp1y[k,:]
        C1[k+1,:,-1] += dt*Jn1y[k,:]
        C2[k+1,0,:] += dt*Jp2x[k,:]
        C2[k+1,-1,:] += dt*Jn2x[k,:]
        C2[k+1,:,0] += dt*Jp2y[k,:]
        C2[k+1,:,-1] += dt*Jn2y[k,:]
        # Remove outflux to the neigbours:
        C1[k+1,:,:] -= dt*K1*C1[k,:,:]
        C2[k+1,:,:] -= dt*K2*C2[k,:,:]
        # Add influx from the neighbours
        C1[k+1,:-1,:] += dt*Kn1x[1:,:]*C1[k,1:,:]
        C1[k+1,1:,:] += dt*Kp1x[:-1,:]*C1[k,:-1,:]
        C1[k+1,:,:-1] += dt*Kn1y[:,1:]*C1[k,:,1:]
        C1[k+1,:,1:] += dt*Kp1y[:,:-1]*C1[k,:,:-1]
        C2[k+1,:-1,:] += dt*Kn2x[1:,:]*C2[k,1:,:]
        C2[k+1,1:,:] += dt*Kp2x[:-1,:]*C2[k,:-1,:]
        C2[k+1,:,:-1] += dt*Kn2y[:,1:]*C2[k,:,1:]
        C2[k+1,:,1:] += dt*Kp2y[:,:-1]*C2[k,:,:-1]
        # Add influx at the same location
        C1[k+1,:,:] += dt*K21*C1[k,:,:]
        C2[k+1,:,:] += dt*K12*C2[k,:,:]
    return C1, C2


        

######### TESTS ############

tmax = 120 # sec
dt = 0.01 # sec
MTT = 20 # sec

weight = 70.0           # Patient weight in kg
conc = 0.25             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
dose = 0.025            # mL per kg bodyweight (quarter dose)
rate = 1                # Injection rate (mL/sec)
start = 20.0        # sec
dispersion = 90    # %


def test_step_inject():
    t = np.arange(0, tmax, dt)
    J = step_inject(t, weight, conc, dose, rate, start)
    assert np.round(np.sum(J)*dt,1) == np.round(weight*dose*conc, 1)


if __name__ == "__main__":

    test_step_inject()















