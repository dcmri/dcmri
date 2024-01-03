import math
import numpy as np
from scipy.special import gamma
from scipy.stats import rv_histogram

import dcmri.tools as tools

# Trap

def conc_trap(J, t=None, dt=1.0):
    return tools.trapz(J, t=t, dt=dt)

def flux_trap(J, t=None, dt=1.0):
    n = len(J)
    return np.zeros(n)

def res_trap(t):
    return np.ones(len(t))

def prop_trap(t):
    return np.zeros(len(t))

# Pass (no dispersion)

def conc_pass(J, T, t=None, dt=1.0):
    return T*np.array(J)

def flux_pass(J, T=None, t=None, dt=1.0):
    return np.array(J)

def res_pass(T, t):
    return T*tools.res_ddelta(t)

def prop_pass(t):
    return tools.prop_ddelta(t)

# Compartment

def conc_comp(J, T, t=None, dt=1.0):
    if T == np.inf:
        return conc_trap(J, t=t, dt=dt)
    return T*tools.expconv(J, T, t=t, dt=dt)

def flux_comp(J, T, t=None, dt=1.0):
    if T == np.inf:
        return flux_trap(J, t=t, dt=dt)
    return tools.expconv(J, T, t=t, dt=dt)

def res_comp(T, t):
    if T == np.inf:
        return res_trap(t)
    if T == 0:
        return tools.res_ddelta(t)
    return np.exp(-t/T)

def prop_comp(T, t):
    if T == np.inf:
        return prop_trap(t)
    if T == 0:
        return tools.prop_ddelta(t)
    return np.exp(-t/T)/T

# Plug flow

def conc_plug(J, T, t=None, dt=1.0):
    if T==np.inf:
        return conc_trap(J, t=t, dt=dt)
    if T==0:
        return 0*J
    t = tools.tarray(J, t=t, dt=dt)
    Jo = np.interp(t-T, t, J, left=0)
    return tools.trapz(t, J-Jo)

def flux_plug(J, T, t=None, dt=1.0):
    if T==np.inf:
        return flux_trap(J, t=t, dt=dt)
    if T==0:
        return J
    t = tools.tarray(J, t=t, dt=dt)
    return np.interp(t-T, t, J, left=0) 

def res_plug(T, t):
    g = np.ones(len(t))
    g[np.where(t>T)] = 0
    return g

def prop_plug(T, t):
    h = tools.prop_ddelta(t)
    return np.interp(t-T, t, h, left=0)

# Chain

def conc_chain(J, T, D, t=None, dt=1.0):
    if D == 0:
        return conc_plug(J, T, t=t, dt=dt)
    if D == 100:
        return conc_comp(J, T, t=t, dt=dt)
    t = tools.tarray(J, t=t, dt=dt)
    r = res_chain(T, D, t)
    return tools.conv(r, J, t)

def flux_chain(J, T, D, t=None, dt=1.0):
    if D == 0:
        return prop_plug(J, T, t=t, dt=dt)
    if D == 100:
        return prop_comp(J, T, t=t, dt=dt)
    t = tools.tarray(J, t=t, dt=dt)
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
    Tx = T/n
    u = t/Tx
    g = u**(n-1) * np.exp(-u)/Tx/gamma(n)
    gfin = np.isfinite(g)
    n_notfin = g.size - np.count_nonzero(gfin)
    if n_notfin > 0:
        raise ValueError('A chain model is not numerically stable with these parameters. Consider restricting the minimal dispersion allowed.')
    return g

# Free

def conc_free(J, H, t=None, dt=1.0, TT=None, TTmin=0, TTmax=None):
    u = tools.tarray(J, t=t, dt=dt)
    r = res_free(H, u, TT=TT, TTmin=TTmin, TTmax=TTmax)
    return tools.conv(r, J, t=t, dt=dt)

def flux_free(J, H, t=None, dt=1.0, TT=None, TTmin=0, TTmax=None):
    u = tools.tarray(J, t=t, dt=dt)
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
    t = tools.tarray(J[:,0], t=t, dt=dt)
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
    t = tools.tarray(J[:,0], t=t, dt=dt)
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
    t = tools.tarray(J[:,0], t=t, dt=dt)
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
    t = tools.tarray(J[:,0], t=t, dt=dt)
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
    t = tools.tarray(J, t=t, dt=dt)
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
    t = tools.tarray(J, t=t, dt=dt)
    J0 = C[:,0]*(1-E)/T[0]
    return J0

def res_2cxm(T, E, t):
    """Concentration in a 2-compartment exchange model system.

    E is the scalar extraction fraction E10
    """
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
    t = tools.tarray(J, t=t, dt=dt)
    C0 = conc_comp(J, T[0], t)
    if E==0:
        C1 = np.zeros(len(t))
    elif T[0]==0:
        C1 = conc_comp(E*J, T[1], t)
    else:
        C1 = conc_comp(C0*E/T[0], T[1], t)
    return np.stack((C0, C1), axis=-1)

def flux_2cfm(J, T, E, t=None, dt=1.0):
    t = tools.tarray(J, t=t, dt=dt)
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
    t = tools.tarray(J, t=t, dt=dt)
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


