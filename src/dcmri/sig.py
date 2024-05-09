from scipy.linalg import expm
import numpy as np
import dcmri as dc

def signal_dsc(R1, R2, S0, TR, TE):
    return S0*np.exp(-np.multiply(TE,R2))*(1-np.exp(-np.multiply(TR,R1)))

def signal_spgress(R1, S0, TR, FA):
    E = np.exp(-np.multiply(TR,R1))
    cFA = np.cos(FA*np.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def signal_spgress_fex(v, R1, S0, TR, FA):
    R1 = np.sum(np.multiply(v,R1))
    return signal_spgress(R1, S0, TR, FA)

def signal_spgress_nex(v, R1, S0, TR, FA):
    if np.size(R1) == np.size(v):
        S = signal_spgress(R1, S0, TR, FA)
        return np.sum(np.multiply(v,S)) 
    # R1 is a numpy array with dimensions (nc,nt)
    nc, nt = R1.shape
    S = np.zeros(nt)
    for c in range(nc):
        S += v[c]*signal_spgress(R1[c,:], S0, TR, FA)
    return S

def signal_spgress_wex(PS, v, R1, S0, TR, FA):
    # Inputs:

    # f1 = fo1 + f21 + R1_1(t)*v1
    # f2 = fo2 + f12 + R1_2(t)*v2
    
    # J1 = fi1*S0*mi1(t) + R1_1(t)*v1*S0*m01
    # J2 = fi2*S0*mi2(t) + R1_2(t)*v2*S0*m02

    # K = [[f1/v1, -f21/v1], [-f12/v2, f2/v2]]
    # J = [J1, J2]

    # K dimensions = (ncomp, ncomp, nt)
    # J dimensions = (ncomp, nt)

    # Returns:

    # M = (1 - cosFA exp(-TR*K))^-1 (1-exp(-TR*K)) K^-1 J
    # M = [(1 - cosFA exp(-TR*K))K]^-1 (1-exp(-TR*K)) J
    nc, nt = R1.shape
    J = np.empty((nc,nt))
    K = np.empty((nc,nc,nt))
    for c in range(nc):
        J[c,:] = S0*v[c]*R1[c,:]
        K[c,c,:] = R1[c,:] + np.sum(PS[:,c])/v[c]
        for d in range(nc):
            if d!=c:
                K[c,d,:] = -PS[c,d]/v[d]

    cFA = np.cos(FA*np.pi/180)
    Mag = np.empty(nt)
    Id = np.eye(nc)
    for t in range(nt):
        Et = expm(-TR*K[:,:,t])
        Mt = np.dot(K[:,:,t], Id-cFA*Et)
        Mt = np.linalg.inv(Mt)
        Vt = np.dot(Id-Et, J[:,t])
        Vt = np.dot(Mt, Vt)
        Mag[t] = np.sum(Vt)
    return Mag


def signal_srspgre(R1, S0, TR, FA, Tsat, TI):
    """Selective FLASH readout with non-selective saturation preparation"""
    if Tsat > TI:
        msg = 'Incorrect sequence parameters.'
        msg += 'Tsat must be smaller than TI.'
        raise ValueError(msg)
    cFA = np.cos(np.pi*FA/180)
    T1_app = TR/(TR*R1-np.log(cFA))

    ER = np.exp(-TR*R1)
    E_sat = np.exp(-Tsat*R1)
    E_center = np.exp(-(TI-Tsat)/T1_app)

    S_sat = S0 * (1-E_sat)
    S_ss = S0 * (1-ER)/(1-cFA*ER)

    return S_ss*(1-E_center) + S_sat*E_center

def signal_eqspgre(R1, S0, TR, FA, TI):
    """Selective FLASH readout from equilibrium (inflow model)"""
    #TI is the residence time in the slab
    cFA = np.cos(np.pi*FA/180)
    R1_app = R1 - np.log(cFA)/TR

    ER = np.exp(-TR*R1)
    EI = np.exp(-TI*R1_app)

    S_ss = S0 * (1-ER)/(1-cFA*ER)

    return S_ss*(1-EI) + S0*EI


def signal_sr(R1, S0, TI):
    """Saturation-recovery sequence ignoring the effect of the readout (for fast flowing magnetization)"""
    E = np.exp(-TI*R1)
    return S0 * (1-E)


def sample(t, S, ts, dts): 
    """Sample the signal assuming sample times are at the start of the acquisition"""

    Ss = np.zeros(len(ts)) 
    for k, tk in enumerate(ts):
        #tacq = (t >= tk) & (t < tk+dts)
        #data = S[np.nonzero(tacq)[0]]
        #Ss[k] = np.average(data)
        data = S[(t >= tk) & (t < tk+dts)]
        if data.size > 0:
            Ss[k] = np.mean(data)
    return Ss 