import numpy as np
from scipy.linalg import expm


# Quantities in this module do not have a time index
# These are internal helper functions not exposed to pacakage users




# Steady-state magnetization of a preparation-recovery SPGR
def Mz_pr_spgr_ss(R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
    if np.isscalar(v):
        return _Mz_pr_spgr_ss_1c(R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA)
    elif np.array(v).size==1:
        Mz = _Mz_pr_spgr_ss_1c(R1[0], v[0], Fw[0,0], j[0], me, PA, TP, TC, TR, FA, TA)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_pr_spgr_ss_nc(R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA)


def _Mz_pr_spgr_ss_1c(R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
    # # Derivation

    # # Preparation pulse
    # M = M * cPA

    # # Free recovery for time TP
    # M = EP * M + (1 - EP) * KinvJ

    # # FA-pulses until end of readout
    # M = Mss + nFA * En * (M - Mss)

    # # Free recovery again until the end
    # M = ED * M + (1 - ED) * KinvJ

    # # Put it all together in one line
    # M = ED * (Mss + nFA * En * ((EP * (M * cPA) + (1 - EP) * KinvJ) - Mss)) + (1 - ED) * KinvJ

    # # Expand brackets in steps
    # M = ED * (Mss + nFA * En * (EP * M * cPA + (1 - EP) * KinvJ - Mss)) + (1 - ED) * KinvJ

    # M = ED * (Mss + nFA * En * EP * M * cPA + nFA * En * (1 - EP) * KinvJ - nFA * En * Mss) + (1 - ED) * KinvJ

    # M = ED * Mss + ED * nFA * En * EP * M * cPA + ED * nFA * En * (1 - EP) * KinvJ - ED * nFA * En * Mss + (1 - ED) * KinvJ

    # # Move M-terms to the left
    # M - ED * nFA * En * EP * M * cPA = ED * Mss + ED * nFA * En * (1 - EP) * KinvJ - ED * nFA * En * Mss + (1 - ED) * KinvJ

    # # Group M, Mss, KinvJ terms
    # (1 - ED * nFA * En * EP  * cPA) * M = (ED - ED * nFA * En) * Mss + (ED * nFA * En * (1 - EP) + (1 - ED)) * KinvJ

    # # Write as A * M = B * Mss + C * KinvJ
    # A = 1 - ED * nFA * En * EP  * cPA
    # B = ED - ED * nFA * En
    # C = ED * nFA * En * (1 - EP) + (1 - ED)

    # Steady-state for a preparation-recovery SPGR
    # Rate constants, influx and SPGR steady-state
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Timings
    T_read = 2 * (TC - TP) # Time for readout 
    TD = TA - TP - T_read # remaining free recovery time

    # Pulses and flip angle cosines
    n = np.floor(T_read / TR) # n pulses for readout
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n
    
    # Exponential factors
    En = np.exp(-T_read * K)
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    # Precompute W = nFA * ED * En
    W = nFA * ED * En

    # Compute A, B, C
    A = 1 - cPA * W * EP
    B = ED - W
    C = W * (1 - EP) + (1 - ED)

    # Solution A @ M = B + C @ KinvJ
    RH = B * Mss + C * KinvJ
    M = (1 / A) * RH if A != 0 else 0

    return M

def _Mz_pr_spgr_ss_nc(R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
    # # Derivation

    # # Preparation pulse
    # M = M * cPA

    # # Free recovery for time TP
    # M = EP @ M + (Id - EP) @ KinvJ

    # # FA-pulses until end of the readout
    # M = Mss + nFA * En @ (M - Mss)

    # # Free recovery again until the end
    # M = ED @ M + (Id - ED) @ KinvJ

    # # Put it all together in one line
    # M = ED @ (Mss + nFA * En @ ((EP @ (M * cPA) + (Id - EP) @ KinvJ) - Mss)) + (Id - ED) @ KinvJ

    # # Expand brackets steps
    # M = ED @ (Mss + nFA * En @ ((cPA * EP @ M  + (Id - EP) @ KinvJ) - Mss)) + (Id - ED) @ KinvJ

    # M = ED @ (Mss + nFA * En @ (cPA * EP @ M  + (Id - EP) @ KinvJ - Mss)) + (Id - ED) @ KinvJ

    # M = ED @ (Mss + nFA * cPA * En @ EP @ M + nFA * En @ (Id - EP) @ KinvJ - nFA * En @ Mss) + (Id - ED) @ KinvJ

    # M = ED @ Mss + nFA * cPA * ED @ En @ EP @ M + nFA * ED @ En @ (Id - EP) @ KinvJ - nFA * ED @ En @ Mss + (Id - ED) @ KinvJ

    # # Move M-terms to the left hand side
    # M - nFA * cPA * ED @ En @ EP @ M = ED @ Mss + nFA * ED @ En @ (Id - EP) @ KinvJ - nFA * ED @ En @ Mss + (Id - ED) @ KinvJ

    # # Groupt terms in M, Mss and KinvJ
    # (Id - nFA * cPA * ED @ En @ EP) @ M = (ED - nFA * ED @ En) @ Mss + (nFA * ED @ En @ (Id - EP) + (Id - ED)) @ KinvJ

    # # Write as A @ M = B @ Mss + C @ KinvJ
    # A = Id - nFA * cPA * ED @ En @ EP
    # B = ED - nFA * ED @ En
    # C = nFA * ED @ En @ (Id - EP) + (Id - ED)

    nc = R1.size
    Id = np.eye(nc)
    # Steady-state for a preparation-recovery SPGR
    # Rate constants, influx and SPGR steady-state
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Timings
    T_read = 2 * (TC - TP) # Time for readout 
    TD = TA - TP - T_read # remaining free recovery time

    # Pulses and flip angle cosines
    n = np.floor(T_read / TR) # n pulses for readout
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n
    
    # Exponential factors
    En = expm(-T_read * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    # Precompute W = nFA * ED @ En
    W = nFA * ED @ En

    # Compute A, B, C
    A = Id - cPA * W @ EP
    B = ED - W
    C = W @ (Id - EP) + (Id - ED)
    
    # Solution A @ M = B + C @ KinvJ
    RH = B @ Mss + C @ KinvJ
    M = np.linalg.solve(A, RH)

    return M.reshape(R1.shape)


# Magnetization of a preparation-recovery SPGR
def Mz_pr_spgr_prop(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
    if np.isscalar(v):
        return _Mz_pr_spgr_prop_1c(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA)
    elif v.size==1:
        M_sig, Mz = _Mz_pr_spgr_prop_1c(M[0], R1[0], v[0], Fw[0,0], j[0], me, PA, TP, TC, TR, FA, TA)
        return np.array(M_sig).reshape(R1.shape), np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_pr_spgr_prop_nc(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA)
    

def _Mz_pr_spgr_prop_1c(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
    # Get some constants
    cPA = np.cos(np.radians(PA))
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Preparation pulse
    M = M * cPA

    # Free recovery for time TP
    if TP > 0: 
        EP = np.exp(-TP * K)
        M = EP * M + (1 - EP) * KinvJ

    # FA-pulses until time TC to get the readout
    n = np.floor((TC - TP) / TR) # n pulses for half of the matrix
    En = np.exp(-(TC - TP) * K)
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n
    M_sig = Mss + nFA * En * (M - Mss)
    
    # Now FA pulses for the other half
    M = Mss + nFA * En * (M_sig - Mss)

    # Free recovery again until the end
    TD = TA - TP - 2 * (TC - TP)
    if TD > 0: 
        ED = np.exp(-TD * K)
        M = ED * M + (1 - ED) * KinvJ

    return M_sig, M


def _Mz_pr_spgr_prop_nc(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):

    cPA = np.cos(np.radians(PA))
    # Get some constants
    nc = R1.size
    Id = np.eye(nc)
    
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Preparation pulse
    M = M * cPA

    # Free recovery for time TP
    if TP > 0:
        EP = expm(-TP * K)
        M = EP @ M + (Id - EP) @ KinvJ

    # FA-pulses until time TC to get the readout
    n = np.floor((TC - TP) / TR) # n pulses for half of the matrix
    En = expm(-(TC - TP) * K)
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n
    M_sig = Mss + nFA * En @ (M - Mss) 
    
    # Now FA pulses for the other half
    M = Mss + nFA * En @ (M_sig - Mss)

    # Free recovery again until the end
    TD = TA - TP - 2 * (TC - TP)
    if TD > 0: 
        ED = expm(-TD * K)
        M = ED @ M + (Id - ED) @ KinvJ

    return M_sig.reshape(R1.shape), M.reshape(R1.shape)


# Steady state of a random pulse sequence
def Mz_ss(R1, v, Fw, j, me, seq):
    if np.isscalar(v):
        return _Mz_ss_1c(R1, v, Fw, j, me, seq)
    elif v.size==1:
        Mz = _Mz_ss_1c(R1[0], v[0], Fw[0,0], j[0], me, seq)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_nc(R1, v, Fw, j, me, seq)


def _Mz_ss_1c(R1, v, Fw, j, me, seq):
    # # Derivation
    # M0 = 1 # M before pulse 0

    # # Apply pulse 0
    # i = 0
    # FA = seq[i][0]
    # c0 = np.cos(np.radians(FA))
    # M = c0 * M0

    # # Free recovery for time TP
    # TR = seq[i][1]
    # E0 = np.exp(-TR * K)
    # F0 = 1 - E0
    # M = E0 * M + F0 * KinvJ

    # # Put together
    # M1 = c0 * E0 * M0 + F0 * KinvJ

    # # Notation
    # Ci = ci * Ei

    # M1 = C0 * M0 + F0 * KinvJ

    # # Apply pulse 1
    # M2 = C1 * M1 + F1 * KinvJ

    # # Insert previous result
    # M2 = C1 * (C0 * M0 + F0 * KinvJ) + F1 * KinvJ

    # # Expand brackets
    # M2 = C1 * C0 * M0 + (C1 * F0 + F1) * KinvJ

    # # One more to see the pattern
    # M3 = C2 * M2 + F2 * KinvJ

    # # Insert M2
    # M3 = C2 * (C1 * C0 * M0 + (C1 * F0 + F1) * KinvJ) + F2 * KinvJ
    
    # # Simplify 
    # M3 = C2 * C1 * C0 * M0 + (C2 * (C1 * F0 + F1) + F2) * KinvJ

    # # Steady-state:
    # (1 - C2 * C1 * C0) * M = (C2 * (C1 * F0 + F1) + F2) * KinvJ

    # # Write as A * M = B * KinvJ
    # A = 1 - C2 * C1 * C0
    # B = C2 * (C1 * F0 + F1) + F2

    # Rate constants and influx
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)

    # Compute parameters
    C, F = [], []
    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = np.exp(-TR * K)
        C.append(cFA * E)
        F.append(1 - E)

    # Compute sequences
    seq = F[0]
    prod = C[0]
    for i in range(1, len(F)):
        seq = C[i] * seq + F[i]
        prod = C[i] * prod
        
    A, B = 1 - prod, seq

    M = (1 / A) * B * KinvJ if A !=0 else 0

    return M


def _Mz_ss_nc(R1, v, Fw, j, me, seq):

    nc = R1.size
    Id = np.eye(nc)
    # Rate constants and influx
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)

    # Compute parameters
    C, F = [], []
    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = expm(-TR * K)
        C.append(cFA * E)
        F.append(Id - E)

    # Compute sequences
    seq = F[0]
    prod = C[0]
    for i in range(1, len(F)):
        seq = C[i] @ seq + F[i]
        prod = C[i] @ prod
            
    A, B = Id - prod, seq

    M = np.linalg.solve(A, B @ KinvJ)

    return np.array(M).reshape(R1.shape)


# Propagator through an arbitrary sequence of pulses
def Mz_prop(M, R1, v, Fw, j, me, seq):
    if np.isscalar(v):
        return _Mz_prop_1c(M, R1, v, Fw, j, me, seq)
    elif v.size==1:
        Mz = _Mz_prop_1c(M[0], R1[0], v[0], Fw[0,0], j[0], me, seq)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_prop_nc(M, R1, v, Fw, j, me, seq)
    
def _Mz_prop_1c(M, R1, v, Fw, j, me, seq):
    # Rate constants and influx
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)

    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = np.exp(-TR * K)
        M = cFA * M
        M = E * M + (1 - E) * KinvJ

    return M

def _Mz_prop_nc(M, R1, v, Fw, j, me, seq):

    nc = R1.size
    Id = np.eye(nc)
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)

    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = expm(-TR * K) # Not efficient if all TR's are the same
        M = cFA * M
        M = E @ M + (Id - E) @ KinvJ

    return np.array(M).reshape(R1.shape)


def Mz_ss_spgr(R1, v, Fw, j, me, TR, FA):
    if np.isscalar(v):
        return _Mz_ss_spgr_1c(R1, v, Fw, j, me, TR, FA)
    elif v.size==1:
        Mz = _Mz_ss_spgr_1c(R1[0], v[0], Fw[0,0], j[0], me, TR, FA)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_spgr_nc(R1, v, Fw, j, me, TR, FA)


def _Mz_ss_spgr_nc(R1, v, Fw, j, me, TR, FA):
    off_diag = ~np.eye(Fw.shape[0], dtype=bool)
    PSw = Fw[off_diag]

    if np.all(PSw == 0):
        return _Mz_ss_nex(R1, v, Fw, j, me, TR, FA)
    
    elif np.all(np.isinf(PSw)):
        return _Mz_ss_fex(R1, v, Fw, j, me, TR, FA)
    
    elif 0 < np.count_nonzero(np.isinf(PSw)):
        raise NotImplementedError(
            'Water exchange with some (but not all) infinite PS '
            'values is currently not implemented.')
    else:
        return _Mz_ss_aex(R1, v, Fw, j, me, TR, FA)


def _Mz_ss_fex(R1, v, Fw, j, me, TR, FA):
    R1fex = np.sum(v * R1) / np.sum(v)
    fo = np.diag(Fw)
    M = _Mz_ss_spgr_1c(R1fex, np.sum(v), np.sum(fo), np.sum(j), me, TR, FA)
    Mc = [M * vc / np.sum(v) for vc in v]
    return np.stack(Mc).reshape(R1.shape)

def _Mz_ss_nex(R1, v, Fw, j, me, TR, FA):
    nc = v.size
    fo = np.diag(Fw)
    Mc = [_Mz_ss_spgr_1c(R1[c], v[c], fo[c], j[c], me, TR, FA) for c in range(nc)]
    return np.stack(Mc).reshape(R1.shape)

def _Mz_ss_aex(R1, v, Fw, j, me, TR, FA):
    K, J = Mz_KJ(R1, v, Fw, j, me)
    E = expm(-TR * K)
    I = np.eye(R1.size)
    cFA = np.cos(np.radians(FA))
    A = K @ (I - cFA * E)
    try:
        Mz = np.linalg.solve(A, (I - E) @ J)  
    except:
        Mz = np.zeros_like(R1)
    return Mz.reshape(R1.shape)

def _Mz_ss_spgr_1c(R1, v, Fw, j, me, TR, FA):
    K, KinvJ = Mz_KinvJ(R1, v, Fw, j, me)
    E = np.exp(-TR * K)
    cFA = np.cos(FA*np.pi/180)
    n = (1-E) / (1-cFA*E)
    return n * KinvJ


def Mz_KinvJ(R1, v, Fw, j, me):
    nc = np.array(v).size
    K, J = Mz_KJ(R1, v, Fw, j, me)

    if nc==1:
        Kinv = np.divide(1.0, K, out=np.zeros_like(K, dtype=float), where=K != 0)
        KinvJ = Kinv * J    
    else:
        KinvJ = np.linalg.solve(K, J)
    return K, KinvJ


def Mz_KJ(R1, v, Fw, j, me):
    K = Mz_K(R1, v, Fw)
    J = (R1 * v + j) * me
    return K, J


def Mz_K(R1, v, Fw):
    nc = np.array(v).size

    # Case 1: Single Compartment
    if nc==1:
        return R1 + Fw / v

    # Case 2: Multi-Compartment
    # Off-diagonal elements: -Fw[row, col] / v[col]
    K = -Fw / v

    # Diagonal elements: R1[i] + (Sum of water leaving i) / v[i]
    # Summing axis=0 gives the total flow out of each compartment (the columns)
    total_outflow = np.sum(Fw, axis=0)
    diag_elements = R1 + total_outflow / v
    
    # Overwrite the diagonal of our K matrix
    np.fill_diagonal(K, diag_elements)
    
    return K