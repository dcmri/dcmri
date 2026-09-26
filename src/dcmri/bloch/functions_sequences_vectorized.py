import numpy as np
from scipy.linalg import expm

def channels(sequence):
    return 2 if sequence in ['Eq-DE-EPI', '2D-DE-EPI', '3D-DE-EPI'] else 1

def pulse_readout(sequence, p):
    if sequence in [
            'ZTE-3D-SPGR-SS',
            '3D-SPGR-SS',
            '2D-SPGR-SS',
            '3D-SPGR',
            '2D-SPGR',
            '3D-SPGR-SSI',
        ]:
        return p['Nk0']
    
    elif sequence in [
            'ZTE-3D-IR-SPGR-SS',
            '3D-IR-SPGR-SS',
            '3D-SR-SPGR-SS',
            '3D-PR-SPGR-SS',
            '3D-IR-SPGR',
            '3D-SR-SPGR',
            '3D-PR-SPGR',
            '2D-SR-SPGR',
        ]:
        return 1 + p['Nk0']

    else:
        return 0

def readout_time(sequence, p):
    if sequence in [
            'ZTE-3D-SPGR-SS',
            '3D-SPGR-SS',
            '2D-SPGR-SS',
            '3D-SPGR',
            '2D-SPGR',
            '3D-SPGR-SSI',
        ]:
        return p['Nk0'] * p['TR']
    
    elif sequence in [
            'ZTE-3D-IR-SPGR-SS',
            '3D-IR-SPGR-SS',
            '3D-SR-SPGR-SS',
            '3D-PR-SPGR-SS',
            '3D-IR-SPGR',
            '3D-SR-SPGR',
            '3D-PR-SPGR',
            '2D-SR-SPGR',
        ]:
        return p['TP'] + p['Nk0'] * p['TR']

    else:
        return 0

def repetition_time(sequence, p):
    if sequence in [
        'ZTE-3D-SPGR-SS',
        '3D-SPGR-SS',
        '2D-SPGR-SS',
        '3D-SPGR',
        '2D-SPGR',
        '3D-SPGR-SSI',
    ]:
        return p['Nph'] * p['TR']

    elif sequence in [
        'ZTE-3D-IR-SPGR-SS',
        '3D-IR-SPGR-SS',
        '3D-SR-SPGR-SS',
        '3D-PR-SPGR-SS',
        '3D-IR-SPGR',
        '3D-SR-SPGR',
        '3D-PR-SPGR',
        '2D-SR-SPGR',
    ]:
        return p['TP'] + p['Nph'] * p['TR'] + p['TD']

    elif sequence in [
        '2D-GE-EPI',
        '2D-SE-EPI',
        '2D-DE-EPI',
        '3D-GE-EPI',
        '3D-SE-EPI',
        '3D-DE-EPI',
    ]:
        return p['TR']
    
    return p['TA']


def acquisition_times(sequence, p, tacq):
    TR = repetition_time(sequence, p)
    t0 = readout_time(sequence, p)
    nt = np.floor(tacq / TR)
    return t0 + TR * np.arange(nt)


def mz_readout(Mz: np.ndarray, R2: np.ndarray, FA, TE):
    # Shapes for Mz, R2: (nc, nt)
    # Other parameters are scalar
    # returns shape (nc, nt,)
    sFA = np.sin(np.radians(FA))
    decay = np.exp(-TE * R2)
    Mxy = decay * sFA * Mz
    return Mxy

    # Mxy = np.sum(Mxy, axis=0) # sum over compartments
    # signal = S0 * np.abs(Mxy)
    # return signal_rice(signal, noise_sdev)
    

def mz_readout_wrapper(Mz, p, seq, R2=None, R2s=None):
    nc, nt = Mz.shape
    FA = p['FA'] * p['B1corr']
    
    if seq in ['2D-SE-EPI', '3D-SE-EPI']:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float) # (channels, components, compartments, times)
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2[c, :], FA, p['TE'])
    
    elif seq in ['2D-DE-EPI', '3D-DE-EPI']:
        Mxy = np.zeros((2, 2, nc, nt), dtype=float)
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, p['TE1'])
            Mxy[1, 0, c, :] = mz_readout(Mz[c, :], R2[c, :], FA, p['TE2'])

    elif seq in ['ZTE-3D-SPGR-SS', 'ZTE-3D-IR-SPGR-SS']:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float)
        R2s = np.zeros(nt)
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, 0)
    
    else:
        Mxy = np.zeros((1, 2, nc, nt), dtype=float) 
        for c in range(nc):
            Mxy[0, 0, c, :] = mz_readout(Mz[c, :], R2s, FA, p['TE'])  

    return Mxy


# ###################
# STEADY STATE MODELS
# ###################

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

    # # Steady-state: M3 = M0
    # (1 - C2 * C1 * C0) * M0 = (C2 * (C1 * F0 + F1) + F2) * KinvJ

    # # Write as A * M = B * KinvJ
    # A = 1 - C2 * C1 * C0
    # B = C2 * (C1 * F0 + F1) + F2


def Mz_ss(R1, v, Fw, j, me, seq):
    """
    Calculate the steady-state longitudinal magnetization (Mz) immediately before the first pulse of a random pulse sequence. 

    This function acts as a dispatcher: it evaluates the steady-state 
    magnetization using a single-compartment model if `v` is a scalar or single-element 
    array, or a multi-compartment model if `v` contains multiple elements.

    Parameters
    ----------
    R1 : float or array-like
        Longitudinal relaxation rate(s) [s^-1].
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : float or array-like
        Exchange flux or compartment-specific flow rate parameters.
    me : float or array-like
        Equilibrium magnetization value(s).
    seq : object or dict
        Pulse sequence parameters (e.g., flip angles, repetition times, timing).

    Returns
    -------
    Mz : float or ndarray
        The steady-state longitudinal magnetization. Returns a scalar if the input
        is scalar, or an array matching the input shape for single/multi-compartment cases.
    """
    if np.isscalar(v):
        return _Mz_ss_1c(R1, v, Fw, j, me, seq)
    elif v.size==1:
        Mz = _Mz_ss_1c(R1[0], v[0], Fw[0,0], j[0], me, seq)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_nc(R1, v, Fw, j, me, seq)

def _Mz_ss_1c(R1, v, Fw, j, me, seq):

    # Rate constants and influx
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)

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
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)

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


def _Mz_KinvJ(R1, v, Fw, j, me):
    nc = np.size(v)
    K, J = _Mz_KJ(R1, v, Fw, j, me)

    if nc==1:
        KinvJ = J / K if K !=0 else 0
        KinvJ = np.array(KinvJ)    
    else:
        KinvJ = np.linalg.solve(K, J)
    return K, KinvJ # magn / cm3

def _Mz_KJ(R1, v, Fw, j, me):
    K = _Mz_K(R1, v, Fw)
    J = R1 * v * me + j   #1/s * mL/cm3 * magn/mL = magn/s/cm3
    return K, J # J/K = (1/s * magn/cm3) / 1/s 

def _Mz_K(R1, v, Fw):
    nc = np.size(v)

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




# Steady-state magnetization of a SPGR
def Mz_ss_spgr(R1, v, Fw, j, me, FA, TR):
    """
    Calculate the steady-state longitudinal magnetization (Mz) for an SPGR sequence.

    This function calculates steady-state magnetization for a Spoiled Gradient 
    Recalled Echo (SPGR) sequence, acting as a dispatcher between single-compartment 
    (`_Mz_ss_spgr_1c`) and multi-compartment (`_Mz_ss_spgr_nc`) models based on 
    the dimension of the volume fraction `v`.

    Parameters
    ----------
    R1 : float or array-like
        Longitudinal relaxation rate(s) [s^-1].
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : float or array-like
        Exchange flux or compartment-specific flow rate parameters.
    me : float or array-like
        Equilibrium magnetization value(s).
    TR : float
        Repetition time [ms or s].
    FA : float
        Flip angle [degrees or radians].

    Returns
    -------
    Mz : float or ndarray
        The steady-state longitudinal magnetization. Returns a scalar for scalar inputs, 
        or an array matching the shape of `R1` for single- and multi-compartment systems.

    """
    if np.isscalar(v):
        return _Mz_ss_spgr_1c(R1, v, Fw, j, me, FA, TR)
    elif v.size==1:
        Mz = _Mz_ss_spgr_1c(R1[0], v[0], Fw[0,0], j[0], me, FA, TR)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_spgr_nc(R1, v, Fw, j, me, FA, TR)

def _Mz_ss_spgr_nc(R1, v, Fw, j, me, FA, TR):
    off_diag = ~np.eye(Fw.shape[0], dtype=bool)
    PSw = Fw[off_diag]

    if np.all(PSw == 0):
        return _Mz_ss_nex(R1, v, Fw, j, me, FA, TR)
    
    elif np.all(np.isinf(PSw)):
        return _Mz_ss_fex(R1, v, Fw, j, me, FA, TR)
    
    elif 0 < np.count_nonzero(np.isinf(PSw)):
        raise NotImplementedError(
            'Water exchange with some (but not all) infinite PS '
            'values is currently not implemented.')
    else:
        return _Mz_ss_aex(R1, v, Fw, j, me, FA, TR)

def _Mz_ss_fex(R1, v, Fw, j, me, FA, TR):
    R1fex = np.sum(v * R1) / np.sum(v)
    fo = np.diag(Fw)
    M = _Mz_ss_spgr_1c(R1fex, np.sum(v), np.sum(fo), np.sum(j), me, FA, TR)
    Mc = [M * vc / np.sum(v) for vc in v]
    return np.stack(Mc).reshape(R1.shape)

def _Mz_ss_nex(R1, v, Fw, j, me, FA, TR):
    nc = v.size
    fo = np.diag(Fw)
    Mc = [_Mz_ss_spgr_1c(R1[c], v[c], fo[c], j[c], me, FA, TR) for c in range(nc)]
    return np.stack(Mc).reshape(R1.shape)

def _Mz_ss_aex(R1, v, Fw, j, me, FA, TR):
    K, J = _Mz_KJ(R1, v, Fw, j, me)
    E = expm(-TR * K)
    I = np.eye(R1.size)
    cFA = np.cos(np.radians(FA))
    A = K @ (I - cFA * E)
    try:
        Mz = np.linalg.solve(A, (I - E) @ J)  
    except:
        Mz = np.zeros_like(R1)
    return Mz.reshape(R1.shape)

def _Mz_ss_spgr_1c(R1, v, Fw, j, me, FA, TR):
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    E = np.exp(-TR * K)
    cFA = np.cos(FA*np.pi/180)
    n = (1-E) / (1-cFA*E)
    return n * KinvJ






# Steady-state magnetization of a preparation-recovery SPGR
    # # Derivation (1D)

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

    # # Derivation (ND)

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


def Mz_ss_pr_spgr(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA):
    """
    Calculate the steady-state longitudinal magnetization (Mz) for a prep-recovery SPGR sequence.

    Evaluates the steady-state magnetization for a Preparation-Recovery Spoiled 
    Gradient Recalled Echo (PR-SPGR) sequence by dispatching to either a 
    single-compartment (`_Mz_ss_pr_spgr_1c`) or multi-compartment (`_Mz_ss_pr_spgr_nc`) solver.

    Parameters
    ----------
    R1 : float or array-like
        Longitudinal relaxation rate(s) [s^-1].
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : float or array-like
        Exchange flux or compartment-specific flow rate parameters.
    me : float or array-like
        Equilibrium magnetization value(s).
    TR : float
        Repetition time of individual SPGR readout pulses [ms or s].
    FA : float
        Flip angle of the SPGR readout pulses [degrees or radians].
    Nph : int or float
        Number of phase encoding steps / readout pulses during acquisition.
    TP : float
        Preparation pulse duration or associated immediate delay [ms or s].
    TD : float
        Delay time following readout before the next preparation pulse [ms or s].
    PA : float
        Preparation pulse flip angle [degrees or radians] (e.g., inversion or saturation pulse).

    Returns
    -------
    Mz : float or ndarray
        The steady-state longitudinal magnetization. Returns a scalar for scalar inputs,
        or an array matching the shape of `R1` for single- and multi-compartment systems.

    Notes
    -----
    - Uses `np.array(v).size` to safely detect scalar/single-element array inputs.
    - Dispatches to `_Mz_ss_pr_spgr_1c` for 1-compartment systems.
    - Dispatches to `_Mz_ss_pr_spgr_nc` for n-compartment systems.
    """
    if np.isscalar(v):
        return _Mz_ss_pr_spgr_1c(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA)
    elif np.array(v).size==1:
        Mz = _Mz_ss_pr_spgr_1c(R1[0], v[0], Fw[0,0], j[0], me, FA, TR, Nph, TP, TD, PA)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_pr_spgr_nc(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA)


def _Mz_ss_pr_spgr_1c(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA):

    # Steady-state for a preparation-recovery SPGR
    # Rate constants, influx and SPGR steady-state
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    # Timings
    T_read = TR * Nph # Time for readout 

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

def _Mz_ss_pr_spgr_nc(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA):
    nc = R1.size
    Id = np.eye(nc)
    # Steady-state for a preparation-recovery SPGR
    # Rate constants, influx and SPGR steady-state
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    # Timings
    T_read = TR * Nph # Time for readout 

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



def Mz_ss_k0_pr_spgr(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, Nk0):
    """
    ... (existing docstring, plus:)

    Nk0 : int or float
        Number of readout (FA) pulses from the start of the readout to the
        center of k-space. The returned Mz is the steady-state magnetization
        immediately before the (Nk0+1)-th readout pulse, i.e. at the k-space
        center, rather than immediately before the next preparation pulse.
    """
    if np.isscalar(v):
        return _Mz_ss_k0_pr_spgr_1c(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, Nk0)
    elif np.array(v).size == 1:
        Mz = _Mz_ss_k0_pr_spgr_1c(R1[0], v[0], Fw[0, 0], j[0], me, FA, TR, Nph, TP, TD, PA, Nk0)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_k0_pr_spgr_nc(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, Nk0)


def _Mz_ss_k0_pr_spgr_1c(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, Nk0):

    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    n0 = Nk0                # pulses from readout start to k-space center
    n1 = Nph - Nk0             # remaining pulses from center to end of readout

    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0 = cFA**n0
    cFA1 = cFA**n1

    E0 = np.exp(-TR * n0 * K)   # readout: start -> center
    E1 = np.exp(-TR * n1 * K)   # readout: center -> end
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    # Cycle, starting and ending at the k-space center:
    #  1. remaining readout pulses (center -> end)
    #  2. delay TD
    #  3. prep pulse
    #  4. delay TP
    #  5. readout pulses (start -> center)
    steps = [
        (cFA1 * E1,  (1 - cFA1 * E1) * Mss),
        (ED,         (1 - ED) * KinvJ),
        (cPA,        0.0),
        (EP,         (1 - EP) * KinvJ),
        (cFA0 * E0,  (1 - cFA0 * E0) * Mss),
    ]

    prod, seq = steps[0]
    for C, b in steps[1:]:
        seq = C * seq + b
        prod = C * prod

    A = 1 - prod
    M_k0 = seq / A if A != 0 else 0

    return M_k0


def _Mz_ss_k0_pr_spgr_nc(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, Nk0):
    nc = R1.size
    Id = np.eye(nc)

    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    n0 = Nk0
    n1 = Nph - Nk0

    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0 = cFA**n0
    cFA1 = cFA**n1

    E0 = expm(-TR * n0 * K)
    E1 = expm(-TR * n1 * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    steps = [
        (cFA1 * E1,   (Id - cFA1 * E1) @ Mss),
        (ED,          (Id - ED) @ KinvJ),
        (cPA * Id,    np.zeros(nc)),
        (EP,          (Id - EP) @ KinvJ),
        (cFA0 * E0,   (Id - cFA0 * E0) @ Mss),
    ]

    prod, seq = steps[0]
    for C, b in steps[1:]:
        seq = C @ seq + b
        prod = C @ prod

    A = Id - prod
    M_k0 = np.linalg.solve(A, seq)

    return M_k0.reshape(R1.shape)








def Mz_ss_spgri(R1, v, Fw, j, me, FA, TR, TF, SA):
    """
    Calculate steady-state longitudinal magnetization including inflow effects.

    Simulates the longitudinal magnetization state during SPGR acquisition with 
    inflow parameters, taking into account saturation from precursor pulses prior 
    to signal readout. Handles both single-compartment (`nc == 1`) and multi-compartment 
    (`nc > 1`) systems.

    Parameters
    ----------
    R1 : float or array-like
        Tissue longitudinal relaxation rate(s) [s^-1].
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : float or array-like
        Exchange flux or compartment-specific flow rate parameters.
    me : float or array-like
        Equilibrium magnetization value(s).
    TR : float
        Repetition time [ms or s].
    FA : float
        Readout flip angle [degrees].
    TF : float
        Time period of pulses applied prior to readout [ms or s].
    SA : float
        Saturation flip angle applied to inflowing or pre-readout spins [degrees].

    Returns
    -------
    M_sig_t : float or ndarray
        The longitudinal magnetization state available for signal generation, 
        accounting for inflow and RF saturation history.
    """
    if np.isscalar(v):
        return _Mz_ss_spgri_1c(R1, v, Fw, j, me, FA, TR, TF, SA)
    elif v.size==1:
        Mz = _Mz_ss_spgri_1c(R1[0], v[0], Fw[0,0], j[0], me, FA, TR, TF, SA)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_spgri_nc(R1, v, Fw, j, me, FA, TR, TF, SA)


def _Mz_ss_spgri_1c(R1, v, Fw, j, me, FA, TR, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR) # n pulses to readout
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me

    K_t = _Mz_K(R1, v, Fw)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    # FA-pulses until time TF to get the Mz before readout
    En_t = np.exp(-TF * K_t)
    M_sig = Mss + nFA * En_t * (M0 - Mss)

    return M_sig


def _Mz_ss_spgri_nc(R1, v, Fw, j, me, FA, TR, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR) # n pulses to readout
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me

    K_t = _Mz_K(R1, v, Fw)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    # FA-pulses until time TF to get the Mz before readout
    En_t = expm(-TF * K_t)
    M_sig = Mss + nFA * En_t @ (M0 - Mss)

    return M_sig



# ###################
# PROPAGATORS
# ###################


# Propagator through an arbitrary sequence of pulses
def Mz_prop(M, R1, v, Fw, j, me, seq):
    """
    Propagate longitudinal magnetization through an arbitrary pulse sequence.

    Calculates the evolution of longitudinal magnetization (Mz) starting from an initial
    state `M` through a specified sequence of RF pulses and relaxation delays.
    Acts as a dispatcher between single-compartment (`_Mz_prop_1c`) and 
    multi-compartment (`_Mz_prop_nc`) propagation models based on the size of `v`.

    Parameters
    ----------
    M : float or array-like
        Initial longitudinal magnetization vector or scalar state.
    R1 : float or array-like
        Longitudinal relaxation rate(s) [s^-1].
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : float or array-like
        Exchange flux or compartment-specific flow rate parameters.
    me : float or array-like
        Equilibrium magnetization value(s).
    seq : list
        Pulse sequence parameters specifying the timings, flip angles, and RF events.

    Returns
    -------
    Mz : float or ndarray
        The propagated longitudinal magnetization after the sequence. Returns a scalar
        for scalar inputs, or an array matching `R1.shape` for single- and multi-compartment systems.

    """
    if len(seq) == 0:
        return M
    
    if np.isscalar(v):
        return _Mz_prop_1c(M, R1, v, Fw, j, me, seq)
    elif v.size==1:
        return _Mz_prop_1c(M[0], R1[0], v[0], Fw[0,0], j[0], me, seq)
    else:
        return _Mz_prop_nc(M, R1, v, Fw, j, me, seq)


def _Mz_prop_1c(M, R1, v, Fw, j, me, seq):
    const = (2 == np.size(R1) + np.size(j))
    if const:
        K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)

    M_calc = np.zeros((1, len(seq)))
    M_curr = M

    for i, pulse in enumerate(seq):
        if not const:
            K, KinvJ = _Mz_KinvJ(R1[i], v, Fw, j[i], me)

        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = np.exp(-TR * K)
        M_next = E * (cFA * M_curr) + (1 - E) * KinvJ 

        M_calc[:, i] = M_next
        M_curr = M_next

    return M_calc


def _Mz_prop_nc(M, R1, v, Fw, j, me, seq):
    nc = R1.shape[0]
    Id = np.eye(nc)

    const = (2 == np.ndim(R1) + np.ndim(j))
    if const:
        K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)

    M_calc = np.zeros((nc, len(seq)))
    M_curr = M

    for i, pulse in enumerate(seq):
        if not const:
            K, KinvJ = _Mz_KinvJ(R1[:,i], v, Fw, j[:,i], me)

        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = expm(-TR * K)
        M_next = E @ (cFA * M_curr) + (Id - E) @ KinvJ

        M_calc[:, i] = M_next
        M_curr = M_next

    return M_calc



# Propagation through a preparation-recovery SPGR
def Mz_prop_pr_spgr(M, R1, v, Fw, j, me, PA, TP, TC, FA, TR, TA):
    """
    Propagate magnetization through a preparation-recovery SPGR sequence.

    Calculates the evolution of longitudinal magnetization from an initial state `M` 
    through a Preparation-Recovery Spoiled Gradient Recalled Echo (PR-SPGR) sequence. 
    Dispatches to single-compartment (`_Mz_prop_pr_spgr_1c`) or multi-compartment 
    (`_Mz_prop_pr_spgr_nc`) implementations depending on the size of `v`.

    Parameters
    ----------
    M : float or array-like
        Initial longitudinal magnetization vector or scalar state.
    R1 : float or array-like
        Longitudinal relaxation rate(s) [s^-1].
    v : float or array-like
        Volume fraction(s) of the compartment(s).
    Fw : float or array-like
        Water exchange rate matrix or values between compartments [s^-1].
    j : float or array-like
        Exchange flux or compartment-specific flow rate parameters.
    me : float or array-like
        Equilibrium magnetization value(s).
    PA : float
        Preparation pulse flip angle [degrees or radians].
    TP : float
        Preparation pulse duration [ms or s].
    TC : float
        Recovery time following the preparation pulse (e.g., inversion delay) [ms or s].
    TR : float
        Repetition time of the SPGR readout pulses [ms or s].
    FA : float
        Flip angle of the SPGR readout pulses [degrees or radians].
    TA : float
        Acquisition duration or total readout period [ms or s].

    Returns
    -------
    M_sig : float or ndarray
        The signal-producing magnetization component during or immediately following readout.
    Mz : float or ndarray
        The final propagated longitudinal magnetization state at the end of the sequence.

    """
    if np.isscalar(v):
        return _Mz_prop_pr_spgr_1c(M, R1, v, Fw, j, me, PA, TP, TC, FA, TR, TA)
    elif v.size==1:
        M_sig, Mz = _Mz_prop_pr_spgr_1c(M[0], R1[0], v[0], Fw[0,0], j[0], me, PA, TP, TC, FA, TR, TA)
        return np.array(M_sig).reshape(R1.shape), np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_prop_pr_spgr_nc(M, R1, v, Fw, j, me, PA, TP, TC, FA, TR, TA)

def _Mz_prop_pr_spgr_1c(M, R1, v, Fw, j, me, PA, TP, TC, FA, TR, TA):
    # Get some constants
    cPA = np.cos(np.radians(PA))
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

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

def _Mz_prop_pr_spgr_nc(M, R1, v, Fw, j, me, PA, TP, TC, FA, TR, TA):
    cPA = np.cos(np.radians(PA))
    # Get some constants
    nc = R1.size
    Id = np.eye(nc)
    
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

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











############################
# VECTORIZED IMPLEMENTATIONS - STEADY STATE
############################








def Mz_ss_vectorized(R1_pulses, v, Fw, j_pulses, me, seq):
    """
    Vectorized steady-state Mz over n_periods.
    R1_pulses, j_pulses: shape (n_comps, n_periods)
    Returns: Mz, shape (n_comps, n_periods)
    """
    if np.isscalar(v) or (hasattr(v, "size") and v.size == 1):
        return _Mz_ss_1c_vec(R1_pulses, v, Fw, j_pulses, me, seq)
    else:
        return _Mz_ss_nc_vec(R1_pulses, v, Fw, j_pulses, me, seq)


def _Mz_ss_1c_vec(R1_pulses, v, Fw, j_pulses, me, seq):
    R1 = R1_pulses.ravel() if R1_pulses.ndim > 1 else R1_pulses  # (n_periods,)
    j = j_pulses.ravel() if j_pulses.ndim > 1 else j_pulses

    # K, KinvJ must now be shape (n_periods,) each
    K, KinvJ = _Mz_KinvJ_vec_1c(R1, v, Fw, j, me)  # <-- needs batching support

    n = K.shape[0]
    prod = np.ones(n)
    acc = np.zeros(n)
    for FA, TR in seq:
        cFA = np.cos(np.radians(FA))
        E = np.exp(-TR * K)          # (n_periods,)
        C = cFA * E
        F = 1 - E
        acc = C * acc + F
        prod = C * prod

    A = 1 - prod
    M = np.where(A != 0, acc / A * KinvJ, 0.0)
    return M.reshape(R1_pulses.shape)


def _Mz_ss_nc_vec(R1_pulses, v, Fw, j_pulses, me, seq):
    nc, n_periods = R1_pulses.shape
    Id = np.eye(nc)[None, :, :].repeat(n_periods, axis=0)   # (n_periods, nc, nc)

    K, KinvJ = _Mz_KinvJ_vec_nc(R1_pulses, v, Fw, j_pulses, me)

    prod = Id.copy()
    acc = np.zeros((n_periods, nc, nc))
    for FA, TR in seq:
        cFA = np.cos(np.radians(FA))
        E = expm(-TR * K)
        C = cFA * E
        F = Id - E
        acc = C @ acc + F
        prod = C @ prod

    A = Id - prod
    rhs = np.einsum('pij,pj->pi', acc, KinvJ)   # (n_periods, nc)

    M = np.linalg.solve(A, rhs[..., None])[..., 0]   # force unambiguous batch shape
    return M.T


def _Mz_K_vec(R1_periods, v, Fw):
    """
    nc==1: R1_periods shape (n_periods,); v, Fw scalars -> returns K shape (n_periods,)
    nc>1:  R1_periods shape (n_periods, nc); v shape (nc,), Fw shape (nc, nc) -> returns K shape (n_periods, nc, nc)
    """
    nc = np.size(v)

    if nc == 1:
        return R1_periods + Fw / v          # R1_periods already (n_periods,); Fw, v scalars

    n_periods = R1_periods.shape[0]
    K_offdiag = -Fw / v
    total_outflow = np.sum(Fw, axis=0)

    K = np.broadcast_to(K_offdiag, (n_periods, nc, nc)).copy()
    diag = R1_periods + total_outflow / v

    idx = np.arange(nc)
    K[:, idx, idx] = diag

    return K


def _Mz_KJ_vec(R1_periods, v, Fw, j_periods, me):
    """
    R1_periods, j_periods: (n_periods, nc)
    v, me: (nc,) or scalar (nc==1)
    Returns K, J with leading n_periods axis
    """
    K = _Mz_K_vec(R1_periods, v, Fw)
    J = R1_periods * v * me + j_periods              # broadcasts fine for both nc==1 and nc>1
    return K, J


def _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me):
    nc = np.size(v)
    K, J = _Mz_KJ_vec(R1_periods, v, Fw, j_periods, me)

    if nc == 1:
        KinvJ = np.where(K != 0, J / K, 0.0)          # (n_periods,)
    else:
        # J: (n_periods, nc) -> (n_periods, nc, 1) to force correct batch broadcasting
        KinvJ = np.linalg.solve(K, J[..., None])       # (n_periods, nc, 1)
        KinvJ = KinvJ[..., 0]                           # back to (n_periods, nc)

    return K, KinvJ


def _Mz_KinvJ_vec_1c(R1_pulses, v, Fw, j_pulses, me):
    R1_periods = R1_pulses.reshape(-1)   # (n_periods,)
    j_periods = j_pulses.reshape(-1)     # (n_periods,)
    return _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)

def _Mz_KinvJ_vec_nc(R1_pulses, v, Fw, j_pulses, me):
    R1_periods = R1_pulses.T   # (n_periods, nc)
    j_periods = j_pulses.T     # (n_periods, nc)
    return _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)





def Mz_ss_spgr_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    """
    R1_pulses, j_pulses: shape (n_comps, n_periods)
    Returns Mz: shape (n_comps, n_periods)
    """
    if np.isscalar(v):
        return _Mz_ss_spgr_1c_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)
    elif v.size == 1:
        Mz = _Mz_ss_spgr_1c_vec(R1_pulses[0], v[0], Fw[0, 0], j_pulses[0], me, FA, TR)
        return Mz.reshape(R1_pulses.shape)
    else:
        return _Mz_ss_spgr_nc_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)


def _Mz_ss_spgr_nc_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    off_diag = ~np.eye(Fw.shape[0], dtype=bool)
    PSw = Fw[off_diag]

    if np.all(PSw == 0):
        return _Mz_ss_nex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)
    
    elif np.all(np.isinf(PSw)):
        return _Mz_ss_fex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)
    
    elif 0 < np.count_nonzero(np.isinf(PSw)):
        raise NotImplementedError(
            'Water exchange with some (but not all) infinite PS '
            'values is currently not implemented.')
    else:
        return _Mz_ss_aex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)


def _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # reuse existing helper
    E = np.exp(-TR * K)
    cFA = np.cos(np.radians(FA))
    n = (1 - E) / (1 - cFA * E)
    return n * KinvJ


def _Mz_ss_fex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    R1fex_periods = np.sum(v[:, None] * R1_pulses, axis=0) / np.sum(v)   # (n_periods,)
    fo = np.diag(Fw)
    j_sum_periods = np.sum(j_pulses, axis=0)                              # (n_periods,)

    M_periods = _Mz_ss_spgr_1c_vec(
        R1fex_periods, np.sum(v), np.sum(fo), j_sum_periods, me, FA, TR
    )                                                                      # (n_periods,)

    Mc = M_periods[None, :] * (v / np.sum(v))[:, None]                    # (nc, n_periods)
    return Mc


def _Mz_ss_nex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    nc = v.size
    fo = np.diag(Fw)
    Mc = [
        _Mz_ss_spgr_1c_vec(R1_pulses[c], v[c], fo[c], j_pulses[c], me, FA, TR)
        for c in range(nc)
    ]
    return np.stack(Mc)   # (nc, n_periods) -- loop is only over nc, not n_periods


def _Mz_ss_aex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    nc, n_periods = R1_pulses.shape
    R1_periods = R1_pulses.T   # (n_periods, nc)
    j_periods = j_pulses.T     # (n_periods, nc)

    K = _Mz_K_vec(R1_periods, v, Fw)             # (n_periods, nc, nc), reuse existing helper
    J = R1_periods * v * me + j_periods           # (n_periods, nc)

    E = expm(-TR * K)                              # batched
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))
    cFA = np.cos(np.radians(FA))

    A = K @ (Id - cFA * E)                          # (n_periods, nc, nc)
    rhs = (Id - E) @ J[..., None]                    # (n_periods, nc, 1)

    Mz = np.zeros((n_periods, nc))
    try:
        Mz = np.linalg.solve(A, rhs)[..., 0]
    except np.linalg.LinAlgError:
        # fall back per-period only where the batched solve fails,
        # mirroring the original's try/except -> zeros behaviour
        for p in range(n_periods):
            try:
                Mz[p] = np.linalg.solve(A[p], rhs[p, :, 0])
            except np.linalg.LinAlgError:
                Mz[p] = 0.0

    return Mz.T   # (nc, n_periods)


# PR-SPGR SS before the prepulse
def Mz_ss_pr_spgr_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, Nph, TP, TD, PA):
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_ss_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        M = _Mz_ss_pr_spgr_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, Nph, TP, TD, PA)
        return M.reshape(R1_pulses.shape)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        M = _Mz_ss_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA)
        return M.T   # (nc, n_periods)


def _Mz_ss_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)          # (n_periods,)
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)  # (n_periods,)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n

    En = np.exp(-T_read * K)
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    W = nFA * ED * En
    A = 1 - cPA * W * EP
    B = ED - W
    C = W * (1 - EP) + (1 - ED)

    RH = B * Mss + C * KinvJ
    return np.where(A != 0, RH / A, 0.0)


def _Mz_ss_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # (n_periods,nc,nc), (n_periods,nc)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T  # (n_periods,nc)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n

    En = expm(-T_read * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    W = nFA * (ED @ En)
    A = Id - cPA * (W @ EP)
    B = ED - W
    C = W @ (Id - EP) + (Id - ED)

    RH = np.einsum('pij,pj->pi', B, Mss) + np.einsum('pij,pj->pi', C, KinvJ)
    return np.linalg.solve(A, RH[..., None])[..., 0]   # (n_periods, nc)



# PR-SPGR SS at the k-space center
def Mz_ss_k0_pr_spgr_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, Nph, TP, TD, PA, Nk0):
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_ss_k0_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        M = _Mz_ss_k0_pr_spgr_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0)
        return M.reshape(R1_pulses.shape)
    else:
        R1_periods = R1_pulses.T
        j_periods = j_pulses.T
        M = _Mz_ss_k0_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0)
        return M.T


def _Mz_ss_k0_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    n0, n1 = Nk0, n - Nk0

    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = np.exp(-TR * n0 * K)
    E1 = np.exp(-TR * n1 * K)
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    steps = [
        (cFA1 * E1, (1 - cFA1 * E1) * Mss),
        (ED,        (1 - ED) * KinvJ),
        (cPA,       0.0),
        (EP,        (1 - EP) * KinvJ),
        (cFA0 * E0, (1 - cFA0 * E0) * Mss),
    ]

    prod, seq = steps[0]
    for C, b in steps[1:]:
        seq = C * seq + b
        prod = C * prod

    A = 1 - prod
    return np.where(A != 0, seq / A, 0.0)


def _Mz_ss_k0_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T  # (n_periods,nc)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    n0, n1 = Nk0, n - Nk0

    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = expm(-TR * n0 * K)
    E1 = expm(-TR * n1 * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    def mv(C, x):  # batched matrix @ vector
        return np.einsum('pij,pj->pi', C, x)

    steps = [
        (cFA1 * E1, mv(Id - cFA1 * E1, Mss)),
        (ED,        mv(Id - ED, KinvJ)),
        (cPA * Id,  np.zeros((n_periods, nc))),
        (EP,        mv(Id - EP, KinvJ)),
        (cFA0 * E0, mv(Id - cFA0 * E0, Mss)),
    ]

    prod, seq = steps[0]
    for C, b in steps[1:]:
        seq = mv(C, seq) + b
        prod = C @ prod

    A = Id - prod
    return np.linalg.solve(A, seq[..., None])[..., 0]






def Mz_ss_spgri_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, TF, SA):
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_ss_spgri_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        M = _Mz_ss_spgri_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, TF, SA)
        return M.reshape(R1_pulses.shape)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        M = _Mz_ss_spgri_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA)
        return M.T                # (nc, n_periods)


def _Mz_ss_spgri_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR)
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me   # scalar, constant across periods

    K_t = _Mz_K_vec(R1_periods, v, Fw)          # (n_periods,) -- reuse existing helper
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)  # (n_periods,)

    En_t = np.exp(-TF * K_t)                     # (n_periods,)
    M_sig = Mss + nFA * En_t * (M0 - Mss)

    return M_sig


def _Mz_ss_spgri_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR)
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me   # (nc,), constant across periods

    K_t = _Mz_K_vec(R1_periods, v, Fw)                                    # (n_periods, nc, nc)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T  # (n_periods, nc)

    En_t = expm(-TF * K_t)                                                 # batched

    diff = M0[None, :] - Mss                                               # (n_periods, nc)
    M_sig = Mss + nFA * np.einsum('pij,pj->pi', En_t, diff)

    return M_sig




############################
# VECTORIZED IMPLEMENTATIONS - PROPAGATOR
############################



def _Mz_prop_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, seq):
    """
    R1_periods, j_periods: (n_periods,)
    Returns A, B: (n_periods,) such that M_end = A * M_start + B
    """
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # (n_periods,)
    n_periods = K.shape[0]

    A = np.ones(n_periods)
    B = np.zeros(n_periods)

    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = np.exp(-TR * K)                # (n_periods,)
        C = cFA * E
        b = (1 - E) * KinvJ

        B = C * B + b
        A = C * A

    return A, B


def _Mz_prop_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, seq):
    """
    R1_periods, j_periods: (n_periods, nc)
    Returns A: (n_periods, nc, nc), B: (n_periods, nc)
    """
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    # (n_periods, nc, nc), (n_periods, nc)

    A = Id.copy()
    B = np.zeros((n_periods, nc))

    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = expm(-TR * K)                                  # batched, (n_periods, nc, nc)
        C = cFA * E
        b = np.einsum('pij,pj->pi', Id - E, KinvJ)          # (n_periods, nc)

        B = np.einsum('pij,pj->pi', C, B) + b
        A = C @ A

    return A, B.T



def _Mz_prop_coeffs_vectorized(R1_pulses, v, Fw, j_pulses, me, seq):
    """
    R1_pulses, j_pulses: (n_comps, n_periods)
    Returns A, B with the batch (n_periods) as the leading axis, ready to
    drive a cheap sequential recurrence over periods.
    """
    if len(seq) == 0:
        # identity map: A=1/Id, B=0
        if np.isscalar(v):
            n_periods = R1_pulses.shape[-1]
            return np.ones(n_periods), np.zeros(n_periods)
        else:
            nc, n_periods = R1_pulses.shape
            return np.broadcast_to(np.eye(nc), (n_periods, nc, nc)).copy(), np.zeros((n_periods, nc))

    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, seq)
    elif v.size == 1:
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_coeffs_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, seq)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        return _Mz_prop_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, seq)



def Mz_prop_vectorized(M0, R1_periods, v, Kw, j_periods, me, seq):
    nc = R1_periods.shape[0]

    R1_periods = R1_periods.reshape((nc, -1))
    j_periods = j_periods.reshape((nc, -1))
    
    # Precompute once, fully vectorized across all periods
    A, B = _Mz_prop_coeffs_vectorized(R1_periods, v, Kw, j_periods, me, seq)

    n_periods = R1_periods.shape[1]
    Mz = np.zeros((M0.size, n_periods))

    for period in range(n_periods):
        M_init = M0 if period==0 else Mz[:, period - 1]

        if np.isscalar(v) or v.size == 1:
            Mz[:, period] = A[period] * M_init  + B[period]
        else:
            Mz[:, period] = A[period] @ M_init + B[:, period]

    return Mz


# SPGR

def _Mz_prop_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # (n_periods,)

    cFA = np.cos(np.radians(FA))
    E = np.exp(-TR * K)          # (n_periods,) -- one exp call instead of n_pulses
    C = cFA * E
    b = (1 - E) * KinvJ

    cFA_n = cFA ** n_pulses
    A = cFA_n * np.exp(-n_pulses * TR * K)
    # geometric series, guarding C == 1
    B = np.where(C != 1, (1 - A) / (1 - C) * b, n_pulses * b)

    return A, B


def _Mz_prop_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)

    cFA = np.cos(np.radians(FA))
    E = expm(-TR * K)                                    # single-pulse map, still needed for C, b
    C = cFA * E
    b = np.einsum('pij,pj->pi', Id - E, KinvJ)

    cFA_n = cFA ** n_pulses
    A = cFA_n * expm(-n_pulses * TR * K)                   # replaces matrix_power(C, n_pulses)

    rhs = np.einsum('pij,pj->pi', Id - A, b)
    B = np.linalg.solve(Id - C, rhs[..., None])[..., 0]

    return A, B.T


def _Mz_prop_spgr_coeffs_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, n_pulses):
    """
    R1_pulses, j_pulses: (n_comps, n_periods)
    Returns A, B with the batch (n_periods) as the leading axis, ready to
    drive a cheap sequential recurrence over periods.
    """
    if n_pulses == 0:
        # identity map: A=1/Id, B=0
        if np.isscalar(v):
            n_periods = R1_pulses.shape[-1]
            return np.ones(n_periods), np.zeros(n_periods)
        else:
            nc, n_periods = R1_pulses.shape
            return np.broadcast_to(np.eye(nc), (n_periods, nc, nc)).copy(), np.zeros((n_periods, nc))

    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses)
    elif v.size == 1:
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_spgr_coeffs_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, n_pulses)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        return _Mz_prop_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses)



def Mz_prop_spgr_vectorized(M0, R1_periods, v, Kw, j_periods, me, FA, TR, n_pulses):
    nc = R1_periods.shape[0]

    R1_periods = R1_periods.reshape((nc, -1))
    j_periods = j_periods.reshape((nc, -1))
    
    # Precompute once, fully vectorized across all periods
    A, B = _Mz_prop_spgr_coeffs_vectorized(R1_periods, v, Kw, j_periods, me, FA, TR, n_pulses)

    n_periods = R1_periods.shape[1]
    Mz = np.zeros((M0.size, n_periods))

    for period in range(n_periods):
        M_init = M0 if period==0 else Mz[:, period - 1]

        if np.isscalar(v) or v.size == 1:
            Mz[:, period] = A[period] * M_init  + B[period]
        else:
            Mz[:, period] = A[period] @ M_init + B[:, period]

    return Mz




# PR-SPGR


def Mz_prop_pr_spgr_coeffs_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, N, TP, TD, PA, Nk0):
    """
    Precompute per-period affine coefficients A, B such that
    M_end = A @ M_start + B for one prepared-SPGR cycle, starting and
    ending at the k-space center.

    R1_pulses, j_pulses: (n_comps, n_periods)
    Returns A, B with n_periods as the leading batch axis.
    """
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_pr_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        return _Mz_prop_pr_spgr_coeffs_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, N, TP, TD, PA, Nk0)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        return _Mz_prop_pr_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0)


def _Mz_prop_pr_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)             # (n_periods,)
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)     # (n_periods,)

    n0, n1 = Nk0, N - Nk0
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = np.exp(-TR * n0 * K)
    E1 = np.exp(-TR * n1 * K)
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    C1 = cFA1 * E1
    C0 = cFA0 * E0

    steps = [
        (C1,             (1 - C1) * Mss),                # remaining (N-Nk0) readout pulses
        (ED,             (1 - ED) * KinvJ),               # TD recovery
        (cPA * np.ones_like(K), np.zeros_like(K)),        # preparation pulse
        (EP,             (1 - EP) * KinvJ),                # TP recovery
        (C0,             (1 - C0) * Mss),                  # Nk0 readout pulses to k-space center
    ]

    A, B = steps[0]
    for C, b in steps[1:]:
        B = C * B + b
        A = C * A

    return A, B


def _Mz_prop_pr_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T   # (n_periods, nc)

    n0, n1 = Nk0, N - Nk0
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = expm(-TR * n0 * K)
    E1 = expm(-TR * n1 * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    C1 = cFA1 * E1
    C0 = cFA0 * E0

    def mv(C, x):
        return np.einsum('pij,pj->pi', C, x)

    steps = [
        (C1,        mv(Id - C1, Mss)),                    # remaining (N-Nk0) readout pulses
        (ED,        mv(Id - ED, KinvJ)),                   # TD recovery
        (cPA * Id,  np.zeros((n_periods, nc))),            # preparation pulse
        (EP,        mv(Id - EP, KinvJ)),                    # TP recovery
        (C0,        mv(Id - C0, Mss)),                       # Nk0 readout pulses to k-space center
    ]

    A, B = steps[0]
    for C, b in steps[1:]:
        B = mv(C, B) + b
        A = C @ A

    return A, B.T   # B as (nc, n_periods)


def Mz_prop_pr_spgr_vectorized(M0, R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0):
    nc = R1_periods.shape[0]
    R1_periods = R1_periods.reshape((nc, -1))
    j_periods = j_periods.reshape((nc, -1))

    A, B = Mz_prop_pr_spgr_coeffs_vectorized(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0)

    n_periods = R1_periods.shape[1]
    Mz = np.zeros((M0.size, n_periods))

    for period in range(n_periods):
        M_init = M0 if period == 0 else Mz[:, period - 1]
        if np.isscalar(v) or v.size == 1:
            Mz[:, period] = A[period] * M_init + B[period]
        else:
            Mz[:, period] = A[period] @ M_init + B[:, period]

    return Mz
