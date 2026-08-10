import numpy as np
from scipy.linalg import expm

def channels(sequence):
    return 2 if sequence in ['Eq-DE-EPI', '2D-DE-EPI'] else 1

def pulse_readout(sequence, pars):
    if sequence in [
            'ZTE-3D-SPGR-SS',
            '3D-SPGR-SS',
            '2D-SPGR-SS',
            '3D-SPGR',
            '2D-SPGR',
            '3D-SPGR-SSI',
        ]:
        return pars['Nk0']
    
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
        return 1 + pars['Nk0']

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
    ]:
        return p['TR']
    
    return p['TA']


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
    

# ###################
# STEADY STATE MODELS
# ###################


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
def Mz_ss_spgr(R1, v, Fw, j, me, TR, FA):
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

def _Mz_ss_spgr_1c(R1, v, Fw, j, me, TR, FA):
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    E = np.exp(-TR * K)
    cFA = np.cos(FA*np.pi/180)
    n = (1-E) / (1-cFA*E)
    return n * KinvJ






# Steady-state magnetization of a preparation-recovery SPGR
def Mz_ss_pr_spgr(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA):
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
        return _Mz_ss_pr_spgr_1c(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA)
    elif np.array(v).size==1:
        Mz = _Mz_ss_pr_spgr_1c(R1[0], v[0], Fw[0,0], j[0], me, TR, FA, Nph, TP, TD, PA)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_pr_spgr_nc(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA)

def _Mz_ss_pr_spgr_1c(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA):
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
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

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

def _Mz_ss_pr_spgr_nc(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA):
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
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

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



def Mz_ss_spgri(R1, v, Fw, j, me, TR, FA, TF, SA):
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
        return _Mz_ss_spgri_1c(R1, v, Fw, j, me, TR, FA, TF, SA)
    elif v.size==1:
        Mz = _Mz_ss_spgri_1c(R1[0], v[0], Fw[0,0], j[0], me, TR, FA, TF, SA)
        return np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_ss_spgri_nc(R1, v, Fw, j, me, TR, FA, TF, SA)


def _Mz_ss_spgri_1c(R1, v, Fw, j, me, TR, FA, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR) # n pulses to readout
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me

    K_t = _Mz_K(R1, v, Fw)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # FA-pulses until time TF to get the Mz before readout
    En_t = np.exp(-TF * K_t)
    M_sig = Mss + nFA * En_t * (M0 - Mss)

    return M_sig


def _Mz_ss_spgri_nc(R1, v, Fw, j, me, TR, FA, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR) # n pulses to readout
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me

    K_t = _Mz_K(R1, v, Fw)
    Mss = Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

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
    seq : object or dict
        Pulse sequence parameters specifying the timings, flip angles, and RF events.

    Returns
    -------
    Mz : float or ndarray
        The propagated longitudinal magnetization after the sequence. Returns a scalar
        for scalar inputs, or an array matching `R1.shape` for single- and multi-compartment systems.

    """
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
def Mz_prop_pr_spgr(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
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
        return _Mz_prop_pr_spgr_1c(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA)
    elif v.size==1:
        M_sig, Mz = _Mz_prop_pr_spgr_1c(M[0], R1[0], v[0], Fw[0,0], j[0], me, PA, TP, TC, TR, FA, TA)
        return np.array(M_sig).reshape(R1.shape), np.array(Mz).reshape(R1.shape)
    else:
        return _Mz_prop_pr_spgr_nc(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA)

def _Mz_prop_pr_spgr_1c(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
    # Get some constants
    cPA = np.cos(np.radians(PA))
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
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

def _Mz_prop_pr_spgr_nc(M, R1, v, Fw, j, me, PA, TP, TC, TR, FA, TA):
    cPA = np.cos(np.radians(PA))
    # Get some constants
    nc = R1.size
    Id = np.eye(nc)
    
    K, KinvJ = _Mz_KinvJ(R1, v, Fw, j, me)
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