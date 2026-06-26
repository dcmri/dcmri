import numpy as np
from scipy.linalg import expm
from scipy.special import i0, i1


from dcmri.bloch import pulse


def mz_readout(Mz: np.ndarray, R2: np.ndarray, S0, FA, TE, noise_sdev):
    # Shapes for Mz, R2: (nc, nt)
    # Other parameters are scalar
    # returns shape (nt,)
    sFA = np.sin(np.radians(FA))
    decay = np.exp(-TE * R2)
    Mxy = decay * sFA * Mz
    Mxy = np.sum(Mxy, axis=0) # sum over compartments
    signal = S0 * np.abs(Mxy)
    return signal_rice(signal, noise_sdev)
    

def signal_rice(nu, sigma)-> np.ndarray:
    if sigma==0:
        return nu
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        K = nu**2 / (2*sigma**2)
        arg = K/2
        pref = sigma * np.sqrt(np.pi/2)
        rice_mean = pref * np.exp(-K/2) * ((1+K)*i0(arg) + K*i1(arg))
    # Nan values are points where the distribution is indistinguisable from Gaussian
    return np.where(np.isnan(rice_mean) | np.isinf(rice_mean), nu, rice_mean)


# def Mz_ge(R1, v, Fw, j, me, TR, FA): 
#     """Mz for one slice in a GE-EPI sequence
#     """
#     if np.isinf(TR):
#         return np.full_like(R1, me)
    
#     pulse_sequence = [
#         [FA, TR]
#     ]
#     def _Mz_ge_t(R1_t, j_t):
#         return pulse.Mz_ss(R1_t, v, Fw, j_t, me, pulse_sequence)

#     nc, nt = R1.shape
#     M = [_Mz_ge_t(R1[:,k].T, j[:,k].T) for k in range(nt)]
#     return np.array(M).T.reshape(nc, nt)


def Mz_se(R1, v, Fw, j, me, TE, TR, FA): 
    """Mz for one slice in a SE-EPI sequence
    """
    if np.isinf(TR):
        return np.full_like(R1, me)
    pulse_sequence = [
        [FA, TE / 2], 
        [180, TR - TE/2]
    ]
    def _Mz_se_t(R1_t, j_t):
        return pulse.Mz_ss(R1_t, v, Fw, j_t, me, pulse_sequence)

    nc, nt = R1.shape
    M = [_Mz_se_t(R1[:,k].T, j[:,k].T) for k in range(nt)]
    return np.array(M).T.reshape(nc, nt)


def Mz_spgr_in_ss(R1, v, Fw, j, me, TR, FA) -> np.ndarray:
    """Spoiled gradient echo sequence in steady state"""

    if np.isinf(TR):
        return np.full_like(R1, me)
    
    nc, nt = R1.shape

    M = [pulse.Mz_ss_spgr(R1[:,t], v, Fw, j[:,t], me, TR, FA) for t in range(nt)]
    return np.array(M).T.reshape(nc, nt)


def Mz_pr_spgr(R1, v, Fw, j, me, TC, TR, FA, TP, TA, PA): 
    """This models SPGR with a preparation pulse and linear k-space ordering

    - A preparation pulse PA at the start of each time interval, 
    - Free recovery over a time TP
    - FA readout pulses separated by TR for a duration of 2 * (TC-TP)
    - Free recovery until the start of the next time interval. 
    - And a readout at time TC after the preparation pulse.

    R1 is assumed to be constant on each time interval.
    """
    nc, nt = R1.shape
    Mt = v * me
    args = (me, PA, TP, TC, TR, FA, TA)

    M = []
    for k in range(nt):
        M_sig, Mt = pulse.Mz_pr_spgr_prop(Mt, R1[:,k].T, v, Fw, j[:,k].T, *args)
        M.append(M_sig)
    
    return np.array(M).T.reshape(nc, nt)


def Mz_pr_spgr_in_ss(R1, v, Fw, j, me, TC, TR, FA, TP, TA, PA):
    """This models SPGR with a preparation pulse and linear k-space ordering
    running in the steady state

    R1 is assumed to be constant on each time interval.
    """
    args = (me, PA, TP, TC, TR, FA, TA)
    def _Mz_pr_spgr_in_ss_t(R1_t, j_t):
        Mss_t = pulse.Mz_pr_spgr_ss(R1_t, v, Fw, j_t, *args)
        M_sig, _ = pulse.Mz_pr_spgr_prop(Mss_t, R1_t, v, Fw, j_t, *args)
        return M_sig
    
    nc, nt = R1.shape
    M = [_Mz_pr_spgr_in_ss_t(R1[:,k].T, j[:,k].T) for k in range(nt)]
    return np.array(M).T.reshape(nc, nt)


def Mz_ssi(R1, v, Fw, j, me, TR, FA, TF, SA): 
    """This models steady-state imaging with inflow effects

    - Initial magnetization determined by saturation slabs outside the imaging volume. Without slabs, n_init=1 
    - FA readout pulses separated by TR for a duration of TF (inflow time)

    Each readout starts the same - no build=up effects

    R1 is assumed to be constant on each time interval.
    """
    nc, nt = R1.shape
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR) # n pulses to readout
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me

    def _Mz_ssi_prop(R1_t, j_t):
        K_t = pulse.Mz_K(R1_t, v, Fw)
        Mss_t = pulse.Mz_ss_spgr(R1_t, v, Fw, j_t, me, TR, FA)
        # FA-pulses until time TF to get the Mz before readout
        if nc==1:
            En_t = np.exp(-TF * K_t)
            M_sig_t = Mss_t + nFA * En_t * (M0 - Mss_t)
        else:
            En_t = expm(-TF * K_t)
            M_sig_t = Mss_t + nFA * En_t @ (M0 - Mss_t)           
        return M_sig_t

    M = [_Mz_ssi_prop(R1[:,k].T, j[:,k].T) for k in range(nt)]
    return np.array(M).T.reshape(nc, nt)