from scipy.special import i0, i1
import numpy as np


def signal(Mz=None, R2=10, S0=None, FA=None, TE=0, noise_sdev=0):
    """Signal after readout of given Mz.

    Args:
        S0 (float): Signal scaling factor (arbitrary units).
        R2 (array-like): Transverse relaxation rate R1 or R2* in 1/sec. 
        FA (float): flip angle (deg)
        TE (float): Echo time (sec)
        noise_sdev (float, optional): standard deviation of the signal noise. 

    Returns:
        np.ndarray: Signal in the same units as S0 and with the same 
        dimensions as Mz.
    """  
    # Possible shapes for Mz:
    # scalar, 1D (nt, ) and 2D (nc, nt)

    input_shape = np.shape(Mz)
    Mz = np.atleast_1d(Mz)
    if Mz.ndim==1:
        Mz_total = Mz
        output_shape = input_shape
    elif Mz.ndim==2:
        Mz_total = Mz.sum(axis=0) 
        output_shape = input_shape[1:]  
    else:
        raise ValueError("Mz must be scalar, 1D or 2D") 
    sFA = np.sin(np.radians(FA))
    Mxy = S0 * np.exp(-TE * R2) * sFA * Mz_total
    signal = _signal_rice(np.abs(Mxy), noise_sdev)

    # Return result in original shape
    if output_shape == ():
        return signal[0]
    else:
        return signal.reshape(output_shape)
    

def _signal_rice(nu, sigma)-> np.ndarray:
    if sigma==0:
        return nu
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        K = nu**2 / (2*sigma**2)
        arg = K/2
        pref = sigma * np.sqrt(np.pi/2)
        rice_mean = pref * np.exp(-K/2) * ((1+K)*i0(arg) + K*i1(arg))
    # Nan values are points where the distribution is indistinguisable from Gaussian
    return np.where(np.isnan(rice_mean) | np.isinf(rice_mean), nu, rice_mean)