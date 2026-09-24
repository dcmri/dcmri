import numpy as np

from dcmri.utils import convolution

def conc_ls(ca, irf, dt):
    ca = np.array(ca)
    # if ca.size != irf.shape[-1]:
    #     raise ValueError("Cannot compute concentrations as IRF and AIF have different number of time points")
    
    ca_mat = dt * convolution.convmat(ca)
    # Reshape IRF to 2D (n_samples, n_times)
    shape = irf.shape
    irf = irf.reshape(-1, irf.shape[-1])
    # Convolve with transpose because for matrix multiplication 
    # we need (rows, columns) = (n_times, n_samples)
    conc = ca_mat @ irf.T
    # Transpose back to get result in standard form (n_samples, n_times)
    # Then convert back to original shape 
    return conc.T.reshape(shape)


def irf_ls(ca, c, dt, tol=1e-2):
    # Reshape c to 2D (n_samples, n_times)
    shape = c.shape
    c = c.reshape(-1, c.shape[-1])
    # Deconvolve with arterial concentration
    # Use transpose because for matrix multiplication 
    # we need (rows, columns) = (n_times, n_samples)
    irf = convolution.deconv(c.T, ca, dt, tol=tol)
    # Transpose back to get result in standard form (n_samples, n_times)
    # Then convert back to original shape 
    return irf.T.reshape(shape)