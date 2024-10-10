import numpy as np


def c_lin(R1, r1) -> np.ndarray:
    """Derive concentrations from relaxation rates using a linear relationship.

    Args:
        R1 (float or array-like): Relaxation rates. For a multi-compartmental 
          tissue, R1 is a 2-dimensional array where the first dimension is the 
          number of compartments.
        r1 (float or array-like): relaxivity. For a multi-compartmental 
          tissue, r1 can be a 1-dimensional array with the relaxation rates 
          for each compartment. If it is a scalar, the assumption is that all 
          compartments have the same r1.

    Returns:
        np.ndarray: concentrations in each tissue compartment, in the same 
        shape as R1.
    """
    if R1.ndim == 2:
        c = np.zeros(R1.shape)
        for i in range(R1.ndim):
            if np.isscalar(r1):
                r1i = r1
            else:
                r1i = r1[i]
            c[i, :] = (R1[i, :]-R1[i, 0])/r1i
    else:
        c = (R1-R1[0])/r1
    return c


def relax(c, R10, r1) -> np.ndarray:
    """Derive longitudinal R1 from tissue concentrations assuming a linear 
    relation.

    Args:
        c (array-like): Concentrations, either as a one-dimensionsal 
          array for one-compartment systems, or two-dimensional where the 1st 
          dimension is the number of compartments.
        R10 (array-like or float): Precontrast R1, either a single value for 
          one-compartment tissues or an array with one value for each tissue 
          compartment.
        r1 (float or array-like): relaxivity, either a single value for 
          one-compartment tissues or an array with one value for each tissue 
          compartment.

    Returns:
        np.ndarray: Array with longitudinal relaxvities, same shape as C.
    """

    if np.isscalar(R10):
        return R10 + r1*c
    else:
        R1 = np.zeros(c.shape)
        for i in range(c.shape[0]):
            R1[i, :] = R10[i] + r1[i] * c[i, :]
        return R1
