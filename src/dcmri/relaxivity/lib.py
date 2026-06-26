import numpy as np


def relax_t2s(c: np.ndarray, R2sb, r2s=None, r2s_quad=None, r2s_vasc=None, r2s_ees=None, model='lin') -> np.ndarray:
    """Transverse R2* from concentrations.

    Note this requires concentrations rather than tissue concentrations, 
    though for linear or quadratic models there is no numerical difference 
    as the concentrations are averaged over the whole region.

    Args:
        c (array-like): Concentrations, either as a one-dimensionsal 
          array for one-compartment systems, or two-dimensional where the 1st 
          dimension is the number of compartments.
        R2sb (array-like or float): Precontrast R2*, either a single value for 
          one-compartment tissues or an array with one value for each tissue 
          compartment.
        r2s (float or array-like): relaxivity, either a single value for 
          one-compartment tissues or an array with one value for each tissue 
          compartment.
        r2s_quad (float or array-like): quadratic relaxivity term.
        r2s_vasc (float or array-like): vascular relaxivity.
        r2s_ees (float or array-like): EES relaxivity.
        model (str): either 'lin', 'quad' or 'leakage'.

    Returns:
        np.ndarray: Array with longitudinal relaxivities, same shape as C.
    """
    if model == 'lin':
        return R2sb + r2s * c
    if model == 'quad':
        return R2sb + r2s * c + r2s_quad * c**2
    if model == 'leakage':
        if c.ndim==1: # Equal concentrations
            return R2sb + r2s_ees * c
        else:
            return R2sb + r2s_vasc * np.abs(c[0,:] - c[1,:]) + r2s_ees * c[1,:]
    

def relax_t2(c, R2b, r2=None, model='lin') -> np.ndarray:
    """Transverse R2* from tissue concentrations assuming a linear 
    relation.

    Args:
        c (array-like): Concentrations, either as a one-dimensionsal 
          array for one-compartment systems, or two-dimensional where the 1st 
          dimension is the number of compartments.
        R2sb (array-like or float): Precontrast R2*, either a single value for 
          one-compartment tissues or an array with one value for each tissue 
          compartment.
        r2s (float or array-like): relaxivity, either a single value for 


    Returns:
        np.ndarray: Array with longitudinal relaxivities, same shape as C.
    """
    if model == 'lin':
        return relax_t1(c, R2b, r2)
    
    raise ValueError(f'Model {model} not recognized. Must be "lin".')


def relax_t1(c, R1b, r1) -> np.ndarray:
    """Derive longitudinal R1 from tissue concentrations assuming a linear 
    relation.

    Args:
        c (array-like): Concentrations, either as a one-dimensionsal 
          array for one-compartment systems, or two-dimensional where the 1st 
          dimension is the number of compartments.
        R1b (array-like or float): Precontrast R1, either a single value for 
          one-compartment tissues or an array with one value for each tissue 
          compartment.
        r1 (float or array-like): relaxivity, either a single value for 
          one-compartment tissues or an array with one value for each tissue 
          compartment.

    Returns:
        np.ndarray: Array with longitudinal relaxivities, same shape as C.
    """
    c = np.array(c)
    # One compartment tissues
    if np.isscalar(r1):
        if np.isscalar(R1b):
            # c is scalar or 1D
            return R1b + r1*c
        else:
            # concentrations at 1 time point
            if c.shape == R1b.shape:
                return R1b + r1*c
            # concentrations at multiple time points
            else:
                return R1b[..., np.newaxis] + r1 * c

    # n-compartment tissues (compartment is first dimension)
    else:

        r1 = np.array(r1)
        R1b = np.array(R1b)
        c = np.array(c)

        n = len(r1)
        R1 = np.zeros(c.shape)
        if R1b.ndim == 1:
            if c.shape == R1b.shape:
                return R1b + r1*c
            else:
                for i in range(n):
                    R1[i, :] = R1b[i] + r1[i] * c[i,:]
                return R1
        else:
            if c.shape == R1b.shape:
                for i in range(n):
                    R1[i,...] = R1b[i,...] + r1[i] * c[i,...]
                return R1
            else:
                for i in range(n):
                    R1[i,...] = R1b[i,...,np.newaxis] + r1[i] * c[i,...]
                return R1

def conc_t1(R1, r1) -> np.ndarray:
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



