import numpy as np


def mix_fast_exchange(v, R, fx):
    """
    Processes compartment volume fractions and 2D properties, automatically 
    including unlisted compartments as standalone entries. Discards any compartment 
    from the group average if its R series contains one or more NaN values.
    
    Parameters:
    - v: 1D array-like, volume fractions of each compartment (length n).
    - R: 2D array-like, properties of each compartment over time (shape: n x time).
    - fx: list of lists, where each inner list contains compartment indices in fast exchange.
    
    Returns:
    - v_grouped: 1D numpy array of grouped volume fractions (sum of ALL compartments in group).
    - R_grouped: 2D numpy array of weighted-average properties for valid compartments in each group.
    """
    v = np.asarray(v, dtype=float)
    R = np.asarray(R, dtype=float)
    n_compartments = len(v)
    
    if R.ndim != 2:
        raise ValueError("Property array R must be 2D with dimensions (compartments, time).")
    
    # (1) Check that the lists in fx are distinct and find all covered indices
    seen_indices = set()
    for group in fx:
        for idx in group:
            if idx in seen_indices:
                raise ValueError(f"Compartment index {idx} appears in multiple fast-exchange groups.")
            seen_indices.add(idx)
            
    # Check bounds
    if seen_indices and (max(seen_indices) >= n_compartments or min(seen_indices) < 0 or max(seen_indices) >= R.shape[0]):
        raise IndexError("Fast-exchange indices are out of bounds for the input arrays.")

    # Identify compartments not present in any fast-exchange group
    all_compartments = set(range(n_compartments))
    isolated_compartments = all_compartments - seen_indices
    
    # Combine the explicit fast-exchange groups with the isolated compartments (as singletons)
    complete_fx = list(fx) + [[idx] for idx in sorted(isolated_compartments)]

    v_grouped = []
    R_grouped = []
    
    # Process each group (both merged fast-exchange and standalone isolated ones)
    for group in complete_fx:
        group_v = v[group]       
        group_R = R[group, :]    
        
        # Identify valid compartments in the group (no NaNs anywhere across time)
        valid_compartments = ~np.isnan(group_R).any(axis=1)
        
        # Filter down to non-NaN compartments for the average
        valid_v = group_v[valid_compartments]
        valid_R = group_R[valid_compartments, :]

        # Sum up volumes of valid compartments
        # TODO: If None return nan
        sum_valid_v = np.sum(valid_v)
        
        # Calculate weighted average of R using only valid compartments
        if sum_valid_v == 0:
            # If all compartments in the group had NaNs, return NaNs for all time points
            weighted_avg_R = np.full(R.shape[1], np.nan)
        else:
            weighted_avg_R = np.sum(valid_v[:, np.newaxis] * valid_R, axis=0) / sum_valid_v

        v_grouped.append(sum_valid_v)
        R_grouped.append(weighted_avg_R)
        
    return np.array(v_grouped), np.array(R_grouped)


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
            return R2sb + r2s_vasc * np.abs(c[0] - c[1]) + r2s_ees * c[1]
    

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
        return np.array(R2b) + np.array(r2) * np.array(c)
    
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
    return np.array(R1b) + np.array(r1) * np.array(c)

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



