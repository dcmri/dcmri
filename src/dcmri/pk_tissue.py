"""PK models built from PK blocks defined in dcmri.pk"""

import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils


def conc_tissue(ca:np.ndarray, *params, t=None, dt=1.0, kinetics='2CX', sum=True)->np.ndarray:
    """Tissue concentration in a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        params (tuple): free model parameters.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        kinetics (str, optional): Kinetics of the tissue, either 'U', 'NX', 'FX', 'WV', 'HFU', 'HF', '2CU', '2CX', '2CF' - see below for detail. Defaults to '2CX'. 
        sum (bool, optional): For two-compartment tissues, set to True to return the total tissue concentration. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where *k* is the number of time points in *ca*. The concentration is returned in units of M.

    Notes:
        Currently implemented kinetic models are: 
        
        - 'U': uptake tissue. params = Fp
        - 'NX': no tracer exchange tissue. params = (Fp, vp, )
        - 'FX': fast tracer exchange tissue. params = (Fp, v, )
        - 'WV': weakly vascularized tissue - also known as *Tofts model*. params = (Ktrans, ve, )
        - 'HFU': high-flow uptake tissue - also known as *Patlak model*. params = (vp, PS, )
        - 'HF': high-flow tissue - also known as *extended Tofts model*, *extended Patlak model* or *general kinetic model*. Params = (vp, PS, ve, )
        - '2CU': two-compartment uptake tissue. params = (Fp, vp, PS, )
        - '2CX': two-compartment exchange tissue. params = (Fp, vp, PS, ve, )
        - '2CF': two-compartment filtration tissue. params = (Fp, vp, PS, Te, )

        The model parameters are:

        - **Fp** (mL/sec/mL): Plasma flow.
        - **PS** (mL/sec/mL): permeability surface-area product.
        - **Ktrans** (mL/mL): volume transfer constant = Fp*PS/(FP+PS) 
        - **vp** (mL/mL): plasma volume fraction.
        - **ve** (mL/mL): extracellular, extravascular volume fraction.
        - **v** (mL/mL): extracellular volume fraction = vp+ve.
        - **Te** (sec): extracellular, extravascular mean transit time.

    Example:

        Plot concentration in cortex and medulla for typical values:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define some tissue parameters: 

        >>> Fp, vp, PS, ve = 0.01, 0.05, 0.005, 0.4
        >>> Ktrans = PS*Fp/(PS+Fp)

        Set up a plot to show concentrations:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        
        Generate plasma and extravascular tissue concentrations with the 2CX model and add to the plot:

        >>> C = dc.conc_tissue(ca, Fp, vp, PS, ve, t=t, sum=False, kinetics='2CX')
        >>> ax0.set_title('2-compartment exchange model')
        >>> ax0.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma')
        >>> ax0.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Extravascular, extracellular space')
        >>> ax0.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Tissue')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Tissue concentration (mM)')
        >>> ax0.legend()

        Generate plasma and extravascular tissue concentrations with the WV model and add to the plot: 

        >>> C = dc.conc_tissue(ca, Ktrans, ve, t=t, sum=False, kinetics='WV')
        >>> ax1.set_title('Weakly vascularised model')
        >>> ax1.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma (WV)')
        >>> ax1.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Extravascular, extracellular space (WV)')
        >>> ax1.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Tissue (WV)')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()

    """
    if kinetics=='U':
        return _conc_u(ca, *params, t=t, dt=dt)
    elif kinetics=='NX':
        return _conc_1c(ca, *params, t=t, dt=dt)
    elif kinetics=='FX':
        return _conc_1c(ca, *params, t=t, dt=dt)
    elif kinetics=='WV':
        return _conc_wv(ca, *params, t=t, dt=dt, sum=sum)
    elif kinetics=='HFU':
        return _conc_hfu(ca, *params, t=t, dt=dt, sum=sum)
    elif kinetics=='HF':
        return _conc_hf(ca, *params, t=t, dt=dt, sum=sum)
    elif kinetics=='2CU':
        return _conc_2cu(ca, *params, t=t, dt=dt, sum=sum)
    elif kinetics=='2CX':
        return _conc_2cx(ca, *params, t=t, dt=dt, sum=sum)
    elif kinetics=='2CF':
        return _conc_2cf(ca, *params, t=t, dt=dt, sum=sum)
    else:
        raise ValueError('Kinetic model ' + kinetics + ' is not currently implemented.')



def flux_tissue(ca:np.ndarray, *params, t=None, dt=1.0, kinetics='2CX')->np.ndarray:
    """Indicator out of a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        params (tuple): free model parameters.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        kinetics (str, optional): Kinetics of the tissue, either 'U', 'NX', 'FX', 'WV', 'HFU', 'HF', '2CU', '2CX', '2CF' - see below for detail. Defaults to '2CX'. 

    Returns:
        numpy.ndarray: For a one-compartmental tissue, outflux out of the compartment as a 1D array in units of mmol/sec/mL or M/sec. For a multi=compartmental tissue, outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    Notes:
        Currently implemented kinetic models are: 
        
        - 'U': uptake tissue. params = Fp
        - 'NX': no tracer exchange tissue. params = (Fp, vp, )
        - 'FX': fast tracer exchange tissue. params = (Fp, v, )
        - 'WV': weakly vascularized tissue - also known as *Tofts model*. params = (Ktrans, ve, )
        - 'HFU': high-flow uptake tissue - also known as *Patlak model*. params = (vp, PS, )
        - 'HF': high-flow tissue - also known as *extended Tofts model*, *extended Patlak model* or *general kinetic model*. Params = (vp, PS, ve, )
        - '2CU': two-compartment uptake tissue. params = (Fp, vp, PS, )
        - '2CX': two-compartment exchange tissue. params = (Fp, vp, PS, ve, )
        - '2CF': two-compartment filtration tissue. params = (Fp, vp, PS, Te, )

        The model parameters are:

        - **Fp** (mL/sec/mL): Plasma flow.
        - **PS** (mL/sec/mL): permeability surface-area product.
        - **Ktrans** (mL/mL): volume transfer constant = Fp*PS/(FP+PS) 
        - **vp** (mL/mL): plasma volume fraction.
        - **ve** (mL/mL): extracellular, extravascular volume fraction.
        - **v** (mL/mL): extracellular volume fraction = vp+ve.
        - **Te** (sec): extracellular, extravascular mean transit time.
        
    """
    if kinetics=='U':
        return _flux_u(ca, *params, t=t, dt=dt)
    elif kinetics=='NX':
        return _flux_1c(ca, *params, t=t, dt=dt)
    elif kinetics=='FX':
        return _flux_1c(ca, *params, t=t, dt=dt)
    elif kinetics=='WV':
        return _flux_wv(ca, *params, t=t, dt=dt)
    elif kinetics=='HFU':
        return _flux_hfu(ca, *params, t=t, dt=dt)
    elif kinetics=='HF':
        return _flux_hf(ca, *params, t=t, dt=dt)
    elif kinetics=='2CU':
        return _flux_2cu(ca, *params, t=t, dt=dt)
    elif kinetics=='2CX':
        return _flux_2cx(ca, *params, t=t, dt=dt)
    elif kinetics=='2CF':
        return _flux_2cf(ca, *params, t=t, dt=dt)
    else:
        raise ValueError('Kinetic model ' + kinetics + ' is not currently implemented.')



def _conc_u(ca, Fp, t=None, dt=1.0):
    return pk.conc_trap(Fp*ca, t=t, dt=dt)

def _conc_1c(ca, Fp, v, t=None, dt=1.0):
    if Fp==0:
        return np.zeros(len(ca))
    return pk.conc_comp(Fp*ca, v/Fp, t=t, dt=dt)

def _conc_wv(ca, Ktrans, ve, t=None, dt=1.0, sum=True):
    Cp = np.zeros(np.size(ca))
    if Ktrans==0:
        Ce = np.zeros(np.size(ca))
    else:
        Ce = pk.conc_comp(Ktrans*ca, ve/Ktrans, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))

def _conc_hfu(ca, vp, Ktrans, t=None, dt=1.0, sum=True):
    Cp = vp*ca
    Ce = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    
def _conc_hf(ca, vp, Ktrans, ve, t=None, dt=1.0, sum=True):
    Cp = vp*ca
    if Ktrans==0:
        Ce = 0*ca
    else:
        Ce = pk.conc_comp(Ktrans*ca, ve/Ktrans, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    
def _conc_2cu(ca, Fp, vp, PS, t=None, dt=1.0, sum=True):
    if Fp+PS==0:
        return np.zeros((2,len(ca)))
    Tp = vp/(Fp+PS)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    if vp==0:
        Ktrans = Fp*PS/(Fp+PS)
        Ce = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ce = pk.conc_trap(PS*Cp/vp, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    
def _conc_2cx(ca, Fp, vp, PS, ve, t=None, dt=1.0, sum=True):
    if Fp+PS == 0:
        if sum:
            return np.zeros(len(ca))
        else:
            return np.zeros((2,len(ca)))
    if PS == 0:
        Cp = _conc_1c(ca, Fp, vp, t=t, dt=dt)
        Ce = np.zeros(len(ca))
        if sum:
            return Cp+Ce
        else:
            return np.stack((Cp,Ce))
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    Te = ve/PS
    J = Fp*ca
    E = PS/(Fp+PS)
    # Build the system matrix K
    T = [Tp, Te]
    E = [
        [1-E, 1],
        [E,   0],
    ]
    Q, K, Qi = pk.K_2comp(T, E)
    # Initialize concentration-time array
    nc, nt = 2, len(J)
    t = utils.tarray(nt, t=t, dt=dt)
    Ei = np.empty((nc,nt))
    # Loop over the eigenvalues
    for d in [0,1]:
        # Calculate elements of diagonal matrix
        Ei[d,:] = pk.conc_comp(J, 1/K[d], t)
        # Right-multiply with inverse eigenvector matrix
        Ei[d,:] *= Qi[d,0]
    # Left-multiply with eigenvector matrix
    C = np.matmul(Q, Ei)
    if sum:
        return np.sum(C, axis=0)
    else:
        return C
    
def _conc_2cf(ca, Fp, vp, PS, Te, t=None, dt=1.0, sum=True):
    if Fp+PS == 0:
        if sum:
            return np.zeros(len(ca))
        else:
            return np.zeros((2,len(ca)))
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)
    J = Fp*ca
    T = [Tp, Te]
    # Solve the system explicitly
    t = utils.tarray(len(J), t=t, dt=dt)
    C0 = pk.conc_comp(J, T[0], t)
    if E==0:
        C1 = np.zeros(len(t))
    elif T[0]==0:
        J10 = E*J
        C1 = pk.conc_comp(J10, T[1], t)
    else:
        J10 = C0*E/T[0]
        C1 = pk.conc_comp(J10, T[1], t)
    if sum:
        return C0+C1
    else:
        return np.stack((C0, C1))





def _flux_u(ca, Fp, t=None, dt=1.0):
    return pk.flux(Fp*ca, kinetics='trap')

def _flux_1c(ca, Fp, v, t=None, dt=1.0):
    if Fp==0:
        return np.zeros(len(ca))
    return pk.flux(Fp*ca, v/Fp, t=t, dt=dt, kinetics='comp')

def _flux_wv(ca, Ktrans, ve, t=None, dt=1.0):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    if Ktrans!=0:
        J[0,1,:] = pk.flux(Ktrans*ca, ve/Ktrans, t=t, dt=dt, kinetics='comp')
    return J

def _flux_hfu(ca, vp, Ktrans, t=None, dt=1.0):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    return J

def _flux_hf(ca, vp, Ktrans, ve, t=None, dt=1.0):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    if Ktrans==0:
        J[0,1,:] = 0*ca
    else:
        J[0,1,:] = pk.flux(Ktrans*ca, ve/Ktrans, t=t, dt=dt, kinetics='comp')
    return J

def _flux_2cu(ca, Fp, vp, PS, t=None, dt=1.0):
    C = _conc_2cu(ca, Fp, vp, PS, t=t, dt=dt, sum=False)
    J = np.zeros(((2,2,len(ca))))
    if vp==0:
        J[0,0,:] = Fp*ca
        Ktrans = Fp*PS/(Fp+PS)
        J[1,0,:] = Ktrans*ca
    else:
        J[0,0,:] = Fp*C[0,:]/vp
        J[1,0,:] = PS*C[0,:]/vp
    return J


def _flux_2cx(ca, Fp, vp, PS, ve, t=None, dt=1.0):
    if Fp+PS == 0:
        return np.zeros((2,2,len(ca)))
    if PS == 0:
        Jp = _flux_1c(ca, Fp, vp, t=t, dt=dt)
        J = np.zeros((2,2,len(ca)))
        J[0,0,:] = Jp
        return J
    C = _conc_2cx(ca, Fp, vp, PS, ve, t=t, dt=dt, sum=False)
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    Te = ve/PS
    J = Fp*ca
    E = PS/(Fp+PS)
    # Build the system matrix K
    T = [Tp, Te]
    E = [
        [1-E, 1],
        [E,   0],
    ]
    return pk.J_ncomp(C, T, E)


def _flux_2cf(ca, Fp, vp, PS, Te, t=None, dt=1.0):
    if Fp+PS == 0:
        return np.zeros((2,2,len(ca)))
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)
    J = Fp*ca
    T = [Tp, Te]
    # Solve the system explicitly
    t = utils.tarray(len(J), t=t, dt=dt)
    Jo = np.zeros((2,2,len(t)))
    J0 = pk.flux(J, T[0], t=t, kinetics='comp')   
    J10 = E*J0
    Jo[1,0,:] = J10
    Jo[1,1,:] = pk.flux(J10, T[1], t=t, kinetics='comp')
    Jo[0,0,:] = (1-E)*J0
    return Jo


