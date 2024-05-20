"""PK models built from PK blocks defined in dcmri.pk"""

import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils


def conc_1cum(ca, Fp, t=None, dt=1.0):
    """Concentration in a one-compartment uptake model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: 1D array with the total concentration at each time point, in units of M.

    See Also:
        `flux_1cum`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL. In the correct units:

        >>> Fp = 1/60

        Calculate the concentrations in units of mM:
        
        >>> 1000*dc.conc_1cum(ca, Fp, t)
        array([0.        , 0.03703704, 0.07407407, 0.11111111, 0.14814815,
       0.18518519, 0.22222222, 0.25925926, 0.2962963 , 0.33333333])
    """
    return pk.conc_trap(Fp*ca, t=t, dt=dt)

def flux_1cum(ca, Fp, t=None, dt=1.0):
    """Outflux out of a one-compartment uptake model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of the compartment as a 1D array in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_1cum`

    Note:
        The outflux out of an uptake model is always zero, so this is a trivial function. It is included in the package for completeness only.

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL. In the correct units:

        >>> Fp = 1/60

        Verify that the outflux out of a 1-compartment uptake model is always zero:
        
        >>> 1000*dc.flux_1cum(ca, Fp, t)
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    return pk.flux_trap(Fp*ca)

def conc_1cm(ca, Fp, v, t=None, dt=1.0):
    """Concentration in a one-compartment model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        v (float): Dimensionless volume of distribution, with values between 0 and 1.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: 1D array with the total concentration at each time point, in units of M.

    See Also:
        `flux_1cm`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, v = 0.3. In the correct units:

        >>> Fp, v = 1/60, 0.5

        Calculate the concentrations in units of mM:
        
        >>> 1000*dc.conc_1cm(ca, Fp, v, t)
        array([0.        , 0.03569855, 0.06884832, 0.0996313 , 0.12821646,
        0.15476072, 0.17940981, 0.20229901, 0.223554  , 0.24329144])
    """
    if Fp==0:
        return np.zeros(len(ca))
    return pk.conc_comp(Fp*ca, v/Fp, t=t, dt=dt)

def flux_1cm(ca, Fp, v, t=None, dt=1.0):
    """Outflux out of a one-compartment model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        v (float): Dimensionless volume of distribution, with values between 0 and 1.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of the compartment as a 1D array in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_1cm`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, v = 0.3. In the correct units:

        >>> Fp, v = 1/60, 0.5

        Calculate the outflux in units of mM/sec:
        
        >>> 1000*dc.flux_1cm(ca, Fp, v, t)
        array([0.        , 0.00118995, 0.00229494, 0.00332104, 0.00427388,
        0.00515869, 0.00598033, 0.0067433 , 0.0074518 , 0.00810971])
    """
    if Fp==0:
        return np.zeros(len(ca))
    return pk.flux_comp(Fp*ca, v/Fp, t=t, dt=dt)

def conc_tofts(ca, Ktrans, ve, t=None, dt=1.0, sum=True):
    """Concentration in a Tofts model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Ktrans (float): Transcapillary transfer constant, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of arterial plasma fully cleared of indicator per unit of time by a unit of tissue.
        ve (float): Extravascular, extracellular volume, in units of mL/mL.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `flux_tofts`

    Note:
        The concentration 'C[0,:]' in the plasma compartment of a Tofts model is always zero by definition.

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Ktrans = 0.1 mL/min/mL, kep = 0.1/min. In the correct units:

        >>> Ktrans, ve = 0.003, 0.3

        Calculate the concentrations in units of mM:
        
        >>> C = 1000*dc.conc_tofts(ca, Ktrans, ve, t, sum=False)

        The concentration in the extravascular compartment:

        >>> C[1,:]
        array([0.        , 0.00659314, 0.01304138, 0.0193479 , 0.02551583,
        0.0315482 , 0.037448  , 0.04321814, 0.04886147, 0.05438077])
    """
    Cp = np.zeros(len(ca))
    if Ktrans==0:
        Ce = np.zeros(len(ca))
    else:
        Ce = pk.conc_comp(Ktrans*ca, ve/Ktrans, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))


def flux_tofts(ca, Ktrans, ve, t=None, dt=1.0):
    """Outfluxes out of a Tofts model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Ktrans (float): Transcapillary transfer constant, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of arterial plasma fully cleared of indicator per unit of time by a unit of tissue.
        ve (float): Extravascular, extracellular volume, in units of mL/mL.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_tofts`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Ktrans = 0.1 mL/min/mL, kep = 0.1/min. In the correct units:

        >>> Ktrans, ve = 0.003, 0.3

        Calculate the outflux in units of mM/sec:
        
        >>> J = 1000*dc.flux_tofts(ca, Ktrans, ve, t)

        The flux into the extravascular space:

        >>> J[1,0,:]
        array([0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
        0.003])
        
    """
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    if Ktrans!=0:
        J[0,1,:] = pk.flux_comp(Ktrans*ca, ve/Ktrans, t=t, dt=dt)
    return J

def conc_patlak(ca, vp, Ktrans, t=None, dt=1.0, sum=True):
    """Concentration in a Patlak model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        Ktrans (float): Transcapillary transfer constant, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of arterial plasma fully cleared of indicator per unit of time by a unit of tissue.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `flux_patlak`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by vp = 0.1, PS = 0.1/min. In the correct units:

        >>> vp, Ktrans = 0.1, 0.1/60

        Calculate the concentrations in units of mM:
        
        >>> C = 1000*dc.conc_patlak(ca, vp, Ktrans, t, sum=False)

        The concentration in the plasma compartment, in units of mM:

        >>> C[0,:]
        array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        And in the extracellular, extracellular compartment compartment (mM):

        >>> C[1,:]
        array([0.        , 0.0037037 , 0.00740741, 0.01111111, 0.01481481,
        0.01851852, 0.02222222, 0.02592593, 0.02962963, 0.03333333]) 
    """
    Cp = vp*ca
    Ce = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))

def flux_patlak(ca, vp, Ktrans, t=None, dt=1.0):
    """Outfluxes out of a Patlak model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M. 
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        Ktrans (float): Transcapillary transfer constant, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of arterial plasma fully cleared of indicator per unit of time by a unit of tissue.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_patlak`

    Note:
        In the two-compartment uptake model, the outfluxes 'J[1,1,:]' and 'J[0,1,:]' from the peripheral compartment are always 0.

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by vp = 0.1, PS = 0.1/min. In the correct units:

        >>> vp, Ktrans = 0.1, 0.1/60

        Calculate the outflux in units of mM/sec:
        
        >>> J = 1000*dc.flux_patlak(ca, vp, Ktrans, t)

        The flux out of the plasma compartment to the extravascular space:

        >>> J[1,0,:]
        array([0.00166667, 0.00166667, 0.00166667, 0.00166667, 0.00166667,
        0.00166667, 0.00166667, 0.00166667, 0.00166667, 0.00166667])

        We can verify that the backflux from the extravascular, extracellular compartment to the plasma compartment is zero:

        >>> J[0,1,:]
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        The venous outlfux from the plasma compartment is unknown because the plasma flow is not a parameter in the Patlak model:

        >>> J[0,0,:]
        array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    """
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    return J

#Use ve instead of kep for consistency with other definitions
def conc_etofts(ca, vp, Ktrans, ve, t=None, dt=1.0, sum=True):
    """Concentration in an extended Tofts model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        Ktrans (float): Transcapillary transfer constant, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of arterial plasma fully cleared of indicator per unit of time by a unit of tissue.
        ve (float): Extravascular extracellular volume fraction.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `flux_etofts`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Ktrans = 0.1 mL/min/mL, vp = 0.1, kep = 0.1/min. In the correct units:

        >>> vp, Ktrans, ve = 0.1, 1/60, 0.2

        Calculate the concentrations in units of mM:
        
        >>> C = 1000*dc.conc_etofts(ca, vp, Ktrans, ve, t, sum=False)

        The concentration in the plasma compartment, in units of mM:

        >>> C[0,:]
        array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        And in the extracellular, extracellular compartment compartment (mM):

        >>> C[1,:]
        array([0.        , 0.03380992, 0.06190429, 0.08524932, 0.10464787,
               0.12076711, 0.1341614 , 0.14529139, 0.15453986, 0.16222488])
        
    """
    Cp = vp*ca
    if Ktrans==0:
        Ce = 0*ca
    else:
        Ce = pk.conc_comp(Ktrans*ca, ve/Ktrans, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))

#Use ve instead of kep for consistency with other definitions
def flux_etofts(ca, vp, Ktrans, ve, t=None, dt=1.0):
    """Outfluxes out of an extended Tofts model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        Ktrans (float): Transcapillary transfer constant, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of arterial plasma fully cleared of indicator per unit of time by a unit of tissue.
        ve (float): Extravascular extracellular volume fraction.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_etofts`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Ktrans = 0.1 mL/min/mL, vp = 0.1, kep = 0.1/min. In the correct units:

        >>> vp, Ktrans, ve = 0.1, 1/60, 0.2

        Calculate the outflux in units of mM/sec:
        
        >>> J = 1000*dc.flux_etofts(ca, vp, Ktrans, ve, t)

        The flux from the plasma to the extravascular space:

        >>> J[1,0,:]
        array([0.01666667, 0.01666667, 0.01666667, 0.01666667, 0.01666667,
               0.01666667, 0.01666667, 0.01666667, 0.01666667, 0.01666667])
        
        Note the venous outflux out of the plasma compartment is undetermined because the Extended Tofts model does not depend on the plasma flow:

        >>> J[0,0,:]
        array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    """
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    if Ktrans==0:
        J[0,1,:] = 0*ca
    else:
        J[0,1,:] = pk.flux_comp(Ktrans*ca, ve/Ktrans, t=t, dt=dt)
    return J

def conc_2cum(ca, Fp, vp, PS, t=None, dt=1.0, sum=True):
    """Concentration in a two-compartment uptake model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        PS (float): Permeability-Surface area product of the capillary wall, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of plasma fully cleared of indicator per unit of time by a unit of tissue.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `flux_2cum`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, vp = 0.1, PS = 0.1/min. In the correct units:

        >>> Fp, vp, PS = 1/60, 0.1, 0.1/60

        Calculate the concentrations in units of mM:
        
        >>> C = 1000*dc.conc_2cum(ca, Fp, vp, PS, t, sum=False)

        The concentration in the plasma compartment, in units of mM:

        >>> C[0,:]
        array([0.        , 0.03042063, 0.0506617 , 0.06412956, 0.07309071,
        0.07905322, 0.08302052, 0.08566025, 0.08741665, 0.08858532])

        And in the extracellular, extracellular compartment compartment (mM):

        >>> C[1,:]
        array([0.        , 0.00056335, 0.00206487, 0.00419063, 0.00673175,
        0.00954923, 0.0125506 , 0.01567431, 0.01887944, 0.02213874])
    """
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

def flux_2cum(ca, Fp, vp, PS, t=None, dt=1.0):
    """Outfluxes out of a 2-compartment uptake model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M. 
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        PS (float): Permeability-Surface area product of the capillary wall, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of plasma fully cleared of indicator per unit of time by a unit of tissue.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_2cum`

    Note:
        In the two-compartment uptake model, the outfluxes 'J[1,1,:]' and 'J[0,1,:]' from the peripheral compartment are always 0.

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, vp = 0.1, PS = 0.1/min. In the correct units:

        >>> Fp, vp, PS = 1/60, 0.1, 0.1/60

        Calculate the outflux in units of mM/sec:
        
        >>> J = 1000*dc.flux_2cum(ca, Fp, vp, PS, t)

        The flux out of the plasma compartment:

        >>> J[0,0,:]
        array([0.        , 0.00507011, 0.00844362, 0.01068826, 0.01218179,
        0.01317554, 0.01383675, 0.01427671, 0.01456944, 0.01476422])

        We can verify that the backflux from the extravascular, extracellular compartment to the plasma compartment is zero:

        >>> J[0,1,:]
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    C = conc_2cum(ca, Fp, vp, PS, t=t, dt=dt, sum=False)
    J = np.zeros(((2,2,len(ca))))
    if vp==0:
        J[0,0,:] = Fp*ca
        Ktrans = Fp*PS/(Fp+PS)
        J[1,0,:] = Ktrans*ca
    else:
        J[0,0,:] = Fp*C[0,:]/vp
        J[1,0,:] = PS*C[0,:]/vp
    return J

def conc_2cxm(ca, Fp, vp, PS, ve, t=None, dt=1.0, sum=True):
    """Concentration in a two-compartment exchange model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        PS (float): Permeability-Surface area product of the capillary wall, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of plasma fully cleared of indicator per unit of time by a unit of tissue.
        ve (float): Dimensionless extravascular, extracellular volume fraction of tissue, with values between 0 and 1.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `flux_2cxm`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, vp = 0.1, PS = 0.1/min and ve = 0.3. In the correct units:

        >>> Fp, vp, PS, ve = 1/60, 0.1, 0.1/60, 0.3

        Calculate the concentrations:
        
        >>> C = dc.conc_2cxm(ca, Fp, vp, PS, ve, t, sum=False)

        The concentration in the plasma compartment, in units of mM:

        >>> C[0,:]*1000
        array([0.        , 0.03042294, 0.05067687, 0.06417205, 0.07317494,
        0.07919185, 0.08322389, 0.0859364 , 0.08777159, 0.08902333])

        And in the extracellular, extracellular compartment (mM) compartment (mM):

        >>> C[1,:]*1000
        array([0.        , 0.00059896, 0.00211018, 0.00421529, 0.00670287,
        0.00943251, 0.01231105, 0.01527671, 0.01828855, 0.0213195 ])

        The total tissue concentration, in mM:

        >>> (C[0,:]+C[1,:])*1000
        array([0.        , 0.0310219 , 0.05278705, 0.06838735, 0.07987781,
        0.08862436, 0.09553494, 0.10121311, 0.10606014, 0.11034283])
    """
    if Fp+PS == 0:
        if sum:
            return np.zeros(len(ca))
        else:
            return np.zeros((2,len(ca)))
    if PS == 0:
        Cp = conc_1cm(ca, Fp, vp, t=t, dt=dt)
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

def flux_2cxm(ca, Fp, vp, PS, ve, t=None, dt=1.0):
    """Outfluxes out of a 2-compartment exchange model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M. 
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        PS (float): Permeability-Surface area product of the capillary wall, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of plasma fully cleared of indicator per unit of time by a unit of tissue.
        ve (float): Dimensionless extravascular, extracellular volume fraction of tissue, with values between 0 and 1.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_2cxm`

    Note:
        In the two-compartment exchange model, the outflux 'J[1,1,:]' from the peripheral compartment to the outside is always 0.

    Example:

        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a 2-compartment exchange model with a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, vp = 0.1, PS = 0.1/min and ve = 0.3. In the correct units:

        >>> Fp, vp, PS, ve = 1/60, 0.1, 0.1/60, 0.3

        Calculate the outflux:
        
        >>> J = dc.flux_2cxm(ca, Fp, vp, PS, ve, t)

        The flux out of the plasma compartment, in units of mM/sec:

        >>> J[0,0,:]*1000
        array([0.        , 0.00507049, 0.00844615, 0.01069534, 0.01219582,
        0.01319864, 0.01387065, 0.01432273, 0.0146286 , 0.01483722])

        The backflux from the extravascular, extracellular compartment to the plasma compartment:

        >>> J[0,1,:]*1000
        array([0.00000000e+00, 3.32758154e-06, 1.17232233e-05, 2.34183019e-05,
        3.72381463e-05, 5.24028209e-05, 6.83947251e-05, 8.48705993e-05,
        1.01603044e-04, 1.18441651e-04])

        And we can verify that there is no leakage from extravascular, extracellular compartment to the environment:

        >>> J[1,1,:]
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    if Fp+PS == 0:
        return np.zeros((2,2,len(ca)))
    if PS == 0:
        Jp = flux_1cm(ca, Fp, vp, t=t, dt=dt)
        J = np.zeros((2,2,len(ca)))
        J[0,0,:] = Jp
        return J
    C = conc_2cxm(ca, Fp, vp, PS, ve, t=t, dt=dt, sum=False)
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


def conc_2cfm(ca, Fp, vp, PS, Te, t=None, dt=1.0, sum=True):
    """Concentration in a two-compartment filtration model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M. 
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        PS (float): Permeability-Surface area product of the capillary wall, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of plasma fully cleared of indicator per unit of time by a unit of tissue.
        Te (float): Mean transit time of the extravascular, extracellular space, in units of sec.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments are returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `flux_2cfm`

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a 2-compartment filtration model with a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, vp = 0.1, PS = 0.1/min and Te = 2min. In the correct units:

        >>> Fp, vp, PS, Te = 1/60, 0.1, 0.1/60, 2*60

        Calculate the concentrations:
        
        >>> C = dc.conc_2cfm(ca, Fp, vp, PS, Te, t, sum=False)

        The concentration in the plasma compartment, in units of mM:

        >>> C[0,:]*1000
        array([0.        , 0.03042063, 0.0506617 , 0.06412956, 0.07309071,
        0.07905322, 0.08302052, 0.08566025, 0.08741665, 0.08858532])

        And in the extravascular, extracellular compartment (mM):

        >>> C[1,:]*1000
        array([0.        , 0.00055988, 0.00203846, 0.00410803, 0.00655089,
        0.00922259, 0.01202734, 0.01490178, 0.0178041 , 0.02070679])

        The total tissue concentration, in mM:

        >>> (C[0,:]+C[1,:])*1000
        array([0.        , 0.03098051, 0.05270016, 0.06823759, 0.07964161,
        0.08827581, 0.09504785, 0.10056203, 0.10522075, 0.10929211])
    """
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

def flux_2cfm(ca, Fp, vp, PS, Te, t=None, dt=1.0):
    """Outfluxes out of a 2-compartment filtration model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec).
        vp (float): Dimensionless plasma volume fraction of tissue, with values between 0 and 1.
        PS (float): Permeability-Surface area product of the capillary wall, in units of mL plasma per sec and per mL tissue (mL/sec/mL or 1/sec). Physically PS is the volume of plasma fully cleared of indicator per unit of time by a unit of tissue.
        Te (float): Mean transit time of the extravascular, extracellular space, in units of sec.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: Outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    See Also:
        `conc_2cfm`

    Note:
        In the two-compartment filtration model, the backflux 'J[0,1,:]' from the extravascular, extracellular compartment to the plasma compartment is always 0.

    Example:
        >>> import numpy as np
        >>> import dcmri as dc

        Consider a measurement with 10 time points from 0 to 20s, and a 2-compartment filtration model with a constant input concentration of 1mM:

        >>> t = np.linspace(0, 20, 10)
        >>> ca = 0.001*np.ones(t.size)

        The tissue is characterized by Fp = 1 mL/min/mL, vp = 0.1, PS = 0.1/min and Te = 2min. In the correct units:

        >>> Fp, vp, PS, Te = 1/60, 0.1, 0.1/60, 2*60

        Calculate the outfluxes:
        
        >>> J = dc.flux_2cfm(ca, Fp, vp, PS, Te, t)

        The flux out of the central compartment in mM/sec:

        >>> J[0,0,:]*1000
        array([0.        , 0.00507011, 0.00844362, 0.01068826, 0.01218179,
        0.01317554, 0.01383675, 0.01427671, 0.01456944, 0.01476422])

        And we can verify that the backflux from the extravascular compartment to the plasma compartment is zero:

        >>> J[0,1,:]
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
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
    J0 = pk.flux_comp(J, T[0], t)   
    J10 = E*J0
    Jo[1,0,:] = J10
    Jo[1,1,:] = pk.flux_comp(J10, T[1], t)
    Jo[0,0,:] = (1-E)*J0
    return Jo


