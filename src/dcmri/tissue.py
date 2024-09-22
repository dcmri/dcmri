"""PK models built from PK blocks defined in dcmri.pk"""

import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils


def relax_tissue(ca:np.ndarray, R10:float, r1:float, t=None, dt=1.0, model='2CX', **params):
    """Relaxation rates for a 2-site exchange tissue and different water exchange regimes

    This function returns the free relaxation rates for each distinct tissue compartment as a function of time. The free relaxation rates are the relaxation rates of the tissue compartments in the absence of water exchange between them.

    Args:
        ca (array-like): concentration in the arterial input.
        R10 (float): precontrast relaxation rate. The tissue is assumed to be in fast exchange before injection of contrast agent.
        r1 (float): contrast agent relaivity. 
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        model (str, optional): Short name of the model to use, as a string specifying the water exchange regime and kinetic model separated by '-' (see `dcmri.model_props` for details).
        params (dict): model parameters (see `dcmri.model_props` for detail on available models).

    Returns:
        np.ndarray: in the fast water exchange limit, the relaxation rates are a 1D array. In all other situations, relaxation rates are a 2D-array with dimensions (k,n), where k is the number of compartments and n is the number of time points in ca.

    Example:

        Plot the free relaxation rates for a 2-compartment exchange model with intermediate water exchange:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define constants and model parameters: 

        >>> R10, r1 = 1/dc.T1(), dc.relaxivity()                    # Constants
        >>> p = {vc:0.3, vp:0.05, vi:0.4, Fp:0.01, Ktrans:0.005}    # Parameters

        Calculate tissue relaxation rate R1a with intermediate water exchange, and also in the fast exchange limit for comparison:

        >>> R1r = dc.relax_tissue(ca, R10, r1, t=t, model='RR-2CX', **p)
        >>> del p['vc']
        >>> R1f = dc.relax_tissue(ca, R10, r1, t=t, model='2CX', **p)

        Plot the relaxation rates in the three compartments, and compare against the fast exchange result:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> ax0.set_title('Restricted water exchange')
        >>> ax0.plot(t/60, R1r[0,:], linestyle='-', linewidth=2.0, color='darkred', label='Blood')
        >>> ax0.plot(t/60, R1r[1,:], linestyle='-', linewidth=2.0, color='darkblue', label='Interstitium')
        >>> ax0.plot(t/60, R1r[2,:], linestyle='-', linewidth=2.0, color='grey', label='Cells')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Compartment relaxation rate (1/sec)')
        >>> ax0.legend()
        >>> ax1.set_title('Fast water exchange')
        >>> ax1.plot(t/60, R1f, linestyle='-', linewidth=2.0, color='black', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue relaxation rate (1/sec)')
        >>> ax1.legend()
        >>> plt.show()

    """

    # FA: PSw = np.array([[0,PSc],[PSc,0]]), v = np.array([1-vc, vc])
    # AF: PSw = np.array([[0,PSe],[PSe,0]]), v = np.array([vb, 1-vb])
    # A: PSw = np.array([[0,PSe,0],[PSe,0,PSc],[0,PSc,0]]), v = np.array([vb, vi, vc])


    # Consider N scenario and FF vs None

    mdl = model.split('-')
    kinetics = mdl[0] if len(mdl)==1 else mdl[1]
    wex = 'FF' if len(mdl)==1 else mdl[0]

    if kinetics not in ['U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU', '2CX']:
        msg = "Kinetic model '" + str(kinetics) + "' is not recognised.\n"
        msg += "Possible values are: 'U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU' and '2CX'."
        raise ValueError(msg)
    
    if wex not in ['FF','NF','RF','FN','NN','RN','FR','NR','RR']:
        msg = "Water exchange regime '" + str(wex) + "' is not recognised.\n"
        msg += "Possible values are: 'FF','NF','RF','FN','NN','RN','FR','NR','RR'."
        raise ValueError(msg)
    
    # No water exchange is the same tissue model as restricted water exchange
    wex = wex.replace('N','R')
    model = wex + '-' + kinetics

    if model=='FF-U':
        return _relax_u_ff(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FF-FX':
        return _relax_fx_ff(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FF-NX':
        return _relax_nx_ff(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FF-WV':
        return _relax_wv_ff(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FF-HFU':
        return _relax_hfu_ff(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FF-HF':
        return _relax_hf_ff(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FF-2CU':
        return _relax_2cu_ff(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FF-2CX':
        return _relax_2cx_ff(ca, R10, r1, t=t, dt=dt, **params)
    
    
    elif model=='RF-U':
        return _relax_u_rf(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RF-FX':
        return _relax_fx_rf(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RF-NX':
        return _relax_nx_rf(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RF-WV':
        return _relax_wv_rf(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RF-HFU':
        return _relax_hfu_rf(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RF-HF':
        return _relax_hf_rf(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RF-2CU':
        return _relax_2cu_rf(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RF-2CX':
        return _relax_2cx_rf(ca, R10, r1, t=t, dt=dt, **params)
    
    
    elif model=='FR-U':
        return _relax_u_fr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FR-FX':
        return _relax_fx_fr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FR-NX':
        return _relax_nx_fr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FR-WV':
        return _relax_wv_fr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FR-HFU':
        return _relax_hfu_fr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FR-HF':
        return _relax_hf_fr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FR-2CU':
        return _relax_2cu_fr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='FR-2CX':
        return _relax_2cx_fr(ca, R10, r1, t=t, dt=dt, **params)
    
    
    elif model=='RR-U':
        return _relax_u_rr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RR-FX':
        return _relax_fx_rr(ca, R10, r1, t=t, dt=dt, **params) 
    elif model=='RR-NX':
        return _relax_nx_rr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RR-WV':
        return _relax_wv_rr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RR-HFU':
        return _relax_hfu_rr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RR-HF':
        return _relax_hf_rr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RR-2CU':
        return _relax_2cu_rr(ca, R10, r1, t=t, dt=dt, **params)
    elif model=='RR-2CX':
        return _relax_2cx_rr(ca, R10, r1, t=t, dt=dt, **params)
        


def _relax_2cx_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_2cx(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_2cu_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_2cu(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_hf_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_hf(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_hfu_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_hfu(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_nx_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_nx(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_wv_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_wv(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_u_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_u(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_fx_ff(ca, R10, r1, t=None, dt=1.0, ve=None, Fp=None):
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    R1 = R10 + r1*C
    return R1



def _relax_2cx_fr(ca, R10, r1, t=None, dt=1.0, vc=None, **params):
    C = _conc_2cx(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1

def _relax_2cu_fr(ca, R10, r1, t=None, dt=1.0, vc=None, **params):
    C = _conc_2cu(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1

def _relax_hf_fr(ca, R10, r1, t=None, dt=1.0, vc=None, **params):
    C = _conc_hf(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1

def _relax_hfu_fr(ca, R10, r1, t=None, dt=1.0, vc=None, **params):
    C = _conc_hfu(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1

def _relax_nx_fr(ca, R10, r1, t=None, dt=1.0, vc=None, **params):
    C = _conc_nx(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1

def _relax_wv_fr(ca, R10, r1, t=None, dt=1.0, vi=None, **params):
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, **params)
    vc = 1-vi
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1

def _relax_u_fr(ca, R10, r1, t=None, dt=1.0, vc=None, **params):
    C = _conc_u(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1

def _relax_fx_fr(ca, R10, r1, t=None, dt=1.0, vc=None, ve=None, Fp=None):
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1



def _relax_2cx_rf(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, **params)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1

def _relax_2cu_rf(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, **params)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1

def _relax_hf_rf(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    C = _conc_hf(ca, t=t, dt=dt, sum=False, **params)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1

def _relax_hfu_rf(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, **params)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1

def _relax_nx_rf(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    C = _conc_nx(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1

def _relax_wv_rf(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_wv(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1

def _relax_u_rf(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    C = _conc_u(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1

def _relax_fx_rf(ca, R10, r1, t=None, dt=1.0, vb=None, vc=None, ve=None, Fp=None):
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    v_rbc = 1-vc-ve
    vp = vb-v_rbc
    vi = ve-vp
    R1 = np.full((2,C.size), R10)
    if ve != 0:
        ce = C/ve
        if vb != 0:
            R1[0,:] += r1*ce*vp/vb
        if 1-vb != 0:
            R1[1,:] += r1*ce*vi/(1-vb)
    return R1



def _relax_2cx_rr(ca, R10, r1, t=None, dt=1.0, vc=None, vi=None, **params):
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, vi=vi, **params)
    vb = 1-vi-vc
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1

def _relax_2cu_rr(ca, R10, r1, t=None, dt=1.0, vb=None, vc=None, **params):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, **params)
    vi = 1-vb-vc
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1
    
def _relax_hf_rr(ca, R10, r1, t=None, dt=1.0, vc=None, vi=None, **params):
    C = _conc_hf(ca, t=t, dt=dt, sum=False, vi=vi, **params)
    vb = 1-vi-vc
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1

def _relax_hfu_rr(ca, R10, r1, t=None, dt=1.0, vb=None, vc=None, **params):
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, **params)
    vi = 1-vb-vc
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1

def _relax_nx_rr(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    # Permeability across cell wall plays no role if ci=0.
    C = _conc_nx(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1

def _relax_wv_rr(ca, R10, r1, t=None, dt=1.0, vi=None, **params):
    # Permeability across endothelium plays no role if vp=0
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, **params)
    R1 = np.full((2,C.size), R10)
    if vi != 0:
        R1[0,:] += r1*C/vi
    return R1

def _relax_u_rr(ca, R10, r1, t=None, dt=1.0, vb=None, **params):
    # Permeability across cell wall plays no role if ci=0
    C = _conc_u(ca, t=t, dt=dt, **params)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1

def _relax_fx_rr(ca, R10, r1, t=None, dt=1.0, vb=None, vc=None, ve=None, Fp=None):
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    vi = 1-vb-vc
    vp = ve-vi
    R1 = np.full((3,C.size), R10)
    if ve != 0:
        ce = C/ve
        if vb != 0:
            R1[0,:] += r1*ce*vp/vb
        R1[1,:] += r1*ce
    return R1






def conc_tissue(ca:np.ndarray, t=None, dt=1.0, model='2CX', sum=True, **params)->np.ndarray:
    """Tissue concentration in a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        model (str, optional): Kinetics of the tissue, either 'U', 'NX', 'FX', 'WV', 'HFU', 'HF', '2CU', '2CX' (see also `dcmri.model_props` for more detail). Defaults to '2CX'. 
        sum (bool, optional): For two-compartment tissues, set to True to return the total tissue concentration. Defaults to True.
        params (dict): free model parameters and their values.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where *k* is the number of time points in *ca*. The concentration is returned in units of M.

    Notes:

        Currently implemented tissue types are: 
        
        - **2CX**: two-compartment exchange tissue. 
        - **2CU**: two-compartment uptake tissue. 
        - **HF**: high-flow tissue - also known as *extended Tofts model*, *extended Patlak model* or *general kinetic model*.
        - **HFU**: high-flow uptake tissue - also known as *Patlak model*.
        - **FX**: fast tracer exchange tissue. 
        - **NX**: no tracer exchange tissue. 
        - **U**: uptake tissue. 
        - **WV**: weakly vascularized tissue - also known as *Tofts model*. 

        Each tissue is determined by one or more of the following parameters (see table below):

        - **Fp** (mL/sec/cm3): plasma flow.
        - **Ktrans** (mL/sec/cm3): volume transfer constant.
        - **vp** (mL/cm3): plasma volume fraction.
        - **vi** (mL/cm3): interstitial volume fraction.
        - **ve** (mL/cm3): extracellular volume fraction (= plasma + interstitium).

    .. list-table:: Available tissue types and their parameters
        :widths: 15 25 50
        :header-rows: 1

        * - Tissue type
          - Parameters
          - Tracer compartments
        * - 2CX
          - (vp, vi, Fp, Ktrans)
          - Plasma, Interstitium
        * - 2CU
          - (vp, Fp, Ktrans)
          - Plasma, Interstitium
        * - HF
          - (vp, vi, Ktrans)
          - Plasma, Interstitium
        * - HFU
          - (vp, Ktrans)
          - Plasma, Interstitium
        * - FX
          - (ve, Fp)
          - Extracellular space      
        * - NX
          - (vp, Fp)
          - Plasma    
        * - U
          - (Fp, )
          - Plasma      
        * - WV
          - (vi, Ktrans)
          - Interstitium      

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

        >>> p = {'vp':0.05, 'vi':0.4, 'Fp':0.01, 'Ktrans':0.005}

        Generate plasma and extravascular tissue concentrations with the 2CX and WV models:

        >>> C0 = dc.conc_tissue(ca, t=t, sum=False, model='2CX', **p)
        >>> C1 = dc.conc_tissue(ca, t=t, sum=False, model='WV', vi=p['vi'], Ktrans=p['Ktrans'])

        Compare them in a plot:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> ax0.set_title('2-compartment exchange model')
        >>> ax0.plot(t/60, 1000*C0[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma')
        >>> ax0.plot(t/60, 1000*C0[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Extravascular, extracellular space')
        >>> ax0.plot(t/60, 1000*(C0[0,:]+C0[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Tissue')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Tissue concentration (mM)')
        >>> ax0.legend()
        >>> ax1.set_title('Weakly vascularised model')
        >>> ax1.plot(t/60, 1000*C1[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma (WV)')
        >>> ax1.plot(t/60, 1000*C1[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Extravascular, extracellular space')
        >>> ax1.plot(t/60, 1000*(C1[0,:]+C1[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()
    """

    if model=='U':
        return _conc_u(ca, t=t, dt=dt, **params)
    elif model=='FX':
        return _conc_1c(ca, t=t, dt=dt, F=params['Fp'], v=params['ve'])
    elif model=='NX':
        return _conc_nx(ca, t=t, dt=dt, sum=sum, **params)
    elif model=='WV':
        return _conc_wv(ca, t=t, dt=dt, sum=sum, **params)
    elif model=='HFU':
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, **params)
    elif model=='HF':
        return _conc_hf(ca, t=t, dt=dt, sum=sum, **params)
    elif model=='2CU':
        return _conc_2cu(ca, t=t, dt=dt, sum=sum, **params)
    elif model=='2CX':
        return _conc_2cx(ca, t=t, dt=dt, sum=sum, **params)
    # elif model=='2CF':
    #     return _conc_2cf(ca, *params, t=t, dt=dt, sum=sum)
    else:
        raise ValueError('Kinetic model ' + model + ' is not currently implemented.')
    

def _conc_u(ca, t=None, dt=1.0, Fp=None):
    return pk.conc_trap(Fp*ca, t=t, dt=dt)

def _conc_1c(ca, t=None, dt=1.0, v=None, F=None):
    if (F==0) or (v==0):
        return np.zeros(len(ca))
    return pk.conc_comp(F*ca, v/F, t=t, dt=dt)

def _conc_nx(ca, t=None, dt=1.0, sum=True, vp=None, Fp=None):
    if Fp==0:
        return np.zeros((2,len(ca)))
    Cp = pk.conc_comp(Fp*ca, vp/Fp, t=t, dt=dt)
    if sum:
        return Cp
    else:
        Ce = np.zeros(len(ca))
        return np.stack((Cp,Ce))

def _conc_wv(ca, t=None, dt=1.0, sum=True, vi=None, Ktrans=None):
    Cp = np.zeros(np.size(ca))
    if Ktrans==0:
        Ce = np.zeros(np.size(ca))
    else:
        Ce = pk.conc_comp(Ktrans*ca, vi/Ktrans, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))

def _conc_hfu(ca, t=None, dt=1.0, sum=True, vp=None, Ktrans=None):
    Cp = vp*ca
    Ce = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    
def _conc_hf(ca, t=None, dt=1.0, sum=True, vp=None, vi=None, Ktrans=None):
    Cp = vp*ca
    if Ktrans==0:
        Ce = 0*ca
    else:
        Ce = pk.conc_comp(Ktrans*ca, vi/Ktrans, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    
def _conc_2cu(ca, t=None, dt=1.0, sum=True, vp=None, Fp=None, Ktrans=None):

    if np.isinf(Fp):
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, vp=vp, Ktrans=Ktrans)   

    if Ktrans==Fp: #E=1. Since i is a trap, the whole system is a trap.
        if sum is False:
            msg = 'With Fp=Ktrans, the system is a trap and the concentration of intra- and extravascular spaces cannot be independently deptermined.'
            raise ValueError(msg)
        return _conc_u(ca, t=None, dt=1.0, Fp=Fp)

    PS = Ktrans*Fp/(Fp-Ktrans) 

    if Fp+PS==0:
        return np.zeros((2,len(ca)))
    Tp = vp/(Fp+PS)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    if vp==0:
        Ce = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ce = pk.conc_trap(PS*Cp/vp, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    

def _conc_2cx(ca, t=None, dt=1.0, sum=True, vp=None, vi=None, Fp=None, Ktrans=None):

    if np.isinf(Fp):
        return _conc_hf(ca, t=t, dt=dt, sum=sum, vp=vp, vi=vi, Ktrans=Ktrans)  

    if Ktrans==Fp: #E=1: FX
        C = _conc_1c(ca, t=t, dt=dt, v=vp+vi, F=Fp)
        if sum:
            return C
        else:
            if vp+vi==0:
                return np.zeros((2,len(ca)))
            else:
                Cp = C*vp/(vp+vi)
                Ce = C*vi/(vp+vi)
                return np.stack((Cp,Ce))

    PS = Ktrans*Fp/(Fp-Ktrans)
    J = Fp*ca

    if Fp+PS == 0:
        Cp = np.zeros(len(ca))
        Ce = np.zeros(len(ca))
        if sum:
            return Cp+Ce
        else:
            return np.stack((Cp,Ce))
        
    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)

    if PS == 0:
        Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
        Ce = np.zeros(len(ca))
        if sum:
            return Cp+Ce
        else:
            return np.stack((Cp,Ce))
    
    Te = vi/PS
    
    C = pk.conc_2cxm(J, [Tp, Te], E, t=t, dt=dt)
    if sum:
        return np.sum(C, axis=0)
    else:
        return C

    
def _conc_2cf(ca, t=None, dt=1.0, sum=True, vp=None, Fp=None, PS=None, Te=None):
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





def flux_tissue(ca:np.ndarray, t=None, dt=1.0, model='2CX', **params)->np.ndarray:
    """Indicator out of a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        model (str, optional): Kinetics of the tissue, either 'U', 'NX', 'FX', 'WV', 'HFU', 'HF', '2CU', '2CX' - see below for detail. Defaults to '2CX'. 
        params (dict): free model parameters and their values.

    Returns:
        numpy.ndarray: For a one-compartmental tissue, outflux out of the compartment as a 1D array in units of mmol/sec/mL or M/sec. For a multi=compartmental tissue, outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    Notes:
        Currently implemented tissue types are: 
        
        - **2CX**: two-compartment exchange tissue. 
        - **2CU**: two-compartment uptake tissue. 
        - **HF**: high-flow tissue - also known as *extended Tofts model*, *extended Patlak model* or *general kinetic model*.
        - **HFU**: high-flow uptake tissue - also known as *Patlak model*.
        - **FX**: fast tracer exchange tissue. 
        - **NX**: no tracer exchange tissue. 
        - **U**: uptake tissue. 
        - **WV**: weakly vascularized tissue - also known as *Tofts model*. 

        Each tissue is determined by one or more of the following parameters (see table below):

        - **Fp** (mL/sec/cm3): plasma flow.
        - **Ktrans** (mL/sec/cm3): volume transfer constant.
        - **vp** (mL/cm3): plasma volume fraction.
        - **vi** (mL/cm3): interstitial volume fraction.
        - **ve** (mL/cm3): extracellular volume fraction (= plasma + interstitium).

    .. list-table:: Available tissue types and their parameters
        :widths: 15 25 50
        :header-rows: 1

        * - Tissue type
          - Parameters
          - Tracer compartments
        * - 2CX
          - (vp, vi, Fp, Ktrans)
          - Plasma, Interstitium
        * - 2CU
          - (vp, Fp, Ktrans)
          - Plasma, Interstitium
        * - HF
          - (vp, vi, Ktrans)
          - Plasma, Interstitium
        * - HFU
          - (vp, Ktrans)
          - Plasma, Interstitium
        * - FX
          - (ve, Fp)
          - Extracellular space      
        * - NX
          - (vp, Fp)
          - Plasma    
        * - U
          - (Fp, )
          - Plasma      
        * - WV
          - (vi, Ktrans)
          - Interstitium      

    """
    if model=='U':
        return _flux_u(ca, **params)
    elif model=='NX':
        return _flux_1c(ca, t=t, dt=dt, v=params['vp'], F=params['Fp'])
    elif model=='FX':
        return _flux_1c(ca, t=t, dt=dt, v=params['ve'], F=params['Fp'])
    elif model=='WV':
        return _flux_wv(ca, t=t, dt=dt, **params)
    elif model=='HFU':
        return _flux_hfu(ca, **params)
    elif model=='HF':
        return _flux_hf(ca, t=t, dt=dt, **params)
    elif model=='2CU':
        return _flux_2cu(ca, t=t, dt=dt, **params)
    elif model=='2CX':
        return _flux_2cx(ca, t=t, dt=dt, **params)
    # elif model=='2CF':
    #     return _flux_2cf(ca, t=t, dt=dt, **params)
    else:
        raise ValueError('Kinetic model ' + model + ' is not currently implemented.')


def _flux_u(ca, Fp=None):
    return pk.flux(Fp*ca, model='trap')

def _flux_1c(ca, t=None, dt=1.0, v=None, F=None):
    if F==0:
        return np.zeros(len(ca))
    return pk.flux(F*ca, v/F, t=t, dt=dt, model='comp')

def _flux_wv(ca, t=None, dt=1.0, vi=None, Ktrans=None):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    if Ktrans!=0:
        J[0,1,:] = pk.flux(Ktrans*ca, vi/Ktrans, t=t, dt=dt, model='comp')
    return J

def _flux_hfu(ca, Ktrans=None):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = Ktrans*ca
    return J

def _flux_hf(ca, t=None, dt=1.0, vi=None, Ktrans=None):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.inf
    J[1,0,:] = Ktrans*ca
    if Ktrans==0:
        J[0,1,:] = 0*ca
    else:
        J[0,1,:] = pk.flux(Ktrans*ca, vi/Ktrans, t=t, dt=dt, model='comp')
    return J

def _flux_2cu(ca, t=None, dt=1.0, vp=None, Fp=None, Ktrans=None):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, vp=vp, Fp=Fp, Ktrans=Ktrans)
    J = np.zeros(((2,2,len(ca))))
    if vp==0:
        J[0,0,:] = Fp*ca
        J[1,0,:] = Ktrans*ca
    else:
        PS = Ktrans*Fp/(Fp-Ktrans) 
        J[0,0,:] = Fp*C[0,:]/vp
        J[1,0,:] = PS*C[0,:]/vp
    return J

def _flux_2cx(ca, t=None, dt=1.0, vp=None, vi=None, Fp=None, Ktrans=None):

    if np.isinf(Fp):
        return _flux_hf(ca, t=t, dt=dt, vi=vi, Ktrans=Ktrans)  
    
    if Fp==0:
        return np.zeros((2,2,len(ca)))
    
    if Ktrans==Fp: #E=1: FX
        J = np.zeros((2,2,len(ca)))
        Jp = _flux_1c(ca, t=t, dt=dt, v=vp+vi, F=Fp)
        J[0,0,:] = Jp
        J[0,1,:] = np.inf
        J[1,0,:] = np.inf
        return J
    
    PS = Ktrans*Fp/(Fp-Ktrans)
    
    if Fp+PS == 0:
        return np.zeros((2,2,len(ca)))
    if PS == 0:
        Jp = _flux_1c(ca, t=t, dt=dt, v=vp, F=Fp)
        J = np.zeros((2,2,len(ca)))
        J[0,0,:] = Jp
        return J
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, vp=vp, vi=vi, Fp=Fp, Ktrans=Ktrans)
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    Te = vi/PS
    E = PS/(Fp+PS)
    # Build the system matrix K
    T = [Tp, Te]
    E = [
        [1-E, 1],
        [E,   0],
    ]
    return pk._J_ncomp(C, T, E)

def _flux_2cf(ca, t=None, dt=1.0, vp=None, Fp=None, PS=None, Te=None):
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
    J0 = pk.flux(J, T[0], t=t, model='comp')   
    J10 = E*J0
    Jo[1,0,:] = J10
    Jo[1,1,:] = pk.flux(J10, T[1], t=t, model='comp')
    Jo[0,0,:] = (1-E)*J0
    return Jo


