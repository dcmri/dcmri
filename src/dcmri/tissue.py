"""PK models built from PK blocks defined in dcmri.pk"""

import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils


# TODO: HF and 2CXM params the same across WX regimes
def relax_tissue(ca:np.ndarray, R10:float, r1:float, t=None, dt=1.0, kinetics='2CX', water_exchange='FF', **params):
    """Free relaxation rates for a 2-site exchange tissue and different water exchange regimes

    Note: the free relaxation rates are the relaxation rates of the tissue compartments in the absence of water exchange between them.

    Args:
        ca (array-like): concentration in the arterial input.
        R10 (float): precontrast relaxation rate. The tissue is assumed to be in fast exchange before injection of contrast agent.
        r1 (float): contrast agent relaivity. 
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        kinetics (str, optional): Kinetic model to use - see below for detailed options. Defaults to '2CX'.
        water_exchange (str, optional): Water exchange model to use - see below for detailed options. Defaults to 'FF'.
        params (dict): model parameters - see below for detailed options.

    Returns:
        tuple: relaxation rates of tissue compartments and their volumes.
            - **R1** (array-like): in the fast water exchange limit, the relaxation rates are a 1D array. In all other situations, relaxation rates are a 2D-array with dimensions (k,n), where k is the number of compartments and n is the number of time points in ca.
            - **v** (list or float): the volume fractions of the tissue compartments. 

    Water exchange across a barrier can either be in the fast exchange limit (F), or restricted (R). Since there are two barriers involved, there are four different values for the parameter `water_exchange`:

    .. list-table:: **water exchange models**
        :widths: 15 40 40
        :header-rows: 1

        * - water_exchange
          - Transendothelial
          - Transcytolemmal
        * - 'FF'
          - Fast
          - Fast
        * - 'RR'
          - Restricted
          - Restricted
        * - 'RF'
          - Restricted
          - Fast
        * - 'FR'
          - Fast
          - Restricted

    Any of these can be combined with any possible kinetic model (see :ref:`two-site-exchange`). The tables below list the possible model parameters, and the different model combinations with their respective free parameters:

    .. list-table:: **tissue parameters**
        :widths: 15 40 20
        :header-rows: 1

        * - Short name
          - Full name
          - Units
        * - Fp
          - Plasma flow
          - mL/sec/cm3
        * - PS
          - Permeability-surface area product
          - mL/sec/cm3
        * - Ktrans
          - Volume transfer constant
          - mL/sec/cm3
        * - vp  
          - Plasma volume fraction
          - mL/cm3
        * - vi
          - Interstitial volume fraction
          - mL/cm3
        * - ve
          - Extracellular volume fraction (= vp + vi)
          - mL/cm3
        * - vb
          - Blood volume fraction
          - mL/cm3
        * - vc
          - Tissue cell volume fraction
          - mL/cm3
        * - H
          - Hematocrit
          - None


    .. list-table:: **Fast endothelial and cytolemmal water exchange (FF).** 
        :widths: 20 20 30 40
        :header-rows: 1 

        * - water_exchange
          - kinetics
          - Water compartments
          - Free parameters
        * - FF
          - 2CX
          - 1
          - vp, vi, Fp, PS
        * - FF
          - 2CU
          - 1
          - vp, Fp, PS
        * - FF
          - HF
          - 1
          - vp, vi, PS
        * - FF
          - HFU
          - 1
          - vp, PS
        * - FF
          - FX
          - 1
          - vp, ve, Fp  
        * - FF
          - NX
          - 1
          - vp, Fp  
        * - FF
          - U
          - 1
          - Fp 
        * - FF
          - WV
          - 1
          - vi, Ktrans

    .. list-table:: **Restricted endothelial and cytolemmal water exchange (RR).** 
        :widths: 20 20 30 40
        :header-rows: 1 

        * - water_exchange
          - kinetics
          - Water compartments
          - Free parameters
        * - RR
          - 2CX
          - vb, vi, vc
          - H, vp, vi, Fp, PS
        * - RR
          - 2CU
          - vb, vi, vc
          - H, vp, vi, Fp, PS
        * - RR
          - HF
          - vb, vi, vc
          - H, vp, vi, PS
        * - RR
          - HFU
          - vb, vi, vc
          - H, vp, vi, PS
        * - RR
          - FX
          - vb, vi, vc 
          - H, vp, ve, Fp  
        * - RR
          - NX
          - vb, 1-vb 
          - H, vp, Fp  
        * - RR
          - U
          - vb, 1-vb 
          - vb, Fp 
        * - RR
          - WV
          - vi, 1-vi
          - vi, Ktrans
        
        
    .. list-table:: **Restricted endothelial and fast cytolemmal water exchange (RF).** 
        :widths: 20 20 30 40
        :header-rows: 1

        * - water_exchange
          - kinetics
          - Water compartments
          - Free parameters
        * - RF
          - 2CX
          - vb, 1-vb
          - H, vp, vi, Fp, PS
        * - RF
          - 2CU
          - vb, 1-vb
          - H, vp, Fp, PS
        * - RF
          - HF
          - vb, 1-vb
          - H, vp, vi, PS
        * - RF
          - HFU
          - vb, 1-vb
          - H, vp, PS
        * - RF
          - FX
          - vb, 1-vb 
          - H, vp, ve, Fp
        * - RF
          - NX
          - vb, 1-vb
          - H, vp, Fp
        * - RF
          - U
          - vb, 1-vb
          - vb, Fp  
        * - RF
          - WV
          - 1  
          - vi, Ktrans 

            
    .. list-table:: **Fast endothelial and restricted cytolemmal water exchange (FR).**
        :widths: 20 20 30 40
        :header-rows: 1

        * - water_exchange
          - kinetics
          - Water compartments
          - Free parameters
        * - FR
          - 2CX
          - 1-vc, vc
          - H, vp, vi, Fp, PS
        * - FR
          - 2CU
          - 1-vc, vc
          - vc, vp, Fp, PS
        * - FR
          - HF
          - 1-vc, vc
          - H, vp, vi, PS
        * - FR
          - HFU
          - 1-vc, vc
          - vc, vp, PS
        * - FR
          - FX
          - 1-vc, vc 
          - vc, ve, Fp
        * - FR
          - NX
          - 1-vc, vc 
          - vc, vp, Fp
        * - FR
          - U
          - 1-vc, vc 
          - vc, Fp
        * - FR
          - WV
          - vi, 1-vi
          - PSc, vi, Ktrans

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

        >>> R10, r1 = 1/dc.T1(), dc.relaxivity()                # Constants
        >>> pf = {'vp':0.05, 'vi':0.3, 'Fp':0.01, 'PS':0.005}   # Kinetic parameters
        >>> pr = {'H':0.6} | pf                                 # All parameters

        Calculate tissue relaxation rates R1r with restrictedwater exchange, and also in the fast exchange limit for comparison:

        >>> R1f, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='FF', **pf)
        >>> R1r, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='RR', **pr)

        Plot the relaxation rates in the three compartments, and compare against the fast exchange result:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

        Plot restricted water exchange in the left panel:

        >>> ax0.set_title('Restricted water exchange')
        >>> ax0.plot(t/60, R1r[0,:], linestyle='-', linewidth=2.0, color='darkred', label='Blood')
        >>> ax0.plot(t/60, R1r[1,:], linestyle='-', linewidth=2.0, color='darkblue', label='Interstitium')
        >>> ax0.plot(t/60, R1r[2,:], linestyle='-', linewidth=2.0, color='grey', label='Cells')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Compartment relaxation rate (1/sec)')
        >>> ax0.legend()

        Plot fast water exchange in the right panel:

        >>> ax1.set_title('Fast water exchange')
        >>> ax1.plot(t/60, R1f, linestyle='-', linewidth=2.0, color='black', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue relaxation rate (1/sec)')
        >>> ax1.legend()
        >>> plt.show()

    """

    if kinetics not in ['U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU', '2CX']:
        msg = "Kinetic model '" + str(kinetics) + "' is not recognised.\n"
        msg += "Possible values are: 'U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU' and '2CX'."
        raise ValueError(msg)
    
    if water_exchange not in ['FF','RF','FR','RR']:
        msg = "Water exchange regime '" + str(water_exchange) + "' is not recognised.\n"
        msg += "Possible values are: 'FF','RF','FR','RR'."
        raise ValueError(msg)

    if water_exchange=='FF':

        if kinetics=='U':
            return _relax_u_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='FX':
            return _relax_fx_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='NX':
            return _relax_nx_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='WV':
            return _relax_wv_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HFU':
            return _relax_hfu_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HF':
            return _relax_hf_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CU':
            return _relax_2cu_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CX':
            return _relax_2cx_ff(ca, R10, r1, t=t, dt=dt, **params)
    
    elif water_exchange=='RF':
    
        if kinetics=='U':
            return _relax_u_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='FX':
            return _relax_fx_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='NX':
            return _relax_nx_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='WV':
            return _relax_wv_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HFU':
            return _relax_hfu_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HF':
            return _relax_hf_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CU':
            return _relax_2cu_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CX':
            return _relax_2cx_rf(ca, R10, r1, t=t, dt=dt, **params)
    
    elif water_exchange=='FR':

        if kinetics=='U':
            return _relax_u_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='FX':
            return _relax_fx_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='NX':
            return _relax_nx_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='WV':
            return _relax_wv_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HFU':
            return _relax_hfu_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HF':
            return _relax_hf_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CU':
            return _relax_2cu_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CX':
            return _relax_2cx_fr(ca, R10, r1, t=t, dt=dt, **params)
    
    elif water_exchange=='RR':

        if kinetics=='U':
            return _relax_u_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='FX':
            return _relax_fx_rr(ca, R10, r1, t=t, dt=dt, **params) 
        elif kinetics=='NX':
            return _relax_nx_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='WV':
            return _relax_wv_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HFU':
            return _relax_hfu_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='HF':
            return _relax_hf_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CU':
            return _relax_2cu_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics=='2CX':
            return _relax_2cx_rr(ca, R10, r1, t=t, dt=dt, **params)
        


def _relax_2cx_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_2cx(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1, 1

def _relax_2cu_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_2cu(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1, 1

def _relax_hf_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_hf(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1, 1

def _relax_hfu_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_hfu(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1, 1

def _relax_nx_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_nx(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1, 1

def _relax_wv_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_wv(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1, 1

def _relax_u_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_u(ca, t=t, dt=dt, **params)
    R1 = R10 + r1*C
    return R1, 1

def _relax_fx_ff(ca, R10, r1, t=None, dt=1.0, ve=None, Fp=None):
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    R1 = R10 + r1*C
    return R1, 1



def _relax_2cx_fr(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, vi=None, Fp=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    vc=np.amax([1-vb-vi,1e-6]) # TODO allow v=0 in signal models and remove this protection
    C = _conc_2cx(ca, t=t, dt=dt, vp=vp, vi=vi, Fp=Fp, PS=PS)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc,vc]

def _relax_2cu_fr(ca, R10, r1, t=None, dt=1.0, vc=None, vp=None, Fp=None, PS=None):
    C = _conc_2cu(ca, t=t, dt=dt, vp=vp, Fp=Fp, PS=PS)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc,vc]

def _relax_hf_fr(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, vi=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    vc=np.amax([1-vb-vi,1e-6])
    C = _conc_hf(ca, t=t, dt=dt, vp=vp, vi=vi, PS=PS)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc,vc]

def _relax_hfu_fr(ca, R10, r1, t=None, dt=1.0, vc=None, vp=None, PS=None):
    C = _conc_hfu(ca, t=t, dt=dt, vp=vp, PS=PS)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc, vc]

def _relax_fx_fr(ca, R10, r1, t=None, dt=1.0, vc=None, ve=None, Fp=None):
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc,vc]

def _relax_nx_fr(ca, R10, r1, t=None, dt=1.0, vc=None, vp=None, Fp=None):
    C = _conc_nx(ca, t=t, dt=dt, vp=vp, Fp=Fp)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc, vc]

def _relax_wv_fr(ca, R10, r1, t=None, dt=1.0, vi=None, Ktrans=None):
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, Ktrans=Ktrans)
    vc = np.amax([1-vi,1e-6])
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc, vc]

def _relax_u_fr(ca, R10, r1, t=None, dt=1.0, vc=None, Fp=None):
    # vc, Fp
    C = _conc_u(ca, t=t, dt=dt, Fp=Fp)
    R1 = np.full((2,C.size), R10)
    if 1-vc != 0:
        R1[0,:] += r1*C/(1-vc)
    return R1, [1-vc,vc]





def _relax_2cx_rf(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, vi=None, Fp=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_2cx(ca, t=t, dt=dt, vp=vp, vi=vi, Fp=Fp, PS=PS, sum=False)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1, [vb, 1-vb]

def _relax_2cu_rf(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, Fp=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, vp=vp, Fp=Fp, PS=PS)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1, [vb, 1-vb]

def _relax_hf_rf(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, vi=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_hf(ca, t=t, dt=dt, vi=vi, vp=vp, PS=PS, sum=False)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1, [vb, 1-vb]

def _relax_hfu_rf(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, vp=vp, PS=PS)
    R1 = np.full((2,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if 1-vb != 0:
        R1[1,:] += r1*C[1,:]/(1-vb)
    return R1, [vb, 1-vb]

def _relax_fx_rf(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, ve=None, Fp=None):
    vb=vp/(1-H) if H!=1 else 0
    vi=np.amax([ve-vp,1e-6])
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    R1 = np.full((2,C.size), R10)
    if ve != 0:
        ce = C/ve
        if vb != 0:
            R1[0,:] += r1*ce*vp/vb
        if 1-vb != 0:
            R1[1,:] += r1*ce*vi/(1-vb)
    return R1, [vb, 1-vb]

def _relax_nx_rf(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, Fp=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_nx(ca, t=t, dt=dt, vp=vp, Fp=Fp)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1, [vb,1-vb]

def _relax_u_rf(ca, R10, r1, t=None, dt=1.0, vb=None, Fp=None):
    C = _conc_u(ca, t=t, dt=dt, Fp=Fp)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1, [vb,1-vb]

def _relax_wv_rf(ca, R10, r1, t=None, dt=1.0, vi=None, Ktrans=None):
    # vi, Ktrans
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, Ktrans=Ktrans)
    R1 = R10 + r1*C
    return R1, 1



def _relax_2cx_rr(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, vi=None, Fp=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_2cx(ca, t=t, dt=dt, vp=vp, vi=vi, Fp=Fp, PS=PS, sum=False)
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1, [vb, vi, 1-vb-vi]

def _relax_2cu_rr(ca, R10, r1, t=None, dt=1.0, H=None, vi=None, vp=None, Fp=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, vp=vp, Fp=Fp, PS=PS)
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1, [vb, vi, 1-vb-vi]
    
def _relax_hf_rr(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, vi=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_hf(ca, t=t, dt=dt, vp=vp, vi=vi, PS=PS, sum=False)
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1, [vb, vi, 1-vb-vi]

def _relax_hfu_rr(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, vi=None, PS=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, vp=vp, PS=PS)
    R1 = np.full((3,C.shape[1]), R10)
    if vb != 0:
        R1[0,:] += r1*C[0,:]/vb
    if vi != 0:
        R1[1,:] += r1*C[1,:]/vi
    return R1, [vb, vi, 1-vb-vi]

def _relax_fx_rr(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, ve=None, Fp=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_1c(ca, t=t, dt=dt, v=ve, F=Fp)
    R1 = np.full((3,C.size), R10)
    vi = ve-vp
    if ve != 0:
        ce = C/ve
        if vb != 0:
            R1[0,:] += r1*ce*vp/vb
        R1[1,:] += r1*ce
    return R1, [vb, vi, 1-vb-vi]

def _relax_nx_rr(ca, R10, r1, t=None, dt=1.0, H=None, vp=None, Fp=None):
    vb=vp/(1-H) if H!=1 else 0
    C = _conc_nx(ca, t=t, dt=dt, vp=vp, Fp=Fp)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1, [vb, 1-vb]

def _relax_u_rr(ca, R10, r1, t=None, dt=1.0, vb=None, Fp=None):
    C = _conc_u(ca, t=t, dt=dt, Fp=Fp)
    R1 = np.full((2,C.size), R10)
    if vb != 0:
        R1[0,:] += r1*C/vb
    return R1, [vb,1-vb]

def _relax_wv_rr(ca, R10, r1, t=None, dt=1.0, vi=None, Ktrans=None):
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, Ktrans=Ktrans)
    R1 = np.full((2,C.size), R10)
    if vi != 0:
        R1[0,:] += r1*C/vi
    return R1, [vi, 1-vi]








def conc_tissue(ca:np.ndarray, t=None, dt=1.0, kinetics='2CX', sum=True, **params)->np.ndarray:
    """Tissue concentration in a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        t (array_like, optional): the time points of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is provided. Defaults to 1.0.
        kinetics (str, optional): The kinetic model of the tissue (see below for possible values). Defaults to '2CX'. 
        sum (bool, optional): For two-compartment tissues, set to True to return the total tissue concentration, and False to return the concentrations in the compartments separately. In one-compartment tissues this keyword has no effect. Defaults to True.
        params (dict): free model parameters and their values (see below for possible).

    Returns:
        numpy.ndarray: If sum=True, or the tissue is one-compartmental, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where *k* is the number of time points in *ca*. 


    The tables below define the possible values of the `kinetics` argument and the corresponding parameters in the `params` dictionary. 

    .. list-table:: **kinetic models**
        :widths: 10 40 20 20
        :header-rows: 1

        * - Kinetics
          - Full name
          - Parameters
          - Assumptions
        * - '2CX'
          - Two-compartment exchange
          - vi, vp, Fp, PS
          - see :ref:`two-site-exchange`
        * - '2CU'
          - Two-compartment uptake
          - vp, Fp, PS
          - :math:`PS` small
        * - 'HF'
          - High-flow, AKA *extended Tofts model*, *extended Patlak model*, *general kinetic model*.
          - vi, vp, Ktrans
          - :math:`F_p = \infty`
        * - 'HFU'
          - High flow uptake, AKA *Patlak model*
          - vp, PS
          - :math:`F_p = \infty`, PS small
        * - 'FX'
          - Fast indicator exchange
          - ve, Fp
          - :math:`PS = \infty`  
        * - 'NX'
          - No indicator exchange
          - vp, Fp
          - :math:`PS = 0`     
        * - 'U'
          - Uptake
          - Fp
          - Fp small    
        * - 'WV'
          - Weakly vascularized, AKA *Tofts model*.
          - vi, Ktrans
          - :math:`v_p = 0`

            
    .. list-table:: **tissue parameters**
        :widths: 15 40 20
        :header-rows: 1

        * - Short name
          - Full name
          - Units
        * - Fp
          - Plasma flow
          - mL/sec/cm3
        * - PS
          - Permeability-surface area product
          - mL/sec/cm3
        * - Ktrans
          - Volume transfer constant
          - mL/sec/cm3
        * - vp  
          - Plasma volume fraction
          - mL/cm3
        * - vi
          - Interstitial volume fraction
          - mL/cm3
        * - ve
          - Extracellular volume fraction (= vp + vi)
          - mL/cm3

    Example:

        We plot the concentrations of 2CX and WV models with the same values for the shared tissue parameters. 

    .. plot::
        :include-source:

        Start by importing the packages:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define some tissue parameters: 

        >>> p2x = {'vp':0.05, 'vi':0.4, 'Fp':0.01, 'PS':0.005}
        >>> pwv = {'vi':0.4, 'Ktrans':0.005*0.01/(0.005+0.01)}

        Generate plasma and extravascular tissue concentrations with the 2CX and WV models:

        >>> C2x = dc.conc_tissue(ca, t=t, sum=False, model='2CX', **p2x)
        >>> Cwv = dc.conc_tissue(ca, t=t, sum=False, model='WV', **pwv)

        Compare them in a plot:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

        Plot 2CX results in the left panel:

        >>> ax0.set_title('2-compartment exchange model')
        >>> ax0.plot(t/60, 1000*C2x[0,:], linestyle='-', linewidth=3.0, color='darkred', label='Plasma')
        >>> ax0.plot(t/60, 1000*C2x[1,:], linestyle='-', linewidth=3.0, color='darkblue', label='Extravascular, extracellular space')
        >>> ax0.plot(t/60, 1000*(C2x[0,:]+C2x[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Tissue')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Tissue concentration (mM)')
        >>> ax0.legend()

        Plot WV results in the right panel:

        >>> ax1.set_title('Weakly vascularised model')
        >>> ax1.plot(t/60, 1000*Cwv[0,:], linestyle='-', linewidth=3.0, color='darkred', label='Plasma (WV)')
        >>> ax1.plot(t/60, 1000*Cwv[1,:], linestyle='-', linewidth=3.0, color='darkblue', label='Extravascular, extracellular space')
        >>> ax1.plot(t/60, 1000*(Cwv[0,:]+Cwv[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()
    """

    if kinetics=='U':
        return _conc_u(ca, t=t, dt=dt, **params)
    elif kinetics=='FX':
        return _conc_1c(ca, t=t, dt=dt, F=params['Fp'], v=params['ve'])
    elif kinetics=='NX':
        return _conc_nx(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics=='WV':
        return _conc_wv(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics=='HFU':
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics=='HF':
        return _conc_hf(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics=='2CU':
        return _conc_2cu(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics=='2CX':
        return _conc_2cx(ca, t=t, dt=dt, sum=sum, **params)
    # elif model=='2CF':
    #     return _conc_2cf(ca, *params, t=t, dt=dt, sum=sum)
    else:
        raise ValueError('Kinetic model ' + kinetics + ' is not currently implemented.')
    

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

def _conc_hfu(ca, t=None, dt=1.0, sum=True, vp=None, PS=None):
    Cp = vp*ca
    Ce = pk.conc_trap(PS*ca, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    
def _conc_hf(ca, t=None, dt=1.0, sum=True, vi=None, vp=None, PS=None):
    Cp = vp*ca
    if PS==0:
        Ce = 0*ca
    else:
        Ce = pk.conc_comp(PS*ca, vi/PS, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    
def _conc_2cu(ca, t=None, dt=1.0, sum=True, vp=None, Fp=None, PS=None):

    if np.isinf(Fp):
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, vp=vp, PS=PS)   
    if Fp+PS==0:
        return np.zeros((2,len(ca)))
    Tp = vp/(Fp+PS)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    if vp==0:
        Ktrans = PS*Fp/(PS+Fp)
        Ce = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ce = pk.conc_trap(PS*Cp/vp, t=t, dt=dt)
    if sum:
        return Cp+Ce
    else:
        return np.stack((Cp,Ce))
    

def _conc_2cx(ca, t=None, dt=1.0, sum=True, vi=None, vp=None, Fp=None, PS=None):

    if np.isinf(Fp):
        return _conc_hf(ca, t=t, dt=dt, sum=sum, vi=vi, vp=vp, PS=PS)

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





def flux_tissue(ca:np.ndarray, t=None, dt=1.0, kinetics='2CX', **params)->np.ndarray:
    """Indicator out of a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        t (array_like, optional): the time points of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is provided. Defaults to 1.0.
        kinetics (str, optional): The kinetic model of the tissue (see below for possible values). Defaults to '2CX'. 
        params (dict): free model parameters and their values (see below for possible).

    Returns:
        numpy.ndarray: For a one-compartmental tissue, outflux out of the compartment as a 1D array in units of mmol/sec/mL or M/sec. For a multi=compartmental tissue, outflux out of each compartment, and at each time point, as a 3D array with dimensions *(2,2,k)*, where *2* is the number of compartments and *k* is the number of time points in *J*. Encoding of the first two indices is the same as for *E*: *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* is the flux from *i* directly to the outside. The flux is returned in units of mmol/sec/mL or M/sec.

    **Notes**: the tables below define the possible values of the `kinetics` argument and the corresponding parameters in the `params` dictionary. 

    .. list-table:: **kinetic models**
        :widths: 10 40 20 20
        :header-rows: 1

        * - Kinetics
          - Full name
          - Parameters
          - Assumptions
        * - '2CX'
          - Two-compartment exchange
          - vi, vp, Fp, PS
          - see :ref:`two-site-exchange`
        * - '2CU'
          - Two-compartment uptake
          - vp, Fp, PS
          - :math:`PS` small
        * - 'HF'
          - High-flow, AKA *extended Tofts model*, *extended Patlak model*, *general kinetic model*.
          - vi, Ktrans
          - :math:`F_p = \infty`
        * - 'HFU'
          - High flow uptake, AKA *Patlak model*
          - PS
          - :math:`F_p = \infty`, PS small
        * - 'FX'
          - Fast indicator exchange
          - ve, Fp
          - :math:`PS = \infty`  
        * - 'NX'
          - No indicator exchange
          - vp, Fp
          - :math:`PS = 0`     
        * - 'U'
          - Uptake
          - Fp
          - Fp small    
        * - 'WV'
          - Weakly vascularized, AKA *Tofts model*.
          - vi, Ktrans
          - :math:`v_p = 0`

            
    .. list-table:: **tissue parameters**
        :widths: 15 40 20
        :header-rows: 1

        * - Short name
          - Full name
          - Units
        * - Fp
          - Plasma flow
          - mL/sec/cm3
        * - PS
          - Permeability-surface area product
          - mL/sec/cm3
        * - Ktrans
          - Volume transfer constant
          - mL/sec/cm3
        * - vp  
          - Plasma volume fraction
          - mL/cm3
        * - vi
          - Interstitial volume fraction
          - mL/cm3
        * - ve
          - Extracellular volume fraction (= vp + vi)
          - mL/cm3

    """
    if kinetics=='U':
        return _flux_u(ca, **params)
    elif kinetics=='NX':
        return _flux_1c(ca, t=t, dt=dt, v=params['vp'], F=params['Fp'])
    elif kinetics=='FX':
        return _flux_1c(ca, t=t, dt=dt, v=params['ve'], F=params['Fp'])
    elif kinetics=='WV':
        return _flux_wv(ca, t=t, dt=dt, **params)
    elif kinetics=='HFU':
        return _flux_hfu(ca, **params)
    elif kinetics=='HF':
        return _flux_hf(ca, t=t, dt=dt, **params)
    elif kinetics=='2CU':
        return _flux_2cu(ca, t=t, dt=dt, **params)
    elif kinetics=='2CX':
        return _flux_2cx(ca, t=t, dt=dt, **params)
    # elif model=='2CF':
    #     return _flux_2cf(ca, t=t, dt=dt, **params)
    else:
        raise ValueError('Kinetic model ' + kinetics + ' is not currently implemented.')


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

def _flux_hfu(ca, PS=None):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.nan
    J[1,0,:] = PS*ca
    return J

def _flux_hf(ca, t=None, dt=1.0, vi=None, PS=None):
    J = np.zeros(((2,2,len(ca))))
    J[0,0,:] = np.inf
    J[1,0,:] = PS*ca
    if PS==0:
        J[0,1,:] = 0*ca
    else:
        J[0,1,:] = pk.flux(PS*ca, vi/PS, t=t, dt=dt, model='comp')
    return J

def _flux_2cu(ca, t=None, dt=1.0, vp=None, Fp=None, PS=None):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, vp=vp, Fp=Fp, PS=PS)
    J = np.zeros(((2,2,len(ca))))
    if vp==0:
        if Fp+PS!=0:
            Ktrans = Fp*PS/(Fp+PS)
            J[0,0,:] = Fp*ca
            J[1,0,:] = Ktrans*ca
    else:
        J[0,0,:] = Fp*C[0,:]/vp
        J[1,0,:] = PS*C[0,:]/vp
    return J

def _flux_2cx(ca, t=None, dt=1.0, vp=None, vi=None, Fp=None, PS=None):

    if np.isinf(Fp):
        return _flux_hf(ca, t=t, dt=dt, vi=vi, PS=PS)  
    
    if Fp==0:
        return np.zeros((2,2,len(ca)))
    
    if Fp+PS == 0:
        return np.zeros((2,2,len(ca)))
    if PS == 0:
        Jp = _flux_1c(ca, t=t, dt=dt, v=vp, F=Fp)
        J = np.zeros((2,2,len(ca)))
        J[0,0,:] = Jp
        return J
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, vp=vp, vi=vi, Fp=Fp, PS=PS)
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


