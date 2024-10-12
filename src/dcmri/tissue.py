"""PK models built from PK blocks defined in dcmri.pk"""

import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils
import dcmri.rel as rel


# TODO: HF and 2CXM params the same across WX regimes
def relax_tissue(ca: np.ndarray, R10: float, r1: float, t=None, dt=1.0, 
                 kinetics='2CX', water_exchange='FF', **params):
    """Free relaxation rates for a 2-site exchange tissue and different water 
    exchange regimes

    Note: the free relaxation rates are the relaxation rates of the tissue 
    compartments in the absence of water exchange between them.

    Args:
        ca (array-like): concentration in the arterial input.
        R10 (float): precontrast relaxation rate. The tissue is assumed to be 
          in fast exchange before injection of contrast agent.
        r1 (float): contrast agent relaivity. 
        t (array_like, optional): the time points in sec of the input function 
        *ca*. If *t* is not provided, the time points are assumed to be 
          uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          explicity provided. Defaults to 1.0.
        kinetics (str, optional): Kinetic model to use - see below for 
          detailed options. Defaults to '2CX'.
        water_exchange (str, optional): Water exchange model to use - see 
          below for detailed options. Defaults to 'FF'.
        params (dict): model parameters. See :ref:`kinetic-regimes` for 
          more detail.

    Returns:
        tuple: relaxation rates of tissue compartments and their volumes.
            - **R1** (numpy.ndarray): in the fast water exchange limit, the 
              relaxation rates are a 1D array. In all other situations, 
              relaxation rates are a 2D-array with dimensions (k,n), where k is 
              the number of compartments and n is the number of time points 
              in ca.
            - **v** (numpy.ndarray or None): the volume fractions of the tissue 
              compartments. Returns None in 'FF' regime. 
            - **PSw** (numpy.ndarray or None): 2D array with water exchange 
              rates between tissue compartments. Returns None in 'FF' regime. 

    Example:

        Plot the free relaxation rates for a 2-compartment exchange model 
        with intermediate water exchange:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define constants and model parameters: 

        >>> R10, r1 = 1/dc.T1(), dc.relaxivity()     
        >>> pf = {'vp':0.05, 'vi':0.3, 'Fp':0.01, 'PS':0.005}   
        >>> pn = {'H':0.5, 'vb':0.1, 'vi':0.3, 'Fp':0.01, 'PS':0.005}                                # All parameters

        Calculate tissue relaxation rates R1r without water exchange, 
        and also in the fast exchange limit for comparison:

        >>> R1f, _, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='FF', **pf)
        >>> R1r, _, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='RR', **pn)

        Plot the relaxation rates in the three compartments, and compare 
        against the fast exchange result:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

        Plot restricted water exchange in the left panel:

        >>> ax0.set_title('Restricted water exchange')
        >>> ax0.plot(t/60, R1r[0,:], linestyle='-', 
        >>>          linewidth=2.0, color='darkred', label='Blood')
        >>> ax0.plot(t/60, R1r[1,:], linestyle='-', 
        >>>          linewidth=2.0, color='darkblue', label='Interstitium')
        >>> ax0.plot(t/60, R1r[2,:], linestyle='-', 
        >>>          linewidth=2.0, color='grey', label='Cells')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Compartment relaxation rate (1/sec)')
        >>> ax0.legend()

        Plot fast water exchange in the right panel:

        >>> ax1.set_title('Fast water exchange')
        >>> ax1.plot(t/60, R1f, linestyle='-', 
        >>>          linewidth=2.0, color='black', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue relaxation rate (1/sec)')
        >>> ax1.legend()
        >>> plt.show()

    """

    # Check configuration
    if kinetics not in ['U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU', '2CX']:
        msg = "Kinetic model '" + str(kinetics) + "' is not recognised.\n"
        msg += "Possible values are: 'U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU' and '2CX'."
        raise ValueError(msg)

    if water_exchange not in ['FF', 'RF', 'FR', 'RR']:
        msg = "Water exchange regime '" + \
            str(water_exchange) + "' is not recognised.\n"
        msg += "Possible values are: 'FF','RF','FR','RR'."
        raise ValueError(msg)

    # Distribute cases
    if water_exchange == 'FF':

        if kinetics == 'U':
            return _relax_u_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_ff(ca, R10, r1, t=t, dt=dt, **params)

    elif water_exchange == 'RF':

        if kinetics == 'U':
            return _relax_u_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_rf(ca, R10, r1, t=t, dt=dt, **params)

    elif water_exchange == 'FR':

        if kinetics == 'U':
            return _relax_u_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_fr(ca, R10, r1, t=t, dt=dt, **params)

    elif water_exchange == 'RR':

        if kinetics == 'U':
            return _relax_u_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_rr(ca, R10, r1, t=t, dt=dt, **params)


def _c(C,v):
    if v==0:
        # In this case the result does not matter
        return C*0
    else:
        return C/v
    

# FF

def _relax_2cx_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_2cx(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_2cu_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_2cu(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_hf_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_hf(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_hfu_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_hfu(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_nx_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_nx(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_wv_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_wv(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_u_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_u(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_fx_ff(ca, R10, r1, t=None, dt=1.0, **params):
    C = _conc_fx(ca, t=t, dt=dt, **params)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

# FR

def _relax_2cx_fr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, Fp=None, PS=None, PSc=0):
    vp = vb * (1-H)
    C = _conc_2cx(ca, t=t, dt=dt, 
                  vp=vp, vi=vi, Fp=Fp, PS=PS)
    v = [vb+vi, 1-vb-vi]
    R1 = (rel.relax(_c(C, v[0]), R10, r1),
          rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_2cu_fr(ca, R10, r1, t=None, dt=1.0, 
                  vc=None, vp=None, Fp=None, PS=None, PSc=0):
    C = _conc_2cu(ca, t=t, dt=dt, 
                  vp=vp, Fp=Fp, PS=PS)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hf_fr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, PS=None, PSc=0):
    vp = vb * (1-H)
    C = _conc_hf(ca, t=t, dt=dt, 
                 vp=vp, vi=vi, PS=PS)
    v = [vb+vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hfu_fr(ca, R10, r1, t=None, dt=1.0, 
                 vc=None, vp=None, PS=None, PSc=0):
    C = _conc_hfu(ca, t=t, dt=dt, 
                 vp=vp, PS=PS)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_wv_fr(ca, R10, r1, t=None, dt=1.0, 
                 vi=None, Ktrans=None, PSc=0):
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, Ktrans=Ktrans)
    v = [vi, 1-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_fx_fr(ca, R10, r1, t=None, dt=1.0, 
                 vc=None, ve=None, Fp=None, PSc=0):
    C = _conc_fx(ca, t=t, dt=dt, ve=ve, Fp=Fp)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_nx_fr(ca, R10, r1, t=None, dt=1.0, 
                 vc=None, vp=None, Fp=None, PSc=0):
    C = _conc_nx(ca, t=t, dt=dt, vp=vp, Fp=Fp)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_u_fr(ca, R10, r1, t=None, dt=1.0, 
                vc=None, Fp=None, PSc=0):
    C = _conc_u(ca, t=t, dt=dt, Fp=Fp)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

# RF

def _relax_2cx_rf(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, Fp=None, PS=None, PSe=0):
    vp = vb * (1-H)
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, 
                  vp=vp, vi=vi, Fp=Fp, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_2cu_rf(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, Fp=None, PS=None, PSe=0):
    vp = vb * (1-H)
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, 
                  vp=vp, Fp=Fp, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hf_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, PS=None, PSe=0):
    vp = vb * (1-H)
    C = _conc_hf(ca, t=t, dt=dt, sum=False, 
                 vp=vp, vi=vi, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hfu_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, PS=None, PSe=0):
    vp = vb * (1-H)
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, 
                 vp=vp, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_wv_rf(ca, R10, r1, t=None, dt=1.0, 
                 vi=None, Ktrans=None):
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, Ktrans=Ktrans)
    R1 = rel.relax(C, R10, r1)
    return R1, None, None

def _relax_fx_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, Fp=None, PSe=0):
    vp = vb * (1-H)
    ve = vp + vi
    C = _conc_fx(ca, t=t, dt=dt, ve=ve, Fp=Fp)
    v = [vb, 1-vb]
    Cp = C*vp/ve
    Ci = C*vi/ve
    R1 = (
        rel.relax(_c(Cp, v[0]), R10, r1),
        rel.relax(_c(Ci, v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_nx_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, Fp=None, PSe=0):
    vp = vb * (1-H)
    C = _conc_nx(ca, t=t, dt=dt, vp=vp, Fp=Fp)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_u_rf(ca, R10, r1, t=None, dt=1.0, 
                vb=None, Fp=None, PSe=0):
    C = _conc_u(ca, t=t, dt=dt, Fp=Fp)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)


# RR


def _relax_2cx_rr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, 
                  Fp=None, PS=None, PSe=0, PSc=0):
    vp = vb * (1-H)
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, 
                  vp=vp, vi=vi, Fp=Fp, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_2cu_rr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, 
                  Fp=None, PS=None, PSe=0, PSc=0):
    vp = vb * (1-H)
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, 
                  vp=vp, Fp=Fp, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hf_rr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, PS=None, PSe=0, PSc=0):
    vp = vb * (1-H)
    C = _conc_hf(ca, t=t, dt=dt, sum=False, 
                 vp=vp, vi=vi, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hfu_rr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, PS=None, PSe=0, PSc=0):
    vp = vb * (1-H)
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, 
                 vp=vp, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_wv_rr(ca, R10, r1, t=None, dt=1.0, 
                 vi=None, Ktrans=None, PSc=0):
    C = _conc_wv(ca, t=t, dt=dt, vi=vi, Ktrans=Ktrans)
    v = [vi, 1-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_fx_rr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, Fp=None, PSe=0, PSc=0):
    vp = vb * (1-H)
    ve = vp + vi
    C = _conc_fx(ca, t=t, dt=dt, ve=ve, Fp=Fp)
    v = [vb, vi, 1-vb-vi]
    Cp = C*vp/ve
    Ci = C*vi/ve
    R1 = (
        rel.relax(_c(Cp, v[0]), R10, r1),
        rel.relax(_c(Ci, v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_nx_rr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, Fp=None, PSe=0):
    vp = vb * (1-H)
    C = _conc_nx(ca, t=t, dt=dt, vp=vp, Fp=Fp)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_u_rr(ca, R10, r1, t=None, dt=1.0, 
                vb=None, Fp=None, PSe=0, PSc=0):
    v = [vb, 1-vb]
    if Fp==0:
        R1 = (
            rel.relax(ca*0, R10, r1),
            rel.relax(ca*0, R10, r1), 
        )
    else:
        C = _conc_u(ca, t=t, dt=dt, Fp=Fp)
        R1 = (
            rel.relax(_c(C, v[0]), R10, r1),
            rel.relax(ca*0, R10, r1), 
        )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)





def conc_tissue(ca: np.ndarray, t=None, dt=1.0, kinetics='2CX', sum=True, 
                **params) -> np.ndarray:
    """Tissue concentration in a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        t (array_like, optional): the time points of the input function *ca*. 
          If *t* is not provided, the time points are assumed to be uniformly 
          spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          provided. Defaults to 1.0.
        kinetics (str, optional): Tracer-kinetic model. Possible values are 
          '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U' (see 
          table :ref:`two-site-exchange-kinetics` for detail). Defaults to 
          '2CX'.
        sum (bool, optional): For two-compartment tissues, set to True to 
          return the total tissue concentration, and False to return the 
          concentrations in the compartments separately. In one-compartment 
          tissues this keyword has no effect. Defaults to True.
        params (dict): free model parameters provided as keyword arguments. 
          Possible parameters depend on **kinetics** as detailed in Table 
          :ref:`two-site-exchange-kinetics`. 

    Returns:
        numpy.ndarray: If sum=True, or the tissue is one-compartmental, this 
        is a 1D array with the total concentration at each time point. If 
        sum=False this is the concentration in each compartment, and at each 
        time point, as a 2D array with dimensions *(2,k)*, where *k* is the 
        number of time points in *ca*. 

    Raises:
        ValueError: if values are not provided for one or more of the model 
          parameters.

    Example:

        We plot the concentrations of 2CX and WV models with the same values 
        for the shared tissue parameters. 

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

        Generate plasma and extravascular tissue concentrations with the 2CX 
        and WV models:

        >>> C2x = dc.conc_tissue(ca, t=t, sum=False, kinetics='2CX', **p2x)
        >>> Cwv = dc.conc_tissue(ca, t=t, kinetics='WV', **pwv)

        Compare them in a plot:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

        Plot 2CX results in the left panel:

        >>> ax0.set_title('2-compartment exchange model')
        >>> ax0.plot(t/60, 1000*C2x[0,:], linestyle='-', linewidth=3.0, 
        >>>          color='darkred', label='Plasma')
        >>> ax0.plot(t/60, 1000*C2x[1,:], linestyle='-', linewidth=3.0, 
        >>>          color='darkblue', 
        >>>          label='Extravascular, extracellular space')
        >>> ax0.plot(t/60, 1000*(C2x[0,:]+C2x[1,:]), linestyle='-', 
        >>>          linewidth=3.0, color='grey', label='Tissue')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Tissue concentration (mM)')
        >>> ax0.legend()

        Plot WV results in the right panel:

        >>> ax1.set_title('Weakly vascularised model')
        >>> ax1.plot(t/60, Cwv*0, linestyle='-', linewidth=3.0, 
        >>>          color='darkred', label='Plasma')
        >>> ax1.plot(t/60, 1000*Cwv, linestyle='-', 
        >>>          linewidth=3.0, color='grey', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()
    """

    if kinetics == 'U':
        return _conc_u(ca, t=t, dt=dt, **params)
    elif kinetics == 'FX':
        return _conc_fx(ca, t=t, dt=dt, **params)
    elif kinetics == 'NX':
        return _conc_nx(ca, t=t, dt=dt, **params)
    elif kinetics == 'WV':
        return _conc_wv(ca, t=t, dt=dt, **params)
    elif kinetics == 'HFU':
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics == 'HF':
        return _conc_hf(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics == '2CU':
        return _conc_2cu(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics == '2CX':
        return _conc_2cx(ca, t=t, dt=dt, sum=sum, **params)
    # elif model=='2CF':
    #     return _conc_2cf(ca, *params, t=t, dt=dt, sum=sum)
    else:
        raise ValueError('Kinetic model ' + kinetics +
                         ' is not currently implemented.')



def _conc_u(ca, t=None, dt=1.0, Fp=None):
    # Tb = vp/Fp
    if Fp is None:
        msg = ('Fp is a required parameter for the tissue concentration '
                + 'in an uptake model. \nPlease provide a value.')
        raise ValueError(msg)
    C = pk.conc_trap(Fp*ca, t=t, dt=dt)
    return C
    

def _conc_fx(ca, t=None, dt=1.0, 
             ve=None, Fp=None):
    # Te = ve/Fp
    if Fp is None:
        msg = ('Fp is a required parameter for the tissue concentration '
                + 'in a fast exchange model. \nPlease provide a value.')
        raise ValueError(msg)
    if Fp == 0:
        ce = ca*0
    else:
        if ve is None:
            msg = ('ve is a required parameter for the tissue '
                + 'concentration in a fast exchange model. '
                + '\nPlease provide a value.')
            raise ValueError(msg)
        ce = pk.flux_comp(ca, ve/Fp, t=t, dt=dt)
    return ve*ce


def _conc_nx(ca, t=None, dt=1.0, 
             vp=None, Fp=None):
    if Fp is None:
        msg = ('Fp is a required parameter for the tissue concentration '
                + 'in a no exchange model. \nPlease provide a value.')
        raise ValueError(msg)
    if Fp == 0:
        Cp = ca*0
    else:
        if vp is None:
            msg = ('vp is a required parameter for the tissue ' 
                    + 'concentration in a no exchange model. '
                    + '\nPlease provide a value.')
            raise ValueError(msg)
        Cp = pk.conc_comp(Fp*ca, vp/Fp, t=t, dt=dt)
    return Cp 


def _conc_wv(ca, t=None, dt=1.0, 
             vi=None, Ktrans=None):
    if Ktrans == 0:
        ci = ca*0
    else:
        ci = pk.flux_comp(ca, vi/Ktrans, t=t, dt=dt)
    return vi*ci


def _conc_hfu(ca, t=None, dt=1.0, sum=True, vp=None, PS=None):
    # Ti=vi/PS
    # up = vp / (vp + vi)
    cp = ca
    Ci = pk.conc_trap(PS*cp, t=t, dt=dt)
    if sum:
        return vp*cp + Ci
    else:
        return np.stack((vp*cp, Ci)) 


def _conc_hf(ca, t=None, dt=1.0, sum=True,
             vi=None, vp=None, PS=None):
    # Ti = vi/PS
    # up = vp/ve
    Cp = vp*ca
    if PS == 0:
        Ci = 0*ca
    else:
        Ci = pk.conc_comp(PS*ca, vi/PS, t=t, dt=dt)
    if sum:
        return Cp+Ci
    else:
        return np.stack((Cp, Ci))


def _conc_2cu(ca, t=None, dt=1.0, sum=True,
              vp=None, Fp=None, PS=None, Tp=None):
    # Ti = vi/PS
    if np.isinf(Fp):
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, vp=vp, PS=PS)
    if Fp+PS == 0:
        return np.zeros((2, len(ca)))
    Tp = vp/(Fp+PS)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    if vp == 0:
        Ktrans = PS*Fp/(PS+Fp)
        Ci = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ci = pk.conc_trap(PS*Cp/vp, t=t, dt=dt)
    if sum:
        return Cp+Ci
    else:
        return np.stack((Cp, Ci))


def _conc_2cx(ca, t=None, dt=1.0, sum=True,
              vi=None, vp=None, Fp=None, PS=None):

    if np.isinf(Fp):
        return _conc_hf(ca, t=t, dt=dt, sum=sum, vi=vi, vp=vp, PS=PS)

    J = Fp*ca

    if Fp+PS == 0:
        Cp = np.zeros(len(ca))
        Ce = np.zeros(len(ca))
        if sum:
            return Cp+Ce
        else:
            return np.stack((Cp, Ce))

    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)

    if PS == 0:
        Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
        Ci = np.zeros(len(ca))
        if sum:
            return Cp+Ci
        else:
            return np.stack((Cp, Ci))

    Ti = vi/PS

    C = pk.conc_2cxm(J, [Tp, Ti], E, t=t, dt=dt)
    if sum:
        return np.sum(C, axis=0)
    else:
        return C
    

def _conc_2cf(ca, t=None, dt=1.0, sum=True, 
              vp=None, Fp=None, PS=None, Te=None):
    if Fp+PS == 0:
        if sum:
            return np.zeros(len(ca))
        else:
            return np.zeros((2, len(ca)))
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)
    J = Fp*ca
    T = [Tp, Te]
    # Solve the system explicitly
    t = utils.tarray(len(J), t=t, dt=dt)
    C0 = pk.conc_comp(J, T[0], t)
    if E == 0:
        C1 = np.zeros(len(t))
    elif T[0] == 0:
        J10 = E*J
        C1 = pk.conc_comp(J10, T[1], t)
    else:
        J10 = C0*E/T[0]
        C1 = pk.conc_comp(J10, T[1], t)
    if sum:
        return C0+C1
    else:
        return np.stack((C0, C1))
    

def _lconc_fx(ca, t=None, dt=1.0, Te=None):
    # Te = ve/Fp
    if Te is None:
        msg = ('Te is a required parameter for the concentration'
                + 'in a fast exchange model. \nPlease provide a value.')
        raise ValueError(msg)
    ce = pk.flux_comp(ca, Te, t=t, dt=dt)
    return ce


def _lconc_u(ca, t=None, dt=1.0, Tb=None):
    # Tb = vp/Fp
    if Tb is None:
        msg = ('Tb is a required parameter for the concentration'
                + 'in an uptake model. \nPlease provide a value.')
        raise ValueError(msg)
    if Tb==0:
        msg = ('An uptake tissue with Tb=0 is not well-defined. \n'
                + 'Consider constraining the parameters.')
        raise ValueError(msg)
    cp = pk.conc_trap(ca, t=t, dt=dt)/Tb
    return cp

    
def _lconc_nx(ca, t=None, dt=1.0, Tb=None):
    cp = pk.flux_comp(ca, Tb, t=t, dt=dt)
    return cp

    
def _lconc_wv(ca, t=None, dt=1.0, Ti=None):
    # Note cp is non-zero and equal to (1-E)*ca
    # But is not returned as it sits in a compartment without dimensions
    # Ti = vi/Ktrans
    ci = pk.flux_comp(ca, Ti, t=t, dt=dt)
    return ci

        
def _lconc_hfu(ca, t=None, dt=1.0, Ti=None):
    # Ti=vi/PS
    # up = vp / (vp + vi)
    cp = ca
    if Ti==0:
        msg = 'An uptake tissue with Ti=0 is not well-defined. \n'
        msg += 'Consider constraining the parameters.'
        raise ValueError(msg)
    ci = pk.conc_trap(cp, t=t, dt=dt)/Ti
    return np.stack((cp, ci))

        
def _lconc_hf(ca, t=None, dt=1.0, Ti=None):
    # Ti = vi/PS
    # up = vp/ve
    cp = ca
    ci = pk.flux_comp(cp, Ti, t=t, dt=dt)
    return np.stack((cp, ci))


def _lconc_2cu(ca, t=None, dt=1.0, 
               Tp=None, E=None, Ti=None):
    # Ti = vi/PS
    cp = (1-E)*pk.flux_comp(ca, Tp, t=t, dt=dt)
    ci = pk.conc_trap(cp, t=t, dt=dt)/Ti
    return np.stack((cp, ci))

        
def _lconc_2cx(ca, t=None, dt=1.0, Tp=None, Ti=None, E=None):
    # c = C/Fp
    # cp = C0/vp = c0*Fp/vp = c0 * (1-E)/Tp
    # ci = C1/vi = c1*Fp/vi = c1 * (1-E)/E/Ti
    c = pk.conc_2cxm(ca, [Tp, Ti], E, t=t, dt=dt)
    cp = c[0,:] * (1-E) / Tp
    ci = c[1,:] * (1-E) / E / Ti
    return np.stack((cp, ci))


def flux_tissue(ca: np.ndarray, t=None, dt=1.0, kinetics='2CX', **params) -> np.ndarray:
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
    if kinetics == 'U':
        return _flux_u(ca, **params)
    elif kinetics == 'NX':
        return _flux_1c(ca, t=t, dt=dt, v=params['vp'], F=params['Fp'])
    elif kinetics == 'FX':
        return _flux_1c(ca, t=t, dt=dt, v=params['ve'], F=params['Fp'])
    elif kinetics == 'WV':
        return _flux_wv(ca, t=t, dt=dt, **params)
    elif kinetics == 'HFU':
        return _flux_hfu(ca, **params)
    elif kinetics == 'HF':
        return _flux_hf(ca, t=t, dt=dt, **params)
    elif kinetics == '2CU':
        return _flux_2cu(ca, t=t, dt=dt, **params)
    elif kinetics == '2CX':
        return _flux_2cx(ca, t=t, dt=dt, **params)
    # elif model=='2CF':
    #     return _flux_2cf(ca, t=t, dt=dt, **params)
    else:
        raise ValueError('Kinetic model ' + kinetics +
                         ' is not currently implemented.')


def _flux_u(ca, Fp=None):
    return pk.flux(Fp*ca, model='trap')


def _flux_1c(ca, t=None, dt=1.0, v=None, F=None):
    if F == 0:
        return np.zeros(len(ca))
    return pk.flux(F*ca, v/F, t=t, dt=dt, model='comp')


def _flux_wv(ca, t=None, dt=1.0, vi=None, Ktrans=None):
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = Ktrans*ca
    if Ktrans != 0:
        J[0, 1, :] = pk.flux(Ktrans*ca, vi/Ktrans, t=t, dt=dt, model='comp')
    return J


def _flux_hfu(ca, PS=None):
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = PS*ca
    return J


def _flux_hf(ca, t=None, dt=1.0, vi=None, PS=None):
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.inf
    J[1, 0, :] = PS*ca
    if PS == 0:
        J[0, 1, :] = 0*ca
    else:
        J[0, 1, :] = pk.flux(PS*ca, vi/PS, t=t, dt=dt, model='comp')
    return J


def _flux_2cu(ca, t=None, dt=1.0, vp=None, Fp=None, PS=None):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, vp=vp, Fp=Fp, PS=PS)
    J = np.zeros(((2, 2, len(ca))))
    if vp == 0:
        if Fp+PS != 0:
            Ktrans = Fp*PS/(Fp+PS)
            J[0, 0, :] = Fp*ca
            J[1, 0, :] = Ktrans*ca
    else:
        J[0, 0, :] = Fp*C[0, :]/vp
        J[1, 0, :] = PS*C[0, :]/vp
    return J


def _flux_2cx(ca, t=None, dt=1.0, vp=None, vi=None, Fp=None, PS=None):

    if np.isinf(Fp):
        return _flux_hf(ca, t=t, dt=dt, vi=vi, PS=PS)

    if Fp == 0:
        return np.zeros((2, 2, len(ca)))

    if Fp+PS == 0:
        return np.zeros((2, 2, len(ca)))
    if PS == 0:
        Jp = _flux_1c(ca, t=t, dt=dt, v=vp, F=Fp)
        J = np.zeros((2, 2, len(ca)))
        J[0, 0, :] = Jp
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
        return np.zeros((2, 2, len(ca)))
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)
    J = Fp*ca
    T = [Tp, Te]
    # Solve the system explicitly
    t = utils.tarray(len(J), t=t, dt=dt)
    Jo = np.zeros((2, 2, len(t)))
    J0 = pk.flux(J, T[0], t=t, model='comp')
    J10 = E*J0
    Jo[1, 0, :] = J10
    Jo[1, 1, :] = pk.flux(J10, T[1], t=t, model='comp')
    Jo[0, 0, :] = (1-E)*J0
    return Jo
