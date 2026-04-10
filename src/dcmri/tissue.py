"""Parameters characterizing a 2-site exchange tissue. 
For more detail see :ref:`two-site-exchange`.

Args:
    kinetics (str, optional): Tracer-kinetic regime. Possible values are
        '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to '2CX'.
    water_exchange (str, optional): Water exchange regime. Any combination
        of two of the letters 'F', 'N', 'R' is allowed. Defaults to 'RR'.

Returns: 
    list: tissue parameters

Raises:
    ValueError: if the configuration is not recognized.

Example:

    Print the parameters of a HFU tissue with restricted water 
    exchange:

    >>> import dcmri as dc

    >>> dc.relax_params('HFU', 'RR')
    ['PSe', 'PSc', 'H', 'vb', 'vi', 'PS']

Notes:

    Table :ref:`tissue-kinetic-regimes` list the water compartments and free 
    parameters for all configurations. Regimes without water exchange 
    across one or both of the barriers are not listed 
    explicitly (FN, NF, FR, NR and NN). They differ from restricted water 
    exchange only in the sense that the respective water permeabilities 
    *PSe* and/or *PSc* are zero. 

    .. _tissue-kinetic-regimes:
    .. list-table:: **Parameters by configuration** 
        :widths: 15 15 30 40
        :header-rows: 1 

        * - Water exchange
            - Indicator exchange
            - Water compartments
            - Free parameters
        * - **FF**
            - 
            - 
            - 
        * - FF
            - 2CX
            - vb + vi + vc
            - H, vb, vi, Fb, PS
        * - FF
            - 2CU
            - vb + vi + vc
            - H, vb, Fb, PS
        * - FF
            - HF
            - vb + vi + vc
            - H, vb, vi, PS
        * - FF
            - HFU
            - vb + vi + vc
            - H, vb, PS
        * - FF
            - FX
            - vb + vi + vc 
            - H, ve, Fb  
        * - FF
            - NX
            - vb + vi + vc
            - vb, Fb  
        * - FF
            - NXP
            - vb + vi + vc
            - vb, Fb  
        * - FF
            - U
            - vb + vi + vc
            - Fb
        * - FF
            - WV
            - vb + vi + vc
            - H, vi, Ktrans
        * - **RR**
            - 
            - 
            - 
        * - RR
            - 2CX
            - vb, vi, vc
            - PSe, PSc, H, vb, vi, Fb, PS
        * - RR
            - 2CU
            - vb, vi, vc
            - PSe, PSc, H, vb, vi, Fb, PS
        * - RR
            - HF
            - vb, vi, vc
            - PSe, PSc, H, vb, vi, PS
        * - RR
            - HFU
            - vb, vi, vc
            - PSe, PSc, H, vb, vi, PS
        * - RR
            - FX
            - vb, vi, vc 
            - PSe, PSc, H, vb, vi, Fb 
        * - RR
            - NX
            - vb, vi, vc 
            - PSe, vb, vi, Fb  
        * - RR
            - NXP
            - vb, vi, vc 
            - PSe, vb, vi, Fb 
        * - RR
            - U
            - vb, vi, vc 
            - PSe, vb, vi, Fb 
        * - RR
            - WV
            - vi, vi+vc
            - PSc, H, vi, Ktrans
        * - **RF**
            - 
            - 
            - 
        * - RF
            - 2CX
            - vb, vi+vc
            - PSe, H, vb, vi, Fb, PS
        * - RF
            - 2CU
            - vb, vi+vc
            - PSe, H, vb, Fb, PS
        * - RF
            - HF
            - vb, vi+vc
            - PSe, H, vb, vi, PS
        * - RF
            - HFU
            - vb, vi+vc
            - PSe, H, vb, PS
        * - RF
            - FX
            - vb, vi+vc
            - PSe, H, vb, vi, Fb
        * - RF
            - NX
            - vb, vi+vc
            - PSe, vb, Fb
        * - RF
            - NXP
            - vb, vi+vc
            - PSe, vb, Fb
        * - RF
            - U
            - vb, vi+vc
            - PSe, vb, Fb 
        * - RF
            - WV
            - vi+vc
            - H, vi, Ktrans
        * - **FR**
            - 
            - 
            -  
        * - FR
            - 2CX
            - vb+vi, vc
            - PSc, H, vb, vi, Fb, PS
        * - FR
            - 2CU
            - vb+vi, vc
            - PSc, H, vb, vi, Fb, PS
        * - FR
            - HF
            - vb+vi, vc
            - PSc, H, vb, vi, PS
        * - FR
            - HFU
            - vb+vi, vc
            - PSc, H, vb, vi, PS
        * - FR
            - FX
            - vb+vi, vc 
            - PSc, H, vb, vi, Fb
        * - FR
            - NX
            - vb+vi, vc 
            - PSc, vb, vi, Fb
        * - FR
            - NXP
            - vb+vi, vc 
            - PSc, vb, vi, Fb
        * - FR
            - U
            - vb+vi, vc 
            - PSc, vc, Fb
        * - FR
            - WV
            - vi, vc
            - PSc, H, vi, Ktrans
"""

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
    params (dict): free model parameters provided as keyword arguments. 
        Possible parameters depend on **kinetics** as detailed in Table 
        :ref:`two-site-exchange-kinetics`. 

Returns:
    numpy.ndarray: concentration
        If sum=True, or the tissue is one-compartmental, this 
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

    >>> p2x = {'H': 0.5, 'vb':0.1, 'vi':0.4, 'Fb':0.02, 'PS':0.005}
    >>> pwv = {'H': 0.5, 'vi':0.4, 'Ktrans':0.005*0.01/(0.005+0.01)}

    Generate plasma and extravascular tissue concentrations with the 2CX 
    and WV models:

    >>> C2x = dc.conc(ca, t=t, sum=False, kinetics='2CX', **p2x)
    >>> Cwv = dc.conc(ca, t=t, kinetics='WV', **pwv)

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


"""Free relaxation rates for a 2-site exchange tissue. For more detail see
:ref:`two-site-exchange`.

Note: the free relaxation rates are the relaxation rates of the tissue 
compartments in the absence of water exchange between them.

Args:
    ca (array-like): concentration in the blood of the arterial input.
    R10 (float): precontrast relaxation rate. The tissue is assumed to be 
    in fast exchange before injection of contrast agent.
    r1 (float): contrast agent relaxivity. 
    t (array_like, optional): the time points in sec of the input function 
    *ca*. If *t* is not provided, the time points are assumed to be 
    uniformly spaced with spacing *dt*. Defaults to None.
    dt (float, optional): spacing in seconds between time points for 
    uniformly spaced time points. This parameter is ignored if *t* is 
    explicity provided. Defaults to 1.0.
    kinetics (str, optional): Tracer-kinetic model. Possible values are
    '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to '2CX'.
    water_exchange (str, optional): Water exchange regime, Any combination
    of two of the letters 'F', 'N', 'R' is allowed. Defaults to 'FF'.
    params (dict): values for the parameters of the tissue,
    specified as keyword parameters. See table :ref:`tissue-kinetic-regimes` 
    for more detail on the parameters that are relevant in each regime. 

Returns: dict with
    numpy.ndarray: relaxation rates
    In the fast water exchange limit, the 
    relaxation rates are a 1D array. In all other situations, 
    relaxation rates are a 2D-array with dimensions (k,n), where k is 
    the number of compartments and n is the number of time points 
    in ca.
    volume fractions
    the volume fractions of the tissue compartments. 
    Returns None in 'FF' regime. 
    water flows
    2D array with water exchange 
    rates between tissue compartments. Returns None in 'FF' regime.
    numpy.ndarray: concentrations

Example:

    Compare the free relaxation rates without water exchange against 
    relaxation rates in fast exchange:

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
    >>> pf = {'H':0.5, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005}   
    >>> pn = {'H':0.5, 'vb':0.1, 'vi':0.3, 'Fb':0.01, 'PS':0.005}

    Calculate tissue relaxation rates without water exchange, 
    and also in the fast exchange limit for comparison:

    >>> R1f = dc.tissue.relax(ca, R10, r1, t=t, water_exchange='FF', **pf)['R1]
    >>> R1n = dc.tissue.relax(ca, R10, r1, t=t, water_exchange='NN', **pn)['R1]

    Plot the relaxation rates in the three compartments, and compare 
    against the fast exchange result:

    >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

    Plot restricted water exchange in the left panel:

    >>> ax0.set_title('Restricted water exchange')
    >>> ax0.plot(t/60, R1n[0,:], linestyle='-', 
    >>>          linewidth=2.0, color='darkred', label='Blood')
    >>> ax0.plot(t/60, R1n[1,:], linestyle='-', 
    >>>          linewidth=2.0, color='darkblue', label='Interstitium')
    >>> ax0.plot(t/60, R1n[2,:], linestyle='-', 
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
        
"""Longitudinal magnetization for a 2-site exchange tissue. For more 
detail see :ref:`two-site-exchange`.

Args:
    ca (array-like): concentration in the blood of the arterial input.
    R10 (float): precontrast relaxation rate. The tissue is assumed to be 
    in fast exchange before injection of contrast agent.
    r1 (float): contrast agent relaxivity. 
    t (array_like, optional): the time points in sec of the input function 
    *ca*. If *t* is not provided, the time points are assumed to be 
    uniformly spaced with spacing *dt*. Defaults to None.
    dt (float, optional): spacing in seconds between time points for 
    uniformly spaced time points. This parameter is ignored if *t* is 
    explicity provided. Defaults to 1.0.
    kinetics (str, optional): Tracer-kinetic model. Possible values are
    '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to '2CX'.
    water_exchange (str, optional): Water exchange regime, Any combination
    of two of the letters 'F', 'N', 'R' is allowed. Defaults to 'FF'.
    sequence (dict): the sequence model and its parameters. The 
    dictionary has one required key 'model' which specifies the signal 
    model. Currently either 'SS' or 'SR'. The other keys are the values 
    of the signal parameter, which depend on the model. See table 
    :ref:`Tissue-signal-parameters` for detail. 
    inflow (dict, optional): inflow model. If not provided, the in- and 
    outflow of magnetization is ignored. To include 
    inflow effects, **inflow** must be dictionary with the signal model 
    parameters for the arterial input. For the 'SS' signal model, 
    required parameters are 'R10a' and 'B1corr_a'. Defaults to None.
    params (dict): model parameters. See :ref:`Tissue-signal-parameters` 
    for more detail. Note: the tissue parameters are keyword 
    arguments for convenience, but a value is required.

Raises:
    ValueError: if a required parameter has no value assigned.
    NotImplementedError: if a combination of regimes is not yet 
    implemented. Currently the sequence type 'SR' only accepts fast 
    water exchange 'FF'.

Returns: dict with
    ndarray: magnetization as a 1D array. 
    ndarray: relaxation rates
    volume fractions
    the volume fractions of the tissue compartments. 
    Returns None in 'FF' regime. 
    water flows
    2D array with water exchange 
    rates between tissue compartments. Returns None in 'FF' regime.
    numpy.ndarray: concentrations 

Example:

    We verify that the effect of inflow is negligible in a steady state 
    sequence:

.. plot::
    :include-source:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import dcmri as dc

    Define constants and model parameters: 

    >>> R10, r1 = 1, 5000
    >>> seq = {'model': 'SS', 'FA':15, 'TR': 0.001, 'B1corr':1}
    >>> pars = {
    >>>     'sequence':seq, 'kinetics':'2CX', 'water_exchange':'NN', 
    >>>     'H':0.045, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005} 
    >>> inflow = {'R10a': 0.7, 'B1corr_a':1}

    Generate arterial blood concentrations:

    >>> t = np.arange(0, 300, 1.5)
    >>> ca = dc.aif_parker(t, BAT=20)/(1-0.45) 

    Calculate the signal with and without inflow:

    >>> Mf = dc.tissue.Mz(ca, R10, r1, t=t, inflow=inflow, **pars)
    >>> Mn = dc.tissue.Mz(ca, R10, r1, t=t, **pars)

    Compare them in a plot:

    >>> plt.figure()
    >>> plt.plot(t/60, np.sum(Mn['Mz'], axis=0), label='No inflow', linewidth=3)
    >>> plt.plot(t/60, np.sum(Mf['Mz'], axis=0), label='Inflow')
    >>> plt.xlabel('Time (min)')
    >>> plt.ylabel('Magnetization (A/cm)')
    >>> plt.legend()
    >>> plt.show()

Notes:

    .. _Mz-signal-parameters:
    .. list-table:: **Tissue Mz parameters**
        :widths: 20 30 30
        :header-rows: 1

        * - Parameters
        - When to use
        - Further detail
        * - Fb, PS, Ktrans, vb, H, vi,
            ve, vc, PSe, PSc.
        - Depends on **kinetics** and **water_exchange**
        - :ref:`tissue-kinetic-regimes`
        * - FA, TR, B1corr
        - Always
        - :ref:`params-per-sequence`
        * - TP, TC
        - If **sequence** is 'SR'
        - :ref:`params-per-sequence`
        * - R10a, B1corr_a
        - If **inflow** is not None
        - :ref:`relaxation-params`, :ref:`params-per-sequence`

"""

"""Indicator flux out of a 2-site exchange tissue.

Args:
    ca (array-like): concentration in the arterial input.
    t (array_like, optional): the time points of the input function *ca*. 
    If *t* is not provided, the time points are assumed to be uniformly 
    spaced with spacing *dt*. Defaults to None.
    dt (float, optional): spacing in seconds between time points for 
    uniformly spaced time points. This parameter is ignored if *t* is 
    provided. Defaults to 1.0.
    kinetics (str, optional): The kinetic model of the tissue (see below 
    for possible values). Defaults to '2CX'. 
    params (dict): free model parameters and their values (see below for 
    possible).

Returns: 
    numpy.ndarray: outflux
    For a one-compartmental tissue, outflux out of the 
    compartment as a 1D array in units of mmol/sec/mL or M/sec. For a 
    multi=compartmental tissue, outflux out of each compartment, and at 
    each time point, as a 3D array with dimensions *(2,2,k)*, where *2* 
    is the number of compartments and *k* is the number of time points 
    in *J*. Encoding of the first two indices is the same as for *E*: 
    *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* 
    is the flux from *i* directly to the outside. The flux is returned in 
    units of mmol/sec/mL or M/sec.
"""
from copy import deepcopy
import numpy as np

from dcmri import pk, rel, mz, sig
from dcmri.func import SuperFunc
from dcmri import pk_tissue
from dcmri.lexicon import SEQUENCES


class Conc(SuperFunc):

    _params_dict = {
        '2CX': ['T_a', 'H', 'vb', 'vi', 'Fb', 'PS'],
        'HF': ['T_a', 'H', 'vb', 'vi', 'PS'],
        'WV': ['T_a', 'H', 'vi', 'Ktrans'],
        '2CU': ['T_a', 'H', 'vb', 'Fb', 'PS'],
        'HFU': ['T_a', 'H', 'vb', 'PS'],
        'FX': ['T_a', 'H', 've', 'Fb'],
        'NX': ['T_a', 'vb', 'Fb'],
        'NXP': ['T_a', 'vb', 'Fb'],
        'U': ['T_a', 'Fb'],
    }

    configs = {'kinetics': deepcopy(list(_params_dict.keys()))}

    def __init__(self, kinetics='2CX', **params):
        cnfg = {'kinetics': kinetics}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        return deepcopy(self._params_dict[self._cnfg['kinetics']])

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params):
        p = self._update_pars(**params)

        ca = pk.flux_plug(ca, p['T_a'], dt=dt)
        params = {k: v for k, v in p.items() if k != 'T_a'}
        
        kinetics = self._cnfg['kinetics']
        if kinetics == 'U': return pk_tissue.Conc_u(ca, t=t, dt=dt, **params)
        if kinetics == 'FX': return pk_tissue.Conc_fx(ca, t=t, dt=dt, **params)
        if kinetics == 'NX': return pk_tissue.Conc_nx(ca, t=t, dt=dt, **params)
        if kinetics == 'NXP': return pk_tissue.Conc_nxp(ca, t=t, dt=dt, **params)
        if kinetics == 'WV': return pk_tissue.Conc_wv(ca, t=t, dt=dt, **params)
        if kinetics == 'HFU': return pk_tissue.Conc_hfu(ca, t=t, dt=dt, **params)
        if kinetics == 'HF': return pk_tissue.Conc_hf(ca, t=t, dt=dt, **params)
        if kinetics == '2CU': return pk_tissue.Conc_2cu(ca, t=t, dt=dt, **params)
        if kinetics == '2CX': return pk_tissue.Conc_2cx(ca, t=t, dt=dt, **params)


class R1(SuperFunc):
    configs = deepcopy(Conc.configs) | {
        'water_exchange': ['FF','RF','NF','FR','RR','NR','FN','RN','NN'],
    }
    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> list:
        p = WaterConc(**self._cnfg)._params()
        p += ['R10', 'r1']
        return p

    def __call__(self, C, **params):
        p = self._update_pars(**params)

        # Compute concentration in water compartments
        c = WaterConc(**self._cnfg)(C, **p)

        # Compute R1 of water compartments
        R1_result = [rel.relax(c[i,:], p['R10'], p['r1']) for i in range(c.shape[0])]
        return np.stack(R1_result)
    

class R2(SuperFunc):
    configs = {}
    
    def __init__(self, **params):
        self._cnfg = {}
        self._pars = self._set_pars(**params)

    def _params(self) -> list:
        return ['R20', 'r2']
    
    def __call__(self, C, **params):
        p = self._update_pars(**params)
        C = np.array(C)
        if C.ndim==2:
            C = C.sum(axis=0)
        R2_result = rel.relax(C, p['R20'], p['r2'])
        return R2_result


class R2s(SuperFunc):
    configs = {
        'transverse_relaxation': ['lin', 'quad', 'leakage'],
        'kinetics': deepcopy(Conc.configs['kinetics'])
    }
    def __init__(self, transverse_relaxation='lin', kinetics=None, **params):
        if transverse_relaxation == 'leakage' and kinetics is None:
            raise ValueError('Kinetic model must be specified for leakage model.')
        cnfg = {'transverse_relaxation': transverse_relaxation, 'kinetics': kinetics}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> list:
        if self._cnfg['transverse_relaxation'] == 'lin':
            p = ['R20s', 'r2s']
        elif self._cnfg['transverse_relaxation'] == 'quad':
            p = ['R20s', 'r2s', 'r2s_quad']
        elif self._cnfg['transverse_relaxation'] == 'leakage':
            p = ['R20s', 'r2s_vasc', 'r2s_ees'] 
            p += ContrastConc(self._cnfg['kinetics'])._params()
        return p
    
    def __call__(self, C, **params):
        p = self._update_pars(**params)
        C = np.array(C)

        if self._cnfg['transverse_relaxation'] == 'lin':
            if C.ndim==2:
                C = C.sum(axis=0)
            R2_result = p['R20s'] + p['r2s'] * C

        elif self._cnfg['transverse_relaxation'] == 'quad':
            if C.ndim==2:
                C = C.sum(axis=0)
            R2_result = p['R20s'] + p['r2s'] * C + p['r2s_quad'] * C**2

        elif self._cnfg['transverse_relaxation'] == 'leakage':
            c = ContrastConc(self._cnfg['kinetics'])(C)
            R2_result = p['R20s'] + p['r2s_vasc'] * np.abs(c[0,:] - c[1,:]) + p['r2s_ees'] * c[1,:]
        
        return R2_result


class Mz(SuperFunc):
    configs = deepcopy(R1.configs) | {'sequence': list(SEQUENCES.keys())}

    def __init__(self, kinetics='2CX', water_exchange='FF', sequence='3D-SPGR-SS', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange, 'sequence': sequence}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> dict:
        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']
        sequence = self._cnfg['sequence']
        seq = SEQUENCES[sequence]['mz_prep_tissue']
        iseq = SEQUENCES[sequence]['mz_prep_inflow']

        pars = []
        if iseq is not None:
            pars += mz.Mz(iseq)._params()
        pars += WaterVolumes(kin, wex)._params()
        pars += WaterFlows(kin, wex)._params() 
        pars += mz.Mz(seq)._params()
        pars = list(set(pars))
        # Exclude derived parameters
        return [p for p in pars if p not in ['v', 'Fw']]

    def __call__(self, R1t, R1a=None, **params) -> np.ndarray:
        p = self._update_pars(**params)

        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']
        seq = SEQUENCES[self._cnfg['sequence']]['mz_prep_tissue']
        iseq = SEQUENCES[self._cnfg['sequence']]['mz_prep_inflow']

        # Add magnetization inflow
        if iseq is None: # Model without inflow
            j = None
        elif R1a is None:
            j = None
        elif 'Fb' not in p:
            raise ValueError('Parameter Fb must be provided for inflow effects. Please set R1a=None or provide a value for Fb.')
        else:
            j = np.zeros_like(R1t)
            pi = p | {'v': 1, 'Fw': 0}
            j[0,:] = p['Fb'] * mz.Mz(iseq)(R1a, **pi)

        # Compartment volumes and flows
        vw = WaterVolumes(kin, wex, **p)() 
        Fw = WaterFlows(kin, wex, **p)()
        return mz.Mz(seq, **p)(R1t, j, v=vw, Fw=Fw)


class Signal(SuperFunc):

    configs = deepcopy(Mz.configs | R2s.configs)

    def __init__(self, kinetics='2CX', water_exchange='FF', sequence='3D-SPGR-SS', transverse_relaxation='lin', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange, 'sequence': sequence, 'transverse_relaxation': transverse_relaxation}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        kin, wex, seq, r2s = self._cnfg['kinetics'], self._cnfg['water_exchange'], self._cnfg['sequence'], self._cnfg['transverse_relaxation']
        p = []
        p += Conc(kin)._params()
        p += ['R10_a', 'r1']
        p += R1(kin, wex)._params()
        if seq in ['SE-EPI', 'DE-EPI']:
            p += R2()._params()
        if seq != 'SE-EPI':
            p += R2s(r2s, kin)._params()      
        p += Mz(kin, wex, seq)._params()
        p += sig.Readout(seq)._params()
        p = list(set(p))
        # Exclude derived
        return [k for k in p if k not in ['Mz', 'R2s', 'R2']]

    def __call__(self, ca, t=None, dt=1.0, **params):
        p = self._update_pars(**params)
        kin, wex, seq, r2s = self._cnfg['kinetics'], self._cnfg['water_exchange'], self._cnfg['sequence'], self._cnfg['transverse_relaxation']

        # Compute concentration
        C = Conc(kin)(ca, t, dt, **p)

        # Compute relaxation rates
        if 'Fb' in p:
            R1a = rel.relax(ca, p['R10_a'], p['r1'])
        else:
            R1a = None

        R1_arr = R1(kin, wex)(C, **p)
        
        if seq in ['SE-EPI', 'DE-EPI']:
            R2_arr = R2()(C, **p)
        else:
            R2_arr = None
        
        if seq == 'SE-EPI':
            R2s_arr = None
        elif seq in ['GE-EPI', 'DE-EPI']:
            R2s_arr = R2s(r2s, kin)(C, **p)
        elif p['TE'] > 0:
            R2s_arr = R2s(r2s, kin)(C, **p)
        else: 
            R2s_arr = None

        # Compute magnetization and signal
        Mz_arr = Mz(kin, wex, seq, **p)(R1_arr, R1a)
        return sig.Readout(seq, **p)(Mz=Mz_arr, R2s=R2s_arr, R2=R2_arr)


class Flux(SuperFunc):

    _params_dict = {
        '2CX': ['T_a', 'H', 'vb', 'vi', 'Fb', 'PS'],
        'HF': ['T_a', 'H', 'vi', 'PS'],
        'WV': ['T_a', 'H', 'vi', 'Ktrans'],
        '2CU': ['T_a', 'H', 'vb', 'Fb', 'PS'],
        'HFU': ['T_a', 'H', 'PS'],
        'FX': ['T_a', 'H', 've', 'Fb'],
        'NX': ['T_a', 'vb', 'Fb'],
        'NXP': ['T_a', 'vb', 'Fb'],
        'U': ['T_a', 'Fb'],
    }

    configs = {'kinetics': deepcopy(list(_params_dict.keys()))}
    
    def __init__(self, kinetics='2CX', **params):
        cnfg = {'kinetics': kinetics}       
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        return deepcopy(Flux._params_dict[self._cnfg['kinetics']])

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        p = self._update_pars(**params)

        ca = pk.flux_plug(ca, p['T_a'], dt=dt)
        params = {k: v for k, v in p.items() if k != 'T_a'}

        kinetics = self._cnfg['kinetics']
        if kinetics == 'U': return pk_tissue.flux_u(ca, **params)
        if kinetics == 'NX': return pk_tissue.flux_nx(ca, t=t, dt=dt, **params)
        if kinetics == 'NXP': return pk_tissue.flux_nxp(ca, t=t, dt=dt, **params)
        if kinetics == 'FX': return pk_tissue.flux_fx(ca, t=t, dt=dt, **params)
        if kinetics == 'WV': return pk_tissue.flux_wv(ca, t=t, dt=dt, **params)
        if kinetics == 'HFU': return pk_tissue.flux_hfu(ca, **params)
        if kinetics == 'HF': return pk_tissue.flux_hf(ca, t=t, dt=dt, **params)
        if kinetics == '2CU': return pk_tissue.flux_2cu(ca, t=t, dt=dt, **params)
        if kinetics == '2CX': return pk_tissue.flux_2cx(ca, t=t, dt=dt, **params)



class ContrastConc(SuperFunc):
    # Convert tissue concentration in blood and interstitium to concentration.
    # For uptake models this introduces a new parameter

    configs = deepcopy(Conc.configs)

    def __init__(self, kinetics='2CX', **params):
        cnfg = {'kinetics': kinetics}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)
        
    def _params(self) -> list:
        kinetics = self._cnfg['kinetics']
        if kinetics == 'FX':
            p = ['H', 've']
        elif kinetics in ['U', 'NX', 'NXP']:
            p = ['vb']
        elif kinetics == 'WV':
            p = ['vi']
        elif kinetics in ['HFU', '2CU', 'HF', '2CX']:
            p = ['vb', 'vi']
        return p

    def __call__(self, C, **params):
        p = self._update_pars(**params)

        kinetics = self._cnfg['kinetics']
    
        def div(Ci, vi):
            return Ci / vi if vi > 0 else Ci * 0

        c = np.zeros((2, C.shape[1]))

        if kinetics == 'FX':
            # vp = p['vb'] * (1 - p['H'])
            # vi = p['ve'] - vp
            # Cp = C[0,:] * vp / p['ve']
            # Ci = C[0,:] * vi / p['ve']
            # c[0,:] = div(Cp, p['vb'])
            # c[1,:] = div(Ci, vi)
            c[0,:] = div(C[0,:] * (1 - p['H']), p['ve'])
            c[1,:] = div(C[0,:], p['ve'])

        elif kinetics in ['U', 'NX', 'NXP']:
            c[0,:] = div(C[0,:], p['vb'])  
        
        elif kinetics == 'WV':
            # c[0,:] = ca but does not contribute to T2* at vb=0 so ignored here
            c[1,:] = div(C[0,:], p['vi']) 
        
        elif kinetics in ['HFU', '2CU', 'HF', '2CX']:
            c[0,:] = div(C[0,:], p['vb'])
            c[1,:] = div(C[1,:], p['vi']) # New parameter vi
        
        return c

class WaterConc(SuperFunc):
    # Convert tissue concentration in kinetic compartmetns to concentration in water compartments.

    configs = deepcopy(R1.configs)

    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)
        
    def _params(self) -> list:
        kinetics = self._cnfg['kinetics']
        wex = self._cnfg['water_exchange'].replace('N','R')

        p = []

        if kinetics == 'FX':
            if wex[0] != 'F':
                p += ['ve', 'vb', 'H']

        if wex == 'FF':
            p += []

        elif wex == 'RF':
            if kinetics == 'WV':
                p += []
            elif kinetics in ['U', 'NX', 'NXP', 'FX', 'HFU', 'HF', '2CU', '2CX']:
                p += ['vb']

        elif wex == 'FR':
            if kinetics == 'WV':
                p += ['vi']
            elif kinetics in ['U']:
                p += ['vc']
            elif kinetics in ['FX']:
                p += ['vb', 'H', 've'] 
            elif kinetics in ['NX', 'NXP', 'HFU', 'HF', '2CU', '2CX']:
                p += ['vb', 'vi']

        elif wex == 'RR':
            if kinetics == 'WV':
                p += ['vi']
            elif kinetics in ['NX', 'NXP', 'U']:
                p += ['vb']
            elif kinetics in ['FX']:
                p += ['vb', 'H', 've']
            elif kinetics in ['HF', 'HFU', '2CU', '2CX']:
                p += ['vb', 'vi']

        return list(set(p))


    def __call__(self, C, **params):
        p = self._update_pars(**params)

        kinetics = self._cnfg['kinetics']
        wex = self._cnfg['water_exchange'].replace('N','R')

        # Define helper functions
        def div(Ci, vi):
            if vi==0:
                # In this case the result does not matter
                return Ci * 0
            else:
                return Ci / vi

        def mix_1_comp(C, v=None):
            C = C.sum(axis=0)
            if v is None:
                return C.reshape(1, -1)
            else:
                c = np.zeros((2, C.size))
                c[0,:] = div(C, v)
                return c

        def mix_2_comp(C, v):
            c = np.zeros((2, C.shape[1]))
            c[0,:] = div(C[0,:], v)
            c[1,:] = div(C[1,:], 1 - v)
            return c

        def mix_3_comp(C, v):
            c = np.zeros((3, C.shape[1]))
            for i in range(C.shape[0]):
                c[i, :] = div(C[i,:], v[i])
            return c

        # Separate well-mixed space if needed
        if kinetics == 'FX': # comp = 'e'
            if wex[0] != 'F':
                if p['ve'] == 0:
                    Cp = C[0,:] * 0
                    Ci = C[0,:] * 0
                else:
                    p['vp'] = (1 - p['H']) * p['vb']
                    p['vi'] = p['ve'] - p['vp']
                    Cp = C[0,:] * p['vp'] / p['ve']
                    Ci = C[0,:] * p['vi'] / p['ve']
                C = np.stack((Cp, Ci))

        # Map indicator compartments to water compartments
        if wex == 'FF':
            return mix_1_comp(C)

        elif wex == 'RF':
            if kinetics == 'WV': # i
                return mix_1_comp(C)
            elif kinetics in ['U', 'NX', 'NXP']: # b
                return mix_1_comp(C, p['vb'])
            elif kinetics in ['FX', 'HFU', 'HF', '2CU', '2CX']: #bi
                return mix_2_comp(C, p['vb'])

        elif wex == 'FR':
            if kinetics == 'WV':
                return mix_1_comp(C, p['vi']) #i
            elif kinetics in ['U']:
                return mix_1_comp(C, 1 - p['vc'])  #b
            elif kinetics == 'FX':
                p['vp'] = (1 - p['H']) * p['vb']
                p['vi'] = p['ve'] - p['vp']
                return mix_1_comp(C, p['vb'] + p['vi'])
            elif kinetics in ['NX', 'NXP', 'HFU', 'HF', '2CU', '2CX']:
                return mix_1_comp(C, p['vb'] + p['vi']) # 1-vc

        elif wex == 'RR':
            if kinetics == 'WV':
                return mix_1_comp(C, p['vi'])
            elif kinetics in ['NX', 'NXP', 'U']:
                return mix_3_comp(C, [p['vb']])
            elif kinetics == 'FX':
                p['vp'] = (1 - p['H']) * p['vb']
                p['vi'] = p['ve'] - p['vp']
                return mix_3_comp(C, [p['vb'], p['vi']])
            elif kinetics in ['HF', 'HFU', '2CU', '2CX']:
                return mix_3_comp(C, [p['vb'], p['vi']])


    

class WaterVolumes(SuperFunc):
    configs = R1.configs

    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        geom = {
            ('2CX', 'FF'): [],
            ('2CU', 'FF'): [],
            ('HF', 'FF'): [],
            ('HFU', 'FF'): [],
            ('NX', 'FF'): [],
            ('NXP', 'FF'): [],
            ('WV', 'FF'): [],
            ('U', 'FF'): [],
            ('FX', 'FF'): [],

            ('2CX', 'FR'): ['vb', 'vi'],
            ('2CU', 'FR'): ['vb', 'vi'],
            ('HF', 'FR'): ['vb', 'vi'],
            ('HFU', 'FR'): ['vb', 'vi'],
            ('NX', 'FR'): ['vb', 'vi'],
            ('NXP', 'FR'): ['vb', 'vi'],
            ('WV', 'FR'): ['vi'],
            ('U', 'FR'): ['vb', 'vi'],
            ('FX', 'FR'): ['vb', 'vi'],

            ('2CX', 'RF'): ['vb'],
            ('2CU', 'RF'): ['vb'],
            ('HF', 'RF'): ['vb'],
            ('HFU', 'RF'): ['vb'],
            ('NX', 'RF'): ['vb'],
            ('NXP', 'RF'): ['vb'],
            ('WV', 'RF'): [],
            ('U', 'RF'): ['vb'],
            ('FX', 'RF'): ['vb'],

            ('2CX', 'RR'): ['vb', 'vi'],
            ('2CU', 'RR'): ['vb', 'vi'],
            ('HF', 'RR'): ['vb', 'vi'],
            ('HFU', 'RR'): ['vb', 'vi'],
            ('NX', 'RR'): ['vb', 'vi'],
            ('NXP', 'RR'): ['vb', 'vi'],
            ('WV', 'RR'): ['vi'],
            ('U', 'RR'): ['vb', 'vi'],
            ('FX', 'RR'): ['vb', 'vi'],
        }
        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange'].replace('N','R')
        return geom[(kin, wex)] 

    def __call__(self, **params):
        p = self._update_pars(**params)

        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']
        wex = wex.replace('N','R')

        # Add derived
        if {'vb', 'vi'}.issubset(p):
            p['vc'] = 1 - p['vb'] - p['vi']

        # Map water compartment volumes
        if (kin, wex) == ('2CX', 'FF'): return np.array([1])
        if (kin, wex) == ('2CU', 'FF'): return np.array([1])
        if (kin, wex) == ('HF', 'FF'): return np.array([1])
        if (kin, wex) == ('HFU', 'FF'): return np.array([1])
        if (kin, wex) == ('NX', 'FF'): return np.array([1])
        if (kin, wex) == ('NXP', 'FF'): return np.array([1])
        if (kin, wex) == ('WV', 'FF'): return np.array([1])
        if (kin, wex) == ('U', 'FF'): return np.array([1])
        if (kin, wex) == ('FX', 'FF'): return np.array([1])

        if (kin, wex) == ('2CX', 'FR'): return np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('2CU', 'FR'): return np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('HF', 'FR'): return np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('HFU', 'FR'): return np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('NX', 'FR'): return np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('NXP', 'FR'): return np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('WV', 'FR'): return np.array([p['vi'], 1-p['vi']])
        if (kin, wex) == ('U', 'FR'): return np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('FX', 'FR'): return np.array([1-p['vc'], p['vc']])

        if (kin, wex) == ('2CX', 'RF'): return np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('2CU', 'RF'): return np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('HF', 'RF'): return np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('HFU', 'RF'): return np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('NX', 'RF'): return np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('NXP', 'RF'): return np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('WV', 'RF'): return np.array([1])
        if (kin, wex) == ('U', 'RF'): return np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('FX', 'RF'): return np.array([p['vb'], 1-p['vb']])

        if (kin, wex) == ('2CX', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('2CU', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('HF', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('HFU', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('NX', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('NXP', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('WV', 'RR'): return np.array([p['vi'], 1-p['vi']])
        if (kin, wex) == ('U', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('FX', 'RR'): return np.array([p['vb'], p['vi'], p['vc']])


class WaterFlows(SuperFunc):
    configs = deepcopy(R1.configs)
    
    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> dict:
        geom = {
            ('2CX', 'FF'): ['Fb'],
            ('2CU', 'FF'): ['Fb'],
            ('HF', 'FF'): [],
            ('HFU', 'FF'): [],
            ('NX', 'FF'): ['Fb'],
            ('NXP', 'FF'): ['Fb'],
            ('WV', 'FF'): [],
            ('U', 'FF'): ['Fb'],
            ('FX', 'FF'): ['Fb'],

            ('2CX', 'FR'): ['Fb', 'PSc'],
            ('2CU', 'FR'): ['Fb', 'PSc'],
            ('HF', 'FR'): ['PSc'],
            ('HFU', 'FR'): ['PSc'],
            ('NX', 'FR'): ['Fb', 'PSc'],
            ('NXP', 'FR'): ['Fb', 'PSc'],
            ('WV', 'FR'): ['PSc'],
            ('U', 'FR'): ['Fb', 'PSc'],
            ('FX', 'FR'): ['Fb', 'PSc'],

            ('2CX', 'RF'): ['Fb', 'PSe'],
            ('2CU', 'RF'): ['Fb', 'PSe'],
            ('HF', 'RF'): ['PSe'],
            ('HFU', 'RF'): ['PSe'],
            ('NX', 'RF'): ['Fb', 'PSe'],
            ('NXP', 'RF'): ['Fb', 'PSe'],
            ('WV', 'RF'): [],
            ('U', 'RF'): ['Fb', 'PSe'],
            ('FX', 'RF'): ['Fb', 'PSe'],

            ('2CX', 'RR'): ['Fb', 'PSe', 'PSc'],
            ('2CU', 'RR'): ['Fb', 'PSe', 'PSc'],
            ('HF', 'RR'): ['PSe', 'PSc'],
            ('HFU', 'RR'): ['PSe', 'PSc'],
            ('NX', 'RR'): ['Fb', 'PSe', 'PSc'],
            ('NXP', 'RR'): ['Fb', 'PSe', 'PSc'],
            ('WV', 'RR'): ['PSc'],
            ('U', 'RR'): ['Fb', 'PSe', 'PSc'],
            ('FX', 'RR'): ['Fb', 'PSe', 'PSc'],
        }
        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']
        p = geom[(kin, deepcopy(wex).replace('N','R'))]
        if wex[0] == 'N' and 'PSe' in p:
            p.remove('PSe')
        if wex[1] == 'N' and 'PSc' in p:
            p.remove('PSc')
        return p

    def __call__(self, **params):
        p = self._update_pars(**params)
        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']

        if wex[0] == 'N':
            p['PSe'] = 0
        if wex[1] == 'N':
            p['PSc'] = 0

        # Map water compartment flows
        wex = wex.replace('N','R')
        if (kin, wex) == ('2CX', 'FF'): return np.full((1, 1), p['Fb'])
        if (kin, wex) == ('2CU', 'FF'): return np.full((1, 1), p['Fb'])
        if (kin, wex) == ('HF', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('HFU', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('NX', 'FF'): return np.full((1, 1), p['Fb'])
        if (kin, wex) == ('NXP', 'FF'): return np.full((1, 1), p['Fb'])
        if (kin, wex) == ('WV', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('U', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('FX', 'FF'): return np.full((1, 1), p['Fb'])

        if (kin, wex) == ('2CX', 'FR'): return np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('2CU', 'FR'): return np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('HF', 'FR'): return np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('HFU', 'FR'): return np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('NX', 'FR'): return np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('NXP', 'FR'): return np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('WV', 'FR'): return np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('U', 'FR'): return np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('FX', 'FR'): return np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])

        if (kin, wex) == ('2CX', 'RF'): return np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('2CU', 'RF'): return np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('HF', 'RF'): return np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('HFU', 'RF'): return np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('NX', 'RF'): return np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('NXP', 'RF'): return np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('WV', 'RF'): return np.array([0])
        if (kin, wex) == ('U', 'RF'): return np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('FX', 'RF'): return np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])

        if (kin, wex) == ('2CX', 'RR'): return np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('2CU', 'RR'): return np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('HF', 'RR'): return np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('HFU', 'RR'): return np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('NX', 'RR'): return np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('NXP', 'RR'): return np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('WV', 'RR'): return np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('U', 'RR'): return np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('FX', 'RR'): return np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])


def add_derived_params(p):

    if {'H', 'vb'}.issubset(p):
        p['vp'] = (1 - p['H']) * p['vb']

    if {'ve', 'vp'}.issubset(p):
        p['vi'] = p['ve'] - p['vp']

    elif {'vp', 'vi'}.issubset(p):
        p['ve'] = p['vp'] + p['vi']

    if {'vb', 'vi'}.issubset(p):
        p['vc'] = 1 - p['vb'] - p['vi']

    return p


# def _set_config(**cnfg):
#     for key, value in cnfg.items():
#         if value not in Signal.configs[key]:
#             raise ValueError(f'Config {value} is not recognized. Options are {list(Signal.configs[key])}.')  
            
#     return cnfg


