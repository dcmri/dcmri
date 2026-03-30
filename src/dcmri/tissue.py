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

import numpy as np

from dcmri import pk, rel, mz, sig
import dcmri.lexicon_utils as lexicon


class Conc:
    def __init__(self, kinetics='2CX', **params):
        cnfg = {'kinetics': kinetics}
        self._cnfg = _set_config(**cnfg)
        self._pars = lexicon.init(self._params())
        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()

    def _params_dict(self):
        return {
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

    def _params(self):
        return self._params_dict()[self._cnfg['kinetics']]

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        ca = pk.flux_plug(ca, p['T_a'], dt=dt)
        params = {k: v for k, v in p.items() if k != 'T_a'}
        
        kinetics = self._cnfg['kinetics']
        if kinetics == 'U':
            return _conc_u(ca, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _conc_fx(ca, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _conc_nx(ca, t=t, dt=dt, **params)
        elif kinetics == 'NXP':
            return _conc_nxp(ca, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _conc_wv(ca, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _conc_hfu(ca, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _conc_hf(ca, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _conc_2cu(ca, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _conc_2cx(ca, t=t, dt=dt, **params)
        # elif model=='2CF':
        #     return _conc_2cf(ca, *params, t=t, dt=dt)


class Relax:
    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange}
        self._cnfg = _set_config(**cnfg)
        self._pars = lexicon.init(self._params())

        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()

    def _params(self) -> list:
        kinetics = self._cnfg['kinetics']
        water_exchange = self._cnfg['water_exchange']
        wex = water_exchange.replace('N','R')

        p = Conc(kinetics)._params()

        if kinetics == 'FX':
            if wex[0] != 'F':
                p += ['ve', 'vb', 'H']

        if wex == 'FF':
            p += []

        elif wex == 'RF':
            if kinetics == 'WV':
                p += []

            elif kinetics in ['U', 'NX', 'NXP']:
                p += ['vb']

            elif kinetics in ['FX', 'HFU', 'HF', '2CU', '2CX']:
                p += ['vb']

        elif wex == 'FR':
            if kinetics == 'WV':
                p += ['vi']
            
            elif kinetics in ['U', 'NX', 'NXP']:
                p += ['vc']

            elif kinetics in ['FX']:
                p += ['vb', 'H', 've'] 
            
            elif kinetics in ['HFU', 'HF', '2CU', '2CX']:
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

        p += ['R10', 'r1']

        return list(set(p))


    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        kinetics = self._cnfg['kinetics']
        water_exchange = self._cnfg['water_exchange']

        C = Conc(kinetics)(ca, t, dt, **p)

        wex = water_exchange.replace('N','R')

        # Separate well-mixed space if needed
        if kinetics == 'FX': # comp = 'e'
            if wex[0] != 'F':
                if p['ve']==0:
                    Cp = C[0,:] * 0
                    Ci = C[0,:] * 0
                else:
                    vp = p['vb'] * (1 - p['H'])
                    Cp = C[0,:] * vp / p['ve']
                    Ci = C[0,:] * vp / p['ve']  
                C = np.stack((Cp, Ci))

        # Map indicator compartments to water compartments
        if wex == 'FF':
            c = _conc_wc_mix(C)

        elif wex == 'RF':
            if kinetics == 'WV': # i
                c = _conc_wc_mix(C)

            elif kinetics in ['U', 'NX', 'NXP']: # b
                c = _conc_wc_mix(C, p['vb'])

            elif kinetics in ['FX', 'HFU', 'HF', '2CU', '2CX']: #bi
                c = _conc_wc_2c(C, p['vb'])

        elif wex == 'FR':
            if kinetics == 'WV':
                c = _conc_wc_mix(C, p['vi']) #i

            elif kinetics in ['U', 'NX', 'NXP']:
                c = _conc_wc_mix(C, 1 - p['vc'])  #b

            elif kinetics == 'FX':
                vp = p['vb'] * (1 - p['H'])
                vi = p['ve'] - vp
                c = _conc_wc_mix(C, p['vb'] + vi)

            elif kinetics in ['HFU', 'HF', '2CU', '2CX']:
                c = _conc_wc_mix(C, p['vb'] + p['vi']) # 1-vc

        elif wex == 'RR':
            if kinetics == 'WV':
                c = _conc_wc_mix(C, p['vi'])

            elif kinetics in ['NX', 'NXP', 'U']:
                c = _conc_wc_3c(C, [p['vb']])

            elif kinetics == 'FX':
                vp = p['vb'] * (1 - p['H'])
                vi = p['ve'] - vp
                c = _conc_wc_3c(C, [p['vb'], vi])

            elif kinetics in ['HF', 'HFU', '2CU', '2CX']:
                c = _conc_wc_3c(C, [p['vb'], p['vi']])

        # Compute R1 of wter compartments
        R1 = [rel.relax(c[i,:], p['R10'], p['r1']) for i in range(c.shape[0])]
        R1 = np.stack(R1)

        return R1

class Mz:
    def __init__(self, kinetics='2CX', water_exchange='FF', sequence='SS', inflow_sequence='SS', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange, 'sequence': sequence, 'inflow_sequence': inflow_sequence}
        self._cnfg = _set_config(**cnfg)
        self._pars = lexicon.init(self._params())
        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()

    def _params(self) -> dict:
        kin = self._cnfg['kinetics']
        wex = self._cnfg['water_exchange']
        seq = self._cnfg['sequence']
        iseq = self._cnfg['inflow_sequence']

        pars = Relax(kin, wex)._params()
        pars += ['R10_a', 'r1']
        pars += mz.Mz(iseq)._params()
        pars += WaterVolumes(kin, wex)._params()
        pars += WaterFlows(kin, wex)._params() 
        pars += mz.Mz(seq)._params()
        
        return list(set(pars))

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        kin = self._cnfg['kinetics']
        wex = self._cnfg['water_exchange']
        seq = self._cnfg['sequence']
        iseq = self._cnfg['inflow_sequence']

        # Compute relaxation rates of tissue and input
        R1 = Relax(kin, wex)(ca, t, dt, **p)
        R1a = rel.relax(ca, p['R10_a'], p['r1'])

        # Add magnetization inflow
        if 'Fb' in p:
            j = np.zeros_like(R1)
            j[0,:] = p['Fb'] * mz.Mz(iseq)(R1a, **p)
        else:
            j = None

        # Compartment volumes and flows
        vw = WaterVolumes(kin, wex, **p)() 
        Fw = WaterFlows(kin, wex, **p)()
        return mz.Mz(seq, **p)(R1, vw, Fw, j)


class Signal:
    def __init__(self, kinetics='2CX', water_exchange='FF', sequence='SS', inflow_sequence='SS', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange, 'sequence': sequence, 'inflow_sequence': inflow_sequence}
        self._cnfg = _set_config(**cnfg)
        self._pars = lexicon.init(self._params())

        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()

    def _params(self):
        p = Mz(**self._cnfg)._params()
        p += sig.Readout()._params()
        return list(set(p))

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        # Compute signal
        Mz_arr = Mz(**(self._cnfg | p))(ca, t, dt)
        signal = sig.Readout(**p)(Mz_arr)

        return signal 


class Flux:
    def __init__(self, kinetics='2CX', **params):
        cnfg = {'kinetics': kinetics}       
        self._cnfg = _set_config(**cnfg)
        self._pars = lexicon.init(self._params())
        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()
    
    def _params_dict(self):
        return {
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

    def _params(self):
        return self._params_dict()[self._cnfg['kinetics']]

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        ca = pk.flux_plug(ca, p['T_a'], dt=dt)

        params = {k: v for k, v in p.items() if k != 'T_a'}

        kinetics = self._cnfg['kinetics']
        if kinetics == 'U':
            return _flux_u(ca, **params)
        elif kinetics == 'NX':
            return _flux_nx(ca, t=t, dt=dt, **params)
        elif kinetics == 'NXP':
            return _flux_nxp(ca, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _flux_fx(ca, t=t, dt=dt, **params)
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


class WaterVolumes:
    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange}
        self._cnfg = _set_config(**cnfg)
        self._pars = lexicon.init(self._params())
        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()

    def _params(self):
        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']
        wex = wex.replace('N','R')

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
        return geom[(kin, wex)] 

    def __call__(self, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']
        wex = wex.replace('N','R')

        # Get some notations
        if 'vb' in p:
            vb = p['vb']
        if 'vi' in p:
            vi = p['vi']
        if ('vb' in p) and ('vi' in p):
            vc = 1 - vb - vi

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

        if (kin, wex) == ('2CX', 'FR'): return np.array([1-vc, vc])
        if (kin, wex) == ('2CU', 'FR'): return np.array([1-vc, vc])
        if (kin, wex) == ('HF', 'FR'): return np.array([1-vc, vc])
        if (kin, wex) == ('HFU', 'FR'): return np.array([1-vc, vc])
        if (kin, wex) == ('NX', 'FR'): return np.array([1-vc, vc])
        if (kin, wex) == ('NXP', 'FR'): return np.array([1-vc, vc])
        if (kin, wex) == ('WV', 'FR'): return np.array([vi, 1-vi])
        if (kin, wex) == ('U', 'FR'): return np.array([1-vc, vc])
        if (kin, wex) == ('FX', 'FR'): return np.array([1-vc, vc])

        if (kin, wex) == ('2CX', 'RF'): return np.array([vb, 1-vb])
        if (kin, wex) == ('2CU', 'RF'): return np.array([vb, 1-vb])
        if (kin, wex) == ('HF', 'RF'): return np.array([vb, 1-vb])
        if (kin, wex) == ('HFU', 'RF'): return np.array([vb, 1-vb])
        if (kin, wex) == ('NX', 'RF'): return np.array([vb, 1-vb])
        if (kin, wex) == ('NXP', 'RF'): return np.array([vb, 1-vb])
        if (kin, wex) == ('WV', 'RF'): return np.array([1])
        if (kin, wex) == ('U', 'RF'): return np.array([vb, 1-vb])
        if (kin, wex) == ('FX', 'RF'): return np.array([vb, 1-vb])

        if (kin, wex) == ('2CX', 'RR'): return np.array([vb, vi, vc])
        if (kin, wex) == ('2CU', 'RR'): return np.array([vb, vi, vc])
        if (kin, wex) == ('HF', 'RR'): return np.array([vb, vi, vc])
        if (kin, wex) == ('HFU', 'RR'): return np.array([vb, vi, vc])
        if (kin, wex) == ('NX', 'RR'): return np.array([vb, vi, vc])
        if (kin, wex) == ('NXP', 'RR'): return np.array([vb, vi, vc])
        if (kin, wex) == ('WV', 'RR'): return np.array([vi, 1-vi])
        if (kin, wex) == ('U', 'RR'): return np.array([vb, vi, vc])
        if (kin, wex) == ('FX', 'RR'): return np.array([vb, vi, vc])


class WaterFlows:
    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {'kinetics': kinetics, 'water_exchange': water_exchange}
        self._cnfg = _set_config(**cnfg)
        self._pars = lexicon.init(self._params())
        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()

    def _params(self) -> dict:
        kin = self._cnfg['kinetics']
        wex = self._cnfg['water_exchange']
        wex = wex.replace('N','R')

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
        return geom[(kin, wex)]

    def __call__(self, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        kin, wex = self._cnfg['kinetics'], self._cnfg['water_exchange']
        wex = wex.replace('N','R')

        # Get some notations
        if 'Fb' in p:
            Fb = p['Fb']
        if 'PSc' in p:
            PSc = p['PSc']
        if 'PSe' in p:
            PSe = p['PSe']

        # Map water compartment flows
        if (kin, wex) == ('2CX', 'FF'): return np.full((1, 1), Fb)
        if (kin, wex) == ('2CU', 'FF'): return np.full((1, 1), Fb)
        if (kin, wex) == ('HF', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('HFU', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('NX', 'FF'): return np.full((1, 1), Fb)
        if (kin, wex) == ('NXP', 'FF'): return np.full((1, 1), Fb)
        if (kin, wex) == ('WV', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('U', 'FF'): return np.full((1, 1), 0)
        if (kin, wex) == ('FX', 'FF'): return np.full((1, 1), Fb)

        if (kin, wex) == ('2CX', 'FR'): return np.array([[Fb, PSc], [PSc, 0]])
        if (kin, wex) == ('2CU', 'FR'): return np.array([[Fb, PSc], [PSc, 0]])
        if (kin, wex) == ('HF', 'FR'): return np.array([[0, PSc], [PSc, 0]])
        if (kin, wex) == ('HFU', 'FR'): return np.array([[0, PSc], [PSc, 0]])
        if (kin, wex) == ('NX', 'FR'): return np.array([[Fb, PSc], [PSc, 0]])
        if (kin, wex) == ('NXP', 'FR'): return np.array([[Fb, PSc], [PSc, 0]])
        if (kin, wex) == ('WV', 'FR'): return np.array([[0, PSc], [PSc, 0]])
        if (kin, wex) == ('U', 'FR'): return np.array([[0, PSc], [PSc, 0]])
        if (kin, wex) == ('FX', 'FR'): return np.array([[Fb, PSc], [PSc, 0]])

        if (kin, wex) == ('2CX', 'RF'): return np.array([[Fb, PSe], [PSe, 0]])
        if (kin, wex) == ('2CU', 'RF'): return np.array([[Fb, PSe], [PSe, 0]])
        if (kin, wex) == ('HF', 'RF'): return np.array([[0, PSe], [PSe, 0]])
        if (kin, wex) == ('HFU', 'RF'): return np.array([[0, PSe], [PSe, 0]])
        if (kin, wex) == ('NX', 'RF'): return np.array([[Fb, PSe], [PSe, 0]])
        if (kin, wex) == ('NXP', 'RF'): return np.array([[Fb, PSe], [PSe, 0]])
        if (kin, wex) == ('WV', 'RF'): return np.array([0])
        if (kin, wex) == ('U', 'RF'): return np.array([[0, PSe], [PSe, 0]])
        if (kin, wex) == ('FX', 'RF'): return np.array([[Fb, PSe], [PSe, 0]])

        if (kin, wex) == ('2CX', 'RR'): return np.array([[Fb, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])
        if (kin, wex) == ('2CU', 'RR'): return np.array([[Fb, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])
        if (kin, wex) == ('HF', 'RR'): return np.array([[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])
        if (kin, wex) == ('HFU', 'RR'): return np.array([[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])
        if (kin, wex) == ('NX', 'RR'): return np.array([[Fb, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])
        if (kin, wex) == ('NXP', 'RR'): return np.array([[Fb, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])
        if (kin, wex) == ('WV', 'RR'): return np.array([[0, PSc], [PSc, 0]])
        if (kin, wex) == ('U', 'RR'): return np.array([[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])
        if (kin, wex) == ('FX', 'RR'): return np.array([[Fb, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]])


def derive_params(p):

    if {'H', 'vb'}.issubset(p):
        p['vp'] = (1 - p['H']) * p['vb']

    if {'ve', 'vp'}.issubset(p):
        p['vi'] = p['ve'] + p['vp']

    elif {'vp', 'vi'}.issubset(p):
        p['ve'] = p['vp'] + p['vi']

    return p


def _set_config(**cnfg):
    if 'kinetics' in cnfg:
        kinetics = cnfg['kinetics']
        if kinetics not in ['WV', 'FX', 'U', 'NX', 'NXP', 'FX', 'HFU', 'HF', '2CU', '2CX']:
            raise ValueError(f'Kinetic model {kinetics} is not currently implemented.')
    
    if 'water_exchange' in cnfg:
        water_exchange = cnfg['water_exchange']
        if (water_exchange[0] not in ['F', 'R', 'N']) or (
            water_exchange[1] not in ['F', 'R', 'N']):
            raise ValueError(
                "Water exchange regime '" +
                str(water_exchange) + "' is not recognised.\n" + 
                "Possible values are: 'FF','RF','NF','FR','RR','NR','FN','RN','NN'"
            )
    
    if 'sequence' in cnfg:
        sequence = cnfg['sequence']
        if sequence not in ['SS', 'SR', 'IR-SS', 'PR-SS', 'PR', 'SSI', 'GE-EPI', 'SE-EPI', 'None']:
            raise ValueError(f'Sequence {sequence} is not currently implemented.')   
    
    if 'inflow_sequence' in cnfg:
        inflow_sequence = cnfg['inflow_sequence']
        if inflow_sequence not in ['SS', 'SR', 'IR-SS', 'PR-SS', 'PR', 'SSI', 'GE-EPI', 'SE-EPI', 'None']:
            raise ValueError(f'Sequence {sequence} is not currently implemented.')     
        
    return cnfg

def _c(C,v):
    if v==0:
        # In this case the result does not matter
        return C*0
    else:
        return C/v
        

def _conc_wc_mix(C, v=None):
    C = C.sum(axis=0)
    if v is None:
        return C.reshape(1, -1)
    else:
        c = np.zeros((2, C.size))
        c[0,:] = _c(C, v)
        return c

def _conc_wc_2c(C, v):
    c0 = _c(C[0,:], v)
    c1 = _c(C[1,:], 1 - v)
    return np.stack((c0, c1))

def _conc_wc_3c(C, v):
    c = np.zeros((3, C.shape[1]))
    for i in range(C.shape[0]):
        c[i, :] = _c(C[i,:], v[i])
    return c





    
def _conc_u(ca, t=None, dt=1.0, Fb=None):
    C = pk.conc_trap(Fb*ca, t=t, dt=dt)
    return C.reshape(1, -1)
    
def _conc_fx(ca, t=None, dt=1.0, H=None, ve=None, Fb=None):
    if Fb == 0:
        ce = ca*0
    else:
        Fp = (1-H)*Fb
        ce = pk.flux_comp(ca/(1-H), ve/Fp, t=t, dt=dt)
    return ve*ce.reshape(1, -1)

def _conc_nx(ca, t=None, dt=1.0, vb=None, Fb=None):
    if Fb == 0:
        Cb = ca*0
    else:
        Cb = pk.conc_comp(Fb*ca, vb/Fb, t=t, dt=dt)
    return Cb.reshape(1, -1)

def _conc_nxp(ca, t=None, dt=1.0, vb=None, Fb=None):
    if Fb == 0:
        Cb = ca*0
    else:
        Cb = pk.conc_plug(Fb*ca, vb/Fb, t=t, dt=dt)
    return Cb.reshape(1, -1)

def _conc_wv(ca, t=None, dt=1.0, H=None, vi=None, Ktrans=None):
    if Ktrans == 0:
        ci = ca*0
    else:
        ci = pk.flux_comp(ca/(1-H), vi/Ktrans, t=t, dt=dt)
    return vi*ci.reshape(1, -1)

def _conc_hfu(ca, t=None, dt=1.0, H=None, vb=None, PS=None):
    vp = vb*(1-H)
    cp = ca/(1-H)
    Ci = pk.conc_trap(PS*cp, t=t, dt=dt)
    return np.stack((vp*cp, Ci)) 

def _conc_hf(ca, t=None, dt=1.0, H=None, vi=None, vb=None, PS=None):
    vp = vb*(1-H)
    ca = ca/(1-H)
    Cp = vp*ca
    if PS == 0:
        Ci = 0*ca
    else:
        Ci = pk.conc_comp(PS*ca, vi/PS, t=t, dt=dt)
    return np.stack((Cp, Ci))

def _conc_2cu(ca, t=None, dt=1.0, H=None, vb=None, Fb=None, PS=None):
    vp = (1-H)*vb
    Fp = (1-H)*Fb
    if np.isinf(Fp):
        return _conc_hfu(ca, t=t, dt=dt, H=H, vb=vb, PS=PS)
    ca = ca/(1-H)
    if Fp+PS == 0:
        return np.zeros((2, len(ca)))
    Tp = vp/(Fp+PS)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    if vp == 0:
        Ktrans = PS*Fp/(PS+Fp)
        Ci = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ci = pk.conc_trap(PS*Cp/vp, t=t, dt=dt)
    return np.stack((Cp, Ci))

def _conc_2cx(ca, t=None, dt=1.0, H=None, vi=None, vb=None, Fb=None, PS=None):
    vp = (1-H)*vb
    Fp = (1-H)*Fb
    if np.isinf(Fp):
        return _conc_hf(ca, t=t, dt=dt, H=H, vi=vi, vb=vb, PS=PS)

    ca = ca/(1-H)
    J = Fp*ca

    if Fp+PS == 0:
        Cp = np.zeros(len(ca))
        Ce = np.zeros(len(ca))
        return np.stack((Cp, Ce))

    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)

    if PS == 0:
        Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
        Ci = np.zeros(len(ca))
        return np.stack((Cp, Ci))

    Ti = vi/PS
    C = pk.conc_2cxm(J, [Tp, Ti], E, t=t, dt=dt)
    return C
    




def _flux_u(ca, Fb=None):
    return pk.flux(Fb*ca, model='trap')


def _flux_nx(ca, t=None, dt=1.0, vb=None, Fb=None):
    if Fb == 0:
        return np.zeros(len(ca))
    return pk.flux(Fb*ca, vb/Fb, t=t, dt=dt, model='comp')

def _flux_nxp(ca, t=None, dt=1.0, vb=None, Fb=None):
    if Fb == 0:
        return np.zeros(len(ca))
    return pk.flux(Fb*ca, vb/Fb, t=t, dt=dt, model='plug')

def _flux_fx(ca, t=None, dt=1.0, H=None, ve=None, Fb=None):
    if Fb == 0:
        return np.zeros(len(ca))
    Fp = Fb*(1-H)
    return pk.flux(Fb*ca, ve/Fp, t=t, dt=dt, model='comp')

def _flux_wv(ca, t=None, dt=1.0, H=None, vi=None, Ktrans=None):
    ca = ca/(1-H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = Ktrans*ca
    if Ktrans != 0:
        J[0, 1, :] = pk.flux(Ktrans*ca, vi/Ktrans, t=t, dt=dt, model='comp')
    return J

def _flux_hfu(ca, H=None, PS=None):
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = PS*ca/(1-H)
    return J


def _flux_hf(ca, t=None, dt=1.0, H=None, vi=None, PS=None):
    ca = ca/(1-H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.inf
    J[1, 0, :] = PS*ca
    if PS == 0:
        J[0, 1, :] = 0*ca
    else:
        J[0, 1, :] = pk.flux(PS*ca, vi/PS, t=t, dt=dt, model='comp')
    return J


def _flux_2cu(ca, t=None, dt=1.0, H=None, vb=None, Fb=None, PS=None):
    C = _conc_2cu(ca, t=t, dt=dt, H=H, vb=vb, Fb=Fb, PS=PS)
    ca = ca/(1-H)
    Fp = Fb*(1-H)
    J = np.zeros(((2, 2, len(ca))))
    if vb == 0:
        if Fp+PS != 0:
            Ktrans = Fp*PS/(Fp+PS)
            J[0, 0, :] = Fp*ca
            J[1, 0, :] = Ktrans*ca
    else:
        J[0, 0, :] = Fp*C[0, :]/vb
        J[1, 0, :] = PS*C[0, :]/vb
    return J


def _flux_2cx(ca, t=None, dt=1.0, H=None, vb=None, vi=None, Fb=None, PS=None):

    if np.isinf(Fb):
        return _flux_hf(ca, t=t, dt=dt, H=H, vi=vi, PS=PS)

    if Fb == 0:
        return np.zeros((2, 2, len(ca)))

    Fp = Fb*(1-H)

    if PS == 0:
        Jp = _flux_nx(ca, t=t, dt=dt, vb=vb, Fb=Fb)
        J = np.zeros((2, 2, len(ca)))
        J[0, 0, :] = Jp
        return J
    C = _conc_2cx(ca, t=t, dt=dt, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    # Derive standard parameters
    vp = vb*(1-H)
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


# def _flux_2cf(ca, t=None, dt=1.0, vp=None, Fp=None, PS=None, Te=None):
#     if Fp+PS == 0:
#         return np.zeros((2, 2, len(ca)))
#     # Derive standard parameters
#     Tp = vp/(Fp+PS)
#     E = PS/(Fp+PS)
#     J = Fp*ca
#     T = [Tp, Te]
#     # Solve the system explicitly
#     t = utils.tarray(len(J), t=t, dt=dt)
#     Jo = np.zeros((2, 2, len(t)))
#     J0 = pk.flux(J, T[0], t=t, model='comp')
#     J10 = E*J0
#     Jo[1, 0, :] = J10
#     Jo[1, 1, :] = pk.flux(J10, T[1], t=t, model='comp')
#     Jo[0, 0, :] = (1-E)*J0
#     return Jo
