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
    >>> ca = dc.aif.parker(t, BAT=20)

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
    >>> ca = dc.aif.parker(t, BAT=20)

    Define constants and model parameters: 

    >>> R10, r1 = 1/dc.const.T1(), dc.const.r1()     
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
    >>> ca = dc.aif.parker(t, BAT=20)/(1-0.45) 

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

from dcmri.bloch.tissue import Longitudinal, Readout
from dcmri.core.layer import LayerFunction
from dcmri.lexicon.dicts import SEQUENCES


class WaterVolumesTissueX(LayerFunction):
    configs = {
        'kinetics': ['2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'],
        'water_exchange': ['FF','RF','NF','FR','RR','NR','FN','RN','NN'],
    }
    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {
            'kinetics': kinetics, 
            'water_exchange': water_exchange,
        }
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

    def __call__(self, **params) -> np.ndarray:
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


class WaterFlowsTissueX(LayerFunction):
    configs = {
        'kinetics': ['2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'],
        'water_exchange': ['FF','RF','NF','FR','RR','NR','FN','RN','NN'],
    }
    def __init__(self, kinetics='2CX', water_exchange='FF', **params):
        cnfg = {
            'kinetics': kinetics, 
            'water_exchange': water_exchange,
        }
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

    def __call__(self, **params) -> np.ndarray:
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




class MzTissueX(LayerFunction):
    configs = deepcopy(WaterVolumesTissueX.configs | Longitudinal.configs)

    def __init__(
        self, 
        kinetics='2CX', 
        water_exchange='FF', 
        sequence='3D-SPGR-SS', 
        **params,
    ):
        cnfg = {
            'kinetics': kinetics, 
            'water_exchange': water_exchange, 
            'sequence': sequence,
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> dict:
        kin, wex, seq = self._cnfg.values()

        pars = WaterVolumesTissueX(kin, wex)._params()
        pars += WaterFlowsTissueX(kin, wex)._params() 
        pars += [p for p in Longitudinal(seq)._params() if p not in ['R1', 'R1i', 'Fi', 'v', 'Fw']]
        return list(set(pars))

    def __call__(self, R1=None, R1a=None, **params) -> np.ndarray:
        p = self._update_pars(**params)
        kin, wex, seq = self._cnfg.values()

        # Check that required parameters are provided
        weighting = SEQUENCES[seq]['parameters']['tissue']
        if 'R1' in weighting:
            if R1 is None:
                raise ValueError("R1 must be provided for a T1*-weighted sequence.")

        # Compartment volumes and flows
        vw = WaterVolumesTissueX(kin, wex)(**p) 
        Fw = WaterFlowsTissueX(kin, wex)(**p)

        # If the sequence does not have T1-weighting, return equilibrium
        if R1 is None:
            return np.full_like(vw.size, p['me'])

        # Inlet flow and relaxation rates
        if 'Fb' in p:
            R1i = np.zeros_like(R1)
            R1i[0,:] = R1a
            Fi = np.zeros(R1i.shape[0])
            Fi[0] = p['Fb']
        else:
            R1i = None
            Fi = None

        return Longitudinal(seq, **p)(R1=R1, R1i=R1i, Fi=Fi, v=vw, Fw=Fw)


class SignalTissueX(LayerFunction):
    configs = deepcopy(MzTissueX.configs)

    def __init__(self, 
        kinetics='2CX', 
        water_exchange='FF', 
        sequence='3D-SPGR-SS', 
        **params,
    ):
        cnfg = {
            'kinetics': kinetics, 
            'water_exchange': water_exchange, 
            'sequence': sequence, 
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        kin, wex, seq = self._cnfg.values()
        p = MzTissueX(kin, wex, seq)._params()
        p += [k for k in Readout(seq)._params() if k not in ['Mz', 'R2s', 'R2']]
        return list(set(p))

    def __call__(self, R1=None, R2=None, R2s=None, R1a=None, **params):
        p = self._update_pars(**params)
        kin, wex, seq = self._cnfg.values()

        if R1 is not None:
            Mz_arr = MzTissueX(kin, wex, seq, **p)(R1, R1a)
        elif R2 is not None:
            Mz_arr = np.full(R2.shape, p['me'], dtype=float)
        elif R2s is not None:
            Mz_arr = np.full(R2s.shape, p['me'], dtype=float).reshape(1, -1)
        
        # Signal
        return Readout(seq, **p)(Mz=Mz_arr, R2=R2, R2s=R2s)