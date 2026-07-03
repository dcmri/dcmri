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
    R1b (float): precontrast relaxation rate. The tissue is assumed to be 
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

    >>> R1b, r1 = 1/dc.const.T1(), dc.const.r1()     
    >>> pf = {'H':0.5, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005}   
    >>> pn = {'H':0.5, 'vb':0.1, 'vi':0.3, 'Fb':0.01, 'PS':0.005}

    Calculate tissue relaxation rates without water exchange, 
    and also in the fast exchange limit for comparison:

    >>> R1f = dc.tissue.relax(ca, R1b, r1, t=t, water_exchange='FF', **pf)['R1]
    >>> R1n = dc.tissue.relax(ca, R1b, r1, t=t, water_exchange='NN', **pn)['R1]

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
    R1b (float): precontrast relaxation rate. The tissue is assumed to be 
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
    required parameters are 'R1ba' and 'B1corr_a'. Defaults to None.
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

    >>> R1b, r1 = 1, 5000
    >>> seq = {'model': 'SS', 'FA':15, 'TR': 0.001, 'B1corr':1}
    >>> pars = {
    >>>     'sequence':seq, 'kinetics':'2CX', 'water_exchange':'NN', 
    >>>     'H':0.045, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005} 
    >>> inflow = {'R1ba': 0.7, 'B1corr_a':1}

    Generate arterial blood concentrations:

    >>> t = np.arange(0, 300, 1.5)
    >>> ca = dc.aif.parker(t, BAT=20)/(1-0.45) 

    Calculate the signal with and without inflow:

    >>> Mf = dc.tissue.Mz(ca, R1b, r1, t=t, inflow=inflow, **pars)
    >>> Mn = dc.tissue.Mz(ca, R1b, r1, t=t, **pars)

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
        * - R1ba, B1corr_a
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
from itertools import combinations

import numpy as np

from dcmri.core.module import Module
from dcmri.relaxivity.modules_tissue import R2, R2s, R1


class R1TissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
        't1_relaxation': {'lin'},
    }
    defaults = {
        'kinetics': '2CX', 
        'water_exchange': 'FF',
        't1_relaxation': 'lin',
    }
    def __init__(self, imap: dict=None, **config):
        self.set_config(config)
        self._conc = WaterConcTissueX(**config)
        self._R1 = R1(fast_water_exchange=False, **config)
        self.map_inputs(imap)

    def inputs(self) -> set:
        inputs = {'C', 'R1b', 'r1'}
        inputs |= self._R1.mapped_inputs()
        inputs |= self._conc.mapped_inputs()
        inputs -= self._conc.outputs()
        return inputs
    
    def outputs(self) -> set:
        return {'R1'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # R1b and r1 are scalar
        # c is dims (nc, nt)

        # Concentration in water compartments
        c = self._conc(p)

        # Assume R1b and r1 is the same in all compartments
        p |= c
        p['R1b'] = np.full(c['c'].shape[0], p['R1b'])
        p['r1'] = np.full(c['c'].shape[0], p['r1'])

        return self._R1(p)
    

class R2TissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
        't2_relaxation': {'lin'},
    }
    defaults = {
        'kinetics': '2CX', 
        'water_exchange': 'FF',
        't2_relaxation': 'lin',
    }
    def __init__(self, imap: dict=None, **config):
        self.set_config(config)
        self._conc = WaterConcTissueX(**config)
        self._R2 = R2(fast_water_exchange=False, **config)
        self.map_inputs(imap)

    def inputs(self) -> set:
        inputs = {'C', 'R2b', 'r2'}
        inputs |= self._R2.mapped_inputs()
        inputs |= self._conc.mapped_inputs()
        inputs -= self._conc.outputs()
        return inputs
    
    def outputs(self) -> set:
        return {'R2'}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # Cconcentration in water compartments
        c = self._conc(p)

        # Assume R1b and r1 is the same in all compartments
        p |= c
        p['R2b'] = np.full(c['c'].shape[0], p['R2b'])
        p['r2'] = np.full(c['c'].shape[0], p['r2'])

        return self._R2(p)


class R2sTissueX(Module): 
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        't2s_relaxation': {'lin', 'quad', 'leakage'},
    }
    defaults = {
        'kinetics': None,
        't2s_relaxation': 'leakage', 
    }
    def __init__(self, imap: dict=None, **config):
        self.set_config(config)

        if self.config['t2s_relaxation'] == 'leakage' and self.config['kinetics'] is None:
            raise ValueError('kinetics must be specified when t2s_relaxation is leakage.')
        
        if self.config['t2s_relaxation'] == 'leakage':
            self._conc = ContrastConcTissueX(**config)

        self._R2s = R2s(**config)
        self.map_inputs(imap)

    def inputs(self) -> set:
        inputs = {'C'}
        inputs |= self._R2s.mapped_inputs()

        if self.config['t2s_relaxation'] == 'leakage':
            inputs |= self._conc.mapped_inputs()
            inputs -= self._conc.outputs()

        return inputs
    
    def outputs(self) -> set:
        return {'R2s'}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        if self.config['t2s_relaxation'] == 'leakage':
            p |= self._conc(p)
            
        return self._R2s(p)


# # Build all possible combinations of relaxation rates
# weighting = ['R1', 'R2', 'R2s']
# all_combinations = [
#     tuple(set(combo))
#     for r in range(1, len(weighting) + 1)
#     for combo in combinations(weighting, r)
# ]

class RelaxTissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
        't2s_relaxation': {None, 'lin', 'quad', 'leakage'},
        't2_relaxation': {None, 'lin'},
        't1_relaxation': {None, 'lin'},
    }
    defaults = {
        'kinetics': '2CX', 
        'water_exchange': 'FF', 
        't2s_relaxation': 'lin',
        't2_relaxation': 'lin',
        't1_relaxation': 'lin',
    }
    def __init__(self, imap: dict=None, **config):
        self.set_config(config)

        if self.config['t1_relaxation']:
            self._R1 = R1TissueX(**config)
        if self.config['t2_relaxation']:
            self._R2 = R2TissueX(**config)
        if self.config['t2s_relaxation']:
            self._R2s = R2sTissueX(**config)
        
        self.map_inputs(imap)

    def inputs(self) -> set:
        inputs = set()
        if self.config['t1_relaxation']:
            inputs |= self._R1.mapped_inputs()
        if self.config['t2_relaxation']:
            inputs |= self._R2.mapped_inputs()
        if self.config['t2s_relaxation']:
            inputs |= self._R2s.mapped_inputs()
        return inputs
    
    def outputs(self) -> set:
        outputs = set()
        if self.config['t1_relaxation']:
            outputs |= self._R1.outputs()
        if self.config['t2_relaxation']:
            outputs |= self._R2.outputs()
        if self.config['t2s_relaxation']:
            outputs |= self._R2s.outputs()
        return outputs
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        R = {}
        if self.config['t1_relaxation']:
            R |= self._R1(p)
        if self.config['t2_relaxation']:
            R |= self._R2(p)
        if self.config['t2s_relaxation']:
            R |= self._R2s(p)

        return R



class ContrastConcTissueX(Module):
    # Convert tissue concentration in blood and interstitium to concentration.
    # For uptake models this introduces a new parameter

    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'}
    }
    defaults = {
        'kinetics': '2CX',
    }    
    def inputs(self) -> set:
        inputs = {'C'}
        if self.config['kinetics'] == 'FX':
            inputs |= {'H', 've'}
        elif self.config['kinetics'] in {'U', 'NX', 'NXP'}:
            inputs |= {'vb'}
        elif self.config['kinetics'] == 'WV':
            inputs |= {'vi'}
        elif self.config['kinetics'] in {'HFU', '2CU', 'HF', '2CX'}:
            inputs |= {'vb', 'vi'}
        return inputs
    
    def outputs(self) -> set:
        return {'c'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        kinetics = self.config['kinetics']

        if p['C'].ndim==1:
            C = np.array(p['C']).reshape(1, -1)
        else:
            C = np.array(p['C'])

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
        
        return {'c': c}


class WaterConcTissueX(Module):
    # Convert tissue concentration in kinetic compartments to concentration in water compartments.

    configs = {
        'kinetics': R1TissueX.configs['kinetics'],
        'water_exchange': R1TissueX.configs['water_exchange'],
    }
    defaults = {
        'kinetics': R1TissueX.defaults['kinetics'],
        'water_exchange': R1TissueX.defaults['water_exchange'],
    }
        
    def inputs(self) -> set:
        kinetics = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        inputs = {'C'}

        if kinetics == 'FX':
            if wex[0] != 'F':
                inputs |= {'ve', 'vb', 'H'}

        if wex == 'FF':
            pass

        elif wex == 'RF':
            if kinetics == 'WV':
                pass
            elif kinetics in ['U', 'NX', 'NXP', 'FX', 'HFU', 'HF', '2CU', '2CX']:
                inputs |= {'vb'}

        elif wex == 'FR':
            if kinetics == 'WV':
                inputs |= {'vi'}
            elif kinetics in ['U']:
                inputs |= {'vc'}
            elif kinetics in ['FX']:
                inputs |= {'vb', 'H', 've'}
            elif kinetics in ['NX', 'NXP', 'HFU', 'HF', '2CU', '2CX']:
                inputs |= {'vb', 'vi'}

        elif wex == 'RR':
            if kinetics == 'WV':
                inputs |= {'vi'}
            elif kinetics in ['NX', 'NXP', 'U']:
                inputs |= {'vb'}
            elif kinetics in ['FX']:
                inputs |= {'vb', 'H', 've'}
            elif kinetics in ['HF', 'HFU', '2CU', '2CX']:
                inputs |= {'vb', 'vi'}

        return inputs
    

    def outputs(self) -> set:
        return {'c'}


    def __call__(self, data: dict) -> dict: # (n_comp, n_times)
        p = self.map_data(data)

        C = np.array(p['C'])
        if C.ndim==1:
            C = C.reshape(1, -1)

        kinetics = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

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
            c = mix_1_comp(C)

        elif wex == 'RF':
            if kinetics == 'WV': # i
                c = mix_1_comp(C)
            elif kinetics in ['U', 'NX', 'NXP']: # b
                c = mix_1_comp(C, p['vb'])
            elif kinetics in ['FX', 'HFU', 'HF', '2CU', '2CX']: #bi
                c = mix_2_comp(C, p['vb'])

        elif wex == 'FR':
            if kinetics == 'WV':
                c = mix_1_comp(C, p['vi']) #i
            elif kinetics in ['U']:
                c = mix_1_comp(C, 1 - p['vc'])  #b
            elif kinetics == 'FX':
                p['vp'] = (1 - p['H']) * p['vb']
                p['vi'] = p['ve'] - p['vp']
                c = mix_1_comp(C, p['vb'] + p['vi'])
            elif kinetics in ['NX', 'NXP', 'HFU', 'HF', '2CU', '2CX']:
                c = mix_1_comp(C, p['vb'] + p['vi']) # 1-vc

        elif wex == 'RR':
            if kinetics == 'WV':
                c = mix_1_comp(C, p['vi'])
            elif kinetics in ['NX', 'NXP', 'U']:
                c = mix_3_comp(C, [p['vb']])
            elif kinetics == 'FX':
                p['vp'] = (1 - p['H']) * p['vb']
                p['vi'] = p['ve'] - p['vp']
                c = mix_3_comp(C, [p['vb'], p['vi']])
            elif kinetics in ['HF', 'HFU', '2CU', '2CX']:
                c = mix_3_comp(C, [p['vb'], p['vi']])

        return {'c': c}