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
import numpy as np

from dcmri.bloch.modules_tissue import MzPrep, MxyReadMz
from dcmri.core.module import Module
from dcmri.core.sequences import SEQUENCES


class MagnetizationTissueX(Module): 
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
        'sequence': set(SEQUENCES.keys()),
        'inflow': {False, True},
    }
    defaults = {
        'kinetics': '2CX', 
        'water_exchange': 'FF', 
        'sequence': '3D-SPGR-SS',
        'inflow': False,
    }
    def __init__(self, imap:dict=None, **config):
        self.set_config(config)
        self._water_vol = WaterVolumesTissueX(**self.config)
        self._mz_prep = MzPrepTissueX(**self.config)
        self._mxy_read = MxyReadMz(**self.config)
        self.map_inputs(imap)  
        
    def inputs(self):
        inputs = self._mz_prep.mapped_inputs()
        inputs |= self._mxy_read.mapped_inputs()
        return inputs - {'Mz'}
    
    def outputs(self):
        return {'M'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        if 'R1' in p:
            Mz = self._mz_prep(p)['Mz'] # (compartments, times)

        elif 'R2' in p:
            # Derived nc, nt dimensions
            v = self._water_vol(p)['v']
            nc = v.size
            nt = np.size(p['R2']) / nc
            if not nt.is_integer():
                raise ValueError("R2 must have one element for each tissue compartment and time point")
            nt = int(nt)
            Mz = np.full((nc, nt), p['me'])

        else:
            v = self._water_vol(p)['v']
            nc = v.size
            nt = np.size(p['R2s'])          
            Mz = np.full((nc, nt), p['me'])

        Mxy = self._mxy_read(p, Mz=Mz)['Mxy'] # (channels, components, compartments, times) or (components, compartments, times)
        if Mxy.ndim==3:
            M = np.zeros((1 + Mxy.shape[0], Mxy.shape[1], Mxy.shape[2]), dtype=Mxy.dtype)
            M[:2, :, :] = Mxy
            M[2, :, :] = Mz
        else:
            M = np.zeros((Mxy.shape[0], 1 + Mxy.shape[1], Mxy.shape[2], Mxy.shape[3]), dtype=Mxy.dtype)
            M[:, :2, :, :] = Mxy
            M[:, 2, :, :] = Mz
        return {'M': M}


class MzPrepTissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
        'sequence': set(SEQUENCES.keys()),
        'inflow': {False, True},
    }
    defaults = {
        'kinetics': '2CX', 
        'water_exchange': 'FF', 
        'sequence': '3D-SPGR-SS',
        'inflow': False,
    }
    def __init__(self, imap:dict=None, **config):
        self.set_config(config)
        self._water_vol = WaterVolumesTissueX(**self.config)
        self._water_flow = WaterFlowsTissueX(**self.config)
        self._mz_prep = MzPrep(**self.config)
        self.map_inputs(imap)

    def inputs(self) -> set:
        inputs = self._water_vol.mapped_inputs()
        inputs |= self._water_flow.mapped_inputs()
        inputs |= self._mz_prep.mapped_inputs()
        
        inputs -= {'v', 'Fw'}
        if self.config['inflow']:
            if 'R1' in inputs:
                inputs |= {'R1a', 'Fb'}
                inputs -= {'R1i', 'Fi'}

        return inputs
    
    def outputs(self):
        return self._mz_prep.outputs()

    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        # Compartment flows
        v = self._water_vol(p)['v']

        # Inlet flow and relaxation rates
        if self.config['inflow']:
            if 'R1' in p:
                p['R1i'] = np.zeros_like(p['R1'])
                p['R1i'][0,...] = p['R1a']
                p['Fi'] = np.zeros(p['R1'].shape[0])
                p['Fi'][0] = p['Fb']

        # Compartment flows
        Fw = self._water_flow(p)['Fw']

        return self._mz_prep(p, v=v, Fw=Fw)




_VOLUMES = {
    ('2CX', 'FF'): set(),
    ('2CU', 'FF'): set(),
    ('HF', 'FF'): set(),
    ('HFU', 'FF'): set(),
    ('NX', 'FF'): set(),
    ('NXP', 'FF'): set(),
    ('WV', 'FF'): set(),
    ('U', 'FF'): set(),
    ('FX', 'FF'): set(),

    ('2CX', 'FR'): {'vb', 'vi'},
    ('2CU', 'FR'): {'vb', 'vi'},
    ('HF', 'FR'): {'vb', 'vi'},
    ('HFU', 'FR'): {'vb', 'vi'},
    ('NX', 'FR'): {'vb', 'vi'},
    ('NXP', 'FR'): {'vb', 'vi'},
    ('WV', 'FR'): {'vi'},
    ('U', 'FR'): {'vb', 'vi'},
    ('FX', 'FR'): {'vb', 'vi'},

    ('2CX', 'RF'): {'vb'},
    ('2CU', 'RF'): {'vb'},
    ('HF', 'RF'): {'vb'},
    ('HFU', 'RF'): {'vb'},
    ('NX', 'RF'): {'vb'},
    ('NXP', 'RF'): {'vb'},
    ('WV', 'RF'): set(),
    ('U', 'RF'): {'vb'},
    ('FX', 'RF'): {'vb'},

    ('2CX', 'RR'): {'vb', 'vi'},
    ('2CU', 'RR'): {'vb', 'vi'},
    ('HF', 'RR'): {'vb', 'vi'},
    ('HFU', 'RR'): {'vb', 'vi'},
    ('NX', 'RR'): {'vb', 'vi'},
    ('NXP', 'RR'): {'vb', 'vi'},
    ('WV', 'RR'): {'vi'},
    ('U', 'RR'): {'vb', 'vi'},
    ('FX', 'RR'): {'vb', 'vi'},
}



class WaterVolumesTissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
    }
    defaults = {
        'kinetics': '2CX',
        'water_exchange': 'FF',
    }
    def inputs(self):
        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')
        return _VOLUMES[(kin, wex)]
    
    def outputs(self):
        return {'v'}

    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        # Add derived
        if {'vb', 'vi'}.issubset(p):
            p['vc'] = 1 - p['vb'] - p['vi']

        # Map water compartment volumes
        if (kin, wex) == ('2CX', 'FF'): v = np.array([1])
        if (kin, wex) == ('2CU', 'FF'): v = np.array([1])
        if (kin, wex) == ('HF', 'FF'): v = np.array([1])
        if (kin, wex) == ('HFU', 'FF'): v = np.array([1])
        if (kin, wex) == ('NX', 'FF'): v = np.array([1])
        if (kin, wex) == ('NXP', 'FF'): v = np.array([1])
        if (kin, wex) == ('WV', 'FF'): v = np.array([1])
        if (kin, wex) == ('U', 'FF'): v = np.array([1])
        if (kin, wex) == ('FX', 'FF'): v = np.array([1])

        if (kin, wex) == ('2CX', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('2CU', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('HF', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('HFU', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('NX', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('NXP', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('WV', 'FR'): v = np.array([p['vi'], 1-p['vi']])
        if (kin, wex) == ('U', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('FX', 'FR'): v = np.array([1-p['vc'], p['vc']])

        if (kin, wex) == ('2CX', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('2CU', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('HF', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('HFU', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('NX', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('NXP', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('WV', 'RF'): v = np.array([1])
        if (kin, wex) == ('U', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('FX', 'RF'): v = np.array([p['vb'], 1-p['vb']])

        if (kin, wex) == ('2CX', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('2CU', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('HF', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('HFU', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('NX', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('NXP', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('WV', 'RR'): v = np.array([p['vi'], 1-p['vi']])
        if (kin, wex) == ('U', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('FX', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])

        return {'v': v}


_FLOWS = {
    ('2CX', 'FF'): {'Fb'},
    ('2CU', 'FF'): {'Fb'},
    ('HF', 'FF'): set(),
    ('HFU', 'FF'): set(),
    ('NX', 'FF'): {'Fb'},
    ('NXP', 'FF'): {'Fb'},
    ('WV', 'FF'): set(),
    ('U', 'FF'): {'Fb'},
    ('FX', 'FF'): {'Fb'},

    ('2CX', 'FR'): {'Fb', 'PSc'},
    ('2CU', 'FR'): {'Fb', 'PSc'},
    ('HF', 'FR'): {'PSc'},
    ('HFU', 'FR'): {'PSc'},
    ('NX', 'FR'): {'Fb', 'PSc'},
    ('NXP', 'FR'): {'Fb', 'PSc'},
    ('WV', 'FR'): {'PSc'},
    ('U', 'FR'): {'Fb', 'PSc'},
    ('FX', 'FR'): {'Fb', 'PSc'},

    ('2CX', 'RF'): {'Fb', 'PSe'},
    ('2CU', 'RF'): {'Fb', 'PSe'},
    ('HF', 'RF'): {'PSe'},
    ('HFU', 'RF'): {'PSe'},
    ('NX', 'RF'): {'Fb', 'PSe'},
    ('NXP', 'RF'): {'Fb', 'PSe'},
    ('WV', 'RF'): set(),
    ('U', 'RF'): {'Fb', 'PSe'},
    ('FX', 'RF'): {'Fb', 'PSe'},

    ('2CX', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('2CU', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('HF', 'RR'): {'PSe', 'PSc'},
    ('HFU', 'RR'): {'PSe', 'PSc'},
    ('NX', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('NXP', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('WV', 'RR'): {'PSc'},
    ('U', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('FX', 'RR'): {'Fb', 'PSe', 'PSc'},
}

class WaterFlowsTissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
    }
    defaults = {
        'kinetics': '2CX', 
        'water_exchange': 'FF',
    }
    def inputs(self) -> dict:
        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')
        inputs = _FLOWS[(kin, wex)]

        if wex[0] == 'N' and 'PSe' in inputs :
            inputs - {'PSe'}
        if wex[1] == 'N' and 'PSc' in inputs:
            inputs - {'PSc'}
        return inputs
    
    def outputs(self) -> dict:
        return {'Fw'}

    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        if wex[0] == 'N':
            p['PSe'] = 0
        if wex[1] == 'N':
            p['PSc'] = 0

        # Map water compartment flows
        if (kin, wex) == ('2CX', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('2CU', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('HF', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('HFU', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('NX', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('NXP', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('WV', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('U', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('FX', 'FF'): Fw = np.full((1, 1), p['Fb'])

        if (kin, wex) == ('2CX', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('2CU', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('HF', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('HFU', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('NX', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('NXP', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('WV', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('U', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('FX', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])

        if (kin, wex) == ('2CX', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('2CU', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('HF', 'RF'): Fw = np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('HFU', 'RF'): Fw = np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('NX', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('NXP', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('WV', 'RF'): Fw = np.array([0])
        if (kin, wex) == ('U', 'RF'): Fw = np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('FX', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])

        if (kin, wex) == ('2CX', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('2CU', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('HF', 'RR'): Fw = np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('HFU', 'RR'): Fw = np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('NX', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('NXP', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('WV', 'RR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('U', 'RR'): Fw = np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('FX', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])

        return {'Fw': Fw}

# class MxyReadMzTissueX(Module):
#     configs = {
#         'kinetics': ['2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'],
#         'water_exchange': ['FF','RF','NF','FR','RR','NR','FN','RN','NN'],
#         'sequence': list(SEQUENCES.keys()),
#         'inflow': [False, True],
#     }

#     def __init__(self, 
#         kinetics='2CX', 
#         water_exchange='FF', 
#         sequence='3D-SPGR-SS', 
#         inflow=False,
#         defaults=None,
#         **params,
#     ):
#         cnfg = {
#             'kinetics': kinetics, 
#             'water_exchange': water_exchange, 
#             'sequence': sequence, 
#             'inflow': inflow,
#         }
#         self._set_config(cnfg)
#         self._set_params(defaults)

#     def params(self):
#         pars = MzPrepTissueX(**self._cnfg).params()
#         pars += MxyReadMz(**self._cnfg).params()
#         return list(set(pars))

#     def __call__(self, **params):
#         p = self._update_params(params)

#         if 'R1' in p:
#             Mz_arr = MzPrepTissueX(**self._cnfg, defaults=p)()
#         elif 'R2' in p:
#             Mz_arr = np.full(np.shape(p['R2']), p['me'], dtype=float)
#         elif 'R2s' in p:
#             Mz_arr = np.full(np.shape(p['R2s']), p['me'], dtype=float).reshape(1, -1)
        
#         # Signal
#         return MxyReadMz(**self._cnfg)(Mz_arr, **p)