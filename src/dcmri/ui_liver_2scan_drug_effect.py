from copy import deepcopy
import json

import matplotlib.pyplot as plt
import numpy as np

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.pk as pk


PARAMS = {
    # Assay parameters
    'dt': {'init': 0.5, 'name': 'Forward model time step', 'unit': 'sec'},
    'C-tmax': {'init': 4 * 60 * 60, 'name': 'Control visit - maximum acquisition time', 'unit': 'sec'},
    'D-tmax': {'init': 4 * 60 * 60, 'name': 'Drug visit - maximum acquisition time', 'unit': 'sec'},
    'D-tmax_1': {'init': 2 * 60 * 60, 'name': 'Drug visit - first scan acquisition time', 'unit': 'sec'},
    'C-t_scan2': {'init': 2 * 60 * 60, 'name': 'Control visit - start of second scan', 'unit': 'sec'},
    'D-t_scan2': {'init': 2 * 60 * 60, 'name': 'Drug visit - start of second scan', 'unit': 'sec'},
    'dose_tolerance': {'init': 0.1, 'name': 'Dose tolerance', 'unit': ''},
    'field_strength': {'init': 3.0, 'name': 'Magnetic field strength', 'unit': 'T'},
    'agent': {'init': 'gadoxetate', 'name': 'Contrast agent', 'unit': None},
    'rate': {'init': 1, 'name': 'Contrast agent injection rate', 'unit': 'mL/sec'},
    'C-dose': {'init': 0.05, 'name': 'Control visit - first contrast agent dose', 'unit': 'mL/kg'},
    'D-dose': {'init': 0.05, 'name': 'Drug visit - first contrast agent dose', 'unit': 'mL/kg'},
    'C-dose2': {'init': 0.05, 'name': 'Control - second contrast agent dose', 'unit': 'mL/kg'},
    'D-dose2': {'init': 0.05, 'name': 'Drug - second contrast agent dose', 'unit': 'mL/kg'},
    'FA': {'init': 15.0, 'name': 'Flip angle', 'unit': 'deg'},
    'FA2': {'init': 15.0, 'name': 'Second flip angle', 'unit': 'deg'},
    'TR': {'init': 0.005, 'name': 'Repetition time', 'unit': 'sec'},
    'TS': {'init': None, 'name': 'Sampling time', 'unit': 'sec'},
    'C-BAT': {'init': 120, 'bounds': [-60, 60], 'name': 'Control visit - first bolus arrival time', 'unit': 'sec'},
    'C-BAT2': {'init': 7200 + 900, 'bounds': [-60, 60], 'name': 'Control visit - second bolus arrival time', 'unit': 'sec'},
    'D-BAT': {'init': 120, 'bounds': [-60, 60], 'name': 'Drug visit - first bolus arrival time', 'unit': 'sec'},
    'D-BAT2': {'init': 7200 + 900, 'bounds': [-60, 60], 'name': 'Drug visit - second bolus arrival time', 'unit': 'sec'},

    # Subject parameters
    'weight': {'init': 70.0, 'name': 'Subject weight', 'unit': 'kg'},
    'H': {'init': 0.45, 'name': 'Hematocrit', 'unit': ''},

    # MRI signal parameters - control visit
    'C-R10(a)': {'init': 1/lib.T1(3.0, 'blood'), 'name': 'Control visit - aorta first baseline R1', 'unit': 'Hz'},
    'C-R10(l)': {'init': 1/lib.T1(3.0, 'liver'), 'name': 'Control visit - liver first baseline R1', 'unit': 'Hz'},
    'C-S0(a)': {'init': 1, 'name': 'Control visit - aorta first signal scale factor', 'unit': 'a.u.'},
    'C-S0(l)': {'init': 1, 'name': 'Control visit - liver first signal scale factor', 'unit': 'a.u.'},
    'C-S02(a)': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta second signal scale factor', 'unit': 'a.u.'},
    'C-S02(l)': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver second signal scale factor', 'unit': 'a.u.'},

    # MRI signal parameters - drug visit
    'D-R10(a)': {'init': 1/lib.T1(3.0, 'blood'), 'name': 'Drug visit - aorta first baseline R1', 'unit': 'Hz'},
    'D-R10(l)': {'init': 1/lib.T1(3.0, 'liver'), 'name': 'Drug visit - liver first baseline R1', 'unit': 'Hz'},
    'D-S0(a)': {'init': 1, 'name': 'Drug visit - aorta first signal scale factor', 'unit': 'a.u.'},
    'D-S0(l)': {'init': 1, 'name': 'Drug visit - liver first signal scale factor', 'unit': 'a.u.'},
    'D-S02(a)': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta second signal scale factor', 'unit': 'a.u.'},
    'D-S02(l)': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver second signal scale factor', 'unit': 'a.u.'},

    # Kinetics - common
    'CO': {'init': 100, 'bounds': [0, 300], 'name': 'Cardiac output', 'unit': 'mL/sec'},
    'GFR': {'init': 2, 'name': 'Glomerular filtration rate', 'unit': 'mL/sec'},
    'T(hl)': {'init': 10, 'bounds': [0, 30], 'name': 'Heart-lung mean transit time', 'unit': 'sec'},
    'D(hl)': {'init': 0.2, 'bounds': [0.05, 0.95], 'name': 'Heart-lung dispersion', 'unit': ''},
    'T(o)': {'init': 20, 'bounds': [0, 60], 'name': 'Organs blood mean transit time', 'unit': 'sec'},
    'E(o)': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Organs extraction fraction', 'unit': ''},
    'T(o,e)': {'init': 120, 'bounds': [0, 800], 'name': 'Organs extravascular mean transit time', 'unit': 'sec'},
    'T(g)': {'init': 30, 'bounds': [0.1, 60], 'name': 'Gut mean transit time', 'unit': 'sec'},
    'D(g)': {'init': 0.85, 'bounds': [0, 1], 'name': 'Gut dispersion', 'unit': ''},
    'v(e)': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Liver extracellular volume fraction', 'unit': 'mL/cm3'},
    'v(h)': {'name': 'Hepatocellular volume fraction', 'unit': 'mL/cm3'},
    'RE-k(he)': {'name': 'Relative effect in hepatocellular uptake rate', 'unit': ''},
    'RE-k(bh)': {'name': 'Relative effect in biliary excretion rate', 'unit': ''},
    'RE-CL': {'name': 'Relative effect in liver plasma clearance', 'unit': ''},
    'AE-k(he)': {'name': 'Absolute effect in hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'AE-k(bh)': {'name': 'Absolute effect in biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'AE-CL': {'name': 'Absolute effect in liver plasma clearance', 'unit': 'mL/sec'},

    # Kinetics - control visit
    'C-vol': {'init': 1000, 'name': 'Control visit - liver volume', 'unit': 'cm3'},
    'C-k(he,i)': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'C-k(he,f)': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'C-k(he)': {'name': 'Control visit - Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'C-dk(he)': {'name': 'Control visit - change in hepatocellular uptake rate', 'unit': ''},
    'C-k(bh)': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Control visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'C-CL': {'name': 'Control visit - liver plasma clearance', 'unit': 'mL/sec'},

    # Kinetics - drug visit
    'D-vol': {'init': 1000, 'name': 'Drug visit - liver volume', 'unit': 'cm3'},
    'D-k(he,i)': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'D-k(he,f)': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'D-k(he)': {'name': 'Drug visit - Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'D-dk(he)': {'name': 'Drug visit - Change in hepatocellular uptake rate', 'unit': ''},
    'D-k(bh,i)': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'D-k(bh,f)': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - final biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'D-k(bh)': {'name': 'Drug visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'D-dk(bh)': {'name': 'Drug visit - change in biliary excretion rate', 'unit': ''},
    'D-CL': {'name': 'Drug visit - liver plasma clearance', 'unit': 'mL/sec'},
}


class Liver2scanDrugEffect():
    """Joint model for aorta and liver signals measured over two scans.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta and liver, measured over two separate scans.

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
        kinetics (str, optional): Tracer-kinetic liver model. See table 
          :ref:`table-liver-models` for options - only single-inlet models 
          are allowed. Defaults to '1I-IC-HFD'.
        stationary (str, optional): For intracellular tracers - stationarity 
          regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
          For more detail see :ref:`liver-tissues`. Defaults to 'UE'.
        stationary (str, optional): Stationarity regime of the hepatocytes. 
          The options are 'UE', 'E', 'U' or None. For more detail 
          see :ref:`liver-tissues`. Defaults to 'UE'.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. See tables :ref:`AortaLiver2scan-parameters` and
          :ref:`AortaLiver2scan-defaults` for a list of parameters and their
          default values.

    See Also:
        `AortaLiver`

    Example:

        Use the model to reconstruct concentrations from experimentally 
        derived signals.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data from 
        experimentally-derived concentrations:

        >>> time, aif, roi, gt = dc.fake_tissue2scan(R10=1/dc.T1(3.0,'liver'))

        Since this model generates four time curves, the x- and y-data are 
        tuples:

        >>> time = (time[0], time[1], time[0], time[1])
        >>> signal = (aif[0], aif[1], roi[0], roi[1])

        Build an aorta-liver model and parameters to match the conditions of 
        the fake tissue data:

        >>> model = dc.AortaLiver2scan(
        ...     dt = 0.5,
        ...     tmax = 420,
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     dose2 = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 15,
        ...     FA2 = 15,
        ...     TS = 0.5,
        ...     Th_i = 120,
        ...     Th_f = 120,
        ... )

        In this case we have defined different initial values for Th as 
        the defaults are optimized for the slow passage through hepatocytes. 
        We also need to reset the parameter bounds:

        >>> model.free['Th_i'] = [0, np.inf]
        >>> model.free['Th_f'] = [0, np.inf]

        Train the model on the data:

        >>> model.train(time, signal, n0=10, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare against 
        the experimentally derived data:

        >>> model.plot(time, signal)

        We can also have a look at the model parameters after training:

        >>> model.print_params(round_to=3)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        Aorta second signal scale factor (S02a): 195.824 (2.025) a.u.
        Liver second signal scale factor (S02l): 297.854 (4.9) a.u.
        Second bolus arrival time (BAT2): 254.512 (0.137) sec
        First bolus arrival time (BAT): 14.288 (0.132) sec
        Cardiac output (CO): 203.199 (5.406) mL/sec
        Heart-lung mean transit time (Thl): 15.236 (0.263) sec
        Heart-lung dispersion (Dhl): 0.381 (0.009)
        Organs blood mean transit time (To): 23.761 (3.052) sec
        Organs extraction fraction (Eo): 0.287 (0.053)
        Organs extravascular mean transit time (Toe): 50.274 (17.44) sec
        Body extraction fraction (Eb): 0.078 (0.015)
        Apparent liver extracellular volume fraction (ve_app): 0.053 (0.008) mL/cm3
        Extracellular mean transit time (Te): 1.298 (0.552) sec
        Extracellular dispersion (De): 1.0 (0.7)
        Initial hepatic plasma clearance (Ktrans_i): 0.005 (0.001) mL/sec/cm3
        Final hepatic plasma clearance (Ktrans_f): 0.005 (0.001) mL/sec/cm3
        Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
        Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        Aorta first baseline R1 (R10a): 0.614 Hz
        Aorta first signal scale factor (S0a): 100.117 a.u.
        Liver first baseline R1 (R10l): 1.33 Hz
        Liver first signal scale factor (S0l): 150.003 a.u.
        Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
        Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec


    Notes:

        Table :ref:`AortaLiver-parameters` lists the parameters that are 
        relevant in each regime. Table :ref:`AortaLiver-defaults` list all 
        possible parameters and their default settings. 

        .. _AortaLiver2scan-parameters:
        .. list-table:: **Aorta-Liver 2-scan parameters**
            :widths: 20 30 30
            :header-rows: 1

            * - Parameters
              - When to use
              - Further detail
            * - dt, tmax
              - Always
              - Time axis for forward model
            * - dose_tolerance
              - Always
              - Stopping criterion for whole-body model
            * - field_strength, weight, agent, dose, dose2, rate
              - Always
              - Injection protocol
            * - R10a, R102a, R10l, R102l, S0a, S02a, S0l, S02l
              - Always
              - Precontrast R1 (:ref:`relaxation-params`) and 
                S0 (:ref:`params-per-sequence`) for aorta and liver 
            * - FA, TR, TS, FA2
              - Always
              - :ref:`params-per-sequence`
            * - TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - BAT, BAT2, CO, Thl, Dhl, To, Eo, Tie, Eb
              - Always
              - :ref:`whole-body-tissues`
            * - H, ve, De
              - Always
              - :ref:`table-liver-models`
            * - khe, khe_i, kh_f, Th, Th_i, Th_f
              - Depends on **stationary**
              - :ref:`table-liver-models`

        .. _AortaLiver2scan-defaults:
        .. list-table:: **Aorta-Liver 2-scan parameter defaults**
            :widths: 5 10 10 10 10
            :header-rows: 1

            * - Parameter
              - Type
              - Value
              - Bounds
              - Free/Fixed
            * - 
              - **Simulation**
              -
              - 
              - 
            * - dt
              - Simulation
              - 0.5
              - [0, inf]
              - Fixed
            * - tmax
              - Simulation
              - 120
              - [0, inf]
              - Fixed
            * - dose_tolerance
              - Simulation
              - 0.1
              - [0, 1]
              - Fixed
            * - 
              - **Injection**
              -
              - 
              - 
            * - field_strength
              - Injection
              - 3
              - [0, inf]
              - Fixed
            * - weight
              - Injection
              - 70
              - [0, inf]
              - Fixed
            * - agent
              - Injection
              - 'gadoxetate'
              - None
              - Fixed
            * - dose
              - Injection
              - 0.0125
              - [0, inf]
              - Fixed
            * - rate
              - Injection
              - 1
              - [0, inf]
              - Fixed
            * - 
              - **Signal**
              -
              - 
              - 
            * - R10a
              - Signal
              - 0.7
              - [0, inf]
              - Fixed
            * - R10l
              - Signal
              - 0.7
              - [0, inf]
              - Fixed
            * - S0a
              - Signal
              - 1
              - [0, inf]
              - Free
            * - S0l
              - Signal
              - 1
              - [0, inf]
              - Free
            * - 
              - **Sequence**
              -
              - 
              - 
            * - FA
              - Sequence
              - 15
              - [0, inf]
              - Fixed
            * - FA2
              - Sequence
              - 15
              - [0, inf]
              - Fixed
            * - S0
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - TC
              - Sequence
              - 0.1
              - [0, inf]
              - Fixed
            * - TR
              - Sequence
              - 0.005
              - [0, inf]
              - Fixed
            * - TS
              - Sequence
              - 0
              - [0, inf]
              - Fixed
            * - 
              - **Whole body**
              -
              - 
              - 
            * - BAT
              - Whole body
              - 1200
              - [0, inf]
              - Free
            * - CO
              - Whole body
              - 100
              - [0, inf]
              - Free
            * - Thl
              - Whole body
              - 10
              - [0, 30]
              - Free
            * - Dhl
              - Whole body
              - 0.2
              - [0.05, 0.95]
              - Free
            * - To
              - Whole body
              - 20
              - [0, 60]
              - Free
            * - Eo
              - Whole body
              - 0.15
              - [0, 0.5]
              - Free
            * - Toe
              - Whole body
              - 120
              - [0, 800]
              - Free
            * - Eb
              - Whole body
              - 0.05
              - [0.01, 0.15]
              - Free
            * - 
              - **Liver**
              -
              - 
              - 
            * - H
              - Kinetic
              - 0.45
              - [0, 1]
              - Fixed
            * - Te
              - Kinetic
              - 30
              - [0.1, 60]
              - Free
            * - De
              - Kinetic
              - 0.85
              - [0, 1]
              - Free
            * - ve
              - Kinetic
              - 0.3
              - [0.01, 0.6]
              - Free
            * - khe
              - Kinetic
              - 0.003
              - [0, 0.1]
              - Free
            * - khe_i
              - Kinetic
              - 0.003
              - [0, 0.1]
              - Free
            * - khe_f
              - Kinetic
              - 0.003
              - [0, 0.1]
              - Free
            * - Th
              - Kinetic
              - 1800
              - [600, 36000]
              - Free
            * - Th_i
              - Kinetic
              - 1800
              - [600, 36000]
              - Free
            * - Th_f
              - Kinetic
              - 1800
              - [600, 36000]
              - Free
            * - vol
              - Kinetic
              - 1000
              - [0, 10000]
              - Free
    """

    def __init__(self, **params):
        self._version = '1.0'

        # Initialize parameters
        self._pars = {p: deepcopy(PARAMS[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            
    def _pars_list(self, select=None):
        if select is None:
            pars_list = [p for p in PARAMS.keys() if 'init' in PARAMS[p]]
        elif select == 'free':
            visit_pars = ['BAT', 'BAT2', 'S02(a)', 'S02(l)', 'k(he,i)', 'k(he,f)']
            pars_list = [f'C-{p}' for p in visit_pars] + [f'D-{p}' for p in visit_pars]
            pars_list += ['CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(o)', 'T(o,e)']  # aorta
            pars_list += ['T(g)', 'D(g)', 'v(e)', 'C-k(bh)', 'D-k(bh,i)', 'D-k(bh,f)']  # liver
        elif select=='export':
            pars_list = self._pars_list('free') + ['C-S0(a)', 'C-S0(l)', 'D-S0(a)', 'D-S0(l)']
        elif select=='first_scan':
            pars_list = ['FA', 'TR']
        elif select=='second_scan':
            pars_list = ['FA2', 'TR']
        # elif select=='aorta_control_fit':
        #     pars_list = ['C-S02(a)', 'C-BAT', 'C-BAT2', 'C-k(he,i)', 'C-k(he,f)']
        #     pars_list += ['CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(o)', 'T(o,e)']  # aorta
        # elif select=='liver_control_fit':
        #     pars_list = ['C-S02(l)', 'C-k(he,i)', 'C-k(he,f)']
        #     pars_list += ['T(g)', 'D(g)', 'v(e)', 'C-k(bh)']  # liver
        # elif select=='control_fit':
        #     pars_list = ['C-S02(a)', 'C-BAT', 'C-BAT2', 'C-k(he,i)', 'C-k(he,f)']
        #     pars_list += ['CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(o)', 'T(o,e)']  # aorta
        #     pars_list += ['C-S02(l)']
        #     pars_list += ['T(g)', 'D(g)', 'v(e)', 'C-k(bh)']  # liver
        # elif select=='aorta_drug_fit':
        #     pars_list = ['D-BAT', 'D-BAT2', 'D-S02(a)', 'D-k(he,i)', 'D-k(he,f)']
        # elif select=='liver_drug_fit':
        #     pars_list = ['D-S02(l)', 'D-k(he,i)', 'D-k(he,f)']
        #     pars_list += ['D-k(bh,i)', 'D-k(bh,f)']  # liver
        # elif select=='drug_fit':
        #     pars_list = ['D-BAT', 'D-BAT2', 'D-S02(a)', 'D-S02(l)', 'D-k(he,i)', 'D-k(he,f)']
        #     pars_list += ['D-k(bh,i)', 'D-k(bh,f)']  # liver
        # elif select=='aorta_fit':
        #     visit_pars = ['BAT', 'BAT2', 'S02(a)', 'k(he,i)', 'k(he,f)']
        #     pars_list = [f'C-{p}' for p in visit_pars] + [f'D-{p}' for p in visit_pars]
        #     pars_list += ['CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(o)', 'T(o,e)']  # aorta
        # elif select=='liver_fit':
        #     visit_pars = ['S02(l)', 'k(he,i)', 'k(he,f)']
        #     pars_list = [f'C-{p}' for p in visit_pars] + [f'D-{p}' for p in visit_pars]
        #     pars_list += ['T(g)', 'D(g)', 'v(e)', 'C-k(bh)', 'D-k(bh,i)', 'D-k(bh,f)']  # liver
        return pars_list
    

    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _set_time(self):
        self._t_control = _time(self._pars, 'C')
        self._t_drug = _time(self._pars, 'D')

    def _compute_conc_aorta_control(self):
        self._t_control, self._ca_control = _conc_aorta(self._pars, 'C')

    def _compute_relax_aorta_control(self):
        self._compute_conc_aorta_control()
        self._R1a_control = _relax_aorta(self._ca_control, self._pars, 'C')

    def _compute_signal_aorta_control(self):
        self._compute_relax_aorta_control()
        self._Sa_control = _signal(self._t_control, self._R1a_control, 'aorta', self._pars, 'C')

    def _predict_aorta_control(self, time):
        self._compute_signal_aorta_control()
        return _sample_signal(time, self._t_control, self._Sa_control, self._pars['TS'])

    # Predict aorta signal in drug scan

    def _compute_conc_aorta_drug(self):
        self._t_drug, self._ca_drug = _conc_aorta(self._pars, 'D')

    def _compute_relax_aorta_drug(self):
        self._compute_conc_aorta_drug()
        self._R1a_drug = _relax_aorta(self._ca_drug, self._pars, 'D')

    def _compute_signal_aorta_drug(self):
        self._compute_relax_aorta_drug()
        self._Sa_drug = _signal(self._t_drug, self._R1a_drug, 'aorta', self._pars, 'D')

    def _predict_aorta_drug(self, time):
        self._compute_signal_aorta_drug()
        return _sample_signal(time, self._t_drug, self._Sa_drug, self._pars['TS'])

    # Predict liver signal in control scan

    def _compute_conc_liver_control(self):
        self._Cl_control = _conc_liver(self._ca_control, self._pars, 'C')

    def _compute_relax_liver_control(self):
        self._compute_conc_liver_control()
        self._R1l_control = _relax_liver(self._Cl_control, self._pars, 'C')

    def _compute_signal_liver_control(self):
        self._compute_relax_liver_control()
        self._Sl_control = _signal(self._t_control, self._R1l_control, 'liver', self._pars, 'C')

    def _predict_liver_control(self, time):
        self._compute_signal_liver_control()
        return _sample_signal(time, self._t_control, self._Sl_control, self._pars['TS'])
    
    # Predict liver signal in drug scan

    def _compute_conc_liver_drug(self):
        self._Cl_drug = _conc_liver(self._ca_drug, self._pars, 'D')

    def _compute_relax_liver_drug(self):
        self._compute_conc_liver_drug()
        self._R1l_drug = _relax_liver(self._Cl_drug, self._pars, 'D')

    def _compute_signal_liver_drug(self):
        self._compute_relax_liver_drug()
        self._Sl_drug = _signal(self._t_drug, self._R1l_drug, 'liver', self._pars, 'D')

    def _predict_liver_drug(self, time):
        self._compute_signal_liver_drug()
        return _sample_signal(time, self._t_drug, self._Sl_drug, self._pars['TS'])

    # Predict all signals

    def _predict_control(self, time):
        Sa = self._predict_aorta_control(time[:2])
        Sl = self._predict_liver_control(time[2:])
        return Sa + Sl
    
    def _predict_drug(self, time):
        Sa = self._predict_aorta_drug(time[:2])
        Sl = self._predict_liver_drug(time[2:])
        return Sa + Sl

    # APIs

    def time(self) -> tuple:
        """Time points in aorta and liver for the two visits"""
        self._set_time()
        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['C-t_scan2'], self._pars['D-t_scan2']
        return (
            tc[tc < t2c], tc[tc >= t2c], tc[tc < t2c], tc[tc >= t2c], 
            td[td < t2d], td[td >= t2d], td[tc < t2d], td[tc >= t2d],
        )

    def conc(self) -> tuple:
        """Concentrations in aorta and liver.

        Returns:
            dict: aorta blood concentrations, liver concentrations.
        """
        self._compute_conc_aorta_control()
        self._compute_conc_liver_control()
        self._compute_conc_aorta_drug()
        self._compute_conc_liver_drug()

        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['C-t_scan2'], self._pars['D-t_scan2']
        cac = self._ca_control[tc < t2c], self._ca_control[tc >= t2c]
        cad = self._ca_drug[td < t2d], self._ca_drug[td >= t2d]
        Clc = self._Cl_control[:, tc < t2c], self._Cl_control[:, tc >= t2c]
        Cld = self._Cl_drug[:, td < t2d], self._Cl_drug[:, td >= t2d]            
        return cac + Clc + cad + Cld
    
    def relax(self) -> tuple:
        """Relaxation rates in aorta and liver.

        Returns:
            dict: aorta blood R1, liver R1.
        """
        self._compute_relax_aorta_control()
        self._compute_relax_liver_control()
        self._compute_relax_aorta_drug()
        self._compute_relax_liver_drug()

        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['C-t_scan2'], self._pars['D-t_scan2']
        R1ac = self._R1a_control[tc < t2c], self._R1a_control[tc >= t2c]
        R1lc = self._R1l_control[tc < t2c], self._R1l_control[tc >= t2c]
        R1ad = self._R1a_drug[td < t2d], self._R1a_drug[td >= t2d]
        R1ld = self._R1l_drug[td < t2d], self._R1l_drug[td >= t2d]
        return R1ac + R1lc + R1ad + R1ld
    
    def signal(self) -> tuple:
        """Signal in aorta and liver.

        Returns:
            tuple: aorta blood signal, liver 
              signal.
        """
        self._compute_signal_aorta_control()
        self._compute_signal_liver_control()
        self._compute_signal_aorta_drug()
        self._compute_signal_liver_drug()

        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['C-t_scan2'], self._pars['D-t_scan2']
        Sac = self._Sa_control[tc < t2c], self._Sa_control[tc >= t2c]
        Slc = self._Sl_control[tc < t2c], self._Sl_control[tc >= t2c]
        Sad = self._Sa_drug[td < t2d], self._Sa_drug[td >= t2d]
        Sld = self._Sl_drug[td < t2d], self._Sl_drug[td >= t2d]
        return Sac + Slc + Sad + Sld
    
    def predict(self, time: tuple) -> tuple:
        """Predict the data at given time points

        Args:
            time (tuple): tuple of 8 arrays with time points. The first 
              four are from the control visit: aorta in 
              the first scan, aorta in the second scan, liver in the first 
              scan, and liver in the second scan, in that order. 
              The second group of 4 is the same data for the treatment visit.

        Returns:
            tuple: tuple of 8 arrays with signals corresponding to time.
        """
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        for i, visit in enumerate(['C', 'D']):
            self._pars[f'{visit}-tmax'] = self._pars['dt'] + np.max(time[4 * i: 4 * i + 4]) + ts
        Sc = self._predict_control(time[:4])
        Sd = self._predict_drug(time[4:])
        return Sc + Sd


    def train(
            self, time: tuple, signal: tuple, free=None, 
            bounds:dict=None, R102a=None, R102l=None, n0=[1, 1], 
            **kwargs,
        ):
        """Train the free parameters

        Args:
            time (tuple): tuple of 4 arrays with time points for aorta in 
              the first scan, aorta in the second stand, liver in the first 
              scan, and liver in the second scan, in that order. The four 
              arrays can be different in length and value.
            signal (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            free (dict, optional): Dictionary with free parameters and their
              bounds. If not provided, a default set of free parameters is used.
              Defaults to None.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        # Estimate BAT and S0 from data
        self._estimate_pars(time, signal, n0, R102a, R102l)

        # Check and update free parameters
        free = self._set_free_pars(free, bounds)

        # # Train control data
        # free_ctrl = {p: v for p, v in free.items() if p in self._pars_list('control_fit')}
        # utils.train(self._predict_control, time[:4], signal[:4], self._pars, free_ctrl, **kwargs)
        
        # # Train drug data
        # free_drug = {p: v for p, v in free.items() if p in self._pars_list('drug_fit')}
        # utils.train(self._predict_drug, time[4:], signal[4:], self._pars, free_drug, **kwargs)

        # Train all parameters on all data
        pcov, sdev = utils.train(self.predict, time, signal, self._pars, free, **kwargs)

        pars = {
            p: {
                'name': deepcopy(PARAMS[p]['name']), 
                'unit': deepcopy(PARAMS[p]['unit']), 
                'value': self._pars[p], 
                'sdev': sdev[p] if sdev is not None else None
            } 
            for p in free
        }
        return pars, pcov


    def _estimate_pars(self, time, signal, n0, R102a, R102l):
        
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        for i, visit in enumerate(['C', 'D']):
            self._pars[f'{visit}-tmax'] = self._pars['dt'] + np.max(time[4 * i: 4 * i + 4]) + ts

            # Estimate BAT and BAT2
            T, D = self._pars['T(hl)'], self._pars['D(hl)']
            self._pars[f'{visit}-BAT'] = time[4 * i][np.argmax(signal[4 * i])] - (1-D)*T
            self._pars[f'{visit}-BAT2'] = time[1 + 4 * i][np.argmax(signal[1 + 4 * i])] - (1-D)*T

            # Estimate S0
            pars = self._pars_dict(select='first_scan')
            Srefb = sig.signal('SS', self._pars[f'{visit}-R10(a)'], 1, **pars)
            Srefl = sig.signal('SS', self._pars[f'{visit}-R10(l)'], 1, **pars)
            self._pars[f'{visit}-S0(a)'] = np.mean(signal[0 + 4 * i][:n0[i]]) / Srefb
            self._pars[f'{visit}-S0(l)'] = np.mean(signal[2 + 4 * i][:n0[i]]) / Srefl

            # Estimate S02
            pars = self._pars_dict(select='second_scan')
            pars['FA'] = pars.pop('FA2', None)
            if R102a is None:
                self._pars[f'{visit}-S02(a)'] = self._pars[f'{visit}-S0(a)']
            else:
                Sref2b = sig.signal('SS', R102a[i], 1, **pars)
                self._pars[f'{visit}-S02(a)'] = np.mean(signal[1 + 4 * i][:n0[i]]) / Sref2b
            if R102l is None:
                self._pars[f'{visit}-S02(l)'] = self._pars[f'{visit}-S0(l)']
            else:
                Sref2l = sig.signal('SS', R102l[i], 1, **pars)
                self._pars[f'{visit}-S02(l)'] = np.mean(signal[3 + 4 * i][:n0[i]]) / Sref2l


    def _set_free_pars(self, free: dict=None, bounds: dict=None):
        # --- 0. Set Defaults ---
        if free is None:
            free = {p: deepcopy(PARAMS[p]['bounds']) for p in self._pars_list('free')}
        
        # --- 1. Update Bounds ---
        if bounds is not None:
            for p, b in bounds.items():
                if b is None:
                    free.pop(p, None)
                else:
                    free[p] = b

        # --- 2. Boundary Validation ---
        for p, bnds in free.items():
            if p not in self._pars:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            
            # Parameters with relative bounds
            elif p[2:] in ['BAT', 'BAT2']: 
                if (bnds[0] > 0) or (bnds[1] < 0):
                    raise ValueError(f"Invalid bounds on {p}: Bounds on BAT are relative and must be (negative, positive).")
            elif p[2:] in ['S0(a)', 'S0(l)', 'S02(a)', 'S02(l)']: 
                if not (0 <= bnds[0] < bnds[1]):
                    raise ValueError(f"Invalid bounds on {p}: Bounds on S0 are relative and must be positive.")

            # Absolute bounds
            elif not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(f"Initial {p} ({self._pars[p]}) is out of bounds {bnds}.")

        # --- 3. Relative to Absolute Bounds
        for visit in ['C', 'D']:

            # Additive
            for par in ['BAT', 'BAT2']:
                par_str = f'{visit}-{par}'
                if par_str in free:
                    free[par_str] = [  
                        self._pars[par_str] + free[par_str][0],
                        self._pars[par_str] + free[par_str][1],
                    ]

            # Multiplicative
            for par in ['S0(a)', 'S0(l)', 'S02(a)', 'S02(l)']:
                par_str = f'{visit}-{par}'
                if par_str in free:
                    free[par_str] = [
                        self._pars[par_str] * free[par_str][0],
                        self._pars[par_str] * free[par_str][1],
                    ]   

        return free         
                

    # ==========================================
    # I/O and Reporting
    # ==========================================



    def save(self, file: str):
        """Save the current state of the model as a json file.

        Args:
            file (str): complete path of the json file. 
        """

        if file.split('.')[-1] != 'json':
            file += '.json'

        data = {
            'model': self.__class__.__name__,
            'version': self._version,
            'pars': self._pars,
        }

        with open(file, "w") as f:
            json.dump(data, f, indent=4)

        return self


    def load(self, file):
        """Load the saved state of the model from a json file

        Args:
            file (str): complete path of the json file. 
        """
        with open(file, "r") as f:
            data = json.load(f) 

        if data['model'] != self.__class__.__name__:
            raise ValueError(f"File belongs to {data['model']}, not {self.__class__.__name__}.")
        if data['version'] != self._version:
            raise ValueError(f"Version mismatch: {data['version']} vs {self._version}.")
             
        self._pars = data['pars']
        return self

    def export_params(self) -> dict:
        """Return model parameters with their descriptions

        Returns:
            dict: Dictionary with one item for each model parameter. The key 
            is the short parameter name, and the value is a dict 
            with long parameter name, value, unit.
        """
            
        # List parameters for export
        pars_deriv = _deriv_params(self._pars)
        export_pars = self._pars_list('export') + list(pars_deriv.keys())

        # Add short name, full name, value, units.
        value = self._pars | pars_deriv
        pars = {
            p: {
                'name': deepcopy(PARAMS[p]['name']),
                'unit': deepcopy(PARAMS[p]['unit']),  
                'value': value[p], 
            } for p in export_pars
        }
        return pars 
    
    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is 
              not provided, the values are not rounded. Defaults to None.
        """
        pars = self.export_params()
        for p, v in pars.items():
            val = v['value']
            if round_to is not None:
                val = round(val, round_to)
            print(f"{v['name']} ({p}) = {val} {v['unit']}")

    def _pars_dict(self, *args, select=None):
        """Return the parameter values"""
        if len(args) == 0:
            pars = deepcopy(self._pars)
        else:
            pars = {k: v for k, v in self._pars.items() if k in list(args)}
        if select is not None:
            pars = {k: v for k, v in pars.items() if k in self._pars_list(select)}
        return pars
  
    def params(self, *args, as_dict=False):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            tuple or dict: values of parameters
        """
        pars = self._pars_dict(*args)
        if as_dict:
            return pars
        elif len(pars)==1:
            return list(pars.values())[0]
        else:
            return tuple(pars.values())
        
    def cost(self, time: tuple, signal: tuple, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            time (tuple): tuple of 4 arrays with time points for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The four arrays can 
              be different in length and value.
            signal (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            metric (str, optional): Which metric to use (see notes for 
              possible values). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.

        Notes:

            Available options are: 
            
            - 'RMS': Root-mean-square.
            - 'NRMS': Normalized root-mean-square. 
            - 'AIC': Akaike information criterion. 
            - 'cAIC': Corrected Akaike information criterion for small 
              models.
            - 'BIC': Baysian information criterion.
        """
        y = self.predict(time)
        if isinstance(signal, tuple):
            y = np.concatenate(y)
            signal = np.concatenate(signal)
        return utils.loss(y, signal, metric)


    def plot(self, time: tuple, signal: tuple,
             xlim=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 4 arrays with time points for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The four arrays can 
              be different in length and value.
            signal (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        for i, visit in enumerate(['C', 'D']):
            self._pars[f'{visit}-tmax'] = self._pars['dt'] + np.max(time[4 * i: 4 * i + 4]) + ts
        
        self.signal()
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.3)

        ax1.set_title('First visit')
        ax2.set_title('Second visit')
        ax3.set_title('First visit')
        ax4.set_title('Second visit')

        _plot_data2scan(self._t_control, self._Sa_control, time[0:2], signal[0:2],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'])
        _plot_data2scan(self._t_drug, self._Sa_drug, time[4:6], signal[4:6],
                        ax2, xlim,
                        color=['lightcoral', 'darkred'])
        _plot_conc_aorta(self._t_control, self._ca_control, ax3, xlim)
        _plot_conc_aorta(self._t_drug, self._ca_drug, ax4, xlim)

        _plot_data2scan(self._t_control, self._Sl_control, time[2:4], signal[2:4],
                        ax5, xlim,
                        color=['cornflowerblue', 'darkblue'])
        _plot_data2scan(self._t_drug, self._Sl_drug, time[6:8], signal[6:8],
                        ax6, xlim,
                        color=['cornflowerblue', 'darkblue'])
        _plot_conc_liver(self._t_control, self._Cl_control, ax7, xlim)
        _plot_conc_liver(self._t_drug, self._Cl_drug, ax8, xlim)

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()




    # def to_dmr(self, file=None, subject='Subject', study='drug_effect'):
    #     dmr = {'data': {}, 'pars': {}, 'sdev':{}}
    #     pars = self.export_params()
    #     for p in pars:
    #         dmr['data'][p] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1], str) else 'float']
    #         dmr['pars'][(subject, study, p)] = pars[p][1]
    #         dmr['sdev'][(subject, study, p)] = pars[p][3]
    #     if file is not None:             
    #         pydmr.write(file, dmr)
    #     return dmr



def _sample_signal(time, t, S, TS) -> tuple:
    return (
        utils.sample(time[0], t, S, TS),
        utils.sample(time[1], t, S, TS),
    )

# Helper functions for plotting

def _plot_conc_aorta(t: np.ndarray, cb: np.ndarray, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cb, linestyle='-',
            color='darkred', linewidth=2.0, label='Aorta')
    ax.legend()

def _plot_conc_liver(t, C, ax, xlim=None):
    color = 'darkblue'
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*C[0, :], linestyle='-.',
            color=color, linewidth=2.0, label='Extracellular')
    ax.plot(t/60, 1000*C[1, :], linestyle='--',
            color=color, linewidth=2.0, label='Hepatocytes')
    ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-',
            color=color, linewidth=2.0, label='Tissue')       
    ax.legend()

def _plot_data2scan(t: tuple[np.ndarray, np.ndarray], 
                    sig: tuple[np.ndarray, np.ndarray],
                    time: tuple[np.ndarray, np.ndarray], 
                    signal: tuple[np.ndarray, np.ndarray],
                    ax, xlim, color=['black', 'black']):
    if xlim is None:
        xlim = [0, t[-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', 
           xlim=np.array(xlim)/60)
    ax.plot(np.concatenate(time)/60, np.concatenate(signal),
            marker='o', color=color[0], label='fitted data', linestyle='None')
    ax.plot(t/60, sig,
            linestyle='-', color=color[1], linewidth=3.0, label='fit')

    ax.legend()


# def sdev_effect(v, dv):
#     # z = (x-y)/y = x/y-1 
#     # dz = sqrt((dy * dz/dy)**2 + (dx * dz/dx)**2)
#     # dz/dx = 1/y
#     # dz/dy = -x/y**2
#     x, y = v[0], v[1]
#     dx, dy = dv[0], dv[1]
#     dz_dx = 1 / y if y != 0 else 0
#     dz_dy = - x / y**2 if y != 0 else 0
#     return np.sqrt((dy * dz_dy)**2 + (dx * dz_dx)**2)
    
def _deriv_params(pars):

    # TODO: return sdev of derived
    
    vh = 1 - pars['v(e)'] / (1 - pars['H'])
    C_khe = np.mean([pars['C-k(he,i)'], pars['C-k(he,f)']])
    C_kbh = pars['C-k(bh)']
    D_khe = pars['D-k(he,i)'] 

    t = _time(pars, 'D')
    Th_i = _div(vh, pars['D-k(bh,i)'])
    Th_f = _div(vh, pars['D-k(bh,f)'])
    Th = utils.interp([Th_i, Th_f], t)
    D_kbh = _div(vh, Th.mean())
    
    C_CL = C_khe * pars['C-vol']
    D_CL = D_khe * pars['D-vol']
    pars_deriv = {
        'v(h)': vh,
        'C-k(he)': C_khe,
        'D-k(he)': D_khe,
        'D-k(bh)': D_kbh,
        'C-CL': C_CL,
        'D-CL': D_CL,
        'RE-k(he)': _div(D_khe - C_khe, C_khe),
        'RE-k(bh)': _div(D_kbh - C_kbh, C_kbh),
        'RE-CL': _div(D_CL - C_CL, C_CL),
        'AE-k(he)': D_khe - C_khe,
        'AE-k(bh)': D_kbh - C_kbh,
        'AE-CL': D_CL - C_CL,
        'C-dk(he)': _div(pars['C-k(he,f)'] - pars['C-k(he,i)'], pars['C-k(he,i)']),
        'D-dk(he)': _div(pars['D-k(he,f)'] - pars['D-k(he,i)'], pars['D-k(he,i)']),
        'D-dk(bh)': _div(pars['D-k(bh,f)'] - pars['D-k(bh,i)'], pars['D-k(bh,i)']),
    }
    return pars_deriv

def _time(pars, visit):
    t = np.arange(0, pars[f'{visit}-tmax'], pars['dt'])
    return t

def _conc_aorta(pars, visit):

    t = _time(pars, visit)
    conc = lib.ca_conc(pars['agent'])

    # Derive Eb
    khe = utils.interp([pars[f'{visit}-k(he,i)'], pars[f'{visit}-k(he,f)']], t)
    CL = khe * pars[f'{visit}-vol'] + pars['GFR']
    Eb = CL / (CL + pars['CO'] * (1 - pars['H']))
    
    # Compute flux
    J1 = lib.ca_injection(
        t, pars['weight'], conc, pars[f'{visit}-dose'], 
        pars['rate'], pars[f'{visit}-BAT'],
    )
    J2 = lib.ca_injection(
        t, pars['weight'], conc, pars[f'{visit}-dose2'], 
        pars['rate'], pars[f'{visit}-BAT2'],
    )
    Jb = pk_aorta.flux_aorta(
        J1 + J2, E=Eb, dt=pars['dt'], 
        tol=pars['dose_tolerance'],
        heartlung=['pfcomp', (pars['T(hl)'], pars['D(hl)'])],
        organs=['2cxm', ([pars['T(o)'], pars['T(o,e)']], pars['E(o)'])],
    )
    return t, Jb/pars['CO']


def _conc_liver(cb, pars, visit):

    t = _time(pars, visit)

    cp = cb / (1 - pars['H'])
    cp = pk.flux_pfcomp(cp, pars['T(g)'], pars['D(g)'], dt=pars['dt'])  
    Ce = pars['v(e)'] * cp
    
    khe = utils.interp([pars[f'{visit}-k(he,i)'], pars[f'{visit}-k(he,f)']], t)
    vh = 1 - pars['v(e)'] / (1 - pars['H'])
    if visit == 'C':
        Th = vh / pars[f'{visit}-k(bh)']
        Ch = pk.conc(khe * cp, Th, dt=pars['dt'], model="comp")
    elif visit == 'D':
        Th_i = vh / pars[f'{visit}-k(bh,i)']
        Th_f = vh / pars[f'{visit}-k(bh,f)']
        Th = utils.interp([Th_i, Th_f], t)
        Ch = pk.conc(khe * cp, Th, dt=pars['dt'], model="nscomp")

    return np.stack((Ce, Ch))


def _relax_aorta(ca, pars, visit):
    rb = lib.relaxivity(pars['field_strength'], 'blood', pars['agent'])
    R1a = pars[f'{visit}-R10(a)'] + rb * ca
    return R1a

def _relax_liver(Cl, pars, visit):
    rp = lib.relaxivity(pars['field_strength'], 'plasma', pars['agent'])
    rh = lib.relaxivity(pars['field_strength'], 'hepatocytes', pars['agent'])
    R1l = pars[f'{visit}-R10(l)'] + rp * Cl[0, :] + rh * Cl[1, :]
    return R1l

def _signal(t, R1, roi, pars, visit):
    t1 = t <= pars[f'{visit}-t_scan2']
    t2 = t > pars[f'{visit}-t_scan2']
    S = np.zeros(t.size)
    S[t1] = sig.signal('SS', R1[t1], pars[f'{visit}-S0({roi[0]})'], FA=pars['FA'], TR=pars['TR'])
    S[t2] = sig.signal('SS', R1[t2], pars[f'{visit}-S02({roi[0]})'], FA=pars['FA2'], TR=pars['TR'])

    return S


def _div(a, b):
    with np.errstate(divide='ignore'):
        return np.divide(a, b)
