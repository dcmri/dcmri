import os
import warnings
from copy import deepcopy
import json

from scipy.optimize import curve_fit, minimize

import matplotlib.pyplot as plt
import numpy as np
import pydmr

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.pk as pk


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

        >>> xdata = (time[0], time[1], time[0], time[1])
        >>> ydata = (aif[0], aif[1], roi[0], roi[1])

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

        >>> model.train(xdata, ydata, n0=10, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare against 
        the experimentally derived data:

        >>> model.plot(xdata, ydata)

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
        self._pars = None
        self._free = None
        self._pcov = None
        self._sdev = None
        self._init_params(**params)

    def _compute_conc_aorta(self):
        self._t_control, self._ca_control = _conc_aorta(self._pars, visit=0)
        self._t_drug, self._ca_drug = _conc_aorta(self._pars, visit=1)

    def _compute_relax_aorta(self):
        self._compute_conc_aorta()
        self._R1a_control = _relax_aorta(self._ca_control, self._pars, visit=0)
        self._R1a_drug = _relax_aorta(self._ca_drug, self._pars, visit=1)

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        self._Sa_control = _signal(self._R1a_control, 'aorta', self._pars, visit=0)
        self._Sa_drug = _signal(self._R1a_drug, 'aorta', self._pars, visit=1)

    def _predict_aorta(self, xdata: tuple) -> tuple:
        self._compute_signal_aorta()
        return _sample_signal(self._Sa_control, self._Sa_drug, self._pars, xdata)

    def _compute_conc_liver(self):
        self._Cl_control = _conc_liver(self._ca_control, self._pars, visit=0)
        self._Cl_drug = _conc_liver(self._ca_drug, self._pars, visit=1)

    def _compute_relax_liver(self):
        self._compute_conc_liver()
        self._R1l_control = _relax_liver(self._Cl_control, self._pars, visit=0)
        self._R1l_drug = _relax_liver(self._Cl_drug, self._pars, visit=1)

    def _compute_signal_liver(self):
        self._compute_relax_liver()
        self._Sl_control = _signal(self._R1l_control, 'liver', self._pars, visit=0)
        self._Sl_drug = _signal(self._R1l_drug, 'liver', self._pars, visit=1)

    def _predict_liver(self, xdata: tuple) -> tuple:
        self._compute_signal_liver()
        return _sample_signal(self._Sl_control, self._Sl_drug, self._pars, xdata)

    
    def conc(self):
        """Concentrations in aorta and liver.

        Returns:
            dict: time points, aorta blood concentrations, liver 
              concentrations.
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        return self._t_control, self._ca_control, self._Cl_control, self._t_drug, self._ca_drug, self._Cl_drug
    
    def relax(self):
        """Relaxation rates in aorta and liver.

        Returns:
            dict: time points, aorta blood R1, liver 
              R1.
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        return self._t_control, self._R1a_control, self._R1l_control, self._t_drug, self._R1a_drug, self._R1l_drug
    
    def signal(self):
        """Signal in aorta and liver.

        Returns:
            dict: time points, aorta blood signal, liver 
              signal.
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        return self._t_control, self._Sa_control, self._Sl_control, self._t_drug, self._Sa_drug, self._Sl_drug
    
    def predict(self, xdata: tuple) -> tuple:
        """Predict the data at given time points

        Args:
            xdata (tuple): tuple of 8 arrays with time points. The first 
              four are from the control visit: aorta in 
              the first scan, aorta in the second stand, liver in the first 
              scan, and liver in the second scan, in that order. 
              The second group of 4 is the same data for the treatment visit.

        Returns:
            tuple: tuple of 8 arrays with signals corresponding to xdata.
        """
        Sa = self._predict_aorta((xdata[0], xdata[1], xdata[4], xdata[5]))
        Sl = self._predict_liver((xdata[2], xdata[3], xdata[6], xdata[7]))
        return Sa[0], Sa[1], Sl[0], Sl[1], Sa[2], Sa[3], Sl[2], Sl[3]


    def _init_params(self, **params):

        # List all parameters (TODO: get from PARAMS instead of list again)
        P = ['dt', 'tmax', 'dose_tolerance', 'field_strength', 'weight', 'agent', 'dose', 'rate']
        P += ['H', 'R10a', 'R10l', 'S0a', 'S0l', 'S02a', 'S02l', 'BAT', 'BAT2']
        P += ['FA', 'TR', 'TS', 't_scan2', 'dose2', 'FA2']
        P += ['CO', 'GFR', 'Thl', 'Dhl', 'To', 'Eo', 'Toe']
        #P += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Kbh_i', 'Kbh_f', 'vol']
        P += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'vol']

        # Initialize parameters
        self._pars = {p: PARAMS[p]['init'] for p in P}
        for p in params:
            if p not in P:
                raise ValueError(
                    f"{p} is not a valid model parameter in this configuration."
                )                
            self._pars[p] = params[p] 

        # Initialize free parameters (get from params?)
        P = ['CO', 'Thl', 'Dhl', 'To', 'Eo', 'Toe'] 
        P += ['BAT', 'BAT2', 'S02a', 'S02l']
        # P += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Kbh_i', 'Kbh_f'] 
        P += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f'] 
        
        self._free = {p: PARAMS[p]['bounds'] for p in P}


    def export_params(self) -> dict:
        """Return model parameters with their descriptions

        Args:
            type (str, optional): Type of output. If 'dict', a dictionary is 
              returned. If 'list', a list is returned. Defaults to 'dict'.

        Returns:
            dict: Dictionary with one item for each model parameter. The key 
            is the short parameter name, and the value is a 
            4-element list with [long parameter name, value, unit, sdev].

        """
        pars_deriv = _deriv_params(self._pars)

        # List parameters for export
        export_params = ['BAT', 'BAT2', 'S0a', 'S02a', 'S0l', 'S02l']
        export_params += ['CO', 'GFR', 'Thl', 'Dhl', 'To', 'Eo', 'Toe']
        # export_params += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Kbh_i', 'Kbh_f', 'vol']
        export_params += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'vol']
        export_params += list(pars_deriv.keys())

        # Add short name, full name, value, units.
        all_pars = self._pars | pars_deriv
        pars = {}
        for p in export_params:
            if np.isscalar(all_pars[p]):
                sdev = 0
            else:
                sdev = [0, 0]
                pars[f'{p}_effect'] = [
                    PARAMS[p]['name'], 
                    100 * (all_pars[p][1] - all_pars[p][0]) / all_pars[p][0], 
                    '%', 
                    0,
                ]
            if self._sdev is not None:
                if p in self._sdev:
                    sdev = self._sdev[p]
            pars[p] = [
                PARAMS[p]['name'], 
                all_pars[p], 
                PARAMS[p]['unit'], 
                sdev,
            ]

        # # Inhibition is computed from the lowest rate in the treatment visit
        # # and the average in the baseline visit.
        # pars['khe_inhibition'] = [
        #     "Inhibition of hepatocellular uptake", 
        #     100 * (min([all_pars['khe_i'][1], all_pars['khe_f'][1]]) - all_pars['khe'][0]) / all_pars['khe'][0], 
        #     '%', 
        #     0,
        # ]  
        # pars['kbh_inhibition'] = [
        #     "Inhibition of biliary excretion", 
        #     100 * (min([all_pars['kbh_i'][1], all_pars['kbh_f'][1]]) - all_pars['kbh'][0]) / all_pars['kbh'][0], 
        #     '%', 
        #     0,
        # ]      

        return pars


    def train(
            self, xdata: tuple, ydata: tuple, n0=[1, 1], 
            R102a=None, R102l=None, free=None, constraints=None, **kwargs,
        ):
        # x,y: (aorta scan 1, aorta scan 2, liver scan 1, liver scan 2)
        """Train the free parameters

        Args:
            xdata (tuple): tuple of 4 arrays with time points for aorta in 
              the first scan, aorta in the second stand, liver in the first 
              scan, and liver in the second scan, in that order. The four 
              arrays can be different in length and value.
            ydata (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            joint (bool, optional): If True, aorta and liver parameters 
              are trained jointly after training them separately. 
              Defaults to True.
            free (dict, optional): Dictionary with free parameters and their
              bounds. If not provided, a default set of free parameters is used.
              Defaults to None.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        if free is not None:
            for p in free:
                if p not in self._pars:
                    raise ValueError(
                        f"{p} is not a valid free parameter in this configuration."
                    ) 
            self._free = deepcopy(free)
        
        for visit in [0, 1]:

            # Estimate BAT and BAT2
            T, D = self._pars['Thl'], self._pars['Dhl']
            self._pars['BAT'][visit] = xdata[4 * visit][np.argmax(ydata[4 * visit])] - (1-D)*T
            self._free['BAT'][visit] = [  
                self._pars['BAT'][visit] + free['BAT'][visit][0],
                self._pars['BAT'][visit] + free['BAT'][visit][1],
            ]
            self._pars['BAT2'][visit] = xdata[1 + 4 * visit][np.argmax(ydata[1 + 4 * visit])] - (1-D)*T
            self._free['BAT2'][visit] = [
                self._pars['BAT2'][visit] + free['BAT2'][visit][0],
                self._pars['BAT2'][visit] + free['BAT2'][visit][1],
            ]

            # Estimate S0
            pars = {'FA':self._pars['FA'], 'TR':self._pars['TR']}
            Srefb = sig.signal('SS', self._pars['R10a'][visit], 1, **pars)
            Srefl = sig.signal('SS', self._pars['R10l'][visit], 1, **pars)
            self._pars['S0a'][visit] = np.mean(ydata[0 + 4 * visit][1:n0[visit]]) / Srefb
            self._pars['S0l'][visit] = np.mean(ydata[2 + 4 * visit][1:n0[visit]]) / Srefl

            # Estimate S02
            pars = {'FA':self._pars['FA2'], 'TR':self._pars['TR']}
            if R102a is not None:
                Sref2b = sig.signal('SS', R102a[visit], 1, **pars)
                self._pars['S02a'][visit] = np.mean(ydata[1 + 4 * visit][1:n0[visit]]) / Sref2b
            else:
                self._pars['S02a'][visit] = self._pars['S0a'][visit]
            if 'S02a' in free:
                self._free['S02a'][visit] = [
                    self._pars['S02a'][visit] * free['S02a'][visit][0],
                    self._pars['S02a'][visit] * free['S02a'][visit][1],
                ]
            if R102l is not None:
                Sref2l = sig.signal('SS', R102l[visit], 1, **pars)
                self._pars['S02l'][visit] = np.mean(ydata[3 + 4 * visit][1:n0[visit]]) / Sref2l
            else:
                self._pars['S02l'][visit] = self._pars['S0l'][visit]
            if 'S02l' in free:
                self._free['S02l'][visit] = [
                    self._pars['S02l'][visit] * free['S02l'][visit][0],
                    self._pars['S02l'][visit] * free['S02l'][visit][1],
                ]
 
        # # Train free aorta parameters on aorta data
        # #khe_i, khe_f = deepcopy(self._pars['khe_i']), deepcopy(self._pars['khe_f'])
        # pars = ['BAT', 'CO', 'GFR', 'Thl', 'Dhl', 'To', 'Eo', 'Toe', 'khe_i', 'khe_f', 'vol', 'BAT2', 'S02a']
        # free_aorta = {p:v for p, v in self._free.items() if p in pars}
        # x_aorta = (xdata[0], xdata[1], xdata[4], xdata[5])
        # y_aorta = (ydata[0], ydata[1], ydata[4], ydata[5])
        # _train(self, self._predict_aorta, x_aorta, y_aorta, free_aorta, **kwargs)

        # # Train free liver parameters on liver data
        # #self._pars['khe_i'], self._pars['khe_f'] = khe_i, khe_f
        # pars = ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'S02l']
        # #pars = ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Kbh_i', 'Kbh_f', 'S02l']
        # free_liver = {p:v for p, v in self._free.items() if p in pars}
        # x_liver = (xdata[2], xdata[3], xdata[6], xdata[7])
        # y_liver = (ydata[2], ydata[3], ydata[6], ydata[7])
        # _train(self, self._predict_liver, x_liver, y_liver, free_liver, **kwargs)

        # Train all parameters on all data
        pcov = _train(self, self.predict, xdata, ydata, self._free, **kwargs)
        self._pcov = [(pcov.tolist(), list(self._free.keys()))]
        self._sdev = _sdev(pcov, self._free)

        return self
    

    def plot(self, xdata: tuple, ydata: tuple,
             ref=None, xlim=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (tuple): tuple of 4 arrays with time points for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The four arrays can 
              be different in length and value.
            ydata (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form 
              (x,y), where x is an array with x-values and y is an array with 
              y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        tmax = max([max(x) for x in xdata[:4]])
        if self._pars['tmax'][0] < tmax:
            raise ValueError(
                f"The largest control visit time point that can be predicted with "
                f"the current configuration is {self._pars['tmax'][0]}. "
                f"To predict these data, set tmax = {tmax} (or larger) " 
                f"when creating the tissue object."
            )
        tmax = max([max(x) for x in xdata[4:]])
        if self._pars['tmax'][1] < tmax:
            raise ValueError(
                f"The largest treatment visit time point that can be predicted with "
                f"the current configuration is {self._pars['tmax'][1]}. "
                f"To predict these data, set tmax = {tmax} (or larger) " 
                f"when creating the tissue object."
            )
        
        self.signal()
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.3)

        ax1.set_title('First visit')
        ax2.set_title('Second visit')
        ax3.set_title('First visit')
        ax4.set_title('Second visit')

        _plot_data2scan(self._t_control, self._Sa_control, xdata[0:2], ydata[0:2],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[0])
        _plot_data2scan(self._t_drug, self._Sa_drug, xdata[4:6], ydata[4:6],
                        ax2, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[1])
        _plot_conc_aorta(self._t_control, self._ca_control, ax3, xlim)
        _plot_conc_aorta(self._t_drug, self._ca_drug, ax4, xlim)

        _plot_data2scan(self._t_control, self._Sl_control, xdata[2:4], ydata[2:4],
                        ax5, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[2])
        _plot_data2scan(self._t_drug, self._Sl_drug, xdata[6:8], ydata[6:8],
                        ax6, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[3])
        _plot_conc_liver(self._t_control, self._Cl_control, ax7, xlim)
        _plot_conc_liver(self._t_drug, self._Cl_drug, ax8, xlim)

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def to_dmr(self, file=None, subject='Subject', study='drug_effect'):
        dmr = {'data': {}, 'pars': {}, 'sdev':{}}
        pars = self.export_params()
        for p in pars:
            if np.isscalar(pars[p][1]):
                dmr['data'][p] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1], str) else 'float']
                dmr['pars'][(subject, study, p)] = pars[p][1]
                dmr['sdev'][(subject, study, p)] = pars[p][3]
            else:
                dmr['data'][f'{p}_control'] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1][0], str) else 'float']
                dmr['pars'][(subject, study, p)] = pars[p][1][0]
                dmr['sdev'][(subject, study, p)] = pars[p][3][0]   
                dmr['data'][f'{p}_drug'] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1][1], str) else 'float']
                dmr['pars'][(subject, study, p)] = pars[p][1][1]
                dmr['sdev'][(subject, study, p)] = pars[p][3][1]  
        if file is not None:             
            pydmr.write(file, dmr)
        return dmr
    
    def cost(self, xdata: tuple, ydata: tuple, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            xdata (tuple): tuple of 4 arrays with time points for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The four arrays can 
              be different in length and value.
            ydata (tuple): tuple of 4 arrays with signals for aorta in the 
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
        y = self.predict(xdata)
        if isinstance(ydata, tuple):
            y = np.concatenate(y)
            ydata = np.concatenate(ydata)
        return utils.loss(y, ydata, metric)
    
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
            'free': self._free,
            'pcov': self._pcov,
            'sdev': self._sdev,
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
            raise ValueError(
                f"File {file} saves the state of model {data['model']}. "
            )
        if data['version'] != self._version:
            raise ValueError(
                f"The file {file} saves the state of a different version {data['version']}. "
            )              

        self._pars = data['pars']
        self._free = data['free']
        self._pcov = data['pcov']
        self._sdev = data['sdev']

        return self

    
    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is 
              not provided, the values are not rounded. Defaults to None.
        """
        pars = self.export_params()
        for p, v in pars.items():
            name, unit = v[0], v[2]
            if round_to is None:
                val = v[1]
                err = v[3]
            else:
                if np.isscalar(v[1]):
                    val = round(v[1], round_to)
                    err = round(v[3], round_to)
                    print(f"{name} ({p}): {val} ({err}) {unit}")
                else:
                    val = [round(v[1][i], round_to) for i in [0,1]]
                    err = [round(v[3][i], round_to) for i in [0,1]] 
                    print(f"{name} ({p}): [{val[0]}, {val[1]}] ({err[0]}, {err[1]}) {unit}")                  
            


    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            list or float: values of parameter values, or a scalar value if 
            only one parameter is required.
        """
        pars = self.export_params()
        if len(args) == 1:
            if round_to is None:
                return pars[args[0]][1]
            elif np.isscalar(pars[args[0]][1]):
                return round(pars[args[0]][1], round_to)
            else:
                return [round(pars[args[0]][1][i], round_to) for i in [0,1]]
        else:
            return {p: self.params(p) for p in list(args)}


    def set_params(self, **pars):
        """set parameter values

        Args:
            pars (dict): parameters to set.
        """
        for p, v in pars.items():
            self._pars[p] = v


def _deriv_params(pars):
    # Compute derived parameters
    vh = 1 - pars['ve'] / (1 - pars['H'])
    khe = [np.mean([pars['khe_i'][i], pars['khe_f'][i]]) for i in [0,1]]
    Th = [np.mean([pars['Th_i'][i], pars['Th_f'][i]]) for i in [0,1]]
    pars_deriv = {
        'khe': khe,
        'Th': Th,
        'vh': vh,
        'kbh_i': [_div(vh, pars['Th_i'][i]) for i in [0,1]],
        'kbh_f': [_div(vh, pars['Th_f'][i]) for i in [0,1]],
        'kbh': [_div(vh, Th[i]) for i in [0,1]],
        'Khe': [_div(khe[i], pars['ve']) for i in [0,1]],
        'Kbh': [_div(1, Th[i]) for i in [0,1]], # Needs integration over t
        'Kbh_i': [_div(1, pars['Th_i'][i]) for i in [0,1]],
        'Kbh_f': [_div(1, pars['Th_f'][i]) for i in [0,1]],
        'CL_i': [pars['khe_i'][i] * pars['vol'][i] for i in [0,1]],
        'CL_f': [pars['khe_f'][i] * pars['vol'][i] for i in [0,1]],
    }
    pars_deriv['CL'] = [np.mean([pars_deriv['CL_i'][i], pars_deriv['CL_f'][i]]) for i in [0,1]]
    pars_deriv['Eb_i'] = [_div(pars_deriv['CL_i'][i], pars_deriv['CL_i'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
    pars_deriv['Eb_f'] = [_div(pars_deriv['CL_f'][i], pars_deriv['CL_f'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
    # Needs integration
    pars_deriv['Eb'] = [_div(pars_deriv['CL'][i], pars_deriv['CL'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
    return pars_deriv


# def _deriv_params(pars):
#     # Compute derived parameters
#     vh = 1 - pars['ve'] / (1 - pars['H'])
#     khe = [np.mean([pars['khe_i'][i], pars['khe_f'][i]]) for i in [0,1]]
#     Kbh = [np.mean([pars['Kbh_i'][i], pars['Kbh_f'][i]]) for i in [0,1]]
#     pars_deriv = {
#         'khe': khe,
#         'Kbh': Kbh,
#         'vh': vh,
#         'kbh_i': [vh * pars['Kbh_i'][i] for i in [0,1]],
#         'kbh_f': [vh * pars['Kbh_f'][i] for i in [0,1]],
#         'kbh': [vh * Kbh[i] for i in [0,1]],
#         'Khe': [_div(khe[i], pars['ve']) for i in [0,1]],
#         'Th': [_div(1, Kbh[i]) for i in [0,1]], # Needs integration over t
#         'Th_i': [_div(1, pars['Kbh_i'][i]) for i in [0,1]],
#         'Th_f': [_div(1, pars['Kbh_f'][i]) for i in [0,1]],
#         'CL_i': [pars['khe_i'][i] * pars['vol'][i] for i in [0,1]],
#         'CL_f': [pars['khe_f'][i] * pars['vol'][i] for i in [0,1]],
#     }
#     pars_deriv['CL'] = [np.mean([pars_deriv['CL_i'][i], pars_deriv['CL_f'][i]]) for i in [0,1]]
#     pars_deriv['Eb_i'] = [_div(pars_deriv['CL_i'][i], pars_deriv['CL_i'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
#     pars_deriv['Eb_f'] = [_div(pars_deriv['CL_f'][i], pars_deriv['CL_f'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
#     # Needs integration
#     pars_deriv['Eb'] = [_div(pars_deriv['CL'][i], pars_deriv['CL'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
#     return pars_deriv


def _interp(ki, kf, t):
    #slope = (kf - ki) / (6 * 60 * 60)
    slope = (kf - ki) / t.max()
    return ki + t * slope


def _conc_aorta(pars, visit=None):

    t = np.arange(0, pars['tmax'][visit], pars['dt'])
    conc = lib.ca_conc(pars['agent'])

    # Derive Eb
    khe = _interp(pars['khe_i'][visit], pars['khe_f'][visit], t)
    CL = khe * pars['vol'][visit] + pars['GFR']
    Eb = CL / (CL + pars['CO'] * (1 - pars['H']))
    
    # Compute flux
    J1 = lib.ca_injection(
        t, pars['weight'], conc, pars['dose'][visit], 
        pars['rate'], pars['BAT'][visit],
    )
    J2 = lib.ca_injection(
        t, pars['weight'], conc, pars['dose2'][visit], 
        pars['rate'], pars['BAT2'][visit],
    )
    Jb = pk_aorta.flux_aorta(
        J1 + J2, E=Eb, dt=pars['dt'], 
        tol=pars['dose_tolerance'],
        heartlung=['pfcomp', (pars['Thl'], pars['Dhl'])],
        organs=['2cxm', ([pars['To'], pars['Toe']], pars['Eo'])],
    )
    return t, Jb/pars['CO']


def _conc_liver(cb, pars, visit=None):

    t = np.arange(0, pars['tmax'][visit], pars['dt']) 
    
    # Determine khe, Kbh over the visit duration
    khe = _interp(pars['khe_i'][visit], pars['khe_f'][visit], t)
    #Kbh = _interp(pars['Kbh_i'][visit], pars['Kbh_f'][visit], t)
   # Th = _div(1, Kbh)
    Th = _interp(pars['Th_i'][visit], pars['Th_f'][visit], t)
    
    # Compute concentrations
    cp = cb / (1 - pars['H'])
    cp = pk.flux_pfcomp(cp, pars['Tg'], pars['Dg'], dt=pars['dt'])  
    Ce = pars['ve'] * cp
    Ch = pk.conc(khe * cp, Th, dt=pars['dt'], model="nscomp")

    # Return results
    return np.stack((Ce, Ch))




def _relax_aorta(ca, pars, visit=None):
    rb = lib.relaxivity(pars['field_strength'], 'blood', pars['agent'])
    R1a = pars['R10a'][visit] + rb * ca
    return R1a

def _relax_liver(Cl, pars, visit=None):
    rp = lib.relaxivity(pars['field_strength'], 'plasma', pars['agent'])
    rh = lib.relaxivity(pars['field_strength'], 'hepatocytes', pars['agent'])
    R1l = pars['R10l'][visit] + rp * Cl[0, :] + rh * Cl[1, :]
    return R1l

def _signal(R1, roi, pars, visit=None):
    
    t = np.arange(0, pars['tmax'][visit], pars['dt'])
    t1 = t <= pars['t_scan2'][visit]
    t2 = t > pars['t_scan2'][visit]

    S = np.zeros(t.size)
    S[t1] = sig.signal('SS', R1[t1], pars[f'S0{roi[0]}'][visit], FA=pars['FA'], TR=pars['TR'])
    S[t2] = sig.signal('SS', R1[t2], pars[f'S02{roi[0]}'][visit], FA=pars['FA2'], TR=pars['TR'])

    return S


def _sample_signal(S_control, S_drug, pars, xdata) -> tuple:

    tmax = max([max(x) for x in xdata[:2]])
    if pars['tmax'][0] < tmax:
        raise ValueError(
            f"The largest control visit time point that can be predicted with "
            f"the current configuration is {pars['tmax'][0]}. "
            f"To predict these data, set tmax = {tmax} (or larger) " 
            f"when creating the tissue object."
        )
    tmax = max([max(x) for x in xdata[2:]])
    if pars['tmax'][1] < tmax:
        raise ValueError(
            f"The largest treatment visit time point that can be predicted with "
            f"the current configuration is {pars['tmax'][1]}. "
            f"To predict these data, set tmax = {tmax} (or larger) " 
            f"when creating the tissue object."
        )
    
    t_control = np.arange(0, pars['tmax'][0], pars['dt'])
    t_drug = np.arange(0, pars['tmax'][1], pars['dt'])
    return (
        utils.sample(xdata[0], t_control, S_control, pars['TS']),
        utils.sample(xdata[1], t_control, S_control, pars['TS']),
        utils.sample(xdata[2], t_drug, S_drug, pars['TS']),
        utils.sample(xdata[3], t_drug, S_drug, pars['TS']),
    )


    

# add constraints=None here
# then call train_with_constraints if not None

# def _fit_func(x, *pars):
#     _set_pars(x['self'], pars, x['free'])
#     yp = x['predict'](x['xdata'])
#     if isinstance(yp, tuple):
#         return np.concatenate(yp)
#     else:
#         return yp
        

def _train(self:Liver2scanDrugEffect, predict, xdata, ydata, free, **kwargs): 

    if isinstance(ydata, tuple):
        y = np.concatenate(ydata)
    else:
        y = ydata

    p0 = []
    for p in free:
        if np.isscalar(self._pars[p]):
            p0.append(_normalize(self._pars[p], free[p]))
        else:
            p0.append(_normalize(self._pars[p][0], free[p][0]))
            p0.append(_normalize(self._pars[p][1], free[p][1]))
 
    def fit_func(_, *pars):
        _set_pars(self, pars, free)
        yp = predict(xdata)
        if isinstance(yp, tuple):
            return np.concatenate(yp)
        else:
            return yp

    try:
        x = {'self':self, 'free':free, 'predict':predict, 'xdata': xdata}
        pars, pcov = curve_fit(
            #_fit_func, x, y, p0, bounds=[0, 1], **kwargs,
            fit_func, y, y, p0, bounds=[0, 1], **kwargs,
        )
    except RuntimeError as e:
        msg = 'Runtime error in curve_fit -- \n'
        msg += str(e) + ' Returning initial values.'
        warnings.warn(msg)
        pars, pcov = p0, np.zeros((len(p0), len(p0)))

    _set_pars(self, pars, free)

    return pcov




def _set_pars(self, pars, free):
    # Should be the same as:
    # _set_pdict(self._pars, pars, free)
    i = 0
    for p in free:
        if np.isscalar(self._pars[p]):
            self._pars[p] = _renormalize(pars[i], free[p])
            i += 1
        else:
            self._pars[p][0] = _renormalize(pars[i], free[p][0])
            self._pars[p][1] = _renormalize(pars[i+1], free[p][1])
            i += 2


def _set_pdict(pdict, pars, free):
    i = 0
    for p in free:
        if np.isscalar(pdict[p]):
            pdict[p] = _renormalize(pars[i], free[p])
            i += 1
        else:
            pdict[p][0] = _renormalize(pars[i], free[p][0])
            pdict[p][1] = _renormalize(pars[i+1], free[p][1])
            i += 2

def _normalize(v, bounds):
    return (v - bounds[0]) / (bounds[1] - bounds[0])

def _renormalize(v, bounds):
    return v * (bounds[1] - bounds[0]) + bounds[0]


def _sdev(pcov, free):
    
    sdev = {}

    i = 0
    for p in free.keys():
        if np.isscalar(free[p][0]):
            sdev[p] = _renormalize(np.sqrt(pcov[i,i]), free[p])
            i += 1
        else:
            sdev[p] = [
                _renormalize(np.sqrt(pcov[i,i]), free[p][0]),
                _renormalize(np.sqrt(pcov[i+1,i+1]), free[p][1]),
            ]
            i += 2

    return sdev




def _train_with_constr(self:Liver2scanDrugEffect, predict, xdata, ydata, free, constr, **kwargs):

    if isinstance(ydata, tuple):
        y = np.concatenate(ydata)
    else:
        y = ydata

    p0 = []
    for p in free:
        if np.isscalar(self._pars[p]):
            p0.append(_normalize(self._pars[p], free[p]))
        else:
            p0.append(_normalize(self._pars[p][0], free[p][0]))
            p0.append(_normalize(self._pars[p][1], free[p][1]))
            

    def obj_func(_, *pars):
        _set_pars(self, pars, free)
        yp = predict(xdata)
        if isinstance(yp, tuple):
            yp = np.concatenate(yp)
        return np.sum((y - yp)**2)
    
    constr = [{'type': c['type'], 'fun': _vector_function(c['fun'], self, free)} for c in constr]

    try:
        bounds = [(0, 1)] * len(p0)
        pars, pcov = minimize(
            obj_func, p0, bounds=bounds, constraints=constr, **kwargs,
        )
    except RuntimeError as e:
        msg = 'Runtime error in curve_fit -- \n'
        msg += str(e) + ' Returning initial values.'
        warnings.warn(msg)
        pars, pcov = p0, np.zeros((len(p0), len(p0)))

    _set_pars(self, pars, free)

    return pcov


def _vector_function(func, self, free):

    def vector_func(pars):
        pdict = deepcopy(self._pars)
        _set_pdict(pdict, pars, free)
        return func(pdict)
    
    return vector_func



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
                    xdata: tuple[np.ndarray, np.ndarray], 
                    ydata: tuple[np.ndarray, np.ndarray],
                    ax, xlim, color=['black', 'black'], test=None):
    if xlim is None:
        xlim = [0, t[-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', 
           xlim=np.array(xlim)/60)
    ax.plot(np.concatenate(xdata)/60, np.concatenate(ydata),
            marker='o', color=color[0], label='fitted data', linestyle='None')
    ax.plot(t/60, sig,
            linestyle='-', color=color[1], linewidth=3.0, label='fit')
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black',
                marker='D', linestyle='None', label='Test data')
    ax.legend()





PARAMS = {

    # Prediction and training

    'dt': {
        'init': 0.5,
        'bounds': [0, np.inf],
        'name': 'Forward model time step',
        'unit': 'sec',
    },
    'tmax': {
        'init': [180, 180],
        'bounds': 2 * [[0, np.inf]],
        'name': 'Maximum acquisition time',
        'unit': 'sec',
    },
    'dose_tolerance': {
        'init': 0.1,
        'bounds': [0, np.inf],
        'name': 'Dose tolerance',
        'unit': '',
    },

    # Injection

    'field_strength': {
        'init': 3.0,
        'bounds': [0, np.inf],
        'name': 'Magnetic field strength',
        'unit': 'T',
    },
    'weight': {
        'init': 70.0,
        'bounds': [0, np.inf],
        'name': 'Subject weight',
        'unit': 'kg',
    },
    'agent': {
        'init': 'gadoxetate',
        'name': 'Contrast agent',
        'unit': None,
    },
    'dose': {
        'init': 2 * [lib.ca_std_dose('gadoxetate')/2],
        'bounds': 2 * [[0, np.inf]],
        'name': 'First contrast agent dose',
        'unit': 'mL/kg',
    },
    'rate': {
        'init': 1,
        'bounds': [0, np.inf],
        'name': 'Contrast agent injection rate',
        'unit': 'mL/sec',
    },
    'H': {
        'init': 0.45,
        'bounds': [0, 1],
        'name': 'Hematocrit',
        'unit': '',
    },

    # Sequence

    'TR': {
        'init': 0.005,
        'bounds': [0, np.inf],
        'name': 'Repetition time',
        'unit': 'sec',
    },
    'FA': {
        'init': 15.0,
        'bounds': [0, 180],
        'name': 'Flip angle',
        'unit': 'deg',
    },
    'TC': {
        'init': 0.180,
        'bounds': [0, np.inf],
        'name': 'Time to center',
        'unit': 'sec',
    },
    'TS': {
        'init': None,
        'bounds': [0, np.inf],
        'name': 'Sampling time',
        'unit': 'sec',
    },

    # Signal 1

    'R10a': {
        'init': 2 * [1/lib.T1(3.0, 'blood')],
        'bounds': 2 * [[0, np.inf]],
        'name': 'Aorta first baseline R1',
        'unit': 'Hz',
    },
    'S0a': {
        'init': 2 * [1],
        'bounds': 2 * [[0, np.inf]],
        'name': 'Aorta first signal scale factor',
        'unit': 'a.u.',
    },
    'R10l': {
        'init': 2 * [1/lib.T1(3.0, 'liver')],
        'bounds': 2 * [[0, np.inf]],
        'name': 'Liver first baseline R1',
        'unit': 'Hz',
    },
    'S0l': {
        'init': 2 * [1],
        'bounds': 2 * [[0, np.inf]],
        'name': 'Liver first signal scale factor',
        'unit': 'a.u.',
    },

    # Signal 2

    't_scan2': {
        'init': 90,
        'name': 'Start of second scan',
        'unit': 'sec',
    },
    'dose2': {
        'init': 2 * [lib.ca_std_dose('gadoxetate')/2],
        'bounds': 2 * [[0, np.inf]],
        'name': 'Second contrast agent dose',
        'unit': 'mL/kg',
    },
    'FA2': {
        'init': 15.0,
        'bounds': [0, np.inf],
        'name': 'Second flip angle',
        'unit': 'deg',
    },
    
    'S02a': {
        'init': 2 * [1],
        'bounds': 2 * [[0, 2]],
        'name': 'Aorta second signal scale factor',
        'unit': 'a.u.',
    },
    'S02l': {
        'init': 2 * [1],
        'bounds': 2 * [[0, 2]],
        'name': 'Liver second signal scale factor',
        'unit': 'a.u.',
    },
    'BAT2': {
        'init': 2 * [1200],
        'bounds': 2 * [[-60, 60]],
        'name': 'Second bolus arrival time',
        'unit': 'sec',
    },

    # Body

    'BAT': {
        'init': 2 * [60],
        'bounds': 2 * [[-60, 60]],
        'name': 'First bolus arrival time',
        'unit': 'sec',
    },
    'CO': {
        'init': 100,
        'bounds': [0, 300],
        'name': 'Cardiac output',
        'unit': 'mL/sec',
    },
    'GFR': {
        'init': 2,
        'bounds': [0, 4],
        'name': 'Glomerular filtration rate',
        'unit': 'mL/sec',
    },
    'Thl': {
        'init': 10,
        'bounds': [0, 30],
        'name': 'Heart-lung mean transit time',
        'unit': 'sec',
    },
    'Dhl': {
        'init': 0.2,
        'bounds': [0.05, 0.95],
        'name': 'Heart-lung dispersion',
        'unit': '',
    },
    'To': {
        'init': 20,
        'bounds': [0, 60],
        'name': 'Organs blood mean transit time',
        'unit': 'sec',
    },
    'Eo': {
        'init': 0.15,
        'bounds': [0, 0.5],
        'name': 'Organs extraction fraction',
        'unit': '',
    },
    'Toe': {
        'init': 120,
        'bounds': [0, 800],
        'name': 'Organs extravascular mean transit time',
        'unit': 'sec',
    },

    # Liver

    've': {
        'init': 0.3,
        'bounds': [0.01, 0.6],
        'name': 'Liver extracellular volume fraction',
        'unit': 'mL/cm3',
    },
    'Tg': {
        'init': 30,
        'bounds': [0.1, 60],
        'name': 'Gut mean transit time',
        'unit': 'sec',
    },
    'Dg': {
        'init': 0.85,
        'bounds': [0, 1],
        'name': 'Gut dispersion',
        'unit': '',
    },
    'khe_i': {
        'init': 2 * [0.002],
        'bounds': 2 * [[0.0, 0.1]],
        'name': 'Initial hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
    },
    'khe_f': {
        'init': 2 * [0.002],
        'bounds': 2 * [[0.0, 0.1]],
        'name': 'Final hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
    },
    'Th_i': {
        'init': 2 * [30 * 60],
        'bounds': 2 * [[10*60, 10*60*60]],
        'name': 'Initial hepatocellular mean transit time',
        'unit': 'sec',
    },
    'Th_f': {
        'init': 2 * [30 * 60],
        'bounds': 2 * [[10*60, 10*60*60]],
        'name': 'Final hepatocellular mean transit time',
        'unit': 'sec',
    },
    'Kbh': {
        'init': 2 * [1e-9],
        'bounds': 2 * [[1e-9, 1e-3]],
        'name': 'Biliary tissue excretion rate',
        'unit': '/sec',
    },
    'Kbh_i': {
        'init': 2 * [1e-9],
        'bounds': 2 * [[1e-9, 1e-3]],
        'name': 'Initial biliary tissue excretion rate',
        'unit': '/sec',
    },
    'Kbh_f': {
        'init': 2 * [1e-9],
        'bounds': 2 * [[1e-9, 1e-3]],
        'name': 'Final biliary tissue excretion rate',
        'unit': '/sec',
    },
    'vol': {
        'init': 2 * [1000],
        'bounds': 2 * [[0, 10000]],
        'name': 'Liver volume',
        'unit': 'cm3',
    },
    'vh': {
        'name': 'Hepatocellular volume fraction',
        'unit': 'mL/cm3',
    },
    'khe': {
        'name': 'Hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
    },
    'kbh': {
        'name': 'Biliary excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'kbh_i': {
        'name': 'Initial biliary excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'kbh_f': {
        'name': 'Final biliary excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'Th': {
        'name': 'Hepatocellular mean transit time',
        'unit': 'sec',
    },
    'Khe': {
        'name': 'Hepatocellular tissue uptake rate',
        'unit': '/sec',
    },
    'CL': {
        'name': 'Liver plasma clearance',
        'unit': 'mL/sec',
    }, 
    'CL_i': {
        'name': 'Initial liver plasma clearance',
        'unit': 'mL/sec',
    }, 
    'CL_f': {
        'name': 'Final liver plasma clearance',
        'unit': 'mL/sec',
    }, 
    'Eb': {
        'name': 'Body extraction fraction',
        'unit': '',
    }, 
    'Eb_i': {
        'name': 'Initial body extraction fraction',
        'unit': '',
    },  
    'Eb_f': {
        'name': 'Final body extraction fraction',
        'unit': '',
    }, 
}


def _div(a, b):
    with np.errstate(divide='ignore'):
        return np.divide(a, b)
