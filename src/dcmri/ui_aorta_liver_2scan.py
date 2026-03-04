import warnings
from copy import deepcopy
import json

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np
import pydmr

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.liver as liver

class AortaLiver2scan():
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

    def __init__(
        self, 
        kinetics = '1I-IC-HFD', 
        non_stationary='UE', 
        sequence='SS', 
        **params,
      ):
        
        # Check inputs
        if sequence not in ['SS', 'SR']:
            raise ValueError(
                'Sequence ' + str(sequence) + ' is not available.')
        if kinetics[0] != '1':
            raise ValueError('Only single-inlet models are allowed.')

        self._version = '1.0'

        # Configuration
        self._kinetics = kinetics
        self._sequence = sequence 
        self._non_stationary = non_stationary

        # State variables
        self._pars = None
        self._free = None
        self._pcov = None
        self._sdev = None

        self._init_params(**params)

    def _compute_conc_aorta(self):

        organs = ['2cxm', ([self._pars['To'], self._pars['Toe']], self._pars['Eo'])]
        self._t = np.arange(0, self._pars['tmax'], self._pars['dt'])
        conc = lib.ca_conc(self._pars['agent'])
        J1 = lib.ca_injection(
            self._t, self._pars['weight'], conc, self._pars['dose'], 
            self._pars['rate'], self._pars['BAT'])
        J2 = lib.ca_injection(
            self._t, self._pars['weight'], conc, self._pars['dose2'], 
            self._pars['rate'], self._pars['BAT2'])
        Jb = pk_aorta.flux_aorta(
            J1 + J2, E=self._pars['Eb'], dt=self._pars['dt'], 
            tol=self._pars['dose_tolerance'],
            heartlung = ['pfcomp', (self._pars['Thl'], self._pars['Dhl'])],
            organs = organs)
        self._ca = Jb/self._pars['CO']

    def _compute_relax_aorta(self):

        self._compute_conc_aorta()
        rb = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        self._R1a = self._pars['R10a'] + rb * self._ca

    def _compute_signal_aorta(self):

        pars1 = _sequence_parameters_first_scan(self._sequence, self._pars)
        pars2 = _sequence_parameters_second_scan(self._sequence, self._pars)
        
        self._compute_relax_aorta()

        # Control visit
        self._Sa = np.zeros(self._t.size)
        t1 = self._t <= self._pars['t_scan2']
        t2 = self._t > self._pars['t_scan2']
        self._Sa[t1] = sig.signal(self._sequence, self._R1a[t1], self._pars['S0a'], **pars1)
        self._Sa[t2] = sig.signal(self._sequence, self._R1a[t2], self._pars['S02a'], **pars2)

    def _predict_aorta(self, xdata: tuple) -> tuple:

        tmax = max([max(x) for x in xdata])
        if self._pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be predicted with "
                f"the current configuration is {self._pars['tmax']}. "
                f"To predict these data, set tmax = {tmax} (or larger) " 
                f"when creating the AortaLiver2Scan object."
              )
        
        self._compute_signal_aorta()

        return (
            utils.sample(xdata[0], self._t, self._Sa, self._pars['TS']),
            utils.sample(xdata[1], self._t, self._Sa, self._pars['TS']),
        )
    
    def _compute_conc_liver(self, sum=True):
        pars = liver.params_liver(self._kinetics, self._non_stationary)
        pars = {k:v for k, v in self._pars.items() if k in pars}
        cp = self._ca / (1 - self._pars['H'])
        self._Cl = liver.conc_liver(
            cp, dt=self._pars['dt'], sum=sum, kinetics=self._kinetics, 
            non_stationary=self._non_stationary, **pars,
        )
        
    def _compute_relax_liver(self):

        self._compute_conc_liver(sum=False)

        rp = lib.relaxivity(self._pars['field_strength'], 'plasma', self._pars['agent'])
        rh = lib.relaxivity(self._pars['field_strength'], 'hepatocytes', self._pars['agent'])

        if 'IC' in self._kinetics:
            self._R1l = self._pars['R10l'] + rp*self._Cl[0, :] + rh*self._Cl[1, :]
        else:
            self._R1l = self._pars['R10l'] + rp*self._Cl

    def _compute_signal_liver(self):
        
        pars1 = _sequence_parameters_first_scan(self._sequence, self._pars)
        pars2 = _sequence_parameters_second_scan(self._sequence, self._pars)
        
        self._compute_relax_liver()

        self._Sl = np.zeros(self._t.size)
        t1 = self._t <= self._pars['t_scan2']
        t2 = self._t > self._pars['t_scan2']
        self._Sl[t1] = sig.signal(self._sequence, self._R1l[t1], self._pars['S0l'], **pars1)
        self._Sl[t2] = sig.signal(self._sequence, self._R1l[t2], self._pars['S02l'], **pars2)


    def _predict_liver(self, xdata: tuple) -> tuple:

        tmax = max([max(x) for x in xdata])
        if self._pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be predicted with "
                f"the current configuration is {self._pars['tmax']}. "
                f"To predict these data, set tmax = {tmax} (or larger) " 
                f"when creating the AortaLiver2Scan object."
              )
        
        self._compute_signal_liver()

        return (
            utils.sample(xdata[0], self._t, self._Sl, self._pars['TS']),
            utils.sample(xdata[1], self._t, self._Sl, self._pars['TS']),
        )
    
    def conc(self, sum=True) -> tuple:
        """Concentrations in aorta and liver.

        Args:
            sum (bool, optional): If set to true, the liver concentrations 
              are the sum over both compartments. If set to false, the 
              compartmental concentrations are returned individually. 
              Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, liver 
              concentrations.
        """
        self._compute_conc_aorta()
        self._compute_conc_liver(sum=sum)
        return self._t, self._ca, self._Cl
    
    def relax(self) -> tuple:
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: time points, aorta blood R1, liver 
              R1.
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        return self._t, self._R1a, self._R1l
    
    def signal(self):
        """Signal in aorta and liver.

        Returns:
            dict: time points, aorta blood signal, liver 
              signal.
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()

        return self._t, self._Sa, self._Sl
    
    def predict(self, xdata: tuple) -> tuple:
        """Predict the data at given time points

        Args:
            xdata (tuple): tuple of 4 arrays with time points for aorta in 
              the first scan, aorta in the second stand, liver in the first 
              scan, and liver in the second scan, in that order. The four 
              arrays can be different in length and value.

        Returns:
            tuple: tuple of 4 arrays with signals for aorta in the first 
            scan, aorta in the second stand, liver in the first scan, and 
            liver in the second scan, in that order. The arrays have the 
            same length as its corresponding array of time points.
        """
        Sa = self._predict_aorta((xdata[0], xdata[1]))
        Sl = self._predict_liver((xdata[2], xdata[3]))
        return Sa[0], Sa[1], Sl[0], Sl[1]
    
    def _init_params(self, **params):

        # List parameters for the configuration
        P = PARAMS | PARAMS_SIGNAL_1 | PARAMS_WHOLE_BODY
        P = P | _sequence_pars(self._sequence)
        P = P | liver.params_liver(self._kinetics, self._non_stationary)
        P['TS'] = PARAMS_SEQUENCE['TS']
        P['vol'] = liver.PARAMS_LIVER['vol']
        P = P | PARAMS_2SCAN | PARAMS_SIGNAL_2

        # Initialize parameters
        self._pars = {p: P[p]['init'] for p in P}
        for p in params:
            if p not in P:
                raise ValueError(
                    f"{p} is not a valid model parameter in this configuration."
                )                
            self._pars[p] = params[p]


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
        # Parameters for export
        P = PARAMS_WHOLE_BODY | liver.PARAMS_LIVER | PARAMS_SIGNAL_1
        P = P | PARAMS_SIGNAL_2
        # Add derived parameters
        pars = liver.derived_params_liver(self._pars, self._kinetics, self._pars['H'])
        # Add short name, full name, value, units.
        pars = {
            p: [P[p]['name'], pars[p], P[p]['unit'], 0]
            for p in pars if p in P
        }
        # Add standard deviation
        if self._sdev is not None:
            for p in self._sdev:
                pars[p][-1] = self._sdev[p]
        return pars


    def to_dmr(self, file, subject='Subject', study='Study'):
        dmr = {'data': {}, 'pars': {}, 'sdev':{}}
        pars = self.export_params()
        for p in pars:
            dmr['data'][p] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1], str) else 'float']
            dmr['pars'][(subject, study, p)] = pars[p][1]
            dmr['sdev'][(subject, study, p)] = pars[p][3]
        pydmr.write(file, dmr)


    def train(self, xdata: tuple, ydata: tuple, joint=True, n0=1, 
              R102a=None, R102l=None, free=None, sigma=None, **kwargs):
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
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            R102a (float, optional): R1 value in arterial blood before the 
                second injection. If provided this is used to estimate the 
                baseline S0a in the artery. Else this is initialized to S0a. 
                Defaults to None.
            R102l (float, optional): R1 value in liver before the 
                second injection. If provided this is used to estimate the 
                baseline S0l in the liver. Else this is initialized to S0l. 
                Defaults to None.
            free (dict, optional): Dictionary with free parameters and their
              bounds. If not provided, a default set of free parameters is used.
              Defaults to None.
            sigma (array, optional): signa values used to weight the time 
                points in the fit. If not provided all time points are weighted 
                equally. Defaults to None.
            kwargs: any other keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        
        # Set free parameters
        if free is None:
            P = PARAMS_WHOLE_BODY | liver.params_liver(self._kinetics, self._non_stationary) | PARAMS_SIGNAL_2
            free = {p: P[p]['bounds'] for p in P}
        for p in free:
            if p not in self._pars:
                raise ValueError(
                    f"{p} is not a valid free parameter in this configuration."
                ) 
        self._free = deepcopy(free)
        
        # Estimate BAT and BAT2
        T, D = self._pars['Thl'], self._pars['Dhl']
        self._pars['BAT'] = xdata[0][np.argmax(ydata[0])] - (1-D)*T
        self._free['BAT'] = [
            self._pars['BAT'] + self._free['BAT'][0],
            self._pars['BAT'] + self._free['BAT'][1],
        ]
        self._pars['BAT2'] = xdata[1][np.argmax(ydata[1])] - (1-D)*T
        self._free['BAT2'] = [
            self._pars['BAT2'] + self._free['BAT2'][0],
            self._pars['BAT2'] + self._free['BAT2'][1],
        ]

        # Estimate S0
        pars = _sequence_parameters_first_scan(self._sequence, self._pars)
        Srefb = sig.signal(self._sequence, self._pars['R10a'], 1, **pars)
        Srefl = sig.signal(self._sequence, self._pars['R10l'], 1, **pars)
        self._pars['S0a'] = np.mean(ydata[0][1:n0]) / Srefb
        self._pars['S0l'] = np.mean(ydata[2][1:n0]) / Srefl

        # Estimate S02
        pars = _sequence_parameters_second_scan(self._sequence, self._pars)
        if R102a is not None:
            Sref2b = sig.signal(self._sequence, R102a, 1, **pars)
            self._pars['S02a'] = np.mean(ydata[1][1:n0]) / Sref2b
        else:
            self._pars['S02a'] = self._pars['S0a']
        self._free['S02a'] = [
            self._pars['S02a'] * self._free['S02a'][0],
            self._pars['S02a'] * self._free['S02a'][1],
        ]
        if R102l is not None:
            Sref2l = sig.signal(self._sequence, R102l, 1, **pars)
            self._pars['S02l'] = np.mean(ydata[3][1:n0]) / Sref2l
        else:
            self._pars['S02l'] = self._pars['S0l']
        self._free['S02l'] = [
            self._pars['S02l'] * self._free['S02l'][0],
            self._pars['S02l'] * self._free['S02l'][1],
        ]
 
        # Train free aorta parameters on aorta data
        pars = list(PARAMS_WHOLE_BODY.keys()) + ['BAT2', 'S02a']
        free_aorta = {p:v for p, v in self._free.items() if p in pars}
        sigma_aorta = (sigma[0], sigma[1]) if sigma is not None else None
        pcov_aorta = _train(self, self._predict_aorta, (xdata[0], xdata[1]), (ydata[0], ydata[1]), free_aorta, sigma=sigma_aorta, **kwargs)
        sdev_aorta = _sdev(pcov_aorta, free_aorta)

        # Train free liver parameters on liver data
        pars = list(liver.PARAMS_LIVER.keys()) + ['S02l']
        free_liver = {p:v for p, v in self._free.items() if p in pars}
        sigma_liver = (sigma[2], sigma[3]) if sigma is not None else None
        pcov_liver = _train(self, self._predict_liver, (xdata[2], xdata[3]), (ydata[2], ydata[3]), free_liver, sigma=sigma_liver, **kwargs)
        sdev_liver = _sdev(pcov_liver, free_liver)

        # Train all parameters on all data
        if joint:
            pcov = _train(self, self.predict, xdata, ydata, self._free, sigma=sigma, **kwargs)
            self._pcov = [(pcov.tolist(), list(self._free.keys()))]
            self._sdev = _sdev(pcov, self._free)
        else:
            self._pcov = [
                (pcov_aorta.tolist(), list(free_aorta.keys())), 
                (pcov_liver.tolist(), list(free_liver.keys())),
            ]
            self._sdev = sdev_aorta | sdev_liver

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
        tmax = max([max(x) for x in xdata])
        if self._pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be plotted with "
                f"the current configuration is {tmax}. To increase set a larger " 
                f"tmax value when creating the AortaLiver object."
              )
        self._compute_signal_aorta()
        self._compute_signal_liver()
        self.conc(sum=False)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        _plot_data2scan(self._t, self._Sa, xdata[:2], ydata[:2],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[0])
        _plot_data2scan(self._t, self._Sl, xdata[2:], ydata[2:],
                        ax3, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[1])
        _plot_conc_aorta(self._t, self._ca, ax2, xlim)
        _plot_conc_liver(self._kinetics, self._t, self._Cl, ax4, xlim)
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

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
            'kinetics': self._kinetics,
            'non_stationary': self._non_stationary,
            'sequence': self._sequence,
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

        self._kinetics = data['kinetics']
        self._non_stationary = data['non_stationary']
        self._sequence = data['sequence']
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
                val = round(v[1], round_to)
                err = round(v[3], round_to)
            print(f"{name} ({p}): {val} ({err}) {unit}")


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
            else:
                return round(pars[args[0]][1], round_to)
        if round_to is None:
            return {p: v[1] for p, v in pars.items() if p in list(args)}
        else:
            return {p: round(v[1], round_to) for p, v in pars.items() if p in list(args)}

    def set_params(self, **pars):
        """set parameter values

        Args:
            pars (dict): parameters to set.
        """
        for p, v in pars.items():
            self._pars[p] = v

def _sequence_pars(sequence):
    pars = {
        'SR': ['FA', 'TR', 'TC'],
        'SS': ['FA', 'TR'], 
    }
    return {p:PARAMS_SEQUENCE[p] for p in pars[sequence]}


def _sequence_parameters_first_scan(sequence, pars):
    if sequence == 'SR':
        return {
            'FA': pars['FA'], 
            'TR': pars['TR'], 
            'TC': pars['TC'], 
        }
    if sequence == 'SS':
        return {
            'FA': pars['FA'], 
            'TR': pars['TR'], 
        }
    if sequence == 'SRC':
        return {
            'TC': pars['TC'], 
        }

def _sequence_parameters_second_scan(sequence, pars):
    if sequence == 'SR': 
        return {
            'FA': pars['FA2'], 
            'TR': pars['TR'], 
            'TC': pars['TC'], 
        }
    if sequence == 'SS':
        return {
            'FA': pars['FA2'], 
            'TR': pars['TR'], 
        }
    if sequence == 'SRC':
        return {
            'TC': pars['TC'], 
        }
    

def _train(self:AortaLiver2scan, predict, xdata, ydata, free, sigma=None, **kwargs):

    if isinstance(ydata, tuple):
        y = np.concatenate(ydata)
    else:
        y = ydata
    if sigma is not None:
        if isinstance(sigma, tuple):
            sigma = np.concatenate(sigma)

    p0 = [_normalize(self._pars[p], free[p]) for p in free] 

    def fit_func(_, *pars):
        _set_pars(self, pars, free)
        yp = predict(xdata)
        if isinstance(yp, tuple):
            return np.concatenate(yp)
        else:
            return yp

    try:
        pars, pcov = curve_fit(
            fit_func, y, y, p0, bounds=[0,1], sigma=sigma, 
            **kwargs,
        )
    except RuntimeError as e:
        msg = 'Runtime error in curve_fit -- \n'
        msg += str(e) + ' Returning initial values.'
        warnings.warn(msg)
        pars, pcov = p0, np.zeros((len(p0), len(p0)))

    _set_pars(self, pars, free)

    return pcov

def _set_pars(self, pars, free):
    i = 0
    for p in free:
      self._pars[p] = _renormalize(pars[i], free[p])
      i += 1

def _normalize(v, bounds):
    return (v - bounds[0]) / (bounds[1] - bounds[0])

def _renormalize(v, bounds):
    return v * (bounds[1] - bounds[0]) + bounds[0]

def _sdev(pcov, free:dict):
    
    sdev = {}

    i = 0
    for p in free.keys():
      sdev[p] = _renormalize(np.sqrt(pcov[i,i]), free[p])
      i += 1

    return sdev





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

def _plot_conc_liver(kinetics, t, C, ax, xlim=None):
    color = 'darkblue'
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    if 'IC' in kinetics:
        ax.plot(t/60, 1000*C[0, :], linestyle='-.',
                color=color, linewidth=2.0, label='Extracellular')
        ax.plot(t/60, 1000*C[1, :], linestyle='--',
                color=color, linewidth=2.0, label='Hepatocytes')
        ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-',
                color=color, linewidth=2.0, label='Tissue')
    else:
        ax.plot(t/60, 1000*C, linestyle='-',
                color=color, linewidth=2.0, label='Tissue')        
    ax.legend()

def _plot_data2scan(t, 
                    sig,
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
        'init': 180,
        'bounds': [0, np.inf],
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
        'init': lib.ca_std_dose('gadoxetate')/2,
        'bounds': [0, np.inf],
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

}


PARAMS_SIGNAL_1 = {

    'R10a': {
        'init': 1/lib.T1(3.0, 'blood'),
        'bounds': [0, np.inf],
        'name': 'Aorta first baseline R1',
        'unit': 'Hz',
    },
    'S0a': {
        'init': 1,
        'bounds': [0, np.inf],
        'name': 'Aorta first signal scale factor',
        'unit': 'a.u.',
    },
    'R10l': {
        'init': 1/lib.T1(3.0, 'liver'),
        'bounds': [0, np.inf],
        'name': 'Liver first baseline R1',
        'unit': 'Hz',
    },
    'S0l': {
        'init': 1,
        'bounds': [0, np.inf],
        'name': 'Liver first signal scale factor',
        'unit': 'a.u.',
    },

}

PARAMS_2SCAN = {
    't_scan2': {
        'init': 90,
        'name': 'Start of second scan',
        'unit': 'sec',
    },
    'dose2': {
        'init': lib.ca_std_dose('gadoxetate')/2,
        'bounds': [0, np.inf],
        'name': 'Second contrast agent dose',
        'unit': 'mL/kg',
    },
    'FA2': {
        'init': 15.0,
        'bounds': [0, np.inf],
        'name': 'Second flip angle',
        'unit': 'deg',
    },
}

PARAMS_SIGNAL_2 = {
    
    'S02a': {
        'init': 1,
        'bounds': [0, 2],
        'name': 'Aorta second signal scale factor',
        'unit': 'a.u.',
    },
    'S02l': {
        'init': 1,
        'bounds': [0, 2],
        'name': 'Liver second signal scale factor',
        'unit': 'a.u.',
    },
    'BAT2': {
        'init': 1200,
        'bounds': [-60, 60],
        'name': 'Second bolus arrival time',
        'unit': 'sec',
    },
}

PARAMS_WHOLE_BODY = {

    # Body
    'BAT': {
        'init': 60,
        'bounds': [-60, 60],
        'name': 'First bolus arrival time',
        'unit': 'sec',
    },
    'CO': {
        'init': 100,
        'bounds': [0, 300],
        'name': 'Cardiac output',
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
    'Eb': {
        'init': 0.05,
        'bounds': [0.01, 0.15],
        'name': 'Body extraction fraction',
        'unit': '',
    },
}


PARAMS_SEQUENCE = {
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
}

