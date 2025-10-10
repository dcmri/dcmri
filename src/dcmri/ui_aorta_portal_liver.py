from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import dcmri.ui as ui
import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.pk as pk
import dcmri.liver as liver


class AortaPortalLiver():
    """Joint model for aorta, portal vein and liver signals.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta, portal vein and liver.  

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
        kinetics (str, optional): Tracer-kinetic liver model. See table 
          :ref:`table-liver-models` for options - only dual-inlet models 
          are allowed. Defaults to '2I-EC'.
        non_stationary (str, optional): For intracellular tracers - stationarity 
          regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
          For more detail see :ref:`liver-tissues`. Defaults to None.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SSI' (steady-state with aortic inflow correction). Defaults 
          to 'SS'.
        free (dict, optional): Dictionary with free parameters and their
          bounds. If not provided, a default set of free parameters is used.
          Defaults to None.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. See tables :ref:`AortaLiver-parameters` and
          :ref:`AortaLiver-defaults` for a list of parameters and their
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

        Use `fake_liver` to generate synthetic test data:

        >>> time, aif, vif, roi, _ = dc.fake_liver(sequence='SSI')

        Since this model generates 3 time curves, the x- and y-data are 
        tuples:

        >>> xdata, ydata = (time, time, time), (aif, vif, roi)

        Build an aorta-portal-liver model and parameters to match the 
        conditions of the fake liver data:

        >>> model = dc.AortaPortalLiver(
        ...     kinetics = '2I-IC',
        ...     sequence = 'SSI',
        ...     dt = 0.5,
        ...     tmax = 180,
        ...     weight = 70,
        ...     agent = 'gadoxetate',
        ...     dose = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 15,
        ...     TS = 0.5,
        ... )

        Train the model on the data:

        >>> model.train(xdata, ydata, n0=10, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare 
        against the experimentally derived data:

        >>> model.plot(xdata, ydata)

        We can also have a look at the model parameters after training:

        >>> model.print_params(round_to=3)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        First bolus arrival time (BAT): 14.616 (1.1) sec
        Cardiac output (CO): 100.09 (2.736) mL/sec
        Heart-lung mean transit time (Thl): 14.402 (1.375) sec
        Heart-lung dispersion (Dhl): 0.391 (0.013) 
        Organs blood mean transit time (To): 27.811 (5.291) sec
        Organs extraction fraction (Eo): 0.29 (0.105)
        Organs extravascular mean transit time (Toe): 70.621 (102.614) sec
        Body extraction fraction (Eb): 0.013 (0.23)
        Aorta inflow time (TF): 0.409 (0.014) sec
        Liver extracellular volume fraction (ve): 0.479 (0.112) mL/cm3
        Liver plasma flow (Fp): 0.018 (0.001) mL/sec/cm3
        Arterial flow fraction (fa): 0.087 (0.074)
        Arterial transit time (Ta): 2.398 (1.356) sec
        Hepatocellular uptake rate (khe): 0.006 (0.003) mL/sec/cm3
        Hepatocellular mean transit time (Th): 683.604 (2554.75) sec
        Gut mean transit time (Tg): 10.782 (0.614) sec
        Gut dispersion (Dg): 0.893 (0.07)
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        Hematocrit (H): 0.45
        Arterial venous blood flow (Fa): 0.002 mL/sec/cm3
        Portal venous blood flow (Fv): 0.016 mL/sec/cm3
        Extracellular mean transit time (Te): 27.216 sec
        Biliary tissue excretion rate (Kbh): 0.001 mL/sec/cm3
        Hepatocellular tissue uptake rate (Khe): 0.012 mL/sec/cm3
        Biliary excretion rate (kbh): 0.001 mL/sec/cm3

    Notes:

        Table :ref:`AortaPortalLiver-parameters` lists the parameters that are 
        relevant in each regime. Table :ref:`AortaPortalLiver-defaults` list 
        all possible parameters and their default settings. 

        .. _AortaPortalLiver-parameters:
        .. list-table:: **Aorta-Liver parameters**
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
              - For estimating baseline signal
            * - field_strength, weight, agent, dose, rate
              - Always
              - Injection protocol
            * - R10a, R10l, S0a, S0v, S0l 
              - Always
              - Precontrast R1 (:ref:`relaxation-params`) and 
                S0 (:ref:`params-per-sequence`)for aorta and liver 
            * - FA, TR, TS
              - Always
              - :ref:`params-per-sequence`
            * - TF
              - If **sequence** is 'SSI'
              - To model aorta inflow effects
            * - BAT, CO, Thl, Dhl, To, Eo, Tie, Eb
              - Always
              - :ref:`whole-body-tissues`
            * - Tg , Dg
              - Always
              - Gut dispersion
            * - H, ve, Fp, fa, Ta, Tg, khe, khe_i, kh_f, Th, Th_i, Th_f.
              - Depends on **kinetics** and **stationary**
              - :ref:`table-liver-models`

        .. _AortaPortalLiver-defaults:
        .. list-table:: **Aorta-Liver parameter defaults**
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
            * - S0v
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
            * - S0
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - TF
              - Sequence
              - 0.5
              - [0, 10]
              - Free
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
              - **Portal vein**
              -
              - 
              - 
            * - Tg
              - Kinetic
              - 15
              - [0.1, 60]
              - Free
            * - Dg
              - Kinetic
              - 0.85
              - [0, 1]
              - Free
            * - uv
              - Kinetic
              - 1
              - [0, 1]
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
            * - ve
              - Kinetic
              - 0.3
              - [0.01, 0.6]
              - Free
            * - Fp
              - Kinetic
              - 0.01
              - [0, 0.1]
              - Free
            * - fa
              - Kinetic
              - 0.2
              - [0, 0.1]
              - Free
            * - Ta
              - Kinetic
              - 0.5
              - [0, 3]
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
        kinetics='2I-EC', 
        non_stationary=None, 
        sequence='SS', 
        free=None, 
        **params,
    ):
        
        # Check configuration
        if sequence not in ['SS', 'SSI']:
            raise ValueError(f'Sequence {sequence} is not available.')
        if kinetics[0] != '2':
            raise ValueError('Only dual-inlet models are allowed.')

        # Set configuration
        self.organs = '2cxm' 
        self.kinetics = kinetics 
        self.sequence = sequence 
        self.non_stationary = non_stationary

        # Set parameters
        P = (PARAMS_CONST | PARAMS_SIGNAL | PARAMS_PORTAL| PARAMS_WHOLE_BODY |
             PARAMS_SIGNAL_CONFIG | liver.PARAMS_LIVER )
        
        self.pars = ui.init_parameters(P, self._model_pars(), **params)
        self.free = ui.init_free_parameters(P, self.pars, free)

        # Parameter covariance not known until fit has been done
        self.pcov = None

        # Internal flags
        self._predict = None

    def _config_pars(self):
        pars = liver.params_liver(self.kinetics, self.non_stationary)
        if self.sequence =='SSI':
            pars += ['TF']
        return pars

    def _model_pars(self):
        P_FIX = PARAMS_CONST | PARAMS_SIGNAL | PARAMS_PORTAL | PARAMS_WHOLE_BODY 
        pars = list(P_FIX.keys())
        pars += self._config_pars()
        return pars
    
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
        P_EXP = (PARAMS_SIGNAL | PARAMS_WHOLE_BODY | liver.PARAMS_LIVER 
                 | PARAMS_PORTAL | PARAMS_SIGNAL_CONFIG)
        # Add derived parameters
        pars = liver.derived_params_liver(self.pars, self.kinetics)
        # Add short name, full name, value, units.
        pars = {
            p: [P_EXP[p]['name'], pars[p], P_EXP[p]['unit'], 0]
            for p in pars if p in P_EXP
        }
        # Add standard deviation
        if self.pcov is not None:
            for i, p in enumerate(self.free):
                pars[p][-1] = np.sqrt(self.pcov[i,i])
        return pars

    def _conc_aorta(self) -> np.ndarray:
        if self.organs == 'comp':
            organs = ['comp', (self.pars['To'],)]
        else:
            organs = ['2cxm', ([self.pars['To'], self.pars['Toe']], self.pars['Eo'])]
        self.t = np.arange(0, self.pars['tmax'], self.pars['dt'])
        conc = lib.ca_conc(self.pars['agent'])
        Ji = lib.ca_injection(
            self.t, self.pars['weight'], conc, self.pars['dose'], 
            self.pars['rate'], self.pars['BAT']
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=self.pars['Eb'], dt=self.pars['dt'], 
            tol=self.pars['dose_tolerance'],
            heartlung = ['pfcomp', (self.pars['Thl'], self.pars['Dhl'])],
            organs = organs)
        self.ca = Jb/self.pars['CO']
        return self.t, self.ca

    def _relax_aorta(self):
        t, cb = self._conc_aorta()
        rb = lib.relaxivity(self.pars['field_strength'], 'blood', self.pars['agent'])
        return t, self.pars['R10a'] + rb*cb

    def _predict_aorta(self, xdata: np.ndarray) -> np.ndarray:
        # TODO: change the approach here and increase tmax insetead to fit xdata
        # Build t on demand every time rather than save in self.t
        if self.pars['tmax'] < max(xdata):
            raise ValueError(
                f"The largest time point that can be predicted is {self.pars['tmax']} as defined by the parameter tmax"
              )
        t, R1b = self._relax_aorta()
        if self.sequence == 'SSI':
            signal = sig.signal_spgr(
                self.pars['S0a'], R1b, self.pars['TF'], 
                self.pars['TR'], self.pars['FA'], n0=1)
        else:
            signal = sig.signal_ss(
                self.pars['S0a'], R1b, self.pars['TR'], self.pars['FA']
            )
        return utils.sample(xdata, t, signal, self.pars['TS'])
    
    def _conc_portal(self):
        self.cv = pk.flux_chain(
            self.ca, self.pars['Tg'], self.pars['Dg'], dt=self.pars['dt']
        )
        return self.cv
    
    def _relax_portal(self):
        rb = lib.relaxivity(
            self.pars['field_strength'], 'blood', self.pars['agent'])
        cv = self._conc_portal()
        return self.pars['R10a'] + rb*self.pars['uv']*cv
    
    def _predict_portal(self, xdata):
        t = np.arange(0, self.pars['tmax'], self.pars['dt'])
        R1v = self._relax_portal()
        signal = sig.signal_ss(
            self.pars['S0v'], R1v, self.pars['TR'], self.pars['FA']
        )
        return utils.sample(xdata, t, signal, self.pars['TS'])

    def _conc_liver(self, sum=True):
        pars = liver.params_liver(self.kinetics, self.non_stationary)
        pars = {k:v for k, v in self.pars.items() if k in pars}
        return liver.conc_liver(
            (self.ca, self.cv), 
            dt=self.pars['dt'], 
            sum=sum, 
            kinetics=self.kinetics, 
            non_stationary=self.non_stationary, 
            **pars
        )
        
    def _relax_liver(self):
        t = np.arange(0, self.pars['tmax'], self.pars['dt'])
        Cl = self._conc_liver(sum=False)
        rp = lib.relaxivity(self.pars['field_strength'], 'plasma', self.pars['agent'])
        rh = lib.relaxivity(self.pars['field_strength'], 'hepatocytes', self.pars['agent'])
        if 'IC' in self.kinetics:
            return t, self.pars['R10l'] + rp*Cl[0, :] + rh*Cl[1, :]
        else:
            return t, self.pars['R10l'] + rp*Cl

    def _predict_liver(self, xdata):
        t, R1l = self._relax_liver()
        signal = sig.signal_ss(
            self.pars['S0l'], R1l, self.pars['TR'], self.pars['FA']
        )
        return utils.sample(xdata, t, signal, self.pars['TS'])

    def conc(self, sum=True):
        """Concentrations in aorta. portal vein and liver.

        Args:
            sum (bool, optional): If set to true, the liver concentrations are 
              the sum over both compartments. If set to false, the 
              compartmental concentrations are returned individually. 
              Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, portal-venous 
            blood concentrations, liver concentrations.
        """
        t, cb = self._conc_aorta()
        cv = self._conc_portal()
        C = self._conc_liver(sum=sum)
        return t, cb, cv, C

    def relax(self):
        """Relaxation rates in aorta, portal vein and liver.

        Returns:
            tuple: time points, aorta blood R1, portal-venous blood R1, liver 
            blood R1.
        """
        t, R1b = self._relax_aorta()
        R1v = self._relax_portal()
        t, R1l = self._relax_liver()
        return t, R1b, R1v, R1l

    def predict(self, time: tuple) -> tuple:
        """Predict the signals at given time

        Args:
            xdata (tuple): tuple of 3 arrays with time points for aorta, 
              portal vein and liver, in that order. The 3 arrays can be 
              different in length and value.

        Returns:
            tuple: tuple of 3 arrays with signals for aorta, portal vein and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
        """
        # Public interface
        if self._predict is None:
            signala = self._predict_aorta(time[0])
            signalv = self._predict_portal(time[1])
            signall = self._predict_liver(time[2])
            return signala, signalv, signall
        
        # Private interface with different input & output types
        elif self._predict == 'aorta':
            return self._predict_aorta(time)
        elif self._predict == 'portal':
            return self._predict_portal(time)
        elif self._predict == 'liver':
            return self._predict_liver(time)

    def train(self, time: tuple, signal: tuple, n0=1, **kwargs):
        """Train the free parameters

        Args:
            time (tuple): tuple of 3 arrays with time points for aorta, 
              portal vein and liver, in that order. The arrays can be 
              different in length and value.
            signal (array-like): tuple of 3 arrays with signals for aorta, 
              portal vein and liver, in that order. The arrays can be 
              different in length and value but each has to have the same 
              length as its corresponding array of time points.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaPortalLiver: the trained model
        """
        # Estimate S0
        if self.sequence == 'SSI':
            Srefa = sig.signal_spgr(
                1, self.pars['R10a'], self.pars['TF'], 
                self.pars['TR'], self.pars['FA'], n0=1,
            )
        else:
            Srefa = sig.signal_ss(
                1, self.pars['R10a'], self.pars['TR'], self.pars['FA'],
            )
        Srefv = sig.signal_ss(1, self.pars['R10a'], self.pars['TR'], self.pars['FA'])
        Srefl = sig.signal_ss(1, self.pars['R10l'], self.pars['TR'], self.pars['FA'])
        self.pars['S0a'] = np.mean(signal[0][:n0]) / Srefa
        self.pars['S0v'] = np.mean(signal[1][:n0]) / Srefv
        self.pars['S0l'] = np.mean(signal[2][:n0]) / Srefl

        # Estimate BAT
        self.pars['BAT'] = time[0][np.argmax(signal[0])] - (1-self.pars['Dhl']) * self.pars['Thl']
        self.pars['BAT'] = max([self.pars['BAT'], 0])

        # Copy the original free parameters to restore later
        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = list(PARAMS_WHOLE_BODY.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui._train(self, time[0], signal[0], **kwargs)

        # Train free aorta parameters on portal venous data
        self._predict = 'portal'
        pars = list(PARAMS_PORTAL.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui._train(self, time[1], signal[1], **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        pars = list(liver.PARAMS_LIVER.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui._train(self, time[2], signal[2], **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        return ui._train(self, time, signal, **kwargs)
    

    def plot(self,
        time: tuple,
        signal: tuple,
        xlim=None, 
        ref=None,
        fname=None, 
        show=True,
    ):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 3 arrays with time points for aorta, 
              portal vein and liver, in that order. The two arrays can be 
              different in length and value.
            signal (array-like): tuple of 3 arrays with signals for aorta, 
              portal vein and liver, in that order. The arrays can be 
              different in length and value but each has to have the same 
              length as its corresponding array of time points.
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
        t, cb, cv, C = self.conc(sum=False)
        sig = self.predict((t, t, t))
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        _plot_data(t, sig[0], time[0], signal[0],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[0])
        _plot_data(t, sig[1], time[1], signal[1],
                        ax3, xlim,
                        color=['orchid', 'purple'],
                        test=None if ref is None else ref[1])
        _plot_data(t, sig[2], time[2], signal[2],
                        ax5, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[2],
                        xlabel='Time (min)')
        _plot_conc_aorta(t, cb, ax2, xlim)
        _plot_conc_portal(t, cv, ax4, xlim)
        _plot_conc_liver(self, t, C, ax6, xlim)
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def cost(self, time, signal, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            time (tuple): tuple of 3 arrays with time points for aorta, 
              portal vein and liver, in that order. The arrays can be 
              different in length and value.
            signal (array-like): tuple of 3 arrays with signals for aorta, 
              portal vein and liver, in that order. The arrays can be 
              different in length and value but each has to have the same 
              length as its corresponding array of time points.
            metric (str, optional): Which metric to use (see notes for 
              possible values). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.

        Notes:

            Available options are: 
            
            - 'RMS': Root-mean-square.
            - 'NRMS': Normalized root-mean-square. 
            - 'AIC': Akaike information criterion. 
            - 'cAIC': Corrected Akaike information criterion for small models.
            - 'BIC': Bayesian information criterion.
        """
        return ui._cost(self, time, signal, metric)

    def save(self, file=None, path=None, filename='Model'):
        """Save the current state of the model

        Args:
            file (str, optional): complete path of the file. If this is not 
              provided, a file is constructure from path and filename 
              variables. Defaults to None.
            path (str, optional): path to store the state if file is not 
              provided. Thos variable is ignored if file is provided. 
              Defaults to current working directory.
            filename (str, optional): filename to store the state if file is 
              not provided. If no extension is included, the extension '.pkl' 
              is automatically added. This variable is ignored if file is 
              provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return ui._save(self, file, path, filename)

    def load(self, file=None, path=None, filename='Model'):
        """Load the saved state of the model

        Args:
            file (str, optional): complete path of the file. If this is not 
              provided, a file is constructure from path and filename 
              variables. Defaults to None.
            path (str, optional): path to store the state if file is not 
              provided. Thos variable is ignored if file is provided. 
              Defaults to current working directory.
            filename (str, optional): filename to store the state if file is 
              not provided. If no extension is included, the extension 
              '.pkl' is automatically added. This variable is ignored if file 
              is provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return ui._load(self, file, path, filename)
    
    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is 
              not provided, the values are not rounded. Defaults to None.
        """
        return ui._print_params(self.export_params(), self.free, round_to=round_to)
    
    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            list or float: values of parameter values, or a scalar value if 
            only one parameter is required.
        """
        return ui._return_params(self.export_params(), *args, round_to=round_to)
    




# Helper functions for plotting

def _plot_conc_aorta(t, cb, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(ylabel='Concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cb, linestyle='-',
            color='darkred', linewidth=2.0, label='Aorta')
    ax.legend()


def _plot_conc_portal(t, cv, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(ylabel='Concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cv, linestyle='-',
            color='purple', linewidth=2.0, label='Portal vein')
    ax.legend()


def _plot_conc_liver(self, t, C, ax, xlim=None):
    color = 'darkblue'
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    if 'IC' in self.kinetics:
        ax.plot(t/60, 1000*C[0, :], linestyle='-.',
                color=color, linewidth=2.0, label='Extracellular')
        ax.plot(t/60, 1000*C[1, :], linestyle='--',
                color=color, linewidth=2.0, label='Hepatocytes')
        ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-',
                color=color, linewidth=2.0, label='Liver')
    else:
        ax.plot(t/60, 1000*C, linestyle='-',
                color=color, linewidth=2.0, label='Liver')
    ax.legend()


def _plot_data(t: np.ndarray, sig: np.ndarray,
                    xdata: np.ndarray, ydata: np.ndarray,
                    ax, xlim, color=['black', 'black'],
                    test=None, xlabel=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel=xlabel, ylabel='MR Signal (a.u.)', 
           xlim=np.array(xlim)/60)
    ax.plot(xdata/60, ydata, marker='o',
            color=color[0], label='fitted data', linestyle='None')
    ax.plot(t/60, sig, linestyle='-',
            color=color[1], linewidth=3.0, label='fit')
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black',
                marker='D', linestyle='None', label='Test data')
    ax.legend()




PARAMS_CONST = {

    'dt': {
        'init': 0.5,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Forward model time step',
        'unit': 'sec',
    },
    'tmax': {
        'init': 120,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Maximum acquisition time',
        'unit': 'sec',
    },
    'dose_tolerance': {
        'init': 0.1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Dose tolerance',
        'unit': '',
    },
    'field_strength': {
        'init': 3.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Magnetic field strength',
        'unit': 'T',
    },
    'weight': {
        'init': 70.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Subject weight',
        'unit': 'kg',
    },
    'agent': {
        'init': 'gadoxetate',
        'default_free': False,
        'bounds': None,
        'name': 'Contrast agent',
        'unit': None,
    },
    'dose': {
        'init': lib.ca_std_dose('gadoxetate')/2,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Contrast agent dose',
        'unit': 'mL/kg',
    },
    'rate': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Contrast agent injection rate',
        'unit': 'mL/sec',
    },
    'TR': {
        'init': 0.005,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Repetition time',
        'unit': 'sec',
    },
    'FA': {
        'init': 15.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Flip angle',
        'unit': 'deg',
    },
    'TS': {
        'init': None,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Sampling time',
        'unit': 'sec',
    },
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [0, 1],
        'name': 'Hematocrit',
        'unit': '',
    },
}


PARAMS_SIGNAL = {

    'R10a': {
        'init': 1/lib.T1(3.0, 'blood'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Aorta baseline R1',
        'unit': 'Hz',
    },
    'S0a': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Aorta signal scale factor',
        'unit': 'a.u.',
    },
    'R10l': {
        'init': 1/lib.T1(3.0, 'liver'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Liver baseline R1',
        'unit': 'Hz',
    },
    'S0l': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Liver signal scale factor',
        'unit': 'a.u.',
    },
    'S0v': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Portal venous signal scale factor',
        'unit': 'a.u.',
    },

}

PARAMS_SIGNAL_CONFIG = {
    'TF': {
        'init': 0.5,
        'default_free': True,
        'bounds': [0, 5],
        'name': 'Aorta inflow time',
        'unit': 'sec',
    },
}


PARAMS_WHOLE_BODY = {

    # Body
    'BAT': {
        'init': 60,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Bolus arrival time',
        'unit': 'sec',
    },
    'CO': {
        'init': 100,
        'default_free': True,
        'bounds': [0, 300],
        'name': 'Cardiac output',
        'unit': 'mL/sec',
    },
    'Thl': {
        'init': 10,
        'default_free': True,
        'bounds': [0, 30],
        'name': 'Heart-lung mean transit time',
        'unit': 'sec',
    },
    'Dhl': {
        'init': 0.2,
        'default_free': True,
        'bounds': [0.05, 0.95],
        'name': 'Heart-lung dispersion',
        'unit': '',
    },
    'To': {
        'init': 20,
        'default_free': True,
        'bounds': [0, 60],
        'name': 'Organs blood mean transit time',
        'unit': 'sec',
    },
    'Eo': {
        'init': 0.15,
        'default_free': True,
        'bounds': [0, 0.5],
        'name': 'Organs extraction fraction',
        'unit': '',
    },
    'Toe': {
        'init': 120,
        'default_free': True,
        'bounds': [0, 800],
        'name': 'Organs extravascular mean transit time',
        'unit': 'sec',
    },
    'Eb': {
        'init': 0.05,
        'default_free': True,
        'bounds': [0.01, 0.15],
        'name': 'Body extraction fraction',
        'unit': '',
    },
}


PARAMS_PORTAL = {
    'Tg': {
        'init': 15.0,
        'default_free': True,
        'bounds': [0.1, 60],
        'name': 'Gut mean transit time',
        'unit': 'sec',
    },
    'Dg': {
        'init': 0.85,
        'default_free': True,
        'bounds': [0, 1],
        'name': 'Gut dispersion',
        'unit': '',
    },
    'uv': {
        'init': 1.0,
        'default_free': True,
        'bounds': [0, 1],
        'name': 'Portal vein volume fraction',
        'unit': '',
    },

}

