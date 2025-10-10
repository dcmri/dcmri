from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import dcmri.ui as ui
import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.liver as liver


class AortaLiver():
    """Joint model for aorta and liver signals.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta and liver.  

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
        kinetics (str, optional): Tracer-kinetic liver model. See table 
          :ref:`table-liver-models` for options - only single-inlet models 
          are allowed. Defaults to '1I-IC-HFD'.
        non_stationary (str, optional): For intracellular tracers - stationarity 
          regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
          For more detail see :ref:`liver-tissues`. Defaults to None.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
        free (dict, optional): Dictionary with free parameters and their
          bounds. If not provided, a default set of free parameters is used.
          Defaults to None.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. See tables :ref:`AortaLiver-parameters` and
          :ref:`AortaLiver-defaults` for a list of parameters and their
          default values.

    See Also:
        `AortaLiver2scan`

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

        >>> time, aif, vif, roi, gt = dc.fake_liver()

        Since this model generates two time curves, the x- and y-data are 
        tuples:

        >>> xdata, ydata = (time,time), (aif,roi)

        Build an aorta-liver model and parameters to match the 
        conditions of the fake liver data:

        >>> model = dc.AortaLiver(
        ...     dt = 0.5,
        ...     tmax = 180,
        ...     weight = 70,
        ...     agent = 'gadoxetate',
        ...     field_strength = 3.0,
        ...     dose = 0.2,
        ...     rate = 3,
        ...     TR = 0.005,
        ...     FA = 15,
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
        First bolus arrival time (BAT): 13.231 (0.266) sec
        Cardiac output (CO): 102.893 (4.182) mL/sec
        Heart-lung mean transit time (Thl): 16.285 (0.409) sec
        Heart-lung dispersion (Dhl): 0.324 (0.016)
        Organs blood mean transit time (To): 19.578 (5.583) sec
        Organs extraction fraction (Eo): 0.363 (0.075)
        Organs extravascular mean transit time (Toe): 46.775 (53.05) sec
        Body extraction fraction (Eb): 0.029 (0.168)
        Apparent liver extracellular volume fraction (ve_app): 0.307 (0.564) mL/cm3
        Extracellular mean transit time (Te): 44.748 (79.644) sec
        Extracellular dispersion (De): 0.916 (0.148)
        Hepatic plasma clearance (Ktrans): 0.002 (0.003) mL/sec/cm3
        Hepatocellular mean transit time (Th): 712.855 (5899.078) sec
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        Aorta first baseline R1 (R10a): 0.614 Hz
        Aorta first signal scale factor (S0a): 100.169 a.u.
        Liver first baseline R1 (R10l): 1.33 Hz
        Liver first signal scale factor (S0l): 150.0 a.u.
        Liver volume (vol): 1000 cm3
        Biliary tissue excretion rate (Kbh): 0.001 mL/sec/cm3

    Notes:

        Table :ref:`AortaLiver-parameters` lists the parameters that are 
        relevant in each regime. Table :ref:`AortaLiver-defaults` list all 
        possible parameters and their default settings. 

        .. _AortaLiver-parameters:
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
              - Stopping criterion on whole-body model
            * - field_strength, weight, agent, dose, rate
              - Always
              - Injection protocol
            * - R10a, R10l, S0a, S0l 
              - Always
              - Precontrast R1 (:ref:`relaxation-params`) and 
                S0 (:ref:`params-per-sequence`)for aorta and liver 
            * - FA, TR, TS
              - Always
              - :ref:`params-per-sequence`
            * - TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - BAT, CO, Thl, Dhl, To, Eo, Tie, Eb
              - Always
              - :ref:`whole-body-tissues`
            * - H, ve, De
              - Always
              - :ref:`table-liver-models`
            * - khe, khe_i, kh_f, Th, Th_i, Th_f
              - Depends on **stationary**
              - :ref:`table-liver-models`

        .. _AortaLiver-defaults:
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
        non_stationary=None, 
        sequence='SS', 
        free=None, 
        **params,
    ):

        # Configuration
        self.organs = '2cxm' # fixed
        self.kinetics = kinetics
        self.sequence = sequence 
        self.non_stationary = non_stationary
        _check_config(self)

        # Set parameters
        P = (PARAMS | PARAMS_SIGNAL_1 | PARAMS_WHOLE_BODY | PARAMS_SEQUENCE 
                | liver.PARAMS_LIVER)
        
        self.pars = ui.init_parameters(P, self._model_pars(), **params)
        self.free = ui.init_free_parameters(P, self.pars, free)

        # Parameter covariance not known until fit has been done
        self.pcov = None

        # Internal flags
        self._predict = None

    def _sequence_pars(self):
        pars = {
            'SR': ['FA', 'TR', 'TC'],
            'SS': ['FA', 'TR'], 
        }
        return pars[self.sequence]
        
    def _model_pars(self):
        pars = list(PARAMS.keys()) 
        pars += list(PARAMS_SIGNAL_1.keys()) 
        pars += list(PARAMS_WHOLE_BODY.keys())
        pars += self._sequence_pars() + ['TS']
        pars += liver.params_liver(self.kinetics, self.non_stationary)
        pars += ['vol']
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
        P = PARAMS_WHOLE_BODY | liver.PARAMS_LIVER | PARAMS_SIGNAL_1
        # Add derived parameters
        pars = liver.derived_params_liver(self.pars, self.kinetics)
        # Add short name, full name, value, units.
        pars = {
            p: [P[p]['name'], pars[p], P[p]['unit'], 0]
            for p in pars if p in P
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
        return _relax_aorta(self)

    def _predict_aorta(self, xdata: np.ndarray) -> np.ndarray:

        t, R1b = self._relax_aorta()
        #seq = 'SRC' if self.sequence=='SR' else 'SS'
        pars = {k: v for k, v in self.pars.items() if k in self._sequence_pars()}
        signal = sig.signal(self.sequence, R1b, self.pars['S0a'], **pars)
        return utils.sample(xdata, t, signal, self.pars['TS'])

    def _conc_liver(*args, **kwargs):
        return _conc_liver(*args, **kwargs)

    def _relax_liver(*args, **kwargs):
        return _relax_liver(*args, **kwargs)

    def _predict_liver(self, xdata: np.ndarray) -> np.ndarray:
        t, R1l = self._relax_liver()
        pars = {k: v for k, v in self.pars.items() if k in self._sequence_pars()}
        signal = sig.signal(self.sequence, R1l, self.pars['S0l'], **pars)
        return utils.sample(xdata, t, signal, self.pars['TS'])

    def conc(self, sum=True):
        """Concentrations in aorta and liver.

        Args:
            sum (bool, optional): If set to true, the liver concentrations are 
              the sum over both compartments. If set to false, the 
              compartmental concentrations are returned individually. 
              Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, liver 
              concentrations.
        """
        t, cb = self._conc_aorta()
        C = self._conc_liver(sum=sum)
        return t, cb, C

    def relax(self):
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: time points, aorta blood R1, liver 
              R1.
        """
        t, R1b = self._relax_aorta()
        t, R1l = self._relax_liver()
        return t, R1b, R1l

    def predict(self, xdata: tuple) -> tuple:
        """Predict the data at given xdata

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.

        Returns:
            tuple: tuple of 2 arrays with signals for aorta and liver, in 
              that order. The arrays can be different in length and value but 
              each has to have the same length as its corresponding array of 
              time points.
        """
        tmax = max([max(x) for x in xdata])
        if self.pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be predicted with "
                f"the current configuration is {self.pars['tmax']}. "
                f"To predict these data, set tmax = {tmax} (or larger) " 
                f"when creating the AortaLiver2Scan object."
              )
        # Public interface
        if self._predict is None:
            signala = self._predict_aorta(xdata[0])
            signall = self._predict_liver(xdata[1])
            return signala, signall
        # Private interface with different input & output types
        elif self._predict == 'aorta':
            return self._predict_aorta(xdata)
        elif self._predict == 'liver':
            return self._predict_liver(xdata)

    def train(self, xdata: tuple, ydata: tuple, joint=True, n0=1, **kwargs):
        """Train the free parameters

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
            joint (bool, optional): If True, the aorta and liver 
              parameters are trained simultaneously after training 
              them separately. Defaults to True.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        tmax = max([max(x) for x in xdata])
        if self.pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be predicted with "
                f"the current configuration is {self.pars['tmax']}. "
                f"To train on these data, set tmax = {tmax} (or larger) " 
                f"when creating the AortaLiver2Scan object."
              )
        
        # Estimate BAT and S0a from data
        pars = {k: v for k,v in self.pars.items() if k in self._sequence_pars()} 
        Srefb = sig.signal(self.sequence, self.pars['R10a'], 1, **pars)
        Srefl = sig.signal(self.sequence, self.pars['R10l'], 1, **pars)

        self.pars['S0a'] = np.mean(ydata[0][:n0]) / Srefb
        self.pars['S0l'] = np.mean(ydata[1][:n0]) / Srefl
        self.pars['BAT'] = xdata[0][np.argmax(ydata[0])] - (1-self.pars['Dhl'])*self.pars['Thl']
        self.pars['BAT'] = max([self.pars['BAT'], 0])

        # Copy the original free to restore later
        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = list(PARAMS_WHOLE_BODY.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui._train(self, xdata[0], ydata[0], **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        pars = list(liver.PARAMS_LIVER.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui._train(self, xdata[1], ydata[1], **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        if joint:
            return ui._train(self, xdata, ydata, **kwargs)
        else:
            return self


    def plot(self,
             xdata: tuple,
             ydata: tuple,
             xlim=None, ref=None,
             fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
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
        if self.pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be plotted with "
                f"the current configuration is {self.pars['tmax']}. "
                f"To plot these data, set tmax = {tmax} (or larger) " 
                f"when creating the AortaLiver2Scan object."
              )
        
        t, cb, C = self.conc(sum=False)
        sig = self.predict((t, t))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        _plot_data1scan(t, sig[0], xdata[0], ydata[0],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[0])
        _plot_data1scan(t, sig[1], xdata[1], ydata[1],
                        ax3, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[1])
        _plot_conc_aorta(t, cb, ax2, xlim)
        _plot_conc_liver(self, t, C, ax4, xlim)
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def cost(self, xdata, ydata, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
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
        return ui._cost(self, xdata, ydata, metric)
    
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
        free (dict, optional): Dictionary with free parameters and their
          bounds. If not provided, a default set of free parameters is used.
          Defaults to None.
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
        free=None, 
        **params,
      ):

        # Configuration
        self.organs = '2cxm' # fixed
        self.kinetics = kinetics
        self.sequence = sequence 
        self.non_stationary = non_stationary
        _check_config(self)

        P = (PARAMS | PARAMS_SIGNAL_1 | PARAMS_2SCAN | 
             PARAMS_SIGNAL_2 | PARAMS_WHOLE_BODY | PARAMS_SEQUENCE | 
             liver.PARAMS_LIVER)
    
        # Set parameters
        self.pars = ui.init_parameters(P, self._model_pars(), **params)

        # Set free parameters
        self.free = ui.init_free_parameters(P, self.pars, free)

        # Parameter covariance not known until fit has been done
        self.pcov = None

        # Internal flags
        self._predict = None

    def _sequence_pars(self):
        pars = {
            'SR': ['FA', 'TR', 'TC'],
            'SS': ['FA', 'TR'], 
        }  
        return pars[self.sequence]      

    def _model_pars(self):
        pars = list(PARAMS.keys()) 
        pars += list(PARAMS_SIGNAL_1.keys())
        pars += list(PARAMS_2SCAN.keys())
        pars += list(PARAMS_SIGNAL_2.keys())
        pars += list(PARAMS_WHOLE_BODY.keys())
        pars += self._sequence_pars() + ['TS']
        pars += liver.params_liver(self.kinetics, self.non_stationary)
        pars += ['vol']
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
        P = PARAMS_WHOLE_BODY | liver.PARAMS_LIVER | PARAMS_SIGNAL_1 | PARAMS_SIGNAL_2
        # Add derived parameters
        pars = liver.derived_params_liver(self.pars, self.kinetics)
        # Add short name, full name, value, units.
        pars = {
            p: [P[p]['name'], pars[p], P[p]['unit'], 0]
            for p in pars if p in P
        }
        # Add standard deviation
        if self.pcov is not None:
            for i, p in enumerate(self.free):
                pars[p][-1] = np.sqrt(self.pcov[i,i])
        return pars
    
    def _sequence_parameters_first_scan(self):
        if self.sequence == 'SR':
            return {
                'FA': self.pars['FA'], 
                'TR': self.pars['TR'], 
                'TC': self.pars['TC'], 
            }
        if self.sequence == 'SS':
            return {
                'FA': self.pars['FA'], 
                'TR': self.pars['TR'], 
            }
        if self.sequence == 'SRC':
            return {
                'TC': self.pars['TC'], 
            }

    def _sequence_parameters_second_scan(self):
        if self.sequence == 'SR': 
            return {
                'FA': self.pars['FA2'], 
                'TR': self.pars['TR'], 
                'TC': self.pars['TC'], 
            }
        if self.sequence == 'SS':
            return {
                'FA': self.pars['FA2'], 
                'TR': self.pars['TR'], 
            }
        if self.sequence == 'SRC':
            return {
                'TC': self.pars['TC'], 
            }

    def _conc_aorta(self) -> tuple[np.ndarray, np.ndarray]:
        if self.organs == 'comp':
            organs = ['comp', (self.pars['To'],)]
        else:
            organs = ['2cxm', ([self.pars['To'], self.pars['Toe']], self.pars['Eo'])]
        self.t = np.arange(0, self.pars['tmax'], self.pars['dt'])
        conc = lib.ca_conc(self.pars['agent'])
        J1 = lib.ca_injection(
            self.t, self.pars['weight'], conc, self.pars['dose'], self.pars['rate'], self.pars['BAT'])
        J2 = lib.ca_injection(
            self.t, self.pars['weight'], conc, self.pars['dose2'], self.pars['rate'], self.pars['BAT2'])
        Jb = pk_aorta.flux_aorta(
            J1 + J2, E=self.pars['Eb'], dt=self.pars['dt'], tol=self.pars['dose_tolerance'],
            heartlung = ['pfcomp', (self.pars['Thl'], self.pars['Dhl'])],
            organs = organs)
        self.ca = Jb/self.pars['CO']
        return self.t, self.ca
    
    def _relax_aorta(self):
        return _relax_aorta(self)

    def _predict_aorta(self,
                       xdata: tuple[np.ndarray, np.ndarray],
                       ) -> tuple[np.ndarray, np.ndarray]:
        t, R1 = self._relax_aorta()
        t1 = t <= xdata[0][-1]
        t2 = t >= xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        #seq = 'SRC' if self.sequence=='SR' else 'SS'
        #pars = self._par_values(seq=self.sequence)
        pars = self._sequence_parameters_first_scan()
        signal1 = sig.signal(self.sequence, R11, self.pars['S0a'], **pars)
        pars = self._sequence_parameters_second_scan()
        signal2 = sig.signal(self.sequence, R12, self.pars['S02a'], **pars)
        return (
            utils.sample(xdata[0], t[t1], signal1, self.pars['TS']),
            utils.sample(xdata[1], t[t2], signal2, self.pars['TS']),
        )

    def _conc_liver(*args, **kwargs):
        return _conc_liver(*args, **kwargs)

    def _relax_liver(*args, **kwargs):
        return _relax_liver(*args, **kwargs)

    def _predict_liver(self, xdata: tuple[np.ndarray, np.ndarray],
                       ) -> tuple[np.ndarray, np.ndarray]:
        t, R1 = self._relax_liver()
        t1 = t <= xdata[0][-1]
        t2 = t >= xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        #pars = self._par_values(seq=self.sequence)
        pars = self._sequence_parameters_first_scan()
        signal1 = sig.signal(self.sequence, R11, self.pars['S0l'], **pars)
        pars = self._sequence_parameters_second_scan()
        signal2 = sig.signal(self.sequence, R12, self.pars['S02l'], **pars)
        return (
            utils.sample(xdata[0], t[t1], signal1, self.pars['TS']),
            utils.sample(xdata[1], t[t2], signal2, self.pars['TS']),
        )
    
    def conc(self, sum=True):
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
        t, cb = self._conc_aorta()
        C = self._conc_liver(sum=sum)
        return t, cb, C
    
    def relax(self):
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: time points, aorta blood R1, liver R1.
        """
        t, R1b = self._relax_aorta()
        t, R1l = self._relax_liver()
        return t, R1b, R1l

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
        tmax = max([max(x) for x in xdata])
        if self.pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be predicted with "
                f"the current configuration is {self.pars['tmax']}. "
                f"To predict these data, set tmax = {tmax} (or larger) " 
                f"when creating the AortaLiver2Scan object."
              )
        # Public interface
        if self._predict is None:
            signal_a = self._predict_aorta((xdata[0], xdata[1]))
            signal_l = self._predict_liver((xdata[2], xdata[3]))
            return signal_a + signal_l
        # Private interface with different in- and outputs
        elif self._predict == 'aorta':
            return self._predict_aorta(xdata)
        elif self._predict == 'liver':
            return self._predict_liver(xdata)

    def train(self, xdata: tuple, ydata: tuple, joint=True, n0=1, **kwargs):
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
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        tmax = max([max(x) for x in xdata])
        if self.pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be predicted with "
                f"the current configuration is {self.pars['tmax']}. "
                f"To train on these data, set tmax = {tmax} (or larger) " 
                f"when creating the AortaLiver2Scan object."
              )
        # Estimate BAT
        T, D = self.pars['Thl'], self.pars['Dhl']
        self.pars['BAT'] = xdata[0][np.argmax(ydata[0])] - (1-D)*T
        self.pars['BAT2'] = xdata[1][np.argmax(ydata[1])] - (1-D)*T

        # Estimate S0
        pars = self._sequence_parameters_first_scan()
        Srefb = sig.signal(self.sequence, self.pars['R10a'], 1, **pars)
        Srefl = sig.signal(self.sequence, self.pars['R10l'], 1, **pars)
        pars = self._sequence_parameters_second_scan()
        Sref2l = sig.signal(self.sequence, self.pars['R102l'], 1, **pars)
        Sref2b = sig.signal(self.sequence, self.pars['R102a'], 1, **pars)

        self.pars['S0a'] = np.mean(ydata[0][1:n0]) / Srefb
        self.pars['S02a'] = np.mean(ydata[1][1:n0]) / Sref2b
        self.pars['S0l'] = np.mean(ydata[2][1:n0]) / Srefl
        self.pars['S02l'] = np.mean(ydata[3][1:n0]) / Sref2l

        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = list(PARAMS_WHOLE_BODY.keys()) + ['BAT2', 'S02a']
        self.free = {s: free[s] for s in pars if s in free}
        ui._train(self, (xdata[0], xdata[1]), (ydata[0], ydata[1]), **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        pars = list(liver.PARAMS_LIVER.keys()) + ['S02l']
        self.free = {s: free[s] for s in pars if s in free}
        ui._train(self, (xdata[2], xdata[3]), (ydata[2], ydata[3]), **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        if joint:
            return ui._train(self, xdata, ydata, **kwargs)
        else:
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
        if self.pars['tmax'] < tmax:
            raise ValueError(
                f"The largest time point that can be plotted with "
                f"the current configuration is {tmax}. To increase set a larger " 
                f"tmax value when creating the AortaLiver object."
              )
        t, cb, C = self.conc(sum=False)
        ta1 = t[t <= xdata[1][0]]
        ta2 = t[(t > xdata[1][0]) & (t <= xdata[1][-1])]
        tl1 = t[t <= xdata[3][0]]
        tl2 = t[(t > xdata[3][0]) & (t <= xdata[3][-1])]
        sig = self.predict((ta1, ta2, tl1, tl2))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        _plot_data2scan((ta1, ta2), sig[:2], xdata[:2], ydata[:2],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[0])
        _plot_data2scan((tl1, tl2), sig[2:], xdata[2:], ydata[2:],
                        ax3, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[1])
        _plot_conc_aorta(t, cb, ax2, xlim)
        _plot_conc_liver(self, t, C, ax4, xlim)
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
        return ui._cost(self, xdata, ydata, metric)
    
    
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
    





def _relax_aorta(self) -> np.ndarray:
    t, cb = self._conc_aorta()
    rb = lib.relaxivity(self.pars['field_strength'], 'blood', self.pars['agent'])
    return t, self.pars['R10a'] + rb*cb

def _conc_liver(self, sum=True):
    pars = liver.params_liver(self.kinetics, self.non_stationary)
    pars = {k:v for k, v in self.pars.items() if k in pars}
    return liver.conc_liver(self.ca, dt=self.pars['dt'], sum=sum, kinetics=self.kinetics, non_stationary=self.non_stationary, **pars)
    
def _relax_liver(self):
    t = np.arange(0, self.pars['tmax'], self.pars['dt'])
    Cl = self._conc_liver(sum=False)
    rp = lib.relaxivity(self.pars['field_strength'], 'plasma', self.pars['agent'])
    rh = lib.relaxivity(self.pars['field_strength'], 'hepatocytes', self.pars['agent'])
    if 'IC' in self.kinetics:
        return t, self.pars['R10l'] + rp*Cl[0, :] + rh*Cl[1, :]
    else:
        return t, self.pars['R10l'] + rp*Cl




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
                color=color, linewidth=2.0, label='Tissue')
    else:
        ax.plot(t/60, 1000*C, linestyle='-',
                color=color, linewidth=2.0, label='Tissue')        
    ax.legend()

def _plot_data2scan(t: tuple[np.ndarray, np.ndarray], 
                    sig: tuple[np.ndarray, np.ndarray],
                    xdata: tuple[np.ndarray, np.ndarray], 
                    ydata: tuple[np.ndarray, np.ndarray],
                    ax, xlim, color=['black', 'black'], test=None):
    if xlim is None:
        xlim = [0, t[1][-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', 
           xlim=np.array(xlim)/60)
    ax.plot(np.concatenate(xdata)/60, np.concatenate(ydata),
            marker='o', color=color[0], label='fitted data', linestyle='None')
    ax.plot(np.concatenate(t)/60, np.concatenate(sig),
            linestyle='-', color=color[1], linewidth=3.0, label='fit')
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black',
                marker='D', linestyle='None', label='Test data')
    ax.legend()


def _plot_data1scan(t: np.ndarray, sig: np.ndarray,
                    xdata: np.ndarray, ydata: np.ndarray,
                    ax, xlim, color=['black', 'black'],
                    test=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', 
           xlim=np.array(xlim)/60)
    ax.plot(xdata/60, ydata, marker='o',
            color=color[0], label='fitted data', linestyle='None')
    ax.plot(t/60, sig, linestyle='-',
            color=color[1], linewidth=3.0, label='fit')
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black',
                marker='D', linestyle='None', label='Test data')
    ax.legend()



def _check_config(self):
    if self.sequence not in ['SS', 'SR']:
        raise ValueError(
            'Sequence ' + str(self.sequence) + ' is not available.')
    if self.kinetics[0] != '1':
        raise ValueError('Only single-inlet models are allowed.')
    liver.params_liver(self.kinetics, self.non_stationary)



PARAMS = {

    # Prediction and training
    'dt': {
        'init': 0.5,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Forward model time step',
        'unit': 'sec',
    },
    'tmax': {
        'init': 180,
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


    # Injection
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
        'name': 'First contrast agent dose',
        'unit': 'mL/kg',
    },
    'rate': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Contrast agent injection rate',
        'unit': 'mL/sec',
    },
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [0, 1],
        'name': 'Hematocrit',
        'unit': '',
    },

}


PARAMS_SIGNAL_1 = {

    'R10a': {
        'init': 1/lib.T1(3.0, 'blood'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Aorta first baseline R1',
        'unit': 'Hz',
    },
    'S0a': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Aorta first signal scale factor',
        'unit': 'a.u.',
    },
    'R10l': {
        'init': 1/lib.T1(3.0, 'liver'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Liver first baseline R1',
        'unit': 'Hz',
    },
    'S0l': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Liver first signal scale factor',
        'unit': 'a.u.',
    },

}

PARAMS_2SCAN = {
    'dose2': {
        'init': lib.ca_std_dose('gadoxetate')/2,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Second contrast agent dose',
        'unit': 'mL/kg',
    },
    'FA2': {
        'init': 15.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Second flip angle',
        'unit': 'deg',
    },
}

PARAMS_SIGNAL_2 = {
    
    'R102a': {
        'init': 1/lib.T1(3.0, 'blood'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Aorta second baseline R1',
        'unit': 'Hz',
    },
    'S02a': {
        'init': 1,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Aorta second signal scale factor',
        'unit': 'a.u.',
    },
    'R102l': {
        'init': 1/lib.T1(3.0, 'liver'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Liver second baseline R1',
        'unit': 'Hz',
    },
    'S02l': {
        'init': 1,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Liver second signal scale factor',
        'unit': 'a.u.',
    },
    'BAT2': {
        'init': 1200,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Second bolus arrival time',
        'unit': 'sec',
    },
}

PARAMS_WHOLE_BODY = {

    # Body
    'BAT': {
        'init': 60,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'First bolus arrival time',
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


PARAMS_SEQUENCE = {
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
    'TC': {
        'init': 0.180,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Time to center',
        'unit': 'sec',
    },
    'TS': {
        'init': None,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Sampling time',
        'unit': 'sec',
    },
}

