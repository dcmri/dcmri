from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import dcmri.ui as ui
import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.liver as liver


class AortaLiver(ui.Model):
    """Joint model for aorta and liver signals.

    This models uses a whole-body model to predict aorta concentrations, and 
    uses those as input for a liver model. The whole body model assumes the 
    organs can be modelled as a 2-compartment exchange model, and the liver 
    is modelled with a single-inlet dispersion model for intracellular agent. 

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
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
        ...     t0 = 10,
        ...     TR = 0.005,
        ...     FA = 15,
        ... )

        Train the model on the data:

        >>> model.train(xdata, ydata, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare 
        against the experimentally derived data:

        >>> model.plot(xdata, ydata)

        We can also have a look at the model parameters after training:

        >>> model.print_params(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Bolus arrival time (BAT): 17.127 (2.838) sec
        Cardiac output (CO): 211.047 (13.49) mL/sec
        Heart-lung mean transit time (Thl): 12.503 (3.527) sec
        Heart-lung transit time dispersion (Dhl): 0.465 (0.063)
        Organs mean transit time (To): 28.413 (10.294) sec
        Extraction fraction (Eb): 0.01 (0.623)
        Liver extracellular mean transit time (Te): 2.915 (0.588) sec
        Liver extracellular dispersion (De): 1.0 (0.19)
        Liver extracellular volume fraction (ve): 0.176 (0.015) mL/cm3
        Hepatocellular uptake rate (khe): 0.005 (0.001) mL/sec/cm3
        Hepatocellular transit time (Th): 600.0 (747.22) sec
        Organs extraction fraction (Eo): 0.261 (0.324)
        Organs extracellular mean transit time (Toe): 81.765 (329.097) sec
        ------------------
        Derived parameters
        ------------------
        Blood precontrast T1 (T10a): 1.629 sec
        Mean circulation time (Tc): 40.916 sec
        Liver precontrast T1 (T10l): 0.752 sec
        Biliary excretion rate (kbh): 0.001 mL/sec/cm3
        Hepatocellular tissue uptake rate (Khe): 0.026 mL/sec/cm3
        Biliary tissue excretion rate (Kbh): 0.002 mL/sec/cm3

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
            * - t0, dose_tolerance
              - Always
              - For estimating baseline signal
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

    def __init__(self, stationary='UE', sequence='SS', free=None, **params):

        # Configuration
        self.organs = '2cxm' # fixed
        self.kinetics = '1I-IC-D' # fixed
        self.sequence = sequence 
        self.stationary = stationary

        self._check_config()
        self._set_defaults(free=free, **params)

        # Internal flags
        self._predict = None

    def _check_config(self):
        _check_config(self)

    def _params(self):
        return (PARAMS | PARAMS_WHOLE_BODY | PARAMS_SEQUENCE 
                | PARAMS_LIVER | PARAMS_DERIVED)
    
    def _model_pars(self):
        pars_sequence = {
            'SR': ['FA', 'TR', 'TS', 'TC'],
            'SS': ['FA', 'TR', 'TS'], 
        }
        pars = list(PARAMS.keys()) 
        pars += list(PARAMS_WHOLE_BODY.keys())
        pars += pars_sequence[self.sequence]
        pars += liver.params_liver(self.kinetics, self.stationary)
        pars += ['vol']
        return pars
    
    def _par_values(*args, **kwargs):
        return _par_values(*args, **kwargs)

    def _conc_aorta(self) -> np.ndarray:
        if self.organs == 'comp':
            organs = ['comp', (self.To,)]
        else:
            organs = ['2cxm', ([self.To, self.Toe], self.Eo)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = lib.ca_conc(self.agent)
        Ji = lib.influx_step(
            self.t, self.weight, conc, self.dose, self.rate, self.BAT)
        Jb = pk_aorta.flux_aorta(
            Ji, E=self.Eb, dt=self.dt, tol=self.dose_tolerance,
            heartlung = ['pfcomp', (self.Thl, self.Dhl)],
            organs = organs)
        self.ca = Jb/self.CO
        return self.t, self.ca

    def _relax_aorta(self):
        return _relax_aorta(self)

    def _predict_aorta(self, xdata: np.ndarray) -> np.ndarray:
        self.tmax = max(xdata)+self.dt
        if self.TS is not None:
            self.tmax += self.TS
        t, R1b = self._relax_aorta()
        #seq = 'SRC' if self.sequence=='SR' else 'SS'
        pars = self._par_values(seq=self.sequence)
        signal = sig.signal(self.sequence, R1b, self.S0a, **pars)
        # if self.sequence == 'SR':
        #     # signal = sig.signal_src(R1b, self.S0a, self.TC, R10=self.R10a)
        #     signal = sig.signal_src(R1b, self.S0a, self.TC)
        # else:
        #     # signal = sig.signal_ss(R1b, self.S0a, self.TR, self.FA, R10=self.R10a)
        #     signal = sig.signal_ss(R1b, self.S0a, self.TR, self.FA)
        return utils.sample(xdata, t, signal, self.TS)

    def _conc_liver(*args, **kwargs):
        return _conc_liver(*args, **kwargs)

    def _relax_liver(*args, **kwargs):
        return _relax_liver(*args, **kwargs)

    def _predict_liver(self, xdata: np.ndarray) -> np.ndarray:
        t, R1l = self._relax_liver()
        pars = self._par_values(seq=self.sequence)
        signal = sig.signal(self.sequence, R1l, self.S0l, **pars)
        # if self.sequence == 'SR':
        #     # signal = sig.signal_sr(R1l, self.S0l, self.TR, self.FA, self.TC, R10=R1l[0])
        #     signal = sig.signal_sr(R1l, self.S0l, self.TR, self.FA, self.TC)
        # else:
        #     # signal = sig.signal_ss(R1l, self.S0l, self.TR, self.FA, R10=R1l[0])
        #     signal = sig.signal_ss(R1l, self.S0l, self.TR, self.FA)
        return utils.sample(xdata, t, signal, self.TS)

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
            tuple: time points, aorta blood concentrations, liver 
              concentrations.
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
              that order. The arrays can be different in length and value but each has to have the same length as its corresponding array of time points.
        """
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

    def train(self, xdata: tuple, ydata: tuple, **kwargs):
        """Train the free parameters

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        # Estimate BAT and S0a from data
        pars = self._par_values(seq=self.sequence)
        Srefb = sig.signal(self.sequence, self.R10a, 1, **pars)
        Srefl = sig.signal(self.sequence, self.R10l, 1, **pars)
        # if self.sequence == 'SR':
        #     Srefb = sig.signal_sr(self.R10a, 1, self.TR, self.FA, self.TC)
        #     Srefl = sig.signal_sr(self.R10l, 1, self.TR, self.FA, self.TC)
        # else:
        #     Srefb = sig.signal_ss(self.R10a, 1, self.TR, self.FA)
        #     Srefl = sig.signal_ss(self.R10l, 1, self.TR, self.FA)

        n0 = max([np.sum(xdata[0] < self.t0), 1])
        self.S0a = np.mean(ydata[0][:n0]) / Srefb
        self.S0l = np.mean(ydata[1][:n0]) / Srefl
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-self.Dhl)*self.Thl
        self.BAT = max([self.BAT, 0])

        # Copy the original free to restore later
        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = list(PARAMS_WHOLE_BODY.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui.train(self, xdata[0], ydata[0], **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        pars = list(PARAMS_LIVER.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui.train(self, xdata[1], ydata[1], **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        return ui.train(self, xdata, ydata, **kwargs)

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
        _plot_conc_liver(t, C, ax4, xlim)
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
            metric (str, optional): Which metric to use - options are: 
                **RMS** (Root-mean-square);
                **NRMS** (Normalized root-mean-square); 
                **AIC** (Akaike information criterion); 
                **cAIC** (Corrected Akaike information criterion for small 
                  models);
                **BIC** (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        return super().cost(xdata, ydata, metric)


class AortaLiver2scan(ui.Model):
    """Joint model for aorta and liver signals measured over two scans.

    This models uses a whole-body model to predict aorta concentrations, and 
    uses those as input for a liver model. The whole body model assumes the 
    organs can be modelled as a 2-compartment exchange model, and the liver 
    is modelled with a single-inlet dispersion model for intracellular agent. 

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
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
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     dose2 = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     t0 = 10,
        ...     TR = 0.005,
        ...     FA = 15,
        ... )

        Train the model on the data:

        >>> model.train(xdata, ydata, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare against 
        the experimentally derived data:

        >>> model.plot(xdata, ydata)

        We can also have a look at the model parameters after training:

        >>> model.print_params(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)  
        -----------------------------------------  
        Bolus arrival time (BAT): 17.13 (1.771) sec
        Cardiac output (CO): 208.547 (9.409) mL/sec
        Heart-lung mean transit time (Thl): 12.406 (2.137) sec
        Heart-lung transit time dispersion (Dhl): 0.459 (0.04)
        Organs mean transit time (To): 30.912 (3.999) sec
        Extraction fraction (Eb): 0.064 (0.032)
        Liver extracellular mean transit time (Te): 2.957 (0.452) sec
        Liver extracellular dispersion (De): 1.0 (0.146)
        Liver extracellular volume fraction (ve): 0.077 (0.007) mL/cm3
        Hepatocellular uptake rate (khe): 0.002 (0.001) mL/sec/cm3
        Hepatocellular transit time (Th): 600.0 (1173.571) sec
        Organs extraction fraction (Eo): 0.2 (0.057)
        Organs extracellular mean transit time (Toe): 87.077 (56.882) sec
        Hepatocellular uptake rate (final) (khe_f): 0.001 (0.001) mL/sec/cm3
        Hepatocellular transit time (final) (Th_f): 600.0 (623.364) sec
        ------------------
        Derived parameters
        ------------------
        Blood precontrast T1 (T10a): 1.629 sec
        Mean circulation time (Tc): 43.318 sec
        Liver precontrast T1 (T10l): 0.752 sec
        Biliary excretion rate (kbh): 0.002 mL/sec/cm3
        Hepatocellular tissue uptake rate (Khe): 0.023 mL/sec/cm3
        Biliary tissue excretion rate (Kbh): 0.002 mL/sec/cm3
        Hepatocellular uptake rate (initial) (khe_i): 0.003 mL/sec/cm3
        Hepatocellular transit time (initial) (Th_i): 600.0 sec
        Hepatocellular uptake rate variance (khe_var): 0.934
        Biliary tissue excretion rate variance (Kbh_var): 0.0
        Biliary excretion rate (initial) (kbh_i): 0.002 mL/sec/cm3
        Biliary excretion rate (final) (kbh_f): 0.002 mL/sec/cm3

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
            * - t0, dose_tolerance
              - Always
              - For estimating baseline signal
            * - field_strength, weight, agent, dose, dose2, rate
              - Always
              - Injection protocol
            * - R10a, R102a, R10l, R102l, S0a, S02a, S0l, S02l
              - Always
              - Precontrast R1 (:ref:`relaxation-params`) and 
                S0 (:ref:`params-per-sequence`) for aorta and liver 
            * - FA, TR, TS
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

    def __init__(self, stationary=None, sequence='SS', free=None, **params):

        # Configuration
        self.organs = '2cxm' # fixed
        self.kinetics = '1I-IC-D' # fixed
        self.sequence = sequence 
        self.stationary = stationary

        self._check_config()
        self._set_defaults(free=free, **params)

        # Internal flags
        self._predict = None

    def _check_config(self):
        _check_config(self)
        
    def _params(self):
        return (PARAMS | PARAMS_2SCAN | PARAMS_WHOLE_BODY | PARAMS_SEQUENCE 
                | PARAMS_LIVER | PARAMS_DERIVED)

    def _model_pars(self):
        pars_sequence = {
            'SR': ['FA', 'TR', 'TS', 'TC'],
            'SS': ['FA', 'TR', 'TS'], 
        }
        pars = list(PARAMS.keys()) 
        pars += list(PARAMS_2SCAN.keys())
        pars += list(PARAMS_WHOLE_BODY.keys())
        pars += pars_sequence[self.sequence]
        pars += liver.params_liver(self.kinetics, self.stationary)
        pars += ['vol']
        return pars
    
    def _par_values(*args, **kwargs):
        return _par_values(*args, **kwargs)

    def _conc_aorta(self) -> tuple[np.ndarray, np.ndarray]:
        if self.organs == 'comp':
            organs = ['comp', (self.To,)]
        else:
            organs = ['2cxm', ([self.To, self.Toe], self.Eo)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = lib.ca_conc(self.agent)
        J1 = lib.influx_step(
            self.t, self.weight, conc, self.dose, self.rate, self.BAT)
        J2 = lib.influx_step(
            self.t, self.weight, conc, self.dose2, self.rate, self.BAT2)
        Jb = pk_aorta.flux_aorta(
            J1 + J2, E=self.Eb, dt=self.dt, tol=self.dose_tolerance,
            heartlung = ['pfcomp', (self.Thl, self.Dhl)],
            organs = organs)
        self.ca = Jb/self.CO
        return self.t, self.ca
    
    def _relax_aorta(self):
        return _relax_aorta(self)

    def _predict_aorta(self,
                       xdata: tuple[np.ndarray, np.ndarray],
                       ) -> tuple[np.ndarray, np.ndarray]:
        self.tmax = max(xdata[1])+self.dt
        if self.TS is not None:
            self.tmax += self.TS
        t, R1 = self._relax_aorta()
        t1 = t <= xdata[0][-1]
        t2 = t >= xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        #seq = 'SRC' if self.sequence=='SR' else 'SS'
        pars = self._par_values(seq=self.sequence)
        signal1 = sig.signal(self.sequence, R11, self.S0a, **pars)
        signal2 = sig.signal(self.sequence, R12, self.S02a, **pars)
        # if self.sequence == 'SR':
        #     signal1 = sig.signal_sr(R11, self.S0a, self.TR, self.FA, self.TC)
        #     signal2 = sig.signal_sr(R12, self.S02a, self.TR, self.FA, self.TC)
        # else:
        #     signal1 = sig.signal_ss(R11, self.S0a, self.TR, self.FA)
        #     signal2 = sig.signal_ss(R12, self.S02a, self.TR, self.FA)
        return (
            utils.sample(xdata[0], t[t1], signal1, self.TS),
            utils.sample(xdata[1], t[t2], signal2, self.TS),
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
        pars = self._par_values(seq=self.sequence)
        signal1 = sig.signal(self.sequence, R11, self.S0l, **pars)
        signal2 = sig.signal(self.sequence, R12, self.S02l, **pars)
        # if self.sequence == 'SR':
        #     signal1 = sig.signal_sr(R11, self.S0l, self.TR, self.FA, self.TC)
        #     signal2 = sig.signal_sr(R12, self.S02l, self.TR, self.FA, self.TC)
        # else:
        #     signal1 = sig.signal_ss(R11, self.S0l, self.TR, self.FA)
        #     signal2 = sig.signal_ss(R12, self.S02l, self.TR, self.FA)
        return (
            utils.sample(xdata[0], t[t1], signal1, self.TS),
            utils.sample(xdata[1], t[t2], signal2, self.TS),
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

    def train(self, xdata: tuple, ydata: tuple, **kwargs):
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
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        # Estimate BAT
        T, D = self.Thl, self.Dhl
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-D)*T
        self.BAT2 = xdata[1][np.argmax(ydata[1])] - (1-D)*T

        # Estimate S0
        pars = self._par_values(seq=self.sequence)
        Srefb = sig.signal(self.sequence, self.R10a, 1, **pars)
        Sref2b = sig.signal(self.sequence, self.R102a, 1, **pars)
        Srefl = sig.signal(self.sequence, self.R10l, 1, **pars)
        Sref2l = sig.signal(self.sequence, self.R102l, 1, **pars)

        # # Estimate S0
        # if self.sequence == 'SR':
        #     Srefb = sig.signal_sr(self.R10a, 1, self.TR, self.FA, self.TC)
        #     Sref2b = sig.signal_sr(self.R102a, 1, self.TR, self.FA, self.TC)
        #     Srefl = sig.signal_sr(self.R10l, 1, self.TR, self.FA, self.TC)
        #     Sref2l = sig.signal_sr(self.R102l, 1, self.TR, self.FA, self.TC)
        # else:
        #     Srefb = sig.signal_ss(self.R10a, 1, self.TR, self.FA)
        #     Sref2b = sig.signal_ss(self.R102a, 1, self.TR, self.FA)
        #     Srefl = sig.signal_ss(self.R10l, 1, self.TR, self.FA)
        #     Sref2l = sig.signal_ss(self.R102l, 1, self.TR, self.FA)

        n0 = max([np.sum(xdata[0] < self.t0), 2])
        self.S0a = np.mean(ydata[0][1:n0]) / Srefb
        self.S02a = np.mean(ydata[1][1:n0]) / Sref2b
        self.S0l = np.mean(ydata[2][1:n0]) / Srefl
        self.S02l = np.mean(ydata[3][1:n0]) / Sref2l

        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = list(PARAMS_WHOLE_BODY.keys()) + ['BAT2', 'S02a']
        self.free = {s: free[s] for s in pars if s in free}
        ui.train(self, (xdata[0], xdata[1]), (ydata[0], ydata[1]), **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        pars = list(PARAMS_LIVER.keys()) + ['S02l']
        self.free = {s: free[s] for s in pars if s in free}
        # added if s in free - add everywhere after testing
        ui.train(self, (xdata[2], xdata[3]), (ydata[2], ydata[3]), **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        return ui.train(self, xdata, ydata, **kwargs)
    

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
        _plot_conc_liver(t, C, ax4, xlim)
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
            metric (str, optional): Which metric to use - options are: 
                **RMS** (Root-mean-square);
                **NRMS** (Normalized root-mean-square); 
                **AIC** (Akaike information criterion); 
                **cAIC** (Corrected Akaike information criterion for small 
                  models);
                **BIC** (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        return super().cost(xdata, ydata, metric)


def _relax_aorta(self) -> np.ndarray:
    t, cb = self._conc_aorta()
    rb = lib.relaxivity(self.field_strength, 'blood', self.agent)
    return t, self.R10a + rb*cb

def _conc_liver(self, sum=True):
    pars = self._par_values(kin=True)
    return liver.conc_liver(self.ca, dt=self.dt, sum=sum, **pars)
    
def _relax_liver(self):
    t = np.arange(0, self.tmax, self.dt)
    Cl = self._conc_liver(sum=False)
    rp = lib.relaxivity(self.field_strength, 'plasma', self.agent)
    rh = lib.relaxivity(self.field_strength, 'hepatocytes', self.agent)
    return t, self.R10l + rp*Cl[0, :] + rh*Cl[1, :]




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

def _plot_conc_liver(t: np.ndarray, C: np.ndarray, ax, xlim=None):
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
        xlim = [0, t[1][-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
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
    liver.params_liver(self.kinetics, self.stationary)


def _par_values(self, kin=False, export=False, seq=None):

    if seq is not None:
        pars = {
            'SR': ['FA', 'TR', 'TC'],
            'SS': ['FA', 'TR'], 
            'SRC': ['TC'],
        }
        pars = pars[self.sequence]
        return {par: getattr(self, par) for par in pars}
    
    if kin:
        pars = liver.params_liver(self.kinetics, self.stationary)
        return {par: getattr(self, par) for par in pars}
    
    if export:
        pars = self._par_values()
        p0 = self._model_pars()
        p1 = liver.params_liver(self.kinetics, self.stationary)
        p2 = PARAMS_WHOLE_BODY.keys()
        discard = set(p0) - set(p1) - set(p2) - {'S02a', 'S02l', 'BAT2'}
        return {p: pars[p] for p in pars if p not in discard}
    
    pars = self._model_pars()
    p = {par: getattr(self, par) for par in pars}

    try:
        p['Kbh'] = _div(1, p['Th'])
    except KeyError:
        pass
    try:
        p['Kbh'] = np.mean([
            _div(1, p['Th_i']), 
            _div(1, p['Th_f']),
        ])
    except KeyError:
        pass
    try:
        p['Th'] = np.mean([p['Th_i'], p['Th_f']])
    except KeyError:
        pass
    try:
        p['khe'] = np.mean([p['khe_i'], p['khe_f']])
    except KeyError:
        pass
    try:
        p['Khe'] = _div(p['khe'], p['ve'])
    except KeyError:
        pass
    try:
        p['kbh'] = (1-p['ve'])*p['Kbh']
    except KeyError:
        pass
    if p['vol'] is not None:
        try:
            p['CL'] = _div(p['khe'], p['vol'])
        except KeyError:
            pass
        
    return p


PARAMS = {

    # Prediction and training
    't0': {
        'init': 0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Baseline duration',
        'unit': 'sec',
    },
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


    # Signal
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

PARAMS_LIVER = {
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [0, 1],
        'name': 'Hematocrit',
        'unit': '',
    },
    'Te': {
        'init': 30.0,
        'default_free': True,
        'bounds': [0.1, 60],
        'name': 'Extracellular mean transit time',
        'unit': 'sec',
    },
    'De': {
        'init': 0.85,
        'default_free': True,
        'bounds': [0, 1],
        'name': 'Extracellular dispersion',
        'unit': '',
    },
    've': {
        'init': 0.3,
        'default_free': True,
        'bounds': [0.01, 0.6],
        'name': 'Liver extracellular volume fraction',
        'unit': 'mL/cm3',
    },
    'khe': {
        'init': 0.003,
        'default_free': True,
        'bounds': [0.0, 0.1],
        'name': 'Hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
    },
    'khe_i': {
        'init': 0.003,
        'default_free': True,
        'bounds': [0.0, 0.1],
        'name': 'Initial hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
    },
    'khe_f': {
        'init': 0.003,
        'default_free': True,
        'bounds': [0.0, 0.1],
        'name': 'Final hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
    },
    'Th': {
        'init': 30*60,
        'default_free': True,
        'bounds': [10*60, 10*60*60],
        'name': 'Hepatocellular mean transit time',
        'unit': 'sec',
    },
    'Th_i': {
        'init': 30*60,
        'default_free': True,
        'bounds': [10*60, 10*60*60],
        'name': 'Initial hepatocellular mean transit time',
        'unit': 'sec',
    },
    'Th_f': {
        'init': 30*60,
        'default_free': True,
        'bounds': [10*60, 10*60*60],
        'name': 'Final hepatocellular mean transit time',
        'unit': 'sec',
    },
    'vol': {
        'init': None,
        'default_free': False,
        'bounds': [0, 10000],
        'name': 'Liver volume',
        'unit': 'cm3',
    },
}

PARAMS_DERIVED = {
    'kbh': {
        'name': 'Biliary excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'kbh': {
        'name': 'Biliary excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'Kbh': {
        'name': 'Biliary tissue excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'Khe': {
        'name': 'Hepatocellular tissue uptake rate',
        'unit': 'mL/sec/cm3',
    },
    'CL': {
        'name': 'Liver blood clearance',
        'unit': 'mL/sec',
    }, 
}


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(a, b)