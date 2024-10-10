from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc


class Aorta(dc.Model):
    """Whole-body model for aorta signal.

    The model represents the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a chain compartment and the organs are modelled as a two-compartment exchange model. Bolus injection into the system is modelled as a step function.

        **Injection parameters**

        - **weight** (float, default=70): Subject weight in kg.
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **dose** (float, default=0.2): Injected contrast agent dose in mL per kg bodyweight.
        - **rate** (float, default=1): Contrast agent injection rate in mL per sec.

        **Acquisition parameters**

        - **sequence** (str, default='SS'): Signal model.
        - **tmax** (float, default=120): Maximum acquisition time in sec.
        - **field_strength** (float, default=3.0): Magnetic field strength in T. 
        - **t0** (float, default=1): Baseline length in secs.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec. 
        - **FA** (float, default=15): Nominal flip angle in degrees.
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence (sec).

        **Signal parameters**

        - **R10** (float, default=1): Precontrast arterial relaxation rate in 1/sec. 
        - **S0** (float, default=1): scale factor for the arterial MR signal in the first scan. 

        **Whole body kinetic parameters**

        - **heartlung** (str, default='pfcomp'): Kinetic model for the heart-lung system (either 'pfcomp' or 'chain').
        - **organs** (str, default='2cxm'): Kinetic model for the organs.
        - **BAT** (float, default=60): Bolus arrival time, i.e. time point where the indicator first arrives in the body. 
        - **BAT2** (float, default=1200): Bolus arrival time in the second scan, i.e. time point where the indicator first arrives in the body. 
        - **CO** (float, default=100): Cardiac output in mL/sec.
        - **Thl** (float, default=10): Mean transit time through heart and lungs.
        - **Dhl** (float, default=0.2): Dispersion through the heart-lung system, with a value in the range [0,1].
        - **To** (float, default=20): average time to travel through the organ's vasculature.
        - **Eb** (float, default=0.05): fraction of indicator extracted from the vasculature in a single pass. 
        - **Eo** (float, default=0.15): Fraction of indicator entering the organs which is extracted from the blood pool.
        - **Teb** (float, default=120): Average time to travel through the organs extravascular space.

        **Prediction and training parameters**

        - **dt** (float, default=1): Internal time resolution of the AIF in sec. 
        - **dose_tolerance** (fload, default=0.1): Stopping criterion for the whole-body model.
        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **free** (array-like): 2-element list with lower and upper free of the free parameters. The default depends on the kinetics parameter.

    Args:
        params (dict, optional): override defaults for any of the parameters.

    See Also:
        `AortaLiver`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, _, gt = dc.fake_tissue()

        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_):

        >>> aorta = dc.Aorta(
        ...     dt = 1.5,
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 15,
        ...     R10 = 1/dc.T1(3.0,'blood'),
        ... )

        Train the model on the data:

        >>> aorta.train(time, aif)

        Plot the reconstructed signals and concentrations and compare against the experimentally derived data:

        >>> aorta.plot(time, aif)

        We can also have a look at the model parameters after training:

        >>> aorta.print_params(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Bolus arrival time (BAT): 18.485 (5.656) sec
        Cardiac output (CO): 228.237 (29.321) mL/sec
        Heart-lung mean transit time (Thl): 9.295 (6.779) sec
        Heart-lung transit time dispersion (Dhl): 0.459 (0.177)
        Organs mean transit time (To): 29.225 (11.646) sec
        Extraction fraction (Eb): 0.013 (0.972)
        Organs extraction fraction (Eo): 0.229 (0.582)
        Extracellular mean transit time (Te): 97.626 (640.454) sec
        ------------------
        Derived parameters
        ------------------
        Mean circulation time (Tc): 38.521 sec

        *Note*: The extracellular mean transit time has a high error, indicating that the acquisition time here is insufficient to resolve the transit through the leakage space.
    """

    free = {}   #: lower- and upper free for all free parameters.

    def __init__(self, organs='2cxm', heartlung='pfcomp', sequence='SS', **params):

        # Define model
        self.organs = organs
        self.heartlung = heartlung
        self.sequence = sequence

        #
        # Set defaults for all parameters
        #

        self.dt = 0.5
        self.tmax = 120
        self.dose_tolerance = 0.1
        self.weight = 70.0
        self.agent = 'gadoterate'
        self.dose = dc.ca_std_dose('gadoterate')
        self.rate = 1
        self.field_strength = 3.0
        self.R10 = 0.7
        self.t0 = 0

        self.TR = 0.005
        self.FA = 15.0
        self.TC = 0.180
        self.TS = None

        self.S0 = 1
        self.BAT = 60.0
        self.CO = 100
        self.Thl = 10
        self.Dhl = 0.2
        self.To = 20
        self.Eb = 0.05
        self.Eo = 0.15
        self.Te = 120

        # TODO: preset free depending on model options
        self.free = {
            'BAT': [0, np.inf],
            'CO': [0, np.inf],
            'Thl': [0, 30],
            'Dhl': [0.05, 0.95],
            'To': [0, 60],
            'Eb': [0.01, 0.15],
            'Eo': [0, 0.5],
            'Te': [0, 800],
        }

        # overide defaults
        for k, v in params.items():
            setattr(self, k, v)

    def conc(self):
        """Aorta blood concentration

        Args:
            t (array-like): Time points of the concentration (sec)

        Returns:
            numpy.ndarray: Concentration in M
        """
        if self.organs == 'comp':
            organs = ['comp', (self.To,)]
        else:
            organs = ['2cxm', ([self.To, self.Te], self.Eo)]
        t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight,
                            conc, self.dose, self.rate, self.BAT)
        Jb = dc.flux_aorta(Ji, E=self.Eb,
                           heartlung=[self.heartlung, (self.Thl, self.Dhl)],
                           organs=organs, dt=self.dt, tol=self.dose_tolerance)
        return t, Jb/self.CO

    def relax(self):
        """Aorta longitudinal relation rate

        Args:
            t (array-like): Time points of the concentration (sec)

        Returns:
            numpy.ndarray: Concentration in M
        """
        t, cb = self.conc()
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        return t, self.R10 + rp*cb

    def predict(self, xdata) -> np.ndarray:
        tacq = xdata[1]-xdata[0]
        self.tmax = max(xdata)+tacq+self.dt
        t, R1 = self.relax()
        if self.sequence == 'SR':
            # signal = dc.signal_src(R1, self.S0, self.TC, R10=self.R10)
            signal = dc.signal_src(R1, self.S0, self.TC)
        else:
            # signal = dc.signal_ss(R1, self.S0, self.TR, self.FA, R10=self.R10)
            signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, t, signal, self.TS)

    def train(self, xdata, ydata, **kwargs):
        n0 = max([np.sum(xdata < self.t0), 1])
        self.BAT = xdata[np.argmax(ydata)] - self.Thl
        if self.sequence == 'SR':
            Sref = dc.signal_src(self.R10, 1, self.TC)
        else:
            Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:n0]) / Sref
        return dc.train(self, xdata, ydata, **kwargs)

    def export_params(self):
        pars = {}
        pars['BAT'] = ['Bolus arrival time', self.BAT, "sec"]
        pars['CO'] = ['Cardiac output', self.CO, "mL/sec"]
        pars['Thl'] = ['Heart-lung mean transit time', self.Thl, "sec"]
        pars['Dhl'] = ['Heart-lung transit time dispersion', self.Dhl, ""]
        pars['To'] = ["Organs mean transit time", self.To, "sec"]
        pars['Eb'] = ["Extraction fraction", self.Eb, ""]
        pars['Tc'] = ["Mean circulation time", self.Thl+self.To, 'sec']
        pars['Eo'] = ["Organs extraction fraction", self.Eo, ""]
        pars['Te'] = ["Extracellular mean transit time", self.Te, "sec"]
        # pars['S0'] = ["Baseline", self.S0, "au"]
        return self._add_sdev(pars)

    def plot(self, xdata: np.ndarray, ydata: np.ndarray,
             ref=None, xlim=None, fname=None, show=True):
        aif = self.predict(xdata)
        t, cb = self.conc()
        if xlim is None:
            xlim = [np.amin(xdata), np.amax(xdata)]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(xdata/60, ydata, 'ro', label='Measurement')
        ax0.plot(xdata/60, aif, 'b-', label='Prediction')
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('MRI signal (a.u.)')
        ax0.legend()
        ax1.set_title('Prediction of the concentrations.')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['cb'], 'ro', label='Ground truth')
        ax1.plot(t/60, 1000*cb, 'b-', label='Prediction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Blood concentration (mM)')
        ax1.legend()
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


class AortaLiver(dc.Model):
    """Joint model for aorta and liver signals.

    The model represents the liver as a two-compartment system and the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a chain compartment and the organs are modelled as a two-compartment exchange model. Bolus injection into the system is modelled as a step function.

        **Injection parameters**

        - **weight** (float, default=70): Subject weight in kg.
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **dose** (float, default=0.2): Injected contrast agent dose in mL per kg bodyweight.
        - **rate** (float, default=1): Contrast agent injection rate in mL per sec.

        **Acquisition parameters**

        - **sequence** (str, default='SS'): Signal model.
        - **tmax** (float, default=120): Maximum acquisition time in sec.
        - **tacq** (float, default=None): Time to acquire a single dynamic in the first scan (sec). If this is not provided, tacq is taken from the difference between the first two data points.
        - **field_strength** (float, default=3.0): Magnetic field strength in T. 
        - **t0** (float, default=1): Baseline length in secs.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec. 
        - **FA** (float, default=15): Nominal flip angle in degrees.
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence.

        **Signal parameters**

        - **R10b** (float, default=1): Precontrast arterial relaxation rate in 1/sec.  
        - **R10l** (float, default=1): Baseline R1 for the liver.
        - **S0b** (float, default=1): scale factor for the arterial MR signal in the first scan.
        - **S0l** (float, default=1): Scale factor for the liver signal.   

        **Whole body kinetic parameters**

        - **organs** (str, default='2cxm'): Kinetic model for the organs.
        - **BAT** (float, default=60): Bolus arrival time, i.e. time point where the indicator first arrives in the body. 
        - **BAT2** (float, default=1200): Bolus arrival time in the second scan, i.e. time point where the indicator first arrives in the body. 
        - **CO** (float, default=100): Cardiac output in mL/sec.
        - **Thl** (float, default=10): Mean transit time through heart and lungs.
        - **Dhl** (float, default=0.2): Dispersion through the heart-lung system, with a value in the range [0,1].
        - **To** (float, default=20): average time to travel through the organ's vasculature.
        - **Eb** (float, default=0.05): fraction of indicator extracted from the vasculature in a single pass. 
        - **Eo** (float, default=0.15): Fraction of indicator entering the organs which is extracted from the blood pool.
        - **Teb** (float, default=120): Average time to travel through the organs extravascular space.

        **Liver kinetic parameters**

        - **kinetics** (str, default='stationary'). Liver kinetic model, either stationary or non-stationary.
        - **Hct** (float, default=0.45): Hematocrit.
        - **Tel** (float, default=30): Mean transit time for extracellular space of liver and gut.
        - **De** (float, default=0.85): Dispersion in the extracellular space of liver an gut, in the range [0,1].
        - **ve** (float, default=0.3): Liver extracellular volume fraction.
        - **khe** (float, default=0.003): Hepatocellular uptake rate (mL/sec/cm3).
        - **Th** (float, default=1800): Hepatocellular transit time.
        - **khe_f** (float, default=0.003): Final hepatocellular uptake rate (non-stationary models) (mL/sec/cm3).
        - **Th_f** (float, default=1800): Final hepatocellular transit time (non-stationary models).
        - **vol** (float, default=None): liver volume in mL (for whole liver export parameters).

        **Prediction and training parameters**

        - **dt** (float, default=1): Internal time resolution of the AIF in sec. 
        - **dose_tolerance** (fload, default=0.1): Stopping criterion for the whole-body model.
        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **free** (array-like): 2-element list with lower and upper free of the free parameters. The default depends on the kinetics parameter.

    Args:
        params (dict, optional): override defaults for any of the parameters.

    See Also:
        `AortaLiver`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, roi, gt = dc.fake_tissue()
        >>> xdata, ydata = (time,time), (aif,roi)

        Build an aorta-liver model and parameters to match the conditions of the fake tissue data:

        >>> model = dc.AortaLiver(
        ...     dt = 0.5,
        ...     tmax = 180,
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     t0 = 10,
        ...     TR = 0.005,
        ...     FA = 15,
        ... )

        Train the model on the data:

        >>> model.train(xdata, ydata, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare against the experimentally derived data:

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
        Liver extracellular mean transit time (Tel): 2.915 (0.588) sec
        Liver extracellular dispersion (De): 1.0 (0.19)
        Liver extracellular volume fraction (ve): 0.176 (0.015) mL/cm3
        Hepatocellular uptake rate (khe): 0.005 (0.001) mL/sec/cm3
        Hepatocellular transit time (Th): 600.0 (747.22) sec
        Organs extraction fraction (Eo): 0.261 (0.324)
        Organs extracellular mean transit time (Teb): 81.765 (329.097) sec
        ------------------
        Derived parameters
        ------------------
        Blood precontrast T1 (T10b): 1.629 sec
        Mean circulation time (Tc): 40.916 sec
        Liver precontrast T1 (T10l): 0.752 sec
        Biliary excretion rate (kbh): 0.001 mL/sec/cm3
        Hepatocellular tissue uptake rate (Khe): 0.026 mL/sec/cm3
        Biliary tissue excretion rate (Kbh): 0.002 mL/sec/cm3

    """

    def __init__(self, kinetics='stationary', organs='2cxm', sequence='SS', **params):

        # Configuration
        self.sequence = sequence
        self.organs = organs
        self.kinetics = kinetics

        # Constants
        self.dt = 0.5
        self.tmax = 120
        self.dose_tolerance = 0.1
        self.weight = 70.0
        self.agent = 'gadoterate'
        self.dose = 0.025
        self.rate = 1
        self.field_strength = 3.0
        self.t0 = 0
        self.TR = 0.005
        self.FA = 15.0
        self.TC = 0.180
        self.TS = None

        # Aorta parameters
        self.R10b = 0.7
        self.S0b = 1
        self.BAT = 60
        self.CO = 100
        self.Thl = 10
        self.Dhl = 0.2
        self.To = 20
        self.Eb = 0.05
        self.Eo = 0.15
        self.Teb = 120

        # Liver parameters
        self.Hct = 0.45
        self.R10l = 1/dc.T1(3.0, 'liver')
        self.S0l = 1
        self.Tel = 30.0
        self.De = 0.85
        self.ve = 0.3
        self.khe = 0.003
        self.Th = 30*60
        self.khe_f = 0.003
        self.Th_f = 30*60
        self.vol = None

        self.free = {
            'BAT': [0, np.inf],
            'CO': [0, 300],
            'Thl': [0, 30],
            'Dhl': [0.05, 0.95],
            'To': [0, 60],
            'Eb': [0.01, 0.15],
            'Tel': [0.1, 60],
            'De': [0, 1],
            've': [0.01, 0.6],
            'khe': [0, 0.1],
            'Th': [10*60, 10*60*60],
            'Eo': [0, 0.5],
            'Teb': [0, 800],
        }
        if kinetics == 'non-stationary':
            self.free['khe_f'] = [0, 10*60]
            self.free['Th_f'] = [0.1, 10*60*60]

        # overide defaults
        for k, v in params.items():
            setattr(self, k, v)

        # Internal flag
        self._predict = None

    def _conc_aorta(self) -> np.ndarray:
        if self.organs == 'comp':
            organs = ['comp', (self.To,)]
        else:
            organs = ['2cxm', ([self.To, self.Teb], self.Eo)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(self.t, self.weight,
                            conc, self.dose, self.rate, self.BAT)
        Jb = dc.flux_aorta(Ji, E=self.Eb,
                           heartlung=['pfcomp', (self.Thl, self.Dhl)],
                           organs=organs,
                           dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/self.CO
        self.ca = cb/(1-self.Hct)
        return self.t, cb

    def _relax_aorta(self) -> np.ndarray:
        t, cb = self._conc_aorta()
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        return t, self.R10b + rp*cb

    def _predict_aorta(self, xdata: np.ndarray) -> np.ndarray:
        self.tmax = max(xdata)+self.dt
        if self.TS is not None:
            self.tmax += self.TS
        t, R1b = self._relax_aorta()
        if self.sequence == 'SR':
            # signal = dc.signal_src(R1b, self.S0b, self.TC, R10=self.R10b)
            signal = dc.signal_src(R1b, self.S0b, self.TC)
        else:
            # signal = dc.signal_ss(R1b, self.S0b, self.TR, self.FA, R10=self.R10b)
            signal = dc.signal_ss(R1b, self.S0b, self.TR, self.FA)
        return dc.sample(xdata, t, signal, self.TS)

    def _conc_liver(self, sum=True):
        t = self.t
        if self.kinetics == 'non-stationary':
            khe = dc.interp(
                [self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], t)
            Kbh = dc.interp([1/self.Th, 1/self.Th_f], t)
            return t, dc.conc_liver(self.ca,
                                    self.ve, self.Tel, self.De, khe, 1/Kbh,
                                    t=self.t, dt=self.dt, kinetics='ICNS', sum=sum)
        elif self.kinetics == 'non-stationary uptake':
            self.Th_f = self.Th
            khe = dc.interp(
                [self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], t)
            return t, dc.conc_liver(self.ca,
                                    self.ve, self.Tel, self.De, khe, self.Th,
                                    t=self.t, dt=self.dt, kinetics='ICNSU', sum=sum)
        elif self.kinetics == 'stationary':
            self.khe_f = self.khe
            self.Th_f = self.Th
            khe = self.khe*(1-self.Hct)
            return t, dc.conc_liver(self.ca,
                                    self.ve, self.Tel, self.De, khe, self.Th,
                                    t=self.t, dt=self.dt, kinetics='IC', sum=sum)

    def _relax_liver(self):
        t, Cl = self._conc_liver(sum=False)
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        return t, self.R10l + rp*Cl[0, :] + rh*Cl[1, :]

    def _predict_liver(self, xdata: np.ndarray) -> np.ndarray:
        t, R1l = self._relax_liver()
        if self.sequence == 'SR':
            # signal = dc.signal_sr(R1l, self.S0l, self.TR, self.FA, self.TC, R10=R1l[0])
            signal = dc.signal_sr(R1l, self.S0l, self.TR, self.FA, self.TC)
        else:
            # signal = dc.signal_ss(R1l, self.S0l, self.TR, self.FA, R10=R1l[0])
            signal = dc.signal_ss(R1l, self.S0l, self.TR, self.FA)
        return dc.sample(xdata, t, signal, self.TS)

    def conc(self, sum=True):
        """Concentrations in aorta and liver.

        Args:
            sum (bool, optional): If set to true, the liver concentrations are the sum over both compartments. If set to false, the compartmental concentrations are returned individually. Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, liver concentrations.
        """
        t, cb = self._conc_aorta()
        t, C = self._conc_liver(sum=sum)
        return t, cb, C

    def relax(self):
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: time points, aorta blood concentrations, liver concentrations.
        """
        t, R1b = self._relax_aorta()
        t, R1l = self._relax_liver()
        return t, R1b, R1l

    def predict(self, xdata: tuple) -> tuple:
        """Predict the data at given xdata

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and liver, in that order. The two arrays can be different in length and value.

        Returns:
            tuple: tuple of 2 arrays with signals for aorta and liver, in that order. The arrays can be different in length and value but each has to have the same length as its corresponding array of time points.
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

    def train(self, xdata: tuple,
              ydata: tuple, **kwargs):
        """Train the free parameters

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and liver, in that order. The two arrays can be different in length and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and liver, in that order. The arrays can be different in length and value but each has to have the same length as its corresponding array of time points.
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        # Estimate BAT and S0b from data
        if self.sequence == 'SR':
            Srefb = dc.signal_sr(self.R10b, 1, self.TR, self.FA, self.TC)
            Srefl = dc.signal_sr(self.R10l, 1, self.TR, self.FA, self.TC)
        else:
            Srefb = dc.signal_ss(self.R10b, 1, self.TR, self.FA)
            Srefl = dc.signal_ss(self.R10l, 1, self.TR, self.FA)
        n0 = max([np.sum(xdata[0] < self.t0), 1])
        self.S0b = np.mean(ydata[0][:n0]) / Srefb
        self.S0l = np.mean(ydata[1][:n0]) / Srefl
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-self.Dhl)*self.Thl
        self.BAT = max([self.BAT, 0])

        # Copy the original free to restore later
        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = ['BAT', 'CO', 'Thl', 'Dhl', 'To', 'Eb', 'Eo', 'Teb']
        self.free = {s: free[s] for s in pars if s in free}
        dc.train(self, xdata[0], ydata[0], **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        pars = ['Tel', 'De', 've', 'khe', 'Th']
        if self.kinetics == 'non-stationary':
            free += ['khe_f', 'Th_f']
        self.free = {s: free[s] for s in pars if s in free}
        dc.train(self, xdata[1], ydata[1], **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        return dc.train(self, xdata, ydata, **kwargs)

    def plot(self,
             xdata: tuple,
             ydata: tuple,
             xlim=None, ref=None,
             fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and liver, in that order. The two arrays can be different in length and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and liver, in that order. The arrays can be different in length and value but each has to have the same length as its corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form (x,y), where x is an array with x-values and y is an array with y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
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

    def export_params(self):
        pars = _aorta_liver_params(self)
        return self._add_sdev(pars)

    def cost(self, xdata, ydata, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and liver, in that order. The two arrays can be different in length and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and liver, in that order. The arrays can be different in length and value but each has to have the same length as its corresponding array of time points.
            metric (str, optional): Which metric to use - options are: 
                **RMS** (Root-mean-square);
                **NRMS** (Normalized root-mean-square); 
                **AIC** (Akaike information criterion); 
                **cAIC** (Corrected Akaike information criterion for small models);
                **BIC** (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        return super().cost(xdata, ydata, metric)


class AortaKidneys(dc.Model):
    """Joint model for aorta and kidneys signals.

    The model represents the kidneys as a two-compartment filtration system and the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a chain compartment and the organs are modelled as a compartment or a two-compartment exchange model. Bolus injection into the system is modelled as a step function.

        **Injection parameters**

        - **weight** (float, default=70): Subject weight in kg.
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **dose** (float, default=0.2): Injected contrast agent dose in mL per kg bodyweight.
        - **rate** (float, default=1): Contrast agent injection rate in mL per sec.

        **Acquisition parameters**

        - **sequence** (str, default='SS'): Signal model.
        - **tmax** (float, default=120): Maximum acquisition time in sec.
        - **tacq** (float, default=None): Time to acquire a single dynamic in the first scan (sec). If this is not provided, tacq is taken from the difference between the first two data points.
        - **field_strength** (float, default=3.0): Magnetic field strength in T. 
        - **n0** (float, default=1): Baseline length in nr of scans.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec. 
        - **FA** (float, default=15): Nominal flip angle in degrees.
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence.

        **Signal parameters**

        - **R10b** (float, default=1): Precontrast arterial relaxation rate in 1/sec. 
        - **S0b** (float, default=1): scale factor for the arterial MR signal in the first scan. 
        - **R10_lk** (float, default=1): Baseline R1 for the left kidney.
        - **S0_lk** (float, default=1): Scale factor for the left kidney signal. 
        - **R10_rk** (float, default=1): Baseline R1 for the right kidney.
        - **S0_rk** (float, default=1): Scale factor for the right kidney signal.   

        **Whole body kinetic parameters**

        - **organs** (str, default='2cxm'): Kinetic model for the organs.
        - **BAT** (float, default=60): Bolus arrival time, i.e. time point where the indicator first arrives in the body. 
        - **BAT2** (float, default=1200): Bolus arrival time in the second scan, i.e. time point where the indicator first arrives in the body. 
        - **CO** (float, default=100): Cardiac output in mL/sec.
        - **Eb** (float, default=0.05): fraction of indicator extracted from the vasculature in a single pass. 
        - **Thl** (float, default=10): Mean transit time through heart and lungs.
        - **Dhl** (float, default=0.2): Dispersion through the heart-lung system, with a value in the range [0,1].
        - **To** (float, default=20): average time to travel through the organ's vasculature.
        - **Eo** (float, default=0.15): Fraction of indicator entering the organs which is extracted from the blood pool.
        - **Teb** (float, default=120): Average time to travel through the organs extravascular space.

        **Kidney kinetic parameters**

        - **kinetics** (str, default='2CFM'). Kidney kinetic model (only one option at this stage).
        - **Hct** (float, default=0.45): Hematocrit.
        - **Fp_lk** (Plasma flow, mL/sec/cm3): Flow of plasma into the plasma compartment (left kidney).
        - **Tp_lk** (Plasma mean transit time, sec): Transit time of the plasma compartment (left kidney). 
        - **Ft_lk** (Tubular flow, mL/sec/cm3): Flow of fluid into the tubuli (left kidney).
        - **Tt_lk** (Tubular mean transit time, sec): Transit time of the tubular compartment (left kidney).
        - **Ta_lk** (Arterial delay time, sec): Transit time through the arterial compartment (left kidney). 
        - **Fp_rk** (Plasma flow, mL/sec/cm3): Flow of plasma into the plasma compartment (right kidney).
        - **Tp_rk** (Plasma mean transit time, sec): Transit time of the plasma compartment (right kidney). 
        - **Ft_rk** (Tubular flow, mL/sec/cm3): Flow of fluid into the tubuli (right kidney).
        - **Tt_rk** (Tubular mean transit time, sec): Transit time of the tubular compartment (right kidney).
        - **Ta_rk** (Arterial delay time, sec): Transit time through the arterial compartment (right kidney). 

        **Prediction and training parameters**

        - **dt** (float, default=1): Internal time resolution of the AIF in sec. 
        - **dose_tolerance** (fload, default=0.1): Stopping criterion for the whole-body model.
        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **free** (array-like): 2-element list with lower and upper free of the free parameters. The default depends on the kinetics parameter.

        **Additional parameters**

        - **vol_lk** (float, optional): Kidney volume in cm3 (left kidney).
        - **vol_rk** (float, optional): Kidney volume in cm3 (right kidney).

    Args:
        params (dict, optional): override defaults for any of the parameters.

    See Also:
        `AortaLiver`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, roi, gt = dc.fake_tissue()
        >>> xdata, ydata = (time,time,time), (aif,roi,roi)

        Build an aorta-kidney model and parameters to match the conditions of the fake tissue data:

        >>> model = dc.AortaKidneys(
        ...     dt = 0.5,
        ...     tmax = 180,
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     t0 = 10,
        ...     TR = 0.005,
        ...     FA = 15,
        ... )

        Train the model on the data:

        >>> model.train(xdata, ydata, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare against the experimentally derived data:

        >>> model.plot(xdata, ydata)

    """

    def __init__(self, organs='comp', heartlung='pfcomp', kidneys='2CF', sequence='SS', **params):

        # Configuration
        self.sequence = sequence
        self.organs = organs
        self.heartlung = heartlung
        self.kidneys = kidneys

        # Constants
        self.dt = 0.5
        self.tmax = 120
        self.dose_tolerance = 0.1
        self.weight = 70.0
        self.agent = 'gadoterate'
        self.dose = 0.025
        self.rate = 1
        self.field_strength = 3.0
        self.n0 = 1
        self.TR = 0.005
        self.FA = 15.0
        self.TC = 0.180
        self.TS = None

        # Aorta parameters
        self.R10b = 1/dc.T1(3.0, 'blood')
        self.S0b = 1
        self.BAT = 60
        self.CO = 100
        self.FF = 0.1
        self.Thl = 20
        self.Dhl = 0.2
        self.To = 20
        self.Eo = 0.15
        self.Teb = 120
        self.Hct = 0.45

        # Kidneys
        self.RPF = 10  # mL/sec
        self.DRF = 0.5
        self.DRPF = 0.5

        # Left kidney parameters
        self.R10_lk = 1/dc.T1(3.0, 'kidney')
        self.S0_lk = 1
        self.Ta_lk = 0
        self.vp_lk = 0.1
        self.Tt_lk = 300
        self.vol_lk = 150

        # Right kidney parameters
        self.R10_rk = 1/dc.T1(3.0, 'kidney')
        self.S0_rk = 1
        self.Ta_rk = 0
        self.vp_rk = 0.1
        self.Tt_rk = 300
        self.vol_rk = 150

        self.free = {
            'BAT': [0, np.inf],
            'CO': [0, 300],
            'Thl': [0, 30],
            'Dhl': [0.05, 0.95],
            'To': [0, 60],
            'FF': [0, 0.5],
            'RPF': [0, np.inf],
            'DRPF': [0, 1],
            'DRF': [0, 1],
            'vp_lk': [0, 1],
            'Tt_lk': [0, np.inf],  # 'Ta_lk',
            'vp_rk': [0, 1],
            'Tt_rk': [0, np.inf],  # 'Ta_rk']
        }

        # overide defaults
        for k, v in params.items():
            setattr(self, k, v)

        # Internal flag
        self._predict = None

    def _conc_aorta(self) -> np.ndarray:
        if self.organs == 'comp':
            organs = ['comp', (self.To,)]
        else:
            organs = ['2cxm', ([self.To, self.Teb], self.Eo)]
        if self.heartlung == 'comp':
            heartlung = ['comp', (self.Thl,)]
        elif self.heartlung == 'pfcomp':
            heartlung = ['pfcomp', (self.Thl, self.Dhl)]
        elif self.heartlung == 'chain':
            heartlung = ['chain', (self.Thl, self.Dhl)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(self.t, self.weight,
                            conc, self.dose, self.rate, self.BAT)
        Jb = dc.flux_aorta(Ji, E=self.FF/(1+self.FF),
                           heartlung=heartlung, organs=organs,
                           dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/self.CO
        self.ca = cb/(1-self.Hct)
        return self.t, cb

    def _relax_aorta(self) -> np.ndarray:
        t, cb = self._conc_aorta()
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        return t, self.R10b + rp*cb

    def _predict_aorta(self, xdata: np.ndarray) -> np.ndarray:
        self.tmax = max(xdata)+self.dt
        if self.TS is not None:
            self.tmax += self.TS
        t, R1b = self._relax_aorta()
        if self.sequence == 'SR':
            # signal = dc.signal_src(R1b, self.S0b, self.TC, R10=self.R10b)
            signal = dc.signal_src(R1b, self.S0b, self.TC)
        else:
            # signal = dc.signal_ss(R1b, self.S0b, self.TR, self.FA, R10=self.R10b)
            signal = dc.signal_ss(R1b, self.S0b, self.TR, self.FA)
        return dc.sample(xdata, t, signal, self.TS)

    def _conc_kidneys(self, sum=True):
        t = self.t
        ca_lk = dc.flux(self.ca, self.Ta_lk, t=self.t,
                        dt=self.dt, model='plug')
        ca_rk = dc.flux(self.ca, self.Ta_rk, t=self.t,
                        dt=self.dt, model='plug')
        GFR = self.RPF * self.FF
        GFR_lk = self.DRF*GFR
        GFR_rk = (1-self.DRF)*GFR
        if self.kidneys == '2CF':
            RPF_lk = self.DRPF*self.RPF
            RPF_rk = (1-self.DRPF)*self.RPF
            Tp_lk = self.vp_lk*self.vol_lk/RPF_lk
            Tp_rk = self.vp_rk*self.vol_rk/RPF_rk
            # TODO reparametrize conc_kidney 2CF with vp instead of Tp
            Nlk = dc.conc_kidney(ca_lk, RPF_lk, Tp_lk, GFR_lk, self.Tt_lk,
                                 t=self.t, dt=self.dt, sum=sum, kinetics='2CF')
            Nrk = dc.conc_kidney(ca_rk, RPF_rk, Tp_rk, GFR_rk, self.Tt_rk,
                                 t=self.t, dt=self.dt, sum=sum, kinetics='2CF')
        if self.kidneys == 'HF':
            Nlk = dc.conc_kidney(ca_lk, self.vp_lk*self.vol_lk, GFR_lk,
                                 self.Tt_lk, t=self.t, dt=self.dt, sum=sum, kinetics='HF')
            Nrk = dc.conc_kidney(ca_rk, self.vp_rk*self.vol_rk, GFR_rk,
                                 self.Tt_rk, t=self.t, dt=self.dt, sum=sum, kinetics='HF')
        return t, Nlk/self.vol_rk, Nrk/self.vol_rk

    def _relax_kidneys(self):
        t, Clk, Crk = self._conc_kidneys()
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        return t, self.R10_lk + rp*Clk, self.R10_rk + rp*Crk

    def _predict_kidneys(self, xdata: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        t, R1_lk, R1_rk = self._relax_kidneys()
        if self.sequence == 'SR':
            signal_lk = dc.signal_sr(
                R1_lk, self.S0_lk, self.TR, self.FA, self.TC)
            signal_rk = dc.signal_sr(
                R1_rk, self.S0_rk, self.TR, self.FA, self.TC)
        else:
            signal_lk = dc.signal_ss(R1_lk, self.S0_lk, self.TR, self.FA)
            signal_rk = dc.signal_ss(R1_rk, self.S0_rk, self.TR, self.FA)
        return (
            dc.sample(xdata[0], t, signal_lk, self.TS),
            dc.sample(xdata[1], t, signal_rk, self.TS))

    def conc(self, sum=True):
        """Concentrations in aorta and kidney.

        Args:
            sum (bool, optional): If set to true, the kidney concentrations are the sum over all compartments. If set to false, the compartmental concentrations are returned individually. Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, left kidney concentrations, right kidney concentrations.
        """
        t, cb = self._conc_aorta()
        t, Clk, Crk = self._conc_kidneys(sum=sum)
        return t, cb, Clk, Crk

    def relax(self):
        """Relaxation rates in aorta and kidney.

        Returns:
            tuple: time points, aorta relaxation rate, left kidney relaxation rate, right kidney relaxation rate.
        """
        t, R1b = self._relax_aorta()
        t, R1_lk, R1_rk = self._relax_kidneys()
        return t, R1b, R1_lk, R1_rk

    def predict(self, xdata: tuple) -> tuple:
        """Predict the data at given xdata

        Args:
            xdata (tuple): Tuple of 3 arrays with time points for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and value.

        Returns:
            tuple: Tuple of 3 arrays with signals for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and value but each has to have the same length as its corresponding array of time points.
        """
        # Public interface
        if self._predict is None:
            signala = self._predict_aorta(xdata[0])
            signal_lk, signal_rk = self._predict_kidneys((xdata[1], xdata[2]))
            return signala, signal_lk, signal_rk
        # Private interface with different input & output types
        elif self._predict == 'aorta':
            return self._predict_aorta(xdata)
        elif self._predict == 'kidneys':
            return self._predict_kidneys(xdata)

    def train(self, xdata: tuple,
              ydata: tuple, **kwargs):
        """Train the free parameters

        Args:
            xdata (tuple): Tuple of 3 arrays with time points for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and value.
            ydata (tuple): Tuple of 3 arrays with signals for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and values but each has to have the same length as its corresponding array of time points.
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        # Estimate BAT and S0b from data
        if self.sequence == 'SR':
            Srefb = dc.signal_sr(self.R10b, 1, self.TR, self.FA, self.TC)
            Sref_lk = dc.signal_sr(self.R10_lk, 1, self.TR, self.FA, self.TC)
            Sref_rk = dc.signal_sr(self.R10_rk, 1, self.TR, self.FA, self.TC)
        else:
            Srefb = dc.signal_ss(self.R10b, 1, self.TR, self.FA)
            Sref_lk = dc.signal_ss(self.R10_lk, 1, self.TR, self.FA)
            Sref_rk = dc.signal_ss(self.R10_rk, 1, self.TR, self.FA)
        self.S0b = np.mean(ydata[0][:self.n0]) / Srefb
        self.S0_lk = np.mean(ydata[1][:self.n0]) / Sref_lk
        self.S0_rk = np.mean(ydata[2][:self.n0]) / Sref_rk
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-self.Dhl)*self.Thl
        self.BAT = max([self.BAT, 0])

        # Copy all free to restor at the end
        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = ['BAT', 'CO', 'Thl', 'Dhl', 'To', 'Eb', 'Eo', 'Teb']
        self.free = {s: free[s] for s in pars if s in free}
        dc.train(self, xdata[0], ydata[0], **kwargs)

        # Train free kidney parameters on kidney data
        self._predict = 'kidneys'
        pars = ['RPF', 'DRPF', 'DRF',
                'vp_lk', 'Tt_lk', 'Ta_lk',
                'vp_rk', 'Tt_rk', 'Ta_rk']
        self.free = {s: free[s] for s in pars if s in free}
        dc.train(self, (xdata[1], xdata[2]), (ydata[1], ydata[2]), **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        return dc.train(self, xdata, ydata, **kwargs)

    def plot(self,
             xdata: tuple,
             ydata: tuple,
             xlim=None, ref=None,
             fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (tuple): Tuple of 3 arrays with time points for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and value.
            ydata (tuple): Tuple of 3 arrays with signals for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and values but each has to have the same length as its corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form (x,y), where x is an array with x-values and y is an array with y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        t, cb, Clk, Crk = self.conc(sum=False)
        sig = self.predict((t, t, t))
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)
              ) = plt.subplots(3, 2, figsize=(10, 12))
        fig.subplots_adjust(wspace=0.3)
        _plot_data1scan(t, sig[0], xdata[0], ydata[0],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[0])
        _plot_data1scan(t, sig[1], xdata[1], ydata[1],
                        ax3, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[1])
        _plot_data1scan(t, sig[2], xdata[2], ydata[2],
                        ax5, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[2])
        _plot_conc_aorta(t, cb, ax2, xlim)
        _plot_conc_kidney(t, Clk, ax4, xlim)
        _plot_conc_kidney(t, Crk, ax6, xlim)
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def export_params(self):
        pars = {}

        # Aorta
        pars['T10b'] = ['Blood precontrast T1', 1/self.R10b, "sec"]
        pars['rate'] = ['Injection rate', self.rate, "mL/sec"]
        pars['BAT'] = ['Bolus arrival time', self.BAT, "sec"]
        pars['CO'] = ['Cardiac output', self.CO, "mL/sec"]
        pars['Thl'] = ['Heart-lung mean transit time', self.Thl, "sec"]
        pars['Dhl'] = ['Heart-lung transit time dispersion', self.Dhl, ""]
        pars['To'] = ["Organs mean transit time", self.To, "sec"]
        pars['Tc'] = ["Mean circulation time", self.Thl+self.To, 'sec']
        pars['Eb'] = ["Body extraction fraction", self.FF/(1+self.FF), ""]
        pars['Eo'] = ["Organs extraction fraction", self.Eo, ""]
        pars['Teb'] = ["Organs extracellular mean transit time", self.Teb, "sec"]

        # Kidneys
        GFR = self.RPF * self.FF
        pars['GFR'] = ['Glomerular filtration rate', GFR, 'mL/sec']
        pars['RPF'] = ['Renal plasma flow', self.RPF, 'mL/sec']
        pars['DRPF'] = ['Differential renal plasma flow', self.DRPF, '']
        pars['DRF'] = ['Differential renal function', self.DRF, '']
        pars['FF'] = ['Filtration fraction', GFR/self.RPF, '']

        # Kidney LK
        RPF_lk = self.DRPF*self.RPF
        GFR_lk = self.DRF*GFR
        Tp_lk = self.vp_lk*self.vol_lk/RPF_lk
        pars['LK-RPF'] = ['LK Single-kidney plasma flow', RPF_lk, 'mL/sec']
        pars['LK-GFR'] = ['LK Single-kidney glomerular filtration rate',
                          GFR_lk, 'mL/sec']
        pars['LK-vol'] = ['LK Single-kidney volume', self.vol_lk, 'cm3']
        pars['LK-Fp'] = ['LK Plasma flow', RPF_lk/self.vol_lk, 'mL/sec/cm3']
        pars['LK-Tp'] = ['LK Plasma mean transit time', Tp_lk, 'sec']
        pars['LK-Ft'] = ['LK Tubular flow', GFR_lk/self.vol_lk, 'mL/sec/cm3']
        pars['LK-ve'] = ['LK Extracellular volume', self.vp_lk, 'mL/cm3']
        pars['LK-FF'] = ['LK Filtration fraction', GFR_lk/RPF_lk, '']
        pars['LK-E'] = ['LK Extraction fraction', GFR_lk/(GFR_lk+RPF_lk), '']
        pars['LK-Tt'] = ['LK Tubular mean transit time', self.Tt_lk, 'sec']
        pars['LK-Ta'] = ['LK Arterial mean transit time', self.Ta_lk, 'sec']

        # Kidney RK
        RPF_rk = (1-self.DRPF)*self.RPF
        GFR_rk = (1-self.DRF)*GFR
        Tp_rk = self.vp_rk*self.vol_rk/RPF_rk
        pars['RK-RPF'] = ['RK Single-kidney plasma flow', RPF_rk, 'mL/sec']
        pars['RK-GFR'] = ['RK Single-kidney glomerular filtration rate',
                          GFR_rk, 'mL/sec']
        pars['RK-vol'] = ['RK Single-kidney volume', self.vol_rk, 'cm3']
        pars['RK-Fp'] = ['RK Plasma flow', RPF_rk/self.vol_rk, 'mL/sec/cm3']
        pars['RK-Tp'] = ['RK Plasma mean transit time', Tp_rk, 'sec']
        pars['RK-Ft'] = ['RK Tubular flow', GFR_rk/self.vol_rk, 'mL/sec/cm3']
        pars['RK-ve'] = ['RK Extracellular volume', self.vp_rk, 'mL/cm3']
        pars['RK-FF'] = ['RK Filtration fraction', GFR_rk/RPF_rk, '']
        pars['RK-E'] = ['RK Extraction fraction', GFR_rk/(GFR_rk+RPF_rk), '']
        pars['RK-Tt'] = ['RK Tubular mean transit time', self.Tt_rk, 'sec']
        pars['RK-Ta'] = ['RK Arterial mean transit time', self.Ta_rk, 'sec']

        return self._add_sdev(pars)

    def cost(self, xdata: tuple, ydata: tuple, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            xdata (tuple): Tuple of 3 arrays with time points for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and value.
            ydata (tuple): Tuple of 3 arrays with signals for aorta, left kidney and right kidney, in that order. The three arrays can all be different in length and values but each has to have the same length as its corresponding array of time points.
            metric (str, optional): Which metric to use - options are: 
                **RMS** (Root-mean-square);
                **NRMS** (Normalized root-mean-square); 
                **AIC** (Akaike information criterion); 
                **cAIC** (Corrected Akaike information criterion for small models);
                **BIC** (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        return super().cost(xdata, ydata, metric)


class AortaLiver2scan(AortaLiver):
    """Joint model for aorta and liver signals measured over two scans.

    The model represents the liver as a two-compartment system and the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a chain compartment and the organs are modelled as a two-compartment exchange model. Bolus injection into the system is modelled as a step function.

        **Injection parameters**

        - **weight** (float, default=70): Subject weight in kg.
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **dose** (float, default=0.2): Injected contrast agent dose in mL per kg bodyweight.
        - **rate** (float, default=1): Contrast agent injection rate in mL per sec.

        **Acquisition parameters**

        - **sequence** (str, default='SS'): Signal model.
        - **tmax** (float, default=120): Maximum acquisition time in sec.
        - **tacq** (float, default=None): Time to acquire a single dynamic in the first scan (sec). If this is not provided, tacq is taken from the difference between the first two data points.
        - **tacq2** (float, default=None): Time to acquire a single dynamic in the second scan (sec). If this is not provided, tacq is taken from the difference between the first two data points.
        - **field_strength** (float, default=3.0): Magnetic field strength in T. 
        - **t0** (float, default=1): Baseline length in secs.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec. 
        - **FA** (float, default=15): Nominal flip angle in degrees.
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence.

        **Signal parameters**

        - **R10b** (float, default=1): Precontrast arterial relaxation rate in 1/sec. 
        - **R102b** (float, default=1): Precontrast arterial relaxation rate in the second scan in 1/sec. 
        - **R10l** (float, default=1): Baseline R1 for the liver.
        - **R102l** (float, default=1): Baseline R1 for the liver (second scan).
        - **S0b** (float, default=1): scale factor for the arterial MR signal in the first scan.
        - **S02b** (float, default=1): scale factor for the arterial MR signal in the second scan.
        - **S0l** (float, default=1): Scale factor for the liver signal.  
        - **S02l** (float, default=1): Scale factor for the liver signal (second scan).  

        **Whole body kinetic parameters**

        - **organs** (str, default='2cxm'): Kinetic model for the organs.
        - **BAT** (float, default=60): Bolus arrival time, i.e. time point where the indicator first arrives in the body. 
        - **BAT2** (float, default=1200): Bolus arrival time in the second scan, i.e. time point where the indicator first arrives in the body. 
        - **CO** (float, default=100): Cardiac output in mL/sec.
        - **Thl** (float, default=10): Mean transit time through heart and lungs.
        - **Dhl** (float, default=0.2): Dispersion through the heart-lung system, with a value in the range [0,1].
        - **To** (float, default=20): average time to travel through the organ's vasculature.
        - **Eb** (float, default=0.05): fraction of indicator extracted from the vasculature in a single pass. 
        - **Eo** (float, default=0.15): Fraction of indicator entering the organs which is extracted from the blood pool.
        - **Teb** (float, default=120): Average time to travel through the organs extravascular space.

        **Liver kinetic parameters**

        - **kinetics** (str, default='non-stationary'). Liver kinetic model, either stationary or non-stationary.
        - **Hct** (float, default=0.45): Hematocrit.
        - **Tel** (float, default=30): Mean transit time for extracellular space of liver and gut.
        - **De** (float, default=0.85): Dispersion in the extracellular space of liver an gut, in the range [0,1].
        - **ve** (float, default=0.3): Liver extracellular volume fraction.
        - **khe** (float, default=0.003): Hepatocellular uptake rate.
        - **Th** (float, default=1800): Hepatocellular transit time.
        - **khe_f** (float, default=0.003): Final hepatocellular uptake rate (non-stationary models).
        - **Th_f** (float, default=1800): Final hepatocellular transit time (non-stationary models).
        - **vol** (float, default=None): liver volume in mL (for whole liver export parameters).

        **Prediction and training parameters**

        - **dt** (float, default=1): Internal time resolution of the AIF in sec. 
        - **dose_tolerance** (fload, default=0.1): Stopping criterion for the whole-body model.
        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **free** (array-like): 2-element list with lower and upper free of the free parameters. The default depends on the kinetics parameter.

    Args:
        params (dict, optional): override defaults for any of the parameters.

    See Also:
        `AortaLiver`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, roi, gt = dc.fake_tissue2scan(R10 = 1/dc.T1(3.0,'liver'))
        >>> xdata = (time[0], time[1], time[0], time[1])
        >>> ydata = (aif[0], aif[1], roi[0], roi[1])

        Build an aorta-liver model and parameters to match the conditions of 
        the fake tissue data:

        >>> model = dc.AortaLiver2scan(
        ...     dt = 0.5,
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = [0.2,0.2],
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
        Liver extracellular mean transit time (Tel): 2.957 (0.452) sec
        Liver extracellular dispersion (De): 1.0 (0.146)
        Liver extracellular volume fraction (ve): 0.077 (0.007) mL/cm3
        Hepatocellular uptake rate (khe): 0.002 (0.001) mL/sec/cm3
        Hepatocellular transit time (Th): 600.0 (1173.571) sec
        Organs extraction fraction (Eo): 0.2 (0.057)
        Organs extracellular mean transit time (Teb): 87.077 (56.882) sec
        Hepatocellular uptake rate (final) (khe_f): 0.001 (0.001) mL/sec/cm3
        Hepatocellular transit time (final) (Th_f): 600.0 (623.364) sec
        ------------------
        Derived parameters
        ------------------
        Blood precontrast T1 (T10b): 1.629 sec
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
    """

    def __init__(self, organs='2cxm', kinetics='non-stationary', sequence='SS', **params):

        self.organs = organs
        self.sequence = sequence
        self.kinetics = kinetics

        # Injection
        self.weight = 70.0
        self.agent = 'gadoterate'
        self.dose = [
            dc.ca_std_dose('gadoterate')/2,
            dc.ca_std_dose('gadoterate')/2]
        self.rate = 1

        # Acquisition
        self.tmax = 120
        self.field_strength = 3.0
        self.TR = 0.005
        self.FA = 15.0
        self.TC = 0.180
        self.TS = None
        self.t0 = 0

        # Signal
        self.R10b = 1/dc.T1(3.0, 'blood')
        self.R102b = 1/dc.T1(3.0, 'blood')
        self.S0b = 1
        self.S02b = 1
        self.R10l = 1/dc.T1(3.0, 'liver')
        self.R102l = 1/dc.T1(3.0, 'liver')
        self.S0l = 1
        self.S02l = 1

        # Body
        self.BAT = 60
        self.BAT2 = 1200
        self.CO = 100
        self.Thl = 10
        self.Dhl = 0.2
        self.To = 20
        self.Eb = 0.05
        self.Eo = 0.15
        self.Teb = 120

        # Liver
        self.Hct = 0.45
        self.Tel = 30.0
        self.De = 0.85
        self.ve = 0.3
        self.khe = 0.003
        self.Th = 30*60
        self.khe_f = 0.003
        self.Th_f = 30*60
        self.vol = None

        # Prediction and training
        self.dt = 0.5
        self.dose_tolerance = 0.1
        self.free = {
            'BAT': [0, np.inf],
            'BAT2': [0, np.inf],
            'S02b': [0, np.inf],
            'S02l': [0, np.inf],
            'CO': [0, 300],
            'Thl': [0, 30],
            'Dhl': [0.05, 0.95],
            'To': [0, 60],
            'Eb': [0.01, 0.15],
            'Tel': [0.1, 60],
            'De': [0, 1],
            've': [0.01, 0.6],
            'khe': [0, 0.1],
            'Th': [10*60, 10*60*60],
            'Eo': [0, 0.5],
            'Teb': [0, 800],
        }
        if kinetics == 'non-stationary':
            self.free['khe_f'] = [0, 0.1]
            self.free['Th_f'] = [10*60, 10*60*60]

        # overide defaults
        for k, v in params.items():
            setattr(self, k, v)

        # Internal flags
        self._predict = None

    def _conc_aorta(self) -> tuple[np.ndarray, np.ndarray]:
        if self.organs == 'comp':
            organs = ['comp', (self.To,)]
        else:
            organs = ['2cxm', ([self.To, self.Teb], self.Eo)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        J1 = dc.influx_step(self.t, self.weight, conc,
                            self.dose[0], self.rate, self.BAT)
        J2 = dc.influx_step(self.t, self.weight, conc,
                            self.dose[1], self.rate, self.BAT2)
        Jb = dc.flux_aorta(J1 + J2, E=self.Eb,
                           heartlung=['pfcomp', (self.Thl, self.Dhl)],
                           organs=organs,
                           dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/self.CO
        self.ca = cb/(1-self.Hct)
        return self.t, cb

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
        if self.sequence == 'SR':
            signal1 = dc.signal_sr(R11, self.S0b, self.TR, self.FA, self.TC)
            signal2 = dc.signal_sr(R12, self.S02b, self.TR, self.FA, self.TC)
        else:
            signal1 = dc.signal_ss(R11, self.S0b, self.TR, self.FA)
            signal2 = dc.signal_ss(R12, self.S02b, self.TR, self.FA)
        return (
            dc.sample(xdata[0], t[t1], signal1, self.TS),
            dc.sample(xdata[1], t[t2], signal2, self.TS),
        )

    def _predict_liver(self,
                       xdata: tuple[np.ndarray, np.ndarray],
                       ) -> tuple[np.ndarray, np.ndarray]:
        t, R1 = self._relax_liver()
        t1 = t <= xdata[0][-1]
        t2 = t >= xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        if self.sequence == 'SR':
            signal1 = dc.signal_sr(R11, self.S0l, self.TR, self.FA, self.TC)
            signal2 = dc.signal_sr(R12, self.S02l, self.TR, self.FA, self.TC)
        else:
            signal1 = dc.signal_ss(R11, self.S0l, self.TR, self.FA)
            signal2 = dc.signal_ss(R12, self.S02l, self.TR, self.FA)
        return (
            dc.sample(xdata[0], t[t1], signal1, self.TS),
            dc.sample(xdata[1], t[t2], signal2, self.TS),
        )

    def predict(self, xdata: tuple) -> tuple:
        """Predict the data at given time points

        Args:
            xdata (tuple): tuple of 4 arrays with time points for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The four arrays can be different in length and value.

        Returns:
            tuple: tuple of 4 arrays with signals for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The arrays have the same length as its corresponding array of time points.
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
            xdata (tuple): tuple of 4 arrays with time points for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The four arrays can be different in length and value.
            ydata (tuple): tuple of 4 arrays with signals for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The arrays can be different in length but each has to have the same length as its corresponding array of time points.
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        # Estimate BAT
        T, D = self.Thl, self.Dhl
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-D)*T
        self.BAT2 = xdata[1][np.argmax(ydata[1])] - (1-D)*T

        # Estimate S0
        if self.sequence == 'SR':
            Srefb = dc.signal_sr(self.R10b, 1, self.TR, self.FA, self.TC)
            Sref2b = dc.signal_sr(self.R102b, 1, self.TR, self.FA, self.TC)
            Srefl = dc.signal_sr(self.R10l, 1, self.TR, self.FA, self.TC)
            Sref2l = dc.signal_sr(self.R102l, 1, self.TR, self.FA, self.TC)
        else:
            Srefb = dc.signal_ss(self.R10b, 1, self.TR, self.FA)
            Sref2b = dc.signal_ss(self.R102b, 1, self.TR, self.FA)
            Srefl = dc.signal_ss(self.R10l, 1, self.TR, self.FA)
            Sref2l = dc.signal_ss(self.R102l, 1, self.TR, self.FA)

        n0 = max([np.sum(xdata[0] < self.t0), 2])
        self.S0b = np.mean(ydata[0][1:n0]) / Srefb
        self.S02b = np.mean(ydata[1][1:n0]) / Sref2b
        self.S0l = np.mean(ydata[2][1:n0]) / Srefl
        self.S02l = np.mean(ydata[3][1:n0]) / Sref2l

        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = ['BAT', 'CO', 'Thl', 'Dhl', 'To',
                'Eb', 'Eo', 'Teb', 'BAT2', 'S02b']
        self.free = {s: free[s] for s in pars if s in free}
        dc.train(self, (xdata[0], xdata[1]), (ydata[0], ydata[1]), **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        pars = ['Tel', 'De', 've', 'khe', 'Th', 'khe_f', 'Th_f', 'S02l']
        self.free = {s: free[s] for s in pars if s in free}
        # added if s in free - add everywhere after testing
        dc.train(self, (xdata[2], xdata[3]), (ydata[2], ydata[3]), **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = free
        return dc.train(self, xdata, ydata, **kwargs)
    
    def export_params(self):
        pars = _aorta_liver_params(self)
        pars['BAT2'] = ['Bolus arrival time - 2nd scan', self.BAT2, "sec"]
        pars['S02b'] = ['Blood signal scale factor - 2nd scan', self.S02b, "a.u."]
        pars['S02l'] = ['Liver signal scale factor - 2nd scan', self.S02l, "a.u."]
        return self._add_sdev(pars)

    def plot(self, xdata: tuple, ydata: tuple,
             ref=None, xlim=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (tuple): tuple of 4 arrays with time points for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The four arrays can be different in length and value.
            ydata (tuple): tuple of 4 arrays with signals for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The arrays can be different in length but each has to have the same length as its corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form (x,y), where x is an array with x-values and y is an array with y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
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
            xdata (tuple): tuple of 4 arrays with time points for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The four arrays can be different in length and value.
            ydata (tuple): tuple of 4 arrays with signals for aorta in the first scan, aorta in the second stand, liver in the first scan, and liver in the second scan, in that order. The arrays can be different in length but each has to have the same length as its corresponding array of time points.
            metric (str, optional): Which metric to use - options are: 
                **RMS** (Root-mean-square);
                **NRMS** (Normalized root-mean-square); 
                **AIC** (Akaike information criterion); 
                **cAIC** (Corrected Akaike information criterion for small models);
                **BIC** (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        return super().cost(xdata, ydata, metric)


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


def _plot_conc_kidney(t: np.ndarray, C: np.ndarray, ax, xlim=None):
    color = 'darkblue'
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*C[0, :], linestyle='-.',
            color=color, linewidth=2.0, label='Plasma')
    ax.plot(t/60, 1000*C[1, :], linestyle='--',
            color=color, linewidth=2.0, label='Tubuli')
    ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-',
            color=color, linewidth=2.0, label='Tissue')
    ax.legend()


def _plot_data2scan(t: tuple[np.ndarray, np.ndarray], sig: tuple[np.ndarray, np.ndarray],
                    xdata: tuple[np.ndarray, np.ndarray], ydata: tuple[np.ndarray, np.ndarray],
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
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
    ax.plot(xdata/60, ydata, marker='o',
            color=color[0], label='fitted data', linestyle='None')
    ax.plot(t/60, sig, linestyle='-',
            color=color[1], linewidth=3.0, label='fit')
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black',
                marker='D', linestyle='None', label='Test data')
    ax.legend()


def _aorta_liver_params(self):
    pars = {}
    # Aorta
    pars['T10b'] = ['Blood precontrast T1', 1/self.R10b, "sec"]
    pars['BAT'] = ['Bolus arrival time', self.BAT, "sec"]
    pars['CO'] = ['Cardiac output', self.CO, "mL/sec"]
    pars['Thl'] = ['Heart-lung mean transit time', self.Thl, "sec"]
    pars['Dhl'] = ['Heart-lung transit time dispersion', self.Dhl, ""]
    pars['To'] = ["Organs mean transit time", self.To, "sec"]
    pars['Eb'] = ["Extraction fraction", self.Eb, ""]
    pars['Tc'] = ["Mean circulation time", self.Thl+self.To, 'sec']
    pars['Eo'] = ["Organs extraction fraction", self.Eo, ""]
    pars['Teb'] = ["Organs extracellular mean transit time", self.Teb, "sec"]
    # Liver
    pars['T10l'] = ['Liver precontrast T1', 1/self.R10l, "sec"]
    pars['Tel'] = ["Liver extracellular mean transit time", self.Tel, 'sec']
    pars['De'] = ["Liver extracellular dispersion", self.De, '']
    pars['ve'] = ["Liver extracellular volume fraction", self.ve, 'mL/cm3']
    if self.kinetics == 'stationary':
        pars['khe'] = ["Hepatocellular uptake rate", self.khe, 'mL/sec/cm3']
        pars['Th'] = ["Hepatocellular transit time", self.Th, 'sec']
        pars['kbh'] = ["Biliary excretion rate",
                        (1-self.ve)/self.Th, 'mL/sec/cm3']
        pars['Khe'] = ["Hepatocellular tissue uptake rate",
                        self.khe/self.ve, 'mL/sec/cm3']
        pars['Kbh'] = ["Biliary tissue excretion rate",
                        1/self.Th, 'mL/sec/cm3']
        if self.vol is not None:
            pars['CL'] = ['Liver blood clearance',
                            self.khe*self.vol, 'mL/sec']
    else:
        khe = [self.khe, self.khe_f]
        Kbh = [1/self.Th, 1/self.Th_f]
        khe_avr = np.mean(khe)
        Kbh_avr = np.mean(Kbh)
        khe_var = (np.amax(khe)-np.amin(khe))/khe_avr
        Kbh_var = (np.amax(Kbh)-np.amin(Kbh))/Kbh_avr
        kbh = np.mean((1-self.ve)*Kbh_avr)
        Th = np.mean(1/Kbh_avr)
        pars['khe'] = ["Hepatocellular uptake rate", khe_avr, 'mL/sec/cm3']
        pars['Th'] = ["Hepatocellular transit time", Th, 'sec']
        pars['kbh'] = ["Biliary excretion rate", kbh, 'mL/sec/cm3']
        pars['Khe'] = ["Hepatocellular tissue uptake rate",
                        khe_avr/self.ve, 'mL/sec/cm3']
        pars['Kbh'] = ["Biliary tissue excretion rate", Kbh_avr, 'mL/sec/cm3']
        pars['khe_i'] = [
            "Hepatocellular uptake rate (initial)", self.khe, 'mL/sec/cm3']
        pars['khe_f'] = [
            "Hepatocellular uptake rate (final)", self.khe_f, 'mL/sec/cm3']
        pars['Th_i'] = [
            "Hepatocellular transit time (initial)", self.Th, 'sec']
        pars['Th_f'] = [
            "Hepatocellular transit time (final)", self.Th_f, 'sec']
        pars['khe_var'] = ["Hepatocellular uptake rate variance", khe_var, '']
        pars['Kbh_var'] = [
            "Biliary tissue excretion rate variance", Kbh_var, '']
        pars['kbh_i'] = [
            "Biliary excretion rate (initial)", (1-self.ve)/self.Th, 'mL/sec/cm3']
        pars['kbh_f'] = [
            "Biliary excretion rate (final)", (1-self.ve)/self.Th_f, 'mL/sec/cm3']
        if self.vol is not None:
            pars['CL'] = ['Liver blood clearance',
                            khe_avr*self.vol, 'mL/sec']
    return pars
