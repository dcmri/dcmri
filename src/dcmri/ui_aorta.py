from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import dcmri.ui as ui
import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk as pk
import dcmri.pk_aorta as pk_aorta
import dcmri.kidney as kidney


class Aorta(ui.Model):
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
        self.dose = lib.ca_std_dose('gadoterate')
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
        conc = lib.ca_conc(self.agent)
        Ji = lib.ca_injection(t, self.weight,
                            conc, self.dose, self.rate, self.BAT)
        Jb = pk_aorta.flux_aorta(Ji, E=self.Eb,
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
        rp = lib.relaxivity(self.field_strength, 'plasma', self.agent)
        return t, self.R10 + rp*cb

    def predict(self, xdata) -> np.ndarray:
        tacq = xdata[1]-xdata[0]
        self.tmax = max(xdata)+tacq+self.dt
        t, R1 = self.relax()
        if self.sequence == 'SR':
            # signal = sig.signal_free(self.S0, R1, self.TC, R10=self.R10)
            signal = sig.signal_free(self.S0, R1, self.TC, self.FA)
        else:
            # signal = sig.signal_ss(self.S0, R1, self.TR, self.FA, R10=self.R10)
            signal = sig.signal_ss(self.S0, R1, self.TR, self.FA)
        return utils.sample(xdata, t, signal, self.TS)

    def train(self, xdata, ydata, **kwargs):
        n0 = max([np.sum(xdata < self.t0), 1])
        self.BAT = xdata[np.argmax(ydata)] - self.Thl
        if self.sequence == 'SR':
            Sref = sig.signal_free(1, self.R10, self.TC, self.FA)
        else:
            Sref = sig.signal_ss(1, self.R10, self.TR, self.FA)
        self.S0 = np.mean(ydata[:n0]) / Sref
        return ui.train(self, xdata, ydata, **kwargs)

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


