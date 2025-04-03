import matplotlib.pyplot as plt
import numpy as np

import dcmri.kidney as pkk
import dcmri.lib as lib
import dcmri.ui as ui
import dcmri.sig as sig
import dcmri.utils as utils


class KidneyCortMed(ui.Model):
    """
    General model for renal cortico-medullary data.

    **warning**: This model is functional but under active 
    development. Future versions may change without warning.

        **Kinetic parameters**

        - **Hct** (float, optional): Hematocrit. 
        - **Fp** Plasma flow in mL/sec/mL. 
        - **Eg** Glomerular extraction fraction. 
        - **fc** Cortical flow fraction. 
        - **Tg** Glomerular mean transit time in sec. 
        - **Tv** Peritubular & venous mean transit time in sec. 
        - **Tpt** Proximal tubuli mean transit time in sec. 
        - **Tlh** Lis of Henle mean transit time in sec. 
        - **Tdt** Distal tubuli mean transit time in sec. 
        - **Tcd** Collecting duct mean transit time in sec.

        **Input function**

        - **aif** (array-like, default=None): Signal-time curve in a feeding artery. If AIF is set to None, then the parameter ca must be provided (arterial concentrations).
        - **ca** (array-like, default=None): Concentration (M) in the arterial input. Must be provided when aif = None, ignored otherwise.

        **Acquisition parameters**

        - **sequence** (str, default='SR'): imaging sequence.
        - **dt** (float, optional): Sampling interval of the AIF in sec. 
        - **agent** (str, optional): Contrast agent generic name.
        - **field_strength** (float, optional): Magnetic field strength in T. 
        - **TR** (float, optional): Repetition time, or time between excitation pulses, in sec. 
        - **FA** (float, optional): Nominal flip angle in degrees.
        - **Tsat** (float, optional): Time before start of readout (sec).
        - **TC** (float, optional): Time to the center of the readout pulse
        - **n0** (float, optional): Baseline length in number of acquisitions.

        **Signal parameters**

        - **R10c** (float, optional): Precontrast cortex relaxation rate in 1/sec. 
        - **R10m** (float, optional): Precontrast medulla relaxation rate in 1/sec.
        - **R10a** (float, optional): Precontrast arterial relaxation rate in 1/sec. 
        - **S0c** (float, optional): Signal scaling factor in the cortex (a.u.).
        - **S0m** (float, optional): Signal scaling factor in the medulla (a.u.).

        **Prediction and training parameters**

        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **free** (array-like): 2-element list with lower and upper free of the free parameters. The default depends on the kinetics parameter.

        **Other parameters**

        - **vol** (float, optional): Liver volume in mL.


    See Also:
        `Kidney`, `Liver`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc

        Use `fake_kidney` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_kidney(CNR=100)

        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.KidneyCortMed(
        ...     aif = aif,
        ...     dt = time[1],
        ...     agent = 'gadoterate',
        ...     TR = 0.005,
        ...     FA = 15,
        ...     TC = 0.2,
        ...     n0 = 10,
        ... )

        Train the model on the ROI data and predict signals and concentrations:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> model.plot(time, roi, ref=gt)
    """

    free = {}   #: lower- and upper free for all free parameters.

    def __init__(self, sequence='SR', **params):

        # Config
        self.sequence = sequence

        # Input function
        self.aif = None
        self.ca = None

        # Acquisition parameters
        self.field_strength = 3.0
        self.t = None
        self.dt = 0.5
        self.agent = 'gadoterate'
        self.n0 = 1
        self.TR = 0.005
        self.FA = 15.0
        self.Tsat = 0
        self.TC = 0.085
        self.TS = None

        # Tracer-kinetic parameters
        self.Hct = 0.45
        self.Fp = 0.03
        self.Eg = 0.15
        self.fc = 0.8
        self.Tg = 4
        self.Tv = 10
        self.Tpt = 60
        self.Tlh = 60
        self.Tdt = 30
        self.Tcd = 30

        # Signal parameters
        self.R10c = 1/lib.T1(3.0, 'kidney')
        self.R10m = 1/lib.T1(3.0, 'kidney')
        self.R10a = 1/lib.T1(3.0, 'blood')
        self.S0c = 1
        self.S0m = 1

        # Other parameters
        self.vol = None

        # training parameters
        self.free = {
            'Fp': [0.01, 1],
            'Eg': [0, 1],
            'fc': [0, 1],
            'Tg': [0, 10],
            'Tv': [0, 30],
            'Tpt': [0, np.inf],
            'Tlh': [0, np.inf],
            'Tdt': [0, np.inf],
            'Tcd': [0, np.inf],
        }

        # overide defaults
        for k, v in params.items():
            setattr(self, k, v)

        # Check inputs
        if (self.aif is None) and (self.ca is None):
            raise ValueError(
                'Please provide either an arterial sigal (aif) or an arterial concentration (ca).')

    def time(self):
        """Array of time points

        Returns:
            np.ndarray: time points in seconds.
        """
        if self.t is None:
            if self.aif is None:
                return self.dt*np.arange(np.size(self.ca))
            else:
                return self.dt*np.arange(np.size(self.aif))
        else:
            return self.t

    def conc(self, sum=True):
        """Cortical and medullary concentration

        Returns:
            tuple: Concentration in cortex, concentration in medulla, in M
        """

        if self.aif is not None:
            r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
            if self.sequence == 'SR':
                cb = sig.conc_src(self.aif, self.TC, 1/self.R10a, r1, self.n0)
            elif self.sequence == 'SS':
                cb = sig.conc_ss(self.aif, self.TR, self.FA,
                                 1/self.R10a, r1, self.n0)
            else:
                raise NotImplementedError(
                    'Signal model ' + self.sequence + 'is not (yet) supported.')
            self.ca = cb/(1-self.Hct)
        return pkk.conc_kidney_cm(self.ca,
                                  self.Fp, self.Eg, self.fc, self.Tg, self.Tv,
                                  self.Tpt, self.Tlh, self.Tdt, self.Tcd,
                                  t=self.t, dt=self.dt, sum=sum, kinetics='7C')

    def predict(self, xdata: np.ndarray) -> tuple:
        """Predict the data at given xdata

        Args:
            xdata (array-like): Either an array with x-values (time points) or a tuple with multiple such arrays

        Returns:
            tuple[np.ndarray,np.ndarray]: tuple of two 1D arrays with signal in corex and medulla, respectively.
        """
        Cc, Cm = self.conc()
        r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
        R1c = self.R10c + r1*Cc
        R1m = self.R10m + r1*Cm
        if self.sequence == 'SR':
            Sc = sig.signal_spgr(self.S0c, R1c, self.TC, self.TR,
                               self.FA, self.Tsat)
            Sm = sig.signal_spgr(self.S0m, R1m, self.TC, self.TR,
                               self.FA, self.Tsat)
        elif self.sequence == 'SS':
            Sc = sig.signal_ss(self.S0c, R1c, self.TR, self.FA)
            Sm = sig.signal_ss(self.S0m, R1m, self.TR, self.FA)
        t = self.time()
        return (
            utils.sample(xdata, t, Sc, self.TS),
            utils.sample(xdata, t, Sm, self.TS),
        )

    def train(self, xdata: np.ndarray, ydata: tuple[np.ndarray, np.ndarray], **kwargs):
        """Train the free parameters

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (tuple): Tuple of two 1D arrays (signal_cortex, signal_mdeulla) with signals in cortex and medulla, respectively.
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """

        if self.sequence == 'SR':
            Scref = sig.signal_spgr(1, self.R10c, self.TC, self.TR,
                                  self.FA, self.Tsat)
            Smref = sig.signal_spgr(1, self.R10m, self.TC, self.TR,
                                  self.FA, self.Tsat)
        elif self.sequence == 'SS':
            Scref = sig.signal_ss(1, self.R10c, self.TR, self.FA)
            Smref = sig.signal_ss(1, self.R10m, self.TR, self.FA)
        self.S0c = np.mean(ydata[0][:self.n0]) / Scref
        self.S0m = np.mean(ydata[1][:self.n0]) / Smref
        return ui.train(self, xdata, ydata, **kwargs)

    def export_params(self):
        pars = {}
        pars['Fp'] = ['Plasma flow', self.Fp, 'mL/sec/cm3']
        pars['Eg'] = ['Glomerular extraction fraction', self.Eg, '']
        pars['fc'] = ['Cortical flow fraction', self.fc, '']
        pars['Tg'] = ['Glomerular mean transit time', self.Tg, 'sec']
        pars['Tv'] = ['Peritubular & venous mean transit time', self.Tv, 'sec']
        pars['Tpt'] = ['Proximal tubuli mean transit time', self.Tpt, 'sec']
        pars['Tlh'] = ['Lis of Henle mean transit time', self.Tlh, 'sec']
        pars['Tdt'] = ['Distal tubuli mean transit time', self.Tdt, 'sec']
        pars['Tcd'] = ['Collecting duct mean transit time', self.Tcd, 'sec']
        pars['FF'] = ['Filtration fraction', self.Eg/(1-self.Eg), '']
        pars['Ft'] = ['Tubular flow', self.Fp *
                      self.Eg/(1-self.Eg), 'mL/sec/cm3']
        pars['CBF'] = ['Cortical blood flow',
                       self.Fp/(1-self.Hct), 'mL/sec/cm3']
        pars['MBF'] = ['Medullary blood flow',
                       (1-self.fc)*(1-self.Eg)*self.Fp/(1-self.Hct), 'mL/sec/cm3']
        if self.vol is None:
            return self._add_sdev(pars)
        pars['SKGFR'] = ['Single-kidney glomerular filtration rate',
                         self.vol*self.Fp*self.Eg/(1-self.Eg), 'mL/sec']
        pars['SKBF'] = ['Single-kidney blood flow',
                        self.vol*self.Fp/(1-self.Hct), 'mL/sec']
        pars['SKMBF'] = ['Single-kidney medullary blood flow', self.vol *
                         (1-self.fc)*(1-self.Eg)*self.Fp/(1-self.Hct), 'mL/sec']

        return self._add_sdev(pars)

    def plot(self, xdata: np.ndarray, ydata: tuple[np.ndarray, np.ndarray],
             ref=None, xlim=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (tuple of arrays): Tuple of two 1D arrays (signal_cortex, signal_mdeulla) with signals in cortex and medulla, respectively.
            xlim (array_like, optional): 2-element array with lower and upper boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form (x,y), where x is an array with x-values and y is an array with y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """

        time = self.time()
        if xlim is None:
            xlim = [np.amin(time), np.amax(time)]
        roic_pred, roim_pred = self.predict(time)
        concc, concm = self.conc()

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(xdata/60, ydata[0], marker='o', linestyle='None',
                 color='cornflowerblue', label='Cortex data')
        ax0.plot(xdata/60, ydata[1], marker='x', linestyle='None',
                 color='cornflowerblue', label='Medulla data')
        ax0.plot(time/60, roic_pred, linestyle='-', linewidth=3.0,
                 color='darkblue', label='Cortex prediction')
        ax0.plot(time/60, roim_pred, linestyle='--', linewidth=3.0,
                 color='darkblue', label='Medulla prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)',
                xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations.')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['Cc'], marker='o', linestyle='None',
                     color='cornflowerblue', label='Cortex ground truth')
            ax1.plot(ref['t']/60, 1000*ref['Cm'], marker='x', linestyle='None',
                     color='cornflowerblue', label='Medulla ground truth')
            ax1.plot(ref['t']/60, 1000*ref['cp'], marker='o', linestyle='None',
                     color='lightcoral', label='Arterial ground truth')
        ax1.plot(time/60, 1000*concc, linestyle='-', linewidth=3.0,
                 color='darkblue', label='Cortex prediction')
        ax1.plot(time/60, 1000*concm, linestyle='--', linewidth=3.0,
                 color='darkblue', label='Medulla prediction')
        ax1.plot(time/60, 1000*self.ca, linestyle='-', linewidth=3.0,
                 color='darkred', label='Arterial prediction')
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)',
                xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def cost(self, xdata: np.ndarray, ydata: tuple[np.ndarray, np.ndarray], metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            xdata (array-like): Array with x-data (time points).
            ydata (tuple): Tuple of two 1D arrays (signal_cortex, signal_mdeulla) with signals in cortex and medulla, respectively.
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
