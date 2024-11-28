import matplotlib.pyplot as plt
import numpy as np

import dcmri.pk as pk
import dcmri.kidney as kidney
import dcmri.lib as lib
import dcmri.ui as ui
import dcmri.sig as sig
import dcmri.utils as utils


class Kidney(ui.Model):
    """General model for whole kidney signals.

        **Input function**

        - **aif** (array-like, default=None): Signal-time curve in a feeding artery. If AIF is set to None, then the parameter ca must be provided (arterial concentrations).
        - **ca** (array-like, default=None): Concentration (M) in the arterial input. Must be provided when aif = None, ignored otherwise. 

        **Acquisition parameters**

        - **t** (array-like, default=None): Time points (sec) of the aif. If t is not provided, the temporal sampling is uniform with interval dt.
        - **dt** (float, default=1.0): Sampling interval of the AIF in sec.
        - **field_strength** (float, default=3.0): Magnetic field strength in T.
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **n0** (int, default=1): Baseline length in nr of acquisitions.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec.
        - **FA** (float, default=15): Nominal flip angle in degrees.

        **Tracer-kinetic parameters**

        - **H** (float, optional): Hematocrit. 
        - **Fp** (Plasma flow, mL/sec/cm3): Flow of plasma into the plasma compartment.
        - **Tp** (Plasma mean transit time, sec): Transit time of the plasma compartment. 
        - **Ft** (Tubular flow, mL/sec/cm3): Flow of fluid into the tubuli.
        - **Ta** (Arterial delay time, sec): Transit time through the arterial compartment.
        - **Tt** (Tubular mean transit time, sec): Transit time of the tubular compartment. 

        **Signal parameters**

        - **R10** (float, default=1): Precontrast tissue relaxation rate in 1/sec.
        - **R10a** (float, default=1): Precontrast arterial relaxation rate in 1/sec. 
        - **S0** (float, default=1): Scale factor for the MR signal (a.u.).

        **Prediction and training parameters**

        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **free** (array-like): 2-element list with lower and upper free of the free parameters. The default depends on the kinetics parameter.

        **Additional parameters**

        - **vol** (float, optional): Kidney volume in mL.

    See Also:
        `Liver`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_tissue(R10=1/dc.T1(3.0,'kidney'))

        Override the parameter defaults to match the experimental conditions of the synthetic test data:

        >>> params = {
        ...     'aif':aif, 
        ...     'dt':time[1], 
        ...     'agent': 'gadodiamide', 
        ...     'TR': 0.005, 
        ...     'FA': 15, 
        ...     'R10': 1/dc.T1(3.0,'kidney'), 
        ...     'n0': 15,
        ... }

        Train a two-compartment filtration model on the ROI data and plot the fit:

        >>> params['kinetics'] = '2CF'
        >>> model = dc.Kidney(**params).train(time, roi)
        >>> model.plot(time, roi, ref=gt)
    """

    def __init__(self, 
                 kinetics='2CF', sequence='SS', 
                 aif=None, ca=None, t=None, dt=1.0,
                 free=None, **params):

        # Config
        self.kinetics = kinetics
        self.sequence = sequence
        self._check_config()
        
        # Input function
        self.aif = aif
        self.ca = ca
        self.t = t
        self.dt = dt

        # overide defaults
        self._set_defaults(free=free, **params)


    def _check_config(self):
        if self.kinetics not in ['2CF']:
            msg = 'Kinetic model ' + str(self.kinetics) + ' is not available.'
            raise ValueError(msg)
        if self.sequence not in ['SS', 'SR', 'lin']:
            msg = 'Sequence ' + str(self.sequence) + ' is not available.'
            raise ValueError(msg)

    def _params(self):
        return PARAMS
    
    def _model_pars(self):
        seq_pars = {
            'SS': ['B1corr', 'FA', 'TR'],
            'SR': ['B1corr', 'FA', 'TR', 'TC', 'TP'],
            'lin': [],
        }
        pars = ['field_strength', 'agent', 'vol', 'H']
        pars += ['R10a', 'B1corr_a']
        pars += ['S0', 'TS'] + seq_pars[self.sequence]
        pars += kidney.params_kidney(self.kinetics)
        pars += ['Ta', 'R10', 'n0']
        return pars
    
    def _par_values(self, export=False):

        if export:
            pars = self._par_values()
            p0 = self._model_pars()
            p1 = kidney.params_kidney(self.kinetics)
            discard = set(p0) - set(p1) - set(self.free.keys())
            return {p: pars[p] for p in pars.keys() if p not in discard}
        
        pars = self._model_pars()
        p = {par: getattr(self, par) for par in pars}
        try:
            p['Fb'] = _div(p['Fp'], 1 - p['H'])
        except KeyError:
            pass
        try:
            p['Ft'] = p['FF']*p['Fp']
        except KeyError:
            pass
        try:
            p['Tp'] = _div(p['vp'], p['Fp']+p['Ft'])
        except KeyError:
            pass
        try:
            p['ve'] = p['Fp'] * p['Tp']
        except KeyError:
            pass
        try:
            p['E'] = _div(p['Ft'], p['Ft'] + p['Fp'])
        except KeyError:
            pass
        if p['vol'] is not None:
            try:
                p['SK-GFR'] = p['Ft'] * p['vol']  
            except KeyError:
                pass      
            try:
                p['SK-RBF'] = _div(p['Fp']*p['vol'], 1-p['H'])
            except KeyError:
                pass  

        return p
    

    def time(self):
        """Array of time points.

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
        
    def _check_ca(self):

        # Arterial blood concentrations
        if self.ca is None:
            if self.aif is None:
                raise ValueError(
                    "Either aif or ca must be provided \
                    to predict signal data.")
            else:
                r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
                if self.sequence == 'SR':
                    self.ca = sig.conc_src(
                        self.aif, self.TC, 1 / self.R10a, r1, self.n0)
                elif self.sequence == 'SS':
                    self.ca = sig.conc_ss(
                        self.aif, self.TR, self.B1corr_a * self.FA,
                        1 / self.R10a, r1, self.n0)
                elif self.sequence == 'lin':
                    self.ca = sig.conc_lin(self.aif, 1/self.R10a, r1, self.n0)


    def conc(self, sum=True):
        """Tissue concentration

        Args:
            sum (bool, optional): If True, returns the total concentrations. 
              Else returns the concentration in the individual compartments. 
              Defaults to True.

        Returns:
            numpy.ndarray: Concentration in M
        """

        self._check_ca()
        if self.kinetics == '2CF':
            ca = pk.flux(self.ca, self.Ta, t=self.t, dt=self.dt, model='plug')
            return kidney.conc_kidney(
                ca / (1-self.H), 
                self.Fp, self.vp, self.FF*self.Fp, self.Tt, 
                t=self.t, dt=self.dt, sum=sum, kinetics='2CF')

    def relax(self):
        """Longitudinal relaxation rate R1(t).

        Returns:
            np.ndarray: Relaxation rate. Dimensions are (nt,) for a tissue in 
              fast water exchange, or (nc,nt) for a multicompartment tissue 
              outside the fast water exchange limit.
        """
        C = self.conc()
        r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
        R1 = self.R10 + r1*C
        return R1

    def signal(self) -> np.ndarray:
        """Pseudocontinuous signal S(t) as a function of time.

        Returns:
            np.ndarray: Signal as a 1D array.
        """
        R1 = self.relax()
        if self.sequence == 'SR':
            return sig.signal_spgr(
                self.S0, R1, self.TC, self.TR, self.B1corr * self.FA, self.TP)
        elif self.sequence == 'SS':
            return sig.signal_ss(self.S0, R1, self.TR, self.B1corr * self.FA)
        elif self.sequence == 'lin':
            return sig.signal_lin(R1, self.S0)

    def predict(self, xdata):
        t = self.time()
        if np.amax(xdata) > np.amax(t):
            raise ValueError(
                'The acquisition window is longer than the duration of '
                'the AIF. The largest time point that can be '
                'predicted is ' + str(np.amax(t)/60) + 'min.')
        sig = self.signal()
        return utils.sample(xdata, t, sig, self.TS)

    def train(self, xdata, ydata, **kwargs):
        if self.sequence == 'SR':
            Sref = sig.signal_spgr(
                1, self.R10, self.TR, self.TC, self.B1corr * self.FA, self.TP)
        elif self.sequence == 'SS':
            Sref = sig.signal_ss(1, self.R10, self.TR, self.B1corr * self.FA)
        elif self.sequence == 'lin':
            Sref = sig.signal_lin(self.R10, 1)
        self.S0 = np.mean(ydata[:self.n0]) / Sref
        return ui.train(self, xdata, ydata, **kwargs)

    def plot(self,
             xdata: np.ndarray,
             ydata: np.ndarray,
             ref=None, xlim=None, fname=None, show=True):
        time = self.time()
        if xlim is None:
            xlim = [np.amin(time), np.amax(time)]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(xdata/60, ydata, marker='o', linestyle='None',
                 color='cornflowerblue', label='Data')
        ax0.plot(time/60, self.predict(time), linestyle='-',
                 linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)',
                xlim=np.array(xlim)/60)
        ax0.legend()

        C = self.conc(sum=False)
        ax1.set_title('Reconstruction of concentrations.')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['C'], marker='o', linestyle='None',
                     color='cornflowerblue', label='Tissue ground truth')
            ax1.plot(ref['t']/60, 1000*ref['cp'], marker='o', linestyle='None',
                     color='lightcoral', label='Arterial ground truth')
        ax1.plot(time/60, 1000*self.ca, linestyle='-', linewidth=3.0,
                 color='lightcoral', label='Artery')
        # ax1.plot(time/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0,
        #          color='darkblue', label='Tissue')
        ax1.plot(time/60, 1000*C[0,:], linestyle='-', linewidth=3.0,
                 color='darkred', label='Plasma')
        ax1.plot(time/60, 1000*C[1,:], linestyle='-', linewidth=3.0,
                 color='darkcyan', label='Tubuli')
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)',
                xlim=np.array(xlim)/60)
        ax1.legend()
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    
PARAMS = {
    'field_strength': {
        'init': 3,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Magnetic field strength',
        'unit': 'T',
    },
    'agent': {
        'init': 'gadoterate',
        'default_free': False,
        'bounds': None,
        'name': 'Contrast agent',
        'unit': '',
    },
    'R10a': {
        'init': 1/lib.T1(3.0, 'blood'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial precontrast R1',
        'unit': 'Hz',
    },
    'B1corr_a': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial B1-correction factor',
        'unit': '',
    },
    'Ta': {
        'init': 0,
        'default_free': True,
        'bounds': [0, 3],
        'name': 'Arterial mean transit time',
        'unit': 'sec',
    },
    'Fp': {
        'init': 0.02,
        'default_free': True,
        'bounds': [0, 0.05],
        'name': 'Plasma flow',
        'unit': 'mL/sec/cm3',
    },
    'vp': {
        'init': 0.15,
        'default_free': True,
        'bounds': [0,0.3],
        'name': 'Plasma volume',
        'unit': 'mL/cm3',
    },
    'Tp': {
        'init': 5,
        'default_free': True,
        'bounds': [0, 10],
        'name': 'Plasma mean transit time',
        'unit': 'sec',
    },
    'FF': {
        'init': 0.1,
        'default_free': True,
        'bounds': [0, 0.3],
        'name': 'Filtration fraction',
        'unit': '',
    },
    'Ft': {
        'init': 30/6000,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Tubular flow',
        'unit': 'mL/sec/cm3',
    },
    'Tt': {
        'init': 120,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Tubular mean transit time',
        'unit': 'sec',
    },
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [1e-3, 1 - 1e-3],
        'name': 'Tissue Hematocrit',
        'unit': '',
    },
    'B1corr': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Tissue B1-correction factor',
        'unit': '',
    },
    'FA': {
        'init': 15,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Flip angle',
        'unit': 'deg',
    },
    'TR': {
        'init': 0.005,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Repetition time',
        'unit': 'sec',
    },
    'TC': {
        'init': 0.2,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Time to k-space center',
        'unit': 'sec',
    },
    'TP': {
        'init': 0.05,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Preparation delay',
        'unit': 'sec',
    },
    'TS': {
        'init': 0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Sampling time',
        'unit': 'sec',
    },
    'R10': {
        'init': 1/lib.T1(3.0, 'kidney'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Tissue precontrast R1',
        'unit': 'Hz',
    },
    'S0': {
        'init': 1.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Signal scaling factor',
        'unit': 'a.u.',
    },
    'n0': {
        'init': 1,
        'default_free': False,
        'bounds': None,
        'name': 'Number of precontrast acquisitions',
        'unit': '',
    },
    'vol': {
        'init': None,
        'default_free': False,
        'bounds': None,
        'name': 'Single-kidney volume',
        'unit': 'mL',
    },

    # Derived parameters

    'Fb': {
        'name': 'Blood flow',
        'unit': 'mL/sec/cm3',
    },
    've': {
        'name': 'Extracellular volume',
        'unit': 'mL/cm3',
    },
    'E': {
        'name': 'Extraction fraction',
        'unit': '',
    },
    'SK-GFR': {
        'name': 'Single-kidney glomerular filtration rate',
        'unit': 'mL/sec',
    },
    'SK-RBF': {
        'name': 'Single-kidney renal blood flow',
        'unit': 'mL/sec',
    },
    'FAcorr': {
        'name': 'B1-corrected Flip Angle',
        'unit': 'deg',
    },

}


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b == 0, 0, np.divide(a, b))



