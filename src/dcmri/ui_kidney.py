import matplotlib.pyplot as plt
import numpy as np

import dcmri.pk as pk
import dcmri.kidney as kidney
import dcmri.lib as lib
import dcmri.ui as ui
import dcmri.sig as sig
import dcmri.utils as utils


class Kidney(ui.Model):
    """General model for whole-kidney signals.

    See Also:
        `Liver`, `Tissue`

    Args:
        kinetics (str, optional): Kinetic model for the kidneys. 
          Options are '2CF' (Two-compartment filtration) and 'HF' 
          (High-flow). Defaults to '2CF'. 
        sequence (str, optional): imaging sequence model. Possible 
          values are 'SS' (steady-state), 'SR' (saturation-recovery), 
          and 'lin' (linear). Defaults to 'SS'.
        params (dict, optional): values for the model parameters,
          specified as keyword parameters. Defaults are used for any 
          that are not provided. See table 
          :ref:`Kidney-defaults` for a list of parameters and 
          their default values.

    Notes:

        In the table below, if **Bounds** is None, the parameter is fixed 
        during training. Otherwise it is allowed to vary between the 
        bounds given.

        .. _Kidney-defaults:
        .. list-table:: Kidney parameters. 
            :widths: 15 10 10 10
            :header-rows: 1

            * - Parameter
              - Value
              - Bounds
              - Usage
            * - **General**
              - 
              - 
              - 
            * - field_strength
              - 3
              - None
              - Always
            * - agent
              - 'gadoterate'
              - None
              - Always
            * - t0
              - 0
              - None
              - Always
            * - **Sequence**
              -
              - 
              - 
            * - TS
              - 0
              - None
              - Always
            * - B1corr
              - 1
              - None
              - sequence in ['SS']
            * - FA
              - 15
              - None
              - sequence in ['SR', 'SS']
            * - TR
              - 0.005
              - None
              - sequence in ['SS']
            * - TC
              - 0.1
              - None
              - sequence == 'SR'
            * - TP
              - 0.05
              - None
              - sequence == 'SR'
            * - **AIF**
              - 
              - 
              - 
            * - B1corr_a
              - 1
              - None
              - sequence in ['SS']
            * - R10a
              - 0.7
              - None
              - Always
            * - **Kidney**
              -
              - 
              - 
            * - H
              - 0.45
              - None
              - Always
            * - Ta
              - 0
              - [0, 3]
              - Always
            * - vol
              - 150
              - None
              - Always
            * - Fp
              - 0.02
              - [0, 0.05]
              - kinetics == '2CF'
            * - vp
              - 0.15
              - [0, 0.3]
              - Always
            * - FF
              - 0.1
              - [0, 0.3]
              - kinetics == '2CF'
            * - Tt
              - 120
              - [0, inf]
              - Always
            * - Ft
              - 0.005
              - [0, 0.05]
              - Always
            * - R10
              - 0.65
              - None
              - Always
            * - S0
              - 1.0
              - [0, inf]
              - Always

    Example:

        Use the model to fit minipig data. The AIF is corrupted by 
        inflow effects so for the purpose of this example we will 
        use a standard input function:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import numpy as np
        >>> import pydmr
        >>> import dcmri as dc

        Read the dataset:

        >>> datafile = dc.fetch('minipig_renal_fibrosis')
        >>> data = pydmr.read(datafile, 'nest')
        >>> rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']
        >>> time = pars['TS'] * np.arange(len(rois['LeftKidney']))

        Generate an AIF at high temporal resolution (250 msec):

        >>> dt = 0.25
        >>> t = np.arange(0, np.amax(time) + dt, dt) 
        >>> ca = dc.aif_tristan(
        ...    t, 
        ...    agent="gadoterate",
        ...    dose=pars['dose'],
        ...    rate=pars['rate'],
        ...    weight=pars['weight'],
        ...    CO=60,
        ...    BAT=time[np.argmax(rois['Aorta'])] - 20,
        >>> )        

        Initialize the tissue:

        >>> kidney = dc.Kidney(
        ...    ca=ca,
        ...    dt=dt,
        ...    kinetics='HF',
        ...    field_strength=pars['B0'],
        ...    agent="gadoterate",
        ...    t0=pars['TS'] * pars['n0'],
        ...    TS=pars['TS'], 
        ...    TR=pars['TR'],
        ...    FA=pars['FA'],
        ...    R10a=1/dc.T1(pars['B0'], 'blood'),
        ...    R10=1/dc.T1(pars['B0'], 'kidney'),
        >>> )

        Train the kidney on the data:

        >>> kidney.set_free(Ta=[0,30])
        >>> kidney.train(time, rois['LeftKidney'])
        
        Plot the reconstructed signals and concentrations:

        >>> kidney.plot(time, rois['LeftKidney'])

        Print the model parameters:

        >>> kidney.print_params(round_to=4)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        Arterial mean transit time (Ta): 13.8658 (0.1643) sec
        Plasma volume (vp): 0.0856 (0.003) mL/cm3
        Tubular flow (Ft): 0.0024 (0.0001) mL/sec/cm3
        Tubular mean transit time (Tt): 116.296 (7.6526) sec

    """

    def __init__(
            self, 
            kinetics='2CF', 
            sequence='SS', 
            aif=None, 
            ca=None, 
            t=None, 
            dt=1.0,
            **params,
        ):

        # Check configuration
        if kinetics not in ['2CF', 'HF']:
            raise ValueError(
                f"Kinetic model {kinetics} is not available."
            )
        if sequence not in ['SS', 'SR', 'lin']:
            raise ValueError(
                f"Sequence ' + str(sequence) + ' is not available."
            )
        
        # Config
        self.kinetics = kinetics
        self.sequence = sequence
        
        # Input function
        self.aif = aif
        self.ca = ca
        self.t = t
        self.dt = dt

        # overide defaults
        self._set_defaults(**params)


    def _params(self):
        return PARAMS
    
    def _model_pars(self):

        # General
        pars = ['field_strength', 'agent', 't0']

        # Sequence
        pars += ['TS']
        if self.sequence == 'SR':
            pars += ['B1corr', 'FA', 'TR', 'TC', 'TP']
        elif self.sequence=='SS':
            pars += ['B1corr', 'FA', 'TR']

        # AIF
        if self.aif is not None:
            pars += ['B1corr_a', 'R10a']

        # Kidney
        pars += ['H', 'Ta', 'vol']
        pars += kidney.params_kidney(self.kinetics)
        pars += ['R10', 'S0']
        
        return pars


    def _par_values(self, export=False):

        if export:
            discard = [
                'field_strength', 'agent', 't0', 'TS', 'FA', 'TR', 
                'TC', 'TP', 'H', 'vol', 'R10', 'S0',
            ]
            pars = self._par_values()
            return {p: pars[p] for p in pars.keys() if p not in discard}
        
        pars = self._model_pars()
        p = {par: getattr(self, par) for par in pars}

        if {'Fp', 'H'}.issubset(p):
            p['Fb'] = _div(p['Fp'], 1 - p['H'])
        if {'FF', 'Fp'}.issubset(p):
            p['Ft'] = p['FF']*p['Fp']
        if {'vp', 'Fp', 'Tt'}.issubset(p):
            p['Tp'] = _div(p['vp'], p['Fp']+p['Ft'])
        if {'vp', 'Fp'}.issubset(p):
            p['Tv'] = _div(p['vp'], p['Fp'])
        if {'Ft', 'Fp'}.issubset(p):
            p['E'] = _div(p['Ft'], p['Ft'] + p['Fp'])
        if p['vol'] is not None:
            if {'Ft'}.issubset(p):
                p['GFR'] = p['Ft'] * p['vol']      
            if {'Fp', 'H', 'vol'}.issubset(p):
                p['RBF'] = _div(p['Fp']*p['vol'], 1-p['H'])
                p['RPF'] = p['Fp']*p['vol']

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
        
    def _compute_ca(self):

        # Arterial blood concentrations
        if self.ca is None:
            if self.aif is None:
                raise ValueError(
                    "Either aif or ca must be provided \
                    to predict signal data.")
            else:
                r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
                t = self.time()
                n0 = int(max([np.sum(t < self.t0), 1]))
                if self.sequence == 'SR':
                    self.ca = sig.conc_src(
                        self.aif, self.TC, 1 / self.R10a, r1, n0)
                elif self.sequence == 'SS':
                    self.ca = sig.conc_ss(
                        self.aif, self.TR, self.B1corr_a * self.FA,
                        1 / self.R10a, r1, n0)
                elif self.sequence == 'lin':
                    self.ca = sig.conc_lin(self.aif, 1/self.R10a, r1, n0)


    def conc(self, sum=True):
        """Tissue concentration

        Args:
            sum (bool, optional): If True, this returns the total 
              concentration. Else the function returns the 
              concentration in the individual compartments. Defaults 
              to True.

        Returns:
            numpy.ndarray: Concentration in M
        """

        self._compute_ca()
        ca = pk.flux(
            self.ca, self.Ta, t=self.t, dt=self.dt, model='plug',
        )
        if self.kinetics == '2CF':
            return kidney.conc_kidney(
                ca / (1-self.H), 
                self.Fp, self.vp, self.FF*self.Fp, self.Tt, 
                t=self.t, dt=self.dt, sum=sum, kinetics='2CF',
            )
        elif self.kinetics == 'HF':
            return kidney.conc_kidney(
                ca / (1-self.H), 
                self.vp, self.Ft, self.Tt,
                t=self.t, dt=self.dt, sum=sum, kinetics='HF',
            )

    def relax(self):
        """Longitudinal relaxation rate R1(t).

        Returns:
            np.ndarray: Relaxation rate. Dimensions are (nt,) for a tissue in 
              fast water exchange, or (nc,nt) for a multicompartment tissue 
              outside the fast water exchange limit.
        """
        C = self.conc()
        rb = lib.relaxivity(self.field_strength, 'blood', self.agent)
        return self.R10 + rb*C

    def signal(self) -> np.ndarray:
        """Pseudocontinuous signal S(t) as a function of time.

        Returns:
            np.ndarray: Signal as a 1D array.
        """
        R1 = self.relax()
        if self.sequence == 'SR':
            return sig.signal_spgr(
                self.S0, R1, self.TC, self.TR, 
                self.B1corr * self.FA, self.TP,
            )
        elif self.sequence == 'SS':
            return sig.signal_ss(
                self.S0, R1, self.TR, self.B1corr * self.FA,
            )
        elif self.sequence == 'lin':
            return sig.signal_lin(self.S0, R1)

    def predict(self, xdata):
        t = self.time()
        if np.amax(xdata) > np.amax(t):
            raise ValueError(
                f"The acquisition window is longer than the duration "
                f"of the AIF. The largest time point that can be "
                f"predicted is {np.amax(t)/60} min."
            )
        sig = self.signal()
        return utils.sample(xdata, t, sig, self.TS)

    def train(self, xdata, ydata, **kwargs):
        if self.sequence == 'SR':
            Sref = sig.signal_spgr(
                1, self.R10, self.TR, self.TC, 
                self.B1corr * self.FA, self.TP,
            )
        elif self.sequence == 'SS':
            Sref = sig.signal_ss(
                1, self.R10, self.TR, self.B1corr * self.FA,
            )
        elif self.sequence == 'lin':
            Sref = sig.signal_lin(1, self.R10)
        n0 = int(max([np.sum(xdata < self.t0), 1]))
        self.S0 = np.mean(ydata[:n0]) / Sref
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
        ax1.set_title('Reconstruction of concentrations')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['C'], marker='o', linestyle='None',
                     color='cornflowerblue', label='Tissue ground truth')
            ax1.plot(ref['t']/60, 1000*ref['cp'], marker='o', linestyle='None',
                     color='lightcoral', label='Arterial ground truth')
        ax1.plot(time/60, 1000*self.ca, linestyle='-', linewidth=3.0,
                 color='lightcoral', label='Artery')
        # ax1.plot(time/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0,
        #          color='darkblue', label='Tissue')
        #vb = self.vp / (1-self.H)
        ax1.plot(time/60, 1000*C[0,:], linestyle='-', linewidth=3.0,
                 color='darkred', label='Blood')
        # ax1.plot(time/60, 1000*C[0,:], linestyle='-', linewidth=3.0,
        #          color='darkred', label='Blood')
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

    # General

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
    't0': {
        'init': 0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Baseline duration',
        'unit': '',
    },

    # Sequence

    'TS': {
        'init': 0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Sampling time',
        'unit': 'sec',
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

    # AIF

    'B1corr_a': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial B1-correction factor',
        'unit': '',
    },
    'R10a': {
        'init': 1/lib.T1(3.0, 'blood'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial precontrast R1',
        'unit': 'Hz',
    },

    # Kidney

    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [1e-3, 1 - 1e-3],
        'name': 'Tissue Hematocrit',
        'unit': '',
    },
    'Ta': {
        'init': 0,
        'default_free': True,
        'bounds': [0, 3],
        'name': 'Arterial mean transit time',
        'unit': 'sec',
    },
    'vol': {
        'init': None,
        'default_free': False,
        'bounds': None,
        'name': 'Single-kidney volume',
        'unit': 'mL',
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
        'bounds': [0, 0.3],
        'name': 'Plasma volume',
        'unit': 'mL/cm3',
    },
    'FF': {
        'init': 0.1,
        'default_free': True,
        'bounds': [0, 0.3],
        'name': 'Filtration fraction',
        'unit': '',
    },
    'Tt': {
        'init': 120,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Tubular mean transit time',
        'unit': 'sec',
    },
    'Ft': {
        'init': 0.005,
        'default_free': True,
        'bounds': [0, 0.05],
        'name': 'Tubular flow',
        'unit': 'mL/sec/cm3',
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


    # Derived parameters

    'Fb': {
        'name': 'Blood flow',
        'unit': 'mL/sec/cm3',
    },
    'Tv': {
        'name': 'Vascular mean transit time',
        'unit': 'sec',
    },
    'Tp': {
        'name': 'Plasma mean transit time',
        'unit': 'sec',
    },
    'E': {
        'name': 'Extraction fraction',
        'unit': '',
    },
    'GFR': {
        'name': 'Glomerular filtration rate',
        'unit': 'mL/sec',
    },
    'RBF': {
        'name': 'Renal blood flow',
        'unit': 'mL/sec',
    },
    'RPF': {
        'name': 'Renal plasma flow',
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



