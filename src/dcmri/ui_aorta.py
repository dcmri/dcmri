from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import dcmri.ui as ui
import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta


class Aorta(ui.Model):
    """Whole-body model for the aorta signal.

    This model uses a whole body model to predict the signal in the 
    aorta (see :ref:`whole-body-tissues`). 

    See Also:
        `AortaKidneys`, `AortaLiver`

    Args:
        organs (str, optional): Model for the organs in the whole-body 
          model. The options are 'comp' (one compartment) and '2cxm' 
          (two-compartment exchange). Defaults to 'comp'.
        heartlung (str, optional): Model for the heart-lung system in 
          the whole-body model. Options are 'comp' (compartment), 
          'pfcomp' (plug-flow compartment) or 'chain'. Defaults to 
          'pfcomp'.
        sequence (str, optional): imaging sequence model. Possible 
          values are 'SS' (steady-state), 'SR' (saturation-recovery), 
          'SSI' (steady state with inflow correction) and 'lin' 
          (linear). Defaults to 'SS'.
        params (dict, optional): values for the model parameters,
          specified as keyword parameters. Defaults are used for any 
          that are not provided. See table 
          :ref:`Aorta-defaults` for a list of parameters and 
          their default values.

    Notes:

        In the table below, if **Bounds** is None, the parameter is fixed 
        during training. Otherwise it is allowed to vary between the 
        bounds given.

        .. _Aorta-defaults:
        .. list-table:: Aorta parameters. 
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
            * - dt
              - 0.25
              - None
              - Always
            * - tmax
              - 120
              - None
              - Always
            * - dose_tolerance
              - 0.1
              - None
              - Always
            * - t0
              - 0
              - None
              - Always
            * - field_strength
              - 3
              - None
              - Always
            * - **Injection**
              -
              - 
              - 
            * - agent
              - 'gadoxetate'
              - None
              - Always
            * - weight
              - 70
              - None
              - Always
            * - dose
              - 0.0125
              - None
              - Always
            * - rate
              - 1
              - None
              - Always
            * - BAT
              - 60
              - [0, inf]
              - Always
            * - **Sequence**
              -
              - 
              - 
            * - TS
              - 0
              - None
              - Always
            * - TR
              - 0.005
              - None
              - sequence in ['SS', 'SSI']
            * - FA
              - 15
              - None
              - sequence in ['SR', 'SS', 'SSI']
            * - TC
              - 0.1
              - None
              - sequence == 'SR'
            * - TF
              - 0
              - None
              - sequence == 'SSI'
            * - **Aorta**
              -
              - 
              - 

            * - CO
              - 100
              - [0, 300]
              - Always
            * - Thl
              - 10
              - [0, 30]
              - Always
            * - Dhl
              - 0.2
              - [0.05, 0.95]
              - heartlung in ['pfcomp', 'chain']
            * - To
              - 20
              - [0, 60]
              - Always
            * - Eo
              - 0.15
              - [0, 0.5]
              - organs == '2cxm'
            * - Toe
              - 120
              - [0, 800]
              - organs == '2cxm'
            * - Eb
              - 0.05
              - [0.01, 0.15]
              - Always
            * - R10
              - 0.7
              - None
              - Always
            * - S0
              - 1
              - None
              - Always


    Example:

        Use the model to fit minipig aorta data with inflow 
        correction:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import numpy as np
        >>> import dcmri as dc

        Read the dataset:

        >>> datafile = dc.fetch('minipig_renal_fibrosis')
        >>> rois, pars = dc.read_dmr(datafile, nest=True, valsonly=True)
        >>> rois, pars = rois['Pig']['Test'], pars['Pig']['Test']

        Initialize the tissue:

        >>> aorta = dc.Aorta(
        ...     sequence='SSI',
        ...     heartlung='chain',
        ...     organs='comp',
        ...     field_strength=pars['B0'],
        ...     t0=15, 
        ...     agent="gadoterate",
        ...     weight=pars['weight'],
        ...     dose=pars['dose'],
        ...     rate=pars['rate'],
        ...     TR=pars['TR'],
        ...     FA=pars['FA'],
        ...     TS=pars['TS'],
        ...     CO=60, 
        ...     R10=1/dc.T1(pars['B0'], 'blood'),
        ... )

        Create an array of time points:

        >>> time = pars['TS'] * np.arange(len(rois['Aorta']))

        Train the system to the data:

        >>> aorta.train(time, rois['Aorta'])

        Plot the reconstructed signals and concentrations:

        >>> aorta.plot(time, rois['Aorta'])

        Print the model parameters:

        >>> aorta.print_params(round_to=4)

    """

    def __init__(
            self, 
            organs='comp', 
            heartlung='pfcomp', 
            sequence='SS', 
            **params,
        ):

        # Check configuration
        if organs not in ['comp','2cxm']:
            raise ValueError(
                f"{organs} is not a valid model for the organs. "
                "Current options are 'comp' and '2cxm'."
            )
        if heartlung not in ['pfcomp', 'chain']:
            raise ValueError(
                f"{heartlung} is not a valid heart-lung system. "
                "Current options are 'pfcomp' and 'chain'."
            )
        if sequence not in ['SS', 'SR', 'SSI', 'lin']:
            raise ValueError(
                f"Sequence {sequence} is not available."
            )

        # Set configuration
        self.sequence = sequence
        self.organs = organs
        self.heartlung = heartlung

        # Set defaults
        self._set_defaults(**params)
        # For SSI, S0 needs to be free because TF affects the baseline
        if self.sequence == 'SSI':
            self.free['S0'] = [0, np.inf]
        
    def _params(self):
        return PARAMS
    
    def _model_pars(self):

        # General
        pars = ['dt', 'tmax', 'dose_tolerance', 't0', 'field_strength']

        # Injection
        pars += ['agent', 'weight', 'dose', 'rate', 'BAT']

        # Sequence
        pars += ['TS']
        if self.sequence == 'SR':
            pars += ['TC', 'FA']
        elif self.sequence=='SS':
            pars += ['TR', 'FA']
        elif self.sequence=='SSI':
            pars += ['TF', 'TR', 'FA']

        # Aorta
        pars += ['CO', 'Thl', 'To', 'Eb', 'R10', 'S0']
        if self.heartlung == 'pfcomp':
            pars += ['Dhl']
        elif self.heartlung == 'chain':
            pars += ['Dhl']
        if self.organs=='2cxm':
            pars += ['Toe', 'Eo']

        return pars


    def _par_values(self, export=False):

        if export:
            discard = [
                'dt', 'tmax', 't0', 'agent', 'weight', 'dose', 
                'rate', 'field_strength', 'dose_tolerance', 'R10', 
                'TS' , 'TC', 'TR', 'FA',
            ]
            pars = self._par_values()
            return {p: pars[p] for p in pars if p not in discard}
        
        pars = self._model_pars()
        p = {par: getattr(self, par) for par in pars}

        return p
    

    def conc(self):
        """Aorta blood concentration

        Args:
            t (array-like): Time points of the concentration (sec)

        Returns:
            numpy.ndarray: Concentration in M
        """
        if self.organs == 'comp':
            organs = ['comp', (self.To,)]
        elif self.organs=='2cxm':
            organs = ['2cxm', ([self.To, self.Toe], self.Eo)]
        if self.heartlung == 'comp':
            heartlung = ['comp', (self.Thl,)]
        elif self.heartlung == 'pfcomp':
            heartlung = ['pfcomp', (self.Thl, self.Dhl)]
        elif self.heartlung == 'chain':
            heartlung = ['chain', (self.Thl, self.Dhl)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = lib.ca_conc(self.agent)
        Ji = lib.ca_injection(
            self.t, self.weight, conc, self.dose, self.rate, self.BAT
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=self.Eb, heartlung=heartlung, organs=organs, 
            dt=self.dt, tol=self.dose_tolerance,
        )
        self.ca = Jb/self.CO
        return self.t, self.ca


    def relax(self):
        """Aorta longitudinal relation rate

        Args:
            t (array-like): Time points of the concentration (sec)

        Returns:
            numpy.ndarray: Concentration in M
        """
        t, cb = self.conc()
        rb = lib.relaxivity(self.field_strength, 'blood', self.agent)
        return t, self.R10 + rb*cb


    def predict(self, xdata) -> np.ndarray:
        tacq = xdata[1]-xdata[0]
        self.tmax = max(xdata)+tacq+self.dt
        if self.TS is not None:
            self.tmax += self.TS
        t, R1 = self.relax()
        if self.sequence == 'SR':
            signal = sig.signal_free(self.S0, R1, self.TC, self.FA)
        elif self.sequence=='SS':
            signal = sig.signal_ss(self.S0, R1, self.TR, self.FA)
        elif self.sequence=='SSI':
            signal = sig.signal_spgr(self.S0, R1, self.TF, self.TR, self.FA, n0=1)
        elif self.sequence == 'lin':
            signal = sig.signal_lin(self.S0, R1)
        return utils.sample(xdata, t, signal, self.TS)


    def train(self, xdata, ydata, **kwargs):
        if self.sequence == 'SR':
            Sref = sig.signal_free(1, self.R10, self.TC, self.FA)
        elif self.sequence=='SS':
            Sref = sig.signal_ss(1, self.R10, self.TR, self.FA)
        elif self.sequence=='SSI':
            Sref = sig.signal_spgr(1, self.R10, self.TF, self.TR, self.FA, n0=1)
        n0 = max([np.sum(xdata < self.t0), 1])
        self.S0 = np.mean(ydata[:n0]) / Sref
        self.BAT = xdata[np.argmax(ydata)] - self.Thl
        return ui.train(self, xdata, ydata, **kwargs)
    

    def plot(self, xdata: np.ndarray, ydata: np.ndarray,
             ref=None, xlim=None, fname=None, show=True):
        aif = self.predict(xdata)
        t, cb = self.conc()
        if xlim is None:
            xlim = [np.amin(xdata), np.amax(xdata)]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(xdata/60, ydata, color='lightcoral', marker='o', label='Data')
        ax0.plot(xdata/60, aif, color='darkred', linestyle='-', linewidth=3.0, label='Prediction')
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('MRI signal (a.u.)')
        ax0.legend()
        ax1.set_title('Prediction of the concentrations.')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['cb'], 'ro', label='Ground truth')
        ax1.plot(t/60, 1000*cb, color='darkred', linestyle='-', label='Prediction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Blood concentration (mM)')
        ax1.legend()
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


PARAMS = {
    # Prediction and training
    'dt': {
        'init': 0.25,
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
    't0': {
        'init': 0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Baseline duration',
        'unit': 'sec',
    },
    'field_strength': {
        'init': 3.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Magnetic field strength',
        'unit': 'T',
    },

    # Injection
    'agent': {
        'init': 'gadoterate',
        'default_free': False,
        'bounds': None,
        'name': 'Contrast agent',
        'unit': None,
    },
    'weight': {
        'init': 70,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Subject weight',
        'unit': 'kg',
    },
    'dose': {
        'init': lib.ca_std_dose('gadoterate'),
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

    # Sequence
    'TR': {
        'init': 0.005,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Repetition time',
        'unit': 'sec',
        'pixel_par': False,
    },
    'FA': {
        'init': 15,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Flip angle',
        'unit': 'deg',
        'pixel_par': False,
    },
    'TC': {
        'init': 0.2,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Time to k-space center',
        'unit': 'sec',
        'pixel_par': False,
    },
    'TS': {
        'init': 0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Sampling time',
        'unit': 'sec',
        'pixel_par': False,
    },
    'TF': {
        'init': 0.50,
        'default_free': True,
        'bounds': [0, 2],
        'name': 'Inflow time',
        'unit': 'sec',
        'pixel_par': False,
    },

    # Aorta
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
    'R10': {
        'init': 0.7,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Precontrast R1',
        'unit': 'Hz',
        'pixel_par': False,
    },
    'S0': {
        'init': 1.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Signal scaling factor',
        'unit': 'a.u.',
        'pixel_par': True,
    },

}
