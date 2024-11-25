import matplotlib.pyplot as plt
import numpy as np
import dcmri.ui as ui
import dcmri.lib as lib
import dcmri.liver as liver
import dcmri.sig as sig
import dcmri.utils as utils


class Liver(ui.Model):
    """General model for liver tissue.

    This is the standard interface for liver tissues with known input 
    function(s). For more detail see :ref:`liver-tissues`.

    Args:
        kinetics (str, optional): Tracer-kinetic model. See table 
          :ref:`table-liver-models` for options. Defaults to '2C-EC'.
        stationary (str, optional): Stationarity regime of the hepatocytes. 
          The options are 'UE', 'E', 'U' or None. For more detail 
          see :ref:`liver-tissues`. Defaults to 'UE'.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
        config (str, optional): configuration option for using pre-defined
          variable values from use cases. Currently, the available options
          for this are 'TRISTAN-rat'. Defaults to None.
        aif (array-like, optional): Signal-time curve in the blood of the
          feeding artery. If *aif* is not provided, the arterial
          blood concentration is *ca*. Defaults to None.
        ca (array-like, optional): Blood concentration in the arterial
          input. *ca* is ignored if *aif* is provided, but is required
          otherwise. Defaults to None.
        vif (array-like, optional): Signal-time curve in the blood of the
          portal vein. If *vif* is not provided, the venous
          blood concentration is *cv*. Defaults to None.
        cv (array-like, optional): Blood concentration in the portal venous
          input. *cv* is ignored if *vif* is provided, but is required
          otherwise. Defaults to None.
        t (array-like, optional): Time points of the arterial input function.
          If *t* is not provided, the temporal sampling is uniform with
          interval *dt*. Defaults to None.
        dt (float, optional): Time interval between values of the arterial
          input function. *dt* is ignored if *t* is provided. Defaults to 1.0.
        free (dict, optional): Dictionary with free parameters and their
          bounds. If not provided, a default set of free parameters is used.
          Defaults to None.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. See tables :ref:`Liver-parameters` and
          :ref:`Liver-defaults` for a list of parameters and their
          default values.

    See Also:
        `Tissue`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_liver` to generate synthetic test data:

        >>> time, aif, vif, roi, gt = dc.fake_liver()

        Build a tissue model and set the constants to match the experimental 
        conditions of the synthetic test data:

        >>> model = dc.Liver(
        ...     aif = aif,
        ...     dt = time[1],
        ...     agent = 'gadoxetate',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 15,
        ...     n0 = 10,
        ...     kinetics = '1I-IC-D',
        ...     R10 = 1/dc.T1(3.0,'liver'),
        ...     R10a = 1/dc.T1(3.0, 'blood'), 
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and 
        compare the concentrations against the noise-free ground truth:

        >>> model.plot(time, roi, ref=gt)

    Notes:

        Table :ref:`Liver-parameters` lists the parameters that are relevant 
        in each regime. Table :ref:`Liver-defaults` list all possible 
        parameters and their default settings. 

        .. _Liver-parameters:
        .. list-table:: **Liver parameters**
            :widths: 20 30 30
            :header-rows: 1

            * - Parameters
              - When to use
              - Further detail
            * - n0
              - Always
              - For estimating baseline signal
            * - field_strength, agent, R10
              - Always
              - :ref:`relaxation-params`
            * - R10a, B1corr_a
              - When aif is provided
              - :ref:`relaxation-params`, :ref:`params-per-sequence`
            * - R10v, B1corr_v
              - When vif is provided
              - :ref:`relaxation-params`, :ref:`params-per-sequence`
            * - S0, FA, TR, TS, B1corr
              - Always
              - :ref:`params-per-sequence`
            * - TP, TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - ve, Fp, fa, Ta, Tg, khe, khe_i, kh_f, Th, Th_i, Th_f.
              - Depends on **kinetics** and **stationary**
              - :ref:`table-liver-models`

        .. _Liver-defaults:
        .. list-table:: **Liver parameter defaults**
            :widths: 5 10 10 10 10
            :header-rows: 1

            * - Parameter
              - Type
              - Value
              - Bounds
              - Free/Fixed
            * - field_strength
              - Injection
              - 3
              - [0, inf]
              - Fixed
            * - agent
              - Injection
              - 'gadoxetate'
              - None
              - Fixed
            * - R10
              - Relaxation
              - 0.7
              - [0, inf]
              - Fixed
            * - R10a
              - Relaxation
              - 0.7
              - [0, inf]
              - Fixed
            * - R10v
              - Relaxation
              - 0.7
              - [0, inf]
              - Fixed
            * - B1corr
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - B1corr_a
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - B1corr_v
              - Sequence
              - 1
              - [0, inf]
              - Fixed
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
            * - TP
              - Sequence
              - 0
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
            * - Ta
              - Kinetic
              - 2
              - [0, inf]
              - Free
            * - Tg
              - Kinetic
              - 10
              - [0, inf]
              - Free
            * - Fp
              - Kinetic
              - 0.008
              - [0, inf]
              - Free
            * - fa
              - Kinetic
              - 0.2
              - [0, inf]
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
            kinetics='2C-EC', stationary='UE', sequence='SS', config=None,
            aif=None, ca=None, vif=None, cv=None, t=None, dt=0.5,
            free=None, **params):

        # Configuration
        if config == 'TRISTAN-rat':

            # Acquisition parameters
            params['agent'] = 'gadoxetate'

            # Kinetic paramaters
            self.kinetics = '1I-IC-HF'
            self.stationary = 'UE'
            self.sequence = 'SS'
            params['H'] = 0.418         # Cremer et al, J Cereb Blood Flow
                                        # Metab 3, 254-256 (1983)
            params['ve'] = 0.23
            params['Fp'] = 0.022019     # mL/sec/cm3
                                        # Fp = (1-H)*Fb, where Fb=2.27 mL/min/mL
                                        # calculated from Table S2 in 
                                        # doi: 10.1021/acs.molpharmaceut.1c00206
            free = {
                'khe': [0, np.inf], 
                'Th': [0, np.inf],
            }

            # Tissue paramaters
            params['R10'] = 1/lib.T1(params['field_strength'], 'liver'),
        
        else:
            self.kinetics = kinetics
            self.stationary = stationary
            self.sequence = sequence
        self._check_config()

        # Input function
        self.aif = aif
        self.ca = ca
        self.vif = vif
        self.cv = cv
        self.t = t
        self.dt = dt

        # Set defaults
        self._set_defaults(free=free, **params)


    def _check_config(self):
        if self.sequence not in ['SS', 'SR']:
            raise ValueError(
                'Sequence ' + str(self.sequence) + ' is not available.')
        liver.params_liver(self.kinetics, self.stationary)

    def _params(self):
        return PARAMS
    
    def _model_pars(self):
        pars_sequence = {
            'SR': ['S0', 'B1corr', 'FA', 'TR', 'TS', 'TC', 'TP'],
            'SS': ['S0', 'B1corr', 'FA', 'TR', 'TS'], 
        }    
        pars = ['field_strength', 'agent']
        pars += ['R10a', 'B1corr_a']
        if self.kinetics[0] == '2':
            pars += ['R10v', 'B1corr_v']
        pars += pars_sequence[self.sequence]
        pars += liver.params_liver(self.kinetics, self.stationary)
        pars += ['vol']
        pars += ['R10', 'n0']
        return pars

    def _par_values(self, kin=False, export=False):

        if kin:
            pars = liver.params_liver(self.kinetics, self.stationary)
            return {par: getattr(self, par) for par in pars}
        
        if export:
            pars = self._par_values()
            p0 = self._model_pars()
            p1 = liver.params_liver(self.kinetics, self.stationary)
            discard = set(p0) - set(p1)
            return {p: pars[p] for p in pars if p not in discard}
        
        pars = self._model_pars()
        p = {par: getattr(self, par) for par in pars}

        try:
            p['Fa'] = p['fa']*p['Fp']
            p['Fv'] = (1-p['fa'])*p['Fp']
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
            p['Kbh'] = _div(1, p['Th'])
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
        try:
            p['E'] = p['khe']/(p['khe']+p['Fp'])
        except KeyError:
            try:
                p['E'] = p['khe']/(p['khe']+self.Fp)
            except:
                pass
        try:
            p['Ktrans'] = (1-p['E'])*p['khe']
        except KeyError:
            pass
        if p['vol'] is not None:
            try:
                p['CL'] = p['khe']*p['vol']
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
        if self.ca is None:
            if self.aif is None:
                raise ValueError(
                    "Either aif or ca must be provided "
                    "to predict signal data.")
            else:
                r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
                if self.sequence == 'SR':
                    self.ca = sig.conc_src(
                        self.aif, self.TC, 1 / self.R10a, r1, self.n0)
                elif self.sequence == 'SS':
                    self.ca = sig.conc_ss(
                        self.aif, self.TR, self.B1corr_a * self.FA,
                        1 / self.R10a, r1, self.n0)
                    
    def _check_cv(self):
        if self.kinetics[0] == '1':
            return
        if self.cv is None:
            if self.vif is None:
                if self.kinetics[1]=='2':
                    raise ValueError(
                        "For a dual-inlet model, either vif or cv must be "
                        "provided.")
            else:
                r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
                if self.sequence == 'SR':
                    self.cv = sig.conc_src(
                        self.vif, self.TC, 1 / self.R10v, r1, self.n0)
                elif self.sequence == 'SS':
                    self.cv = sig.conc_ss(
                        self.vif, self.TR, self.B1corr_v * self.FA,
                        1 / self.R10v, r1, self.n0)
                    
    def conc(self, sum=True):
        """Tissue concentrations

        Args:
            sum (bool, optional): If True, returns the total concentrations. 
              If False, returns concentration in both compartments separately. 
              Defaults to True.

        Returns:
            numpy.ndarray: Concentration in M
        """
        self._check_ca()
        self._check_cv()
        pars = self._par_values(kin=True)
        return liver.conc_liver(
            self.ca, t=self.t, dt=self.dt, sum=sum, cv=self.cv, **pars)

    def relax(self):
        """Tissue relaxation rate

        Returns:
            numpy.ndarray: Relaxation rate in 1/sec
        """
        r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
        C = self.conc(sum=False)
        if 'IC' in self.kinetics:
            r1h = lib.relaxivity(self.field_strength, 'hepatocytes', 
                                 self.agent)
            return self.R10 + r1*C[0, :] + r1h*C[1, :]
        else:
            return self.R10 + r1*C
    
    def signal(self) -> np.ndarray:
        """Pseudocontinuous signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        R1 = self.relax()
        if self.sequence == 'SR':
            return sig.signal_spgr(self.S0, R1, self.TC, self.TR, 
                                 self.B1corr * self.FA)
        else:
            return sig.signal_ss(self.S0, R1, self.TR, 
                                 self.B1corr * self.FA)

    def predict(self, time: np.ndarray):
        """Predict the data at specific time points

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Array of predicted data for each element of *time*.
        """
        t = self.time()
        if np.amax(time) > np.amax(t):
            raise ValueError(
                "The acquisition window is longer than the duration "
                "of the AIF. The largest time point that can be "
                "predicted is " + str(np.amax(t) / 60) + "min.")
        sig = self.signal()
        return utils.sample(time, t, sig, self.TS)

    def train(self, time, signal, **kwargs):
        """Train the free parameters

        Args:
            time (array-like): Array with time points
            signal (array-like): Array with signal values
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            Liver: A reference to the model instance.
        """
        if self.sequence == 'SR':
            Sref = sig.signal_spgr(
                1, self.R10, self.TC, self.TR, self.B1corr * self.FA)
        else:
            Sref = sig.signal_ss(1, self.R10, self.TR, 
                                self.B1corr * self.FA)
        self.S0 = np.mean(signal[:self.n0]) / Sref if Sref > 0 else 0
        return ui.train(self, time, signal, **kwargs)



    def plot(self,
             time: np.ndarray,
             signal: np.ndarray,
             ref=None, xlim=None, fname=None, show=True):

        C = self.conc(sum=True)
        t = self.time()
        if xlim is None:
            xlim = [np.amin(time), np.amax(time)]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(time/60, signal, marker='o', linestyle='None',
                 color='cornflowerblue', label='Data')
        ax0.plot(t/60, self.predict(t), linestyle='-',
                 linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)',
                xlim=np.array(xlim)/60)
        ax0.legend()
        ax1.set_title('Reconstruction of concentrations.')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['C'], marker='o', linestyle='None',
                     color='cornflowerblue', label='Tissue ground truth')
            ax1.plot(ref['t']/60, 1000*ref['cb'], marker='o', linestyle='None',
                     color='lightcoral', label='Arterial blood')
        ax1.plot(t/60, 1000*C, linestyle='-', linewidth=3.0,
                 color='darkblue', label='Tissue prediction')
        ax1.plot(t/60, 1000*self.ca, linestyle='-', linewidth=3.0,
                 color='darkred', label='Arterial blood measurement')
        if self.cv is not None:
            ax1.plot(t/60, 1000*self.cv, linestyle='-', linewidth=3.0,
                 color='blue', label='Portal-venous blood measurement')
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
        'init': 3.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Magnetic field strength',
        'unit': 'T',
    },
    'agent': {
        'init': 'gadoxetate',
        'default_free': False,
        'bounds': None,
        'name': 'Contrast agent',
        'unit': None,
    },
    'R10a': {
        'init': 0.7,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial precontrast R1',
        'unit': 'Hz',
        'pixel_par': False,
    },
    'B1corr_a': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial B1-correction factor',
        'unit': '',
        'pixel_par': False,
    },
    'R10v': {
        'init': 0.7,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Portal venous precontrast R1',
        'unit': 'Hz',
        'pixel_par': False,
    },
    'B1corr_v': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Portal venous B1-correction factor',
        'unit': '',
        'pixel_par': False,
    },
    'B1corr': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Tissue B1-correction factor',
        'unit': '',
        'pixel_par': True,
    },
    'FA': {
        'init': 15,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Flip angle',
        'unit': 'deg',
        'pixel_par': False,
    },
    'TR': {
        'init': 0.005,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Repetition time',
        'unit': 'sec',
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
    'TP': {
        'init': 0.05,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Preparation delay',
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
    'R10': {
        'init': 0.7,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Tissue precontrast R1',
        'unit': 'Hz',
        'pixel_par': True,
    },
    'S0': {
        'init': 1.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Signal scaling factor',
        'unit': 'a.u.',
        'pixel_par': True,
    },
    'n0': {
        'init': 1,
        'default_free': False,
        'bounds': None,
        'name': 'Number of precontrast acquisitions',
        'unit': '',
        'pixel_par': False,
    },
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [0, 1],
        'name': 'Hematocrit',
        'unit': '',
        'pixel_par': False,
    },
    'Te': {
        'init': 30.0,
        'default_free': True,
        'bounds': [0.1, 60],
        'name': 'Extracellular mean transit time',
        'unit': 'sec',
        'pixel_par': True,
    },
    'De': {
        'init': 0.85,
        'default_free': True,
        'bounds': [0, 1],
        'name': 'Extracellular dispersion',
        'unit': '',
        'pixel_par': True,
    },
    've': {
        'init': 0.3,
        'default_free': True,
        'bounds': [0.01, 0.6],
        'name': 'Liver extracellular volume fraction',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'Ta': {
        'init': 2,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Arterial mean transit time',
        'unit': 'sec',
        'pixel_par': True,
    },
    'Tg': {
        'init': 10,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Gut mean transit time',
        'unit': 'sec',
        'pixel_par': True,
    },
    'Fp': {
        'init': 0.008,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Liver plasma flow',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'fa': {
        'init': 0.2,
        'default_free': True,
        'bounds': [0, 1],
        'name': 'Arterial flow fraction',
        'unit': '',
        'pixel_par': True,
    },
    'khe': {
        'init': 0.003,
        'default_free': True,
        'bounds': [0.0, 0.1],
        'name': 'Hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'khe_i': {
        'init': 0.003,
        'default_free': True,
        'bounds': [0.0, 0.1],
        'name': 'Initial hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'khe_f': {
        'init': 0.003,
        'default_free': True,
        'bounds': [0.0, 0.1],
        'name': 'Final hepatocellular uptake rate',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'Th': {
        'init': 30*60,
        'default_free': True,
        'bounds': [10*60, 10*60*60],
        'name': 'Hepatocellular mean transit time',
        'unit': 'sec',
        'pixel_par': True,
    },
    'Th_i': {
        'init': 30*60,
        'default_free': True,
        'bounds': [10*60, 10*60*60],
        'name': 'Initial hepatocellular mean transit time',
        'unit': 'sec',
        'pixel_par': True,
    },
    'Th_f': {
        'init': 30*60,
        'default_free': True,
        'bounds': [10*60, 10*60*60],
        'name': 'Final hepatocellular mean transit time',
        'unit': 'sec',
        'pixel_par': True,
    },
    'vol': {
        'init': None,
        'default_free': False,
        'bounds': [0, 10000],
        'name': 'Liver volume',
        'unit': 'cm3',
        'pixel_par': False,
    },
    'E': {
        'init': 0.4,
        'default_free': False,
        'bounds': [0.0, 1.0],
        'name': 'Liver extraction fraction',
        'unit': 'unitless',
        'pixel_par': False,
    },

    # Derived parameters
    'Fa': {
        'name': 'Arterial plasma flow',
        'unit': 'mL/sec/cm3',
    },
    'Fv': {
        'name': 'Venous plasma flow',
        'unit': 'mL/sec/cm3',
    },
    'kbh': {
        'name': 'Biliary excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'kbh': {
        'name': 'Biliary excretion rate',
        'unit': 'mL/sec/cm3',
    },
    'Ktrans': {
        'name': 'Hepatic plasma clearance',
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
