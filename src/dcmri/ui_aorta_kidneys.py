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


class AortaKidneys(ui.Model):
    """Joint model for signals from aorta and both kidneys.

    This model uses a whole body model to simultaneously predict 
    signals in aorta and kidneys (see :ref:`whole-body-tissues`). 

    See Also:
        `Aorta`, `Kidney`

    Args:
        organs (str, optional): Model for the organs in the whole-body 
          model. The options are 'comp' (one compartment) and '2cxm' 
          (two-compartment exchange). Defaults to 'comp'.
        heartlung (str, optional): Model for the heart-lung system in 
          the whole-body model. Options are 'pfcomp' (plug-flow 
          compartment) or 'chain'. Defaults to 'pfcomp'.
        kidneys (str, optional): Model for the kidneys. Options are 
          '2CF' (Two-compartment filtration) and 'HF' (High-flow). 
          Defaults to '2CF'. 
        sequence (str, optional): imaging sequence model. Possible 
          values are 'SS' (steady-state), 'SR' (saturation-recovery), 
          'SSI' (steady state with inflow correction) and 'lin' 
          (linear). Defaults to 'SS'.
        agent (str, optional): Generic name of the contrast agent 
          injected. Defaults to 'gadoterate'.
        params (dict, optional): values for the model parameters,
          specified as keyword parameters. Defaults are used for any 
          that are not provided. See table 
          :ref:`AortaKidneys-defaults` for a list of parameters and 
          their default values.

    Notes:

        In the table below, if **Bounds** is None, the parameter is fixed 
        during training. Otherwise it is allowed to vary between the 
        bounds given.

        .. _AortaKidneys-defaults:
        .. list-table:: AortaKidneys parameters. 
            :widths: 5 10 5 5 5 10
            :header-rows: 1

            * - Parameter
              - Description
              - Value
              - Unit
              - Bounds
              - Usage
            * - **General**
              - 
              - 
              - 
              - 
              - 
            * - dt
              - Prediction time step
              - 0.25
              - sec
              - None
              - Always
            * - tmax
              - Maximum time predicted
              - 120
              - sec
              - None
              - Always
            * - dose_tolerance
              - Stopping criterion whole body model
              - 0.1
              -
              - None
              - Always
            * - t0
              - Baseline duration
              - 0
              - 
              - None
              - Always
            * - field_strength
              - B0-field
              - 3
              - T
              - None
              - Always
            * - **Injection**
              - 
              -
              - 
              - 
              -
            * - weight
              - Subject weight
              - 70
              - kg
              - None
              - Always
            * - dose
              - Contrast agent dose
              - 0.0125
              - mL/kg
              - None
              - Always
            * - rate
              - Contrast agent injection rate
              - 1
              - mL/kg
              - None
              - Always
            * - **Sequence**
              - 
              -
              - 
              - 
              - 
            * - TR
              - Repetition time
              - 0.005
              - sec
              - None
              - sequence in ['SS', 'SSI']
            * - FA
              - Flip angle
              - 15
              - deg
              - None
              - sequence in ['SR', 'SS', 'SSI']
            * - TC
              - Time to k-space center
              - 0.1
              - sec
              - None
              - sequence == 'SR'
            * - TS
              - Sampling duration
              - 0
              - sec
              - None
              - Always
            * - TF
              - Inflow time
              - 0
              - sec
              - None
              - sequence == 'SSI'
            * - **Aorta**
              - 
              -
              - 
              - 
              -
            * - BAT
              - Bolus arrival time
              - 60
              - sec
              - [0, inf]
              - Always
            * - CO
              - Cardiac output
              - 100
              - mL/sec
              - [0, 300]
              - Always
            * - Thl
              - Heart-lung transit time
              - 10
              - sec
              - [0, 30]
              - Always
            * - Dhl
              - Heart-lung dispersion
              - 0.2
              - 
              - [0.05, 0.95]
              - heartlung in ['pfcomp', 'chain']
            * - To
              - Organs transit time
              - 20
              - sec
              - [0, 60]
              - Always
            * - Eo
              - Organs extraction fraction
              - 0.15
              - 
              - [0, 0.5]
              - organs == '2cxm'
            * - Toe
              - Organs extracellular transit time
              - 120
              - sec
              - [0, 800]
              - organs == '2cxm'
            * - Eb
              - Body extraction fraction
              - 0.05
              - 
              - [0.01, 0.15]
              - Always
            * - R10a
              - Arterial precontrast R1
              - 0.7
              - /sec
              - None
              - Always
            * - S0a
              - Arterial signal scale factor
              - 1
              - a.u.
              - None
              - Always
            * - **Kidneys**
              -
              -
              - 
              - 
              -
            * - H
              - Hematocrit
              - 0.45
              - 
              - None
              - Always
            * - RPF
              - Renal plasma flow
              - 20
              - mL/sec
              - [0, 100]
              - Always
            * - DRF
              - Differential renal function
              - 0.5
              - 
              - [0, 1.0]
              - Always
            * - DRPF
              - Differential renal plasma flow
              - 0.5
              - 
              - [0, 1.0]
              - kidneys == '2CF'
            * - FF
              - Filtration fraction
              - 0.1
              - 
              - [0, 0.3]
              - agent in ['gadoxetate', 'gadobenate']
            * - **Left kidney**
              - 
              -
              - 
              - 
              - 
            * - Ta_lk
              - Left kidney arterial delay
              - 0
              - sec
              - [0, 3]
              - Always
            * - vp_lk
              - Left kidney plasma volume
              - 0.15
              - mL/cm3
              - [0, 0.3]
              - Always
            * - Tt_lk
              - Left kidney tubular transit time
              - 120
              - sec
              - [0, inf]
              - Always
            * - R10_lk
              - Left kidney precontrast R1
              - 0.65
              - 1/sec
              - None
              - Always
            * - S0_lk
              - Left kidney signal scale factor
              - 1.0
              - a.u.
              - [0, inf]
              - Always
            * - vol_lk
              - Left kidney volume
              - 150
              - cm3
              - None
              - Always
            * - **Right kidney**
              - 
              -
              - 
              - 
              -
            * - Ta_rk
              - Right kidney arterial delay
              - 0
              - sec
              - [0, 3]
              - Always
            * - vp_rk
              - Right kidney plasma volume
              - 0.15
              - mL/cm3
              - [0, 0.3]
              - Always
            * - Tt_rk
              - Right kidney tubular transit time
              - 120
              - sec
              - [0, inf]
              - Always
            * - R10_rk
              - Right kidney precontrast R1
              - 0.65
              - /sec
              - None
              - Always
            * - S0_rk
              - Right kidney signal scale factor
              - 1.0
              - a.u.
              - [0, inf]
              - Always
            * - vol_rk
              - Right kidney volume
              - 150
              - cm3
              - None
              - Always


    Example:

        Use the model to fit minipig data with inflow correction:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import numpy as np
        >>> import dcmri as dc

        Read the dataset:

        >>> datafile = dc.fetch('minipig_renal_fibrosis')
        >>> data = dc.read_dmr(datafile, 'nest')
        >>> rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']

        Create an array of time points:

        >>> time = pars['TS'] * np.arange(len(rois['Aorta']))

        Initialize the tissue:

        >>> aorta_kidneys = dc.AortaKidneys(
        ...     sequence='SSI',
        ...     heartlung='chain',
        ...     organs='comp',
        ...     agent="gadoterate",
        ...     dt=0.25,
        ...     field_strength=pars['B0'],
        ...     weight=pars['weight'],
        ...     dose=pars['dose'],
        ...     rate=pars['rate'],
        ...     R10a=1/dc.T1(pars['B0'], 'blood'),
        ...     R10_lk=1/dc.T1(pars['B0'], 'kidney'),
        ...     R10_rk=1/dc.T1(pars['B0'], 'kidney'),
        ...     vol_lk=85,
        ...     vol_rk=85,
        ...     TR=pars['TR'],
        ...     FA=pars['FA'],
        ...     TS=pars['TS'],
        ...     CO=60,   
        ...     t0=15, 
        ... )

        Define time and signal data

        >>> t = (time, time, time)
        >>> signal = (rois['Aorta'], rois['LeftKidney'], rois['RightKidney'])

        Train the system to the data:

        >>> aorta_kidneys.train(t, signal)

        Plot the reconstructed signals and concentrations:

        >>> aorta_kidneys.plot(t, signal)

        Print the model parameters:

        >>> aorta_kidneys.print_params(round_to=4)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        Bolus arrival time (BAT): 16.7422 (0.2853) sec
        Inflow time (TF): 0.2801 (0.0133) sec
        Cardiac output (CO): 72.762 (12.4426) mL/sec
        Heart-lung mean transit time (Thl): 16.2249 (0.3069) sec
        Organs blood mean transit time (To): 14.3793 (1.2492) sec
        Body extraction fraction (Eb): 0.0751 (0.0071)
        Heart-lung dispersion (Dhl): 0.0795 (0.0041)
        Renal plasma flow (RPF): 3.3489 (0.7204) mL/sec
        Differential renal function (DRF): 0.9085 (0.0212)
        Differential renal plasma flow (DRPF): 0.812 (0.0169)
        Left kidney arterial mean transit time (Ta_lk): 0.6509 (0.2228) sec
        Left kidney plasma volume (vp_lk): 0.099 (0.0186) mL/cm3
        Left kidney tubular mean transit time (Tt_lk): 46.9705 (3.3684) sec
        Right kidney arterial mean transit time (Ta_rk): 1.4206 (0.2023) sec
        Right kidney plasma volume (vp_rk): 0.1294 (0.0175) mL/cm3
        Right kidney tubular mean transit time (Tt_rk): 4497.8301 (39890.3818) sec
        Aorta signal scaling factor (S0a): 4912.776 (254.2363) a.u.
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        Filtration fraction (FF): 0.0812
        Glomerular Filtration Rate (GFR): 0.2719 mL/sec
        Left kidney plasma flow (RPF_lk): 2.7194 mL/sec
        Right kidney plasma flow (RPF_rk): 0.6295 mL/sec
        Left kidney glomerular filtration rate (GFR_lk): 0.247 mL/sec
        Right kidney glomerular filtration rate (GFR_rk): 0.0249 mL/sec
        Left kidney plasma flow (Fp_lk): 0.032 mL/sec/cm3
        Left kidney plasma mean transit time (Tp_lk): 2.838 sec
        Left kidney vascular mean transit time (Tv_lk): 3.0958 sec
        Left kidney tubular flow (Ft_lk): 0.0029 mL/sec/cm3
        Left kidney filtration fraction (FF_lk): 0.0908
        Left kidney extraction fraction (E_lk): 0.0833
        Right kidney plasma flow (Fp_rk): 0.0074 mL/sec/cm3
        Right kidney plasma mean transit time (Tp_rk): 16.8121 sec
        Right kidney vascular mean transit time (Tv_rk): 17.4762 sec
        Right kidney tubular flow (Ft_rk): 0.0003 mL/sec/cm3
        Right kidney filtration fraction (FF_rk): 0.0395
        Right kidney extraction fraction (E_rk): 0.038
    """

    def __init__(
            self, 
            organs='comp', 
            heartlung='pfcomp', 
            kidneys='2CF', 
            sequence='SS', 
            agent='gadoterate',
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
        if kidneys not in ['2CF', 'HF']:
            raise ValueError(
                f"Kinetic model {kidneys} is not available."
            )
        if sequence not in ['SS', 'SR', 'SSI', 'lin']:
            raise ValueError(
                f"Sequence {sequence} is not available."
            )

        # Set configuration
        self.sequence = sequence
        self.organs = organs
        self.heartlung = heartlung
        self.kidneys = kidneys
        self.agent = agent
    
        # Set defaults
        self._set_defaults(**params)
        # For SSI, S0 needs to be free because TF affects the baseline
        if self.sequence == 'SSI':
            self.free['S0a'] = [0, np.inf]

        # Internal flag
        self._predict = None

    def _params(self):
        return PARAMS_AORTA | PARAMS_KIDNEYS | PARAMS_DERIVED
    
    def _model_pars(self):

        # General
        pars = ['dt', 'tmax', 'dose_tolerance', 't0', 'field_strength']

        # Injection
        pars += ['weight', 'dose', 'rate', 'BAT']

        # Sequence
        pars += ['TS']
        if self.sequence == 'SR':
            pars += ['TC', 'FA']
        elif self.sequence=='SS':
            pars += ['TR', 'FA']
        elif self.sequence=='SSI':
            pars += ['TF', 'TR', 'FA']

        # Aorta
        pars += ['CO', 'Thl', 'To', 'Eb', 'R10a', 'S0a']
        if self.heartlung == 'pfcomp':
            pars += ['Dhl']
        elif self.heartlung == 'chain':
            pars += ['Dhl']
        if self.organs=='2cxm':
            pars += ['Toe', 'Eo']

        # Kidneys
        pars += ['RPF', 'DRF', 'H']
        if self.agent in ['gadoxetate', 'gadobenate']:
            pars += ['FF']
        if self.kidneys == '2CF':
            pars += ['DRPF']
        pars += ['Ta_lk', 'vol_lk', 'vp_lk', 'Tt_lk', 'R10_lk']
        pars += ['Ta_rk', 'vol_rk', 'vp_rk', 'Tt_rk', 'R10_rk']

        return pars
    

    def _par_values(self, export=False):

        if export:
            discard = [
                'dt', 'tmax', 't0', 'weight', 'dose', 
                'rate', 'field_strength', 'dose_tolerance', 'R10a', 
                'TS' , 'TC', 'TR', 'FA', 
                'R10_lk', 'R10_rk', 'H', 
                'vol_rk', 'vol_lk',
            ]
            pars = self._par_values()
            return {p: pars[p] for p in pars if p not in discard}

        pars = self._model_pars()
        p = {par: getattr(self, par) for par in pars}

        # Kidneys
        if 'FF' not in p:
            p['FF'] = _div(p['Eb'], 1-p['Eb'])
        if {'RPF', 'FF'}.issubset(p):   
            p['GFR'] =  p['RPF'] * p['FF']
        if {'DRPF', 'RPF'}.issubset(p): 
            p['RPF_lk'] = p['DRPF'] * p['RPF']
            p['RPF_rk'] = (1 - p['DRPF']) * p['RPF']
        if {'DRF', 'GFR'}.issubset(p):
            p['GFR_lk'] = p['DRF'] * p['GFR']
            p['GFR_rk'] = (1 - p['DRF']) * p['GFR']

        # Kidney LK
        if {'RPF_lk', 'vol_lk'}.issubset(p):
            p['Fp_lk'] = _div(p['RPF_lk'], p['vol_lk'])
        if {'RPF_lk', 'GFR_lk', 'vp_lk', 'vol_lk'}.issubset(p):
            p['Tp_lk'] = _div(p['vp_lk'] * p['vol_lk'], p['RPF_lk']+p['GFR_lk'])
        if {'RPF_lk', 'vp_lk', 'vol_lk'}.issubset(p):
            p['Tv_lk'] = _div(p['vp_lk'] * p['vol_lk'], p['RPF_lk'])
        if {'GFR_lk', 'vol_lk'}.issubset(p):
            p['Ft_lk'] = _div(p['GFR_lk'], p['vol_lk'])
        if {'GFR_lk', 'RPF_lk'}.issubset(p):
            p['FF_lk'] = _div(p['GFR_lk'], p['RPF_lk'])
            p['E_lk'] = _div(p['GFR_lk'], p['GFR_lk']+p['RPF_lk'])

        # Kidney RK
        if {'RPF_rk', 'vol_rk'}.issubset(p):
            p['Fp_rk'] = _div(p['RPF_rk'], p['vol_rk'])
        if {'RPF_rk', 'GFR_rk', 'vp_rk', 'vol_rk'}.issubset(p):
            p['Tp_rk'] = _div(p['vp_rk'] * p['vol_rk'], p['RPF_rk']+p['GFR_rk'])
        if {'RPF_rk', 'vp_rk', 'vol_rk'}.issubset(p):
            p['Tv_rk'] = _div(p['vp_rk'] * p['vol_rk'], p['RPF_rk'])
        if {'GFR_rk', 'vol_rk'}.issubset(p):
            p['Ft_rk'] = _div(p['GFR_rk'], p['vol_rk'])
        if {'GFR_rk', 'RPF_rk'}.issubset(p):
            p['FF_rk'] = _div(p['GFR_rk'], p['RPF_rk'])
            p['E_rk'] = _div(p['GFR_rk'], p['GFR_rk']+p['RPF_rk'])

        return p


    def _conc_aorta(self) -> np.ndarray:
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
            self.t, self.weight, conc, self.dose, self.rate, self.BAT,
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=self.Eb, heartlung=heartlung, organs=organs,
            dt=self.dt, tol=self.dose_tolerance,
        )
        self.ca = Jb/self.CO
        return self.t, self.ca

    def _relax_aorta(self):
        t, cb = self._conc_aorta()
        rb = lib.relaxivity(self.field_strength, 'blood', self.agent)
        return t, self.R10a + rb*cb

    def _predict_aorta(self, xdata: np.ndarray) -> np.ndarray:
        tacq = xdata[1]-xdata[0]
        self.tmax = max(xdata)+tacq+self.dt
        if self.TS is not None:
            self.tmax += self.TS
        t, R1a = self._relax_aorta()
        if self.sequence == 'SR':
            signal = sig.signal_free(self.S0a, R1a, self.TC, self.FA)
        elif self.sequence=='SS':
            signal = sig.signal_ss(self.S0a, R1a, self.TR, self.FA)
        elif self.sequence=='SSI':
            signal = sig.signal_spgr(self.S0a, R1a, self.TF, self.TR, self.FA, n0=1)
        elif self.sequence == 'lin':
            signal = sig.signal_lin(self.S0a, R1a)
        return utils.sample(xdata, t, signal, self.TS)


    def _conc_kidneys(self, sum=True):

        if self.agent in ['gadoxetate', 'gadobenate']:
            FF = self.FF
        else:
            FF = self.Eb/(1-self.Eb)

        t = self.t
        ca_lk = pk.flux(
            self.ca, self.Ta_lk, t=self.t, dt=self.dt, model='plug',
        )
        ca_rk = pk.flux(
            self.ca, self.Ta_rk, t=self.t, dt=self.dt, model='plug',
        )
        GFR = FF * self.RPF
        Ft_lk = self.DRF * GFR / self.vol_lk
        Ft_rk = (1-self.DRF) * GFR / self.vol_rk
        if self.kidneys == '2CF':
            Fp_lk = self.DRPF * self.RPF / self.vol_lk
            C_lk = kidney.conc_kidney(
                ca_lk / (1-self.H), 
                Fp_lk, self.vp_lk, Ft_lk, self.Tt_lk,
                t=self.t, dt=self.dt, sum=sum, kinetics='2CF',
            )
            Fp_rk = (1-self.DRPF) * self.RPF / self.vol_rk
            C_rk = kidney.conc_kidney(
                ca_rk / (1-self.H), 
                Fp_rk, self.vp_rk, Ft_rk, self.Tt_rk,
                t=self.t, dt=self.dt, sum=sum, kinetics='2CF',
            )
        if self.kidneys == 'HF':
            C_lk = kidney.conc_kidney(
                ca_lk / (1-self.H), 
                self.vp_lk, Ft_lk, self.Tt_lk,
                t=self.t, dt=self.dt, sum=sum, kinetics='HF',
            )
            C_rk = kidney.conc_kidney(
                ca_rk / (1-self.H), 
                self.vp_rk, Ft_rk, self.Tt_rk,
                t=self.t, dt=self.dt, sum=sum, kinetics='HF',
            )
        return t, C_lk, C_rk


    def _relax_kidneys(self):
        t, Clk, Crk = self._conc_kidneys()
        rb = lib.relaxivity(self.field_strength, 'blood', self.agent)
        return t, self.R10_lk + rb*Clk, self.R10_rk + rb*Crk


    def _predict_kidneys(self, xdata: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        t, R1_lk, R1_rk = self._relax_kidneys()
        if self.sequence == 'SR':
            signal_lk = sig.signal_spgr(
                self.S0_lk, R1_lk, self.TC, self.TR, self.FA,
            )
            signal_rk = sig.signal_spgr(
                self.S0_rk, R1_rk, self.TC, self.TR, self.FA,
            )
        elif self.sequence in ['SS', 'SSI']:
            signal_lk = sig.signal_ss(self.S0_lk, R1_lk, self.TR, self.FA)
            signal_rk = sig.signal_ss(self.S0_rk, R1_rk, self.TR, self.FA)
        elif self.sequence == 'lin':
            signal_lk = sig.signal_lin(self.S0_lk, R1_lk)
            signal_rk = sig.signal_lin(self.S0_rk, R1_rk)
        return (
            utils.sample(xdata[0], t, signal_lk, self.TS),
            utils.sample(xdata[1], t, signal_rk, self.TS))

    def conc(self, sum=True):
        """Concentrations in aorta and kidney.

        Args:
            sum (bool, optional): If set to true, the kidney 
              concentrations are the sum over all compartments. If 
              set to false, the compartmental concentrations are 
              returned individually. Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, left 
              kidney concentrations, right kidney concentrations.
        """
        t, cb = self._conc_aorta()
        t, Clk, Crk = self._conc_kidneys(sum=sum)
        return t, cb, Clk, Crk

    def relax(self):
        """Relaxation rates in aorta and kidney.

        Returns:
            tuple: time points, aorta relaxation rate, left kidney 
              relaxation rate, right kidney relaxation rate.
        """
        t, R1b = self._relax_aorta()
        t, R1_lk, R1_rk = self._relax_kidneys()
        return t, R1b, R1_lk, R1_rk

    def predict(self, xdata: tuple) -> tuple:
        """Predict the data at given xdata

        Args:
            xdata (tuple): Tuple of 3 arrays with time points for 
              aorta, left kidney and right kidney, in that order. 
              The three arrays can all be different in length and value.

        Returns:
            tuple: Tuple of 3 arrays with signals for aorta, left 
              kidney and right kidney, in that order. The three 
              arrays can all be different in length and value but 
              each has to have the same length as its corresponding 
              array of time points.
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


    def train(self, xdata: tuple, ydata: tuple, **kwargs):
        """Train the free parameters

        Args:
            xdata (tuple): Tuple of 3 arrays with time points for 
              aorta, left kidney and right kidney, in that order. 
              The three arrays can all be different in length and value.
            ydata (tuple): Tuple of 3 arrays with signals for aorta, 
              left kidney and right kidney, in that order. The three 
              arrays can all be different in length and values but 
              each has to have the same length as its corresponding 
              array of time points.
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        # Estimate BAT and S0a from data
        if self.sequence == 'SR':
            Srefb = sig.signal_spgr(1, self.R10a, self.TC, self.TR, self.FA)
            Sref_lk = sig.signal_spgr(1, self.R10_lk, self.TC, self.TR, self.FA)
            Sref_rk = sig.signal_spgr(1, self.R10_rk, self.TC, self.TR, self.FA)
        elif self.sequence=='SS':
            Srefb = sig.signal_ss(1, self.R10a, self.TR, self.FA)
            Sref_lk = sig.signal_ss(1, self.R10_lk, self.TR, self.FA)
            Sref_rk = sig.signal_ss(1, self.R10_rk, self.TR, self.FA)
        elif self.sequence=='SSI':
            Srefb = sig.signal_spgr(1, self.R10a, self.TF, self.TR, self.FA)
            Sref_lk = sig.signal_ss(1, self.R10_lk, self.TR, self.FA)
            Sref_rk = sig.signal_ss(1, self.R10_rk, self.TR, self.FA)
        elif self.sequence=='lin':
            Srefb = sig.signal_lin(1, self.R10a)
            Sref_lk = sig.signal_lin(1, self.R10_lk)
            Sref_rk = sig.signal_lin(1, self.R10_rk)
        n0 = max([np.sum(xdata[0] < self.t0), 1])
        self.S0a = np.mean(ydata[0][:n0]) / Srefb
        n0 = max([np.sum(xdata[1] < self.t0), 1])
        self.S0_lk = np.mean(ydata[1][:n0]) / Sref_lk
        n0 = max([np.sum(xdata[2] < self.t0), 1])
        self.S0_rk = np.mean(ydata[2][:n0]) / Sref_rk
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-self.Dhl)*self.Thl
        self.BAT = max([self.BAT, 0])

        # Copy all free to restor at the end
        free = deepcopy(self.free)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        pars = list(PARAMS_AORTA.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui.train(self, xdata[0], ydata[0], **kwargs)

        # Train free kidney parameters on kidney data
        self._predict = 'kidneys'
        pars = list(PARAMS_KIDNEYS.keys())
        self.free = {s: free[s] for s in pars if s in free}
        ui.train(self, (xdata[1], xdata[2]), (ydata[1], ydata[2]), **kwargs)

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
                        'Aorta', ax1, xlim,
                        color=['lightcoral', 'darkred'],
                        test=None if ref is None else ref[0])
        _plot_data1scan(t, sig[1], xdata[1], ydata[1],
                        'Left kidney', ax3, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[1])
        _plot_data1scan(t, sig[2], xdata[2], ydata[2],
                        'Right kidney', ax5, xlim,
                        color=['cornflowerblue', 'darkblue'],
                        test=None if ref is None else ref[2])
        cb_lk = Clk[0,:] / (self.vp_lk/(1-self.H))
        cb_rk = Crk[0,:] / (self.vp_rk/(1-self.H))
        _plot_conc_aorta(t, cb, cb_lk, cb_rk, ax2, xlim)
        _plot_conc_kidney(t, Clk, 'Left kidney', ax4, xlim)
        _plot_conc_kidney(t, Crk, 'Right kidney', ax6, xlim)
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

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



    


# Helper functions for plotting

def _plot_conc_aorta(t, cb, cb_lk, cb_rk, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Blood concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cb, linestyle='-',
            color='darkred', linewidth=2.0, label='Aorta')
    ax.plot(t/60, 1000*cb_lk, linestyle='--',
            color='lightcoral', linewidth=2.0, label='Left kidney')
    ax.plot(t/60, 1000*cb_rk, linestyle='-.',
            color='lightcoral', linewidth=2.0, label='Right kidney')
    ax.legend()


def _plot_conc_kidney(t, C, kid, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    
    ax.set(xlabel='Time (min)', ylabel=f'{kid} concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*C[0, :], linestyle='-',
            color='darkred', linewidth=2.0, label='Blood')
    ax.plot(t/60, 1000*C[1, :], linestyle='-',
            color='darkcyan', linewidth=2.0, label='Tubuli')
    ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-',
            color='darkblue', linewidth=2.0, label='Tissue')
    ax.legend()


def _plot_data1scan(t: np.ndarray, sig: np.ndarray,
                    xdata: np.ndarray, ydata: np.ndarray,
                    roi, ax, xlim, color=['black', 'black'],
                    test=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel=f'{roi} signal (a.u.)', xlim=np.array(xlim)/60)
    ax.plot(xdata/60, ydata, marker='o',
            color=color[0], label='Data', linestyle='None')
    ax.plot(t/60, sig, linestyle='-',
            color=color[1], linewidth=3.0, label='Prediction')
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black',
                marker='D', linestyle='None', label='Test data')
    ax.legend()



PARAMS_AORTA = {
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
        'bounds': None,
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

    # Kinetics
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


    # Signal
    'R10a': {
        'init': 0.7,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Aorta precontrast R1',
        'unit': 'Hz',
        'pixel_par': False,
    },
    'S0a': {
        'init': 1.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Aorta signal scaling factor',
        'unit': 'a.u.',
        'pixel_par': True,
    },

}


PARAMS_KIDNEYS = {

    # Both kidneys
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [1e-3, 1 - 1e-3],
        'name': 'Tissue Hematocrit',
        'unit': '',
    },
    'RPF': {
        'init': 20,
        'default_free': True,
        'bounds': [0, 100],
        'name': 'Renal plasma flow',
        'unit': 'mL/sec',
    },
    'DRPF': {
        'init': 0.5,
        'default_free': True,
        'bounds': [0, 1],
        'name': 'Differential renal plasma flow',
        'unit': '',
    },
    'DRF': {
        'init': 0.5,
        'default_free': True,
        'bounds': [0, 1.0],
        'name': 'Differential renal function',
        'unit': '',
    },
    'FF': {
        'init': 0.10,
        'default_free': True,
        'bounds': [0.0, 0.5],
        'name': 'Filtration fraction',
        'unit': '',
    },

    # Left kidney

    'Ta_lk': {
        'init': 0,
        'default_free': True,
        'bounds': [0, 3],
        'name': 'Left kidney arterial mean transit time',
        'unit': 'sec',
    },
    'vp_lk': {
        'init': 0.15,
        'default_free': True,
        'bounds': [0, 0.3],
        'name': 'Left kidney plasma volume',
        'unit': 'mL/cm3',
    },
    'Tt_lk': {
        'init': 120,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Left kidney tubular mean transit time',
        'unit': 'sec',
    },
    'R10_lk': {
        'init': 1/lib.T1(3.0, 'kidney'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Left kidney tissue precontrast R1',
        'unit': 'Hz',
    },
    'S0_lk': {
        'init': 1.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Left kidney signal scaling factor',
        'unit': 'a.u.',
    }, 
    'vol_lk': {
        'init': 150,
        'default_free': False,
        'bounds': None,
        'name': 'Left kidney volume',
        'unit': 'mL',
    },
    
    # Right kidney

    'Ta_rk': {
        'init': 0,
        'default_free': True,
        'bounds': [0, 3],
        'name': 'Right kidney arterial mean transit time',
        'unit': 'sec',
    },
    'vp_rk': {
        'init': 0.15,
        'default_free': True,
        'bounds': [0, 0.3],
        'name': 'Right kidney plasma volume',
        'unit': 'mL/cm3',
    },
    'Tt_rk': {
        'init': 120,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Right kidney tubular mean transit time',
        'unit': 'sec',
    },
    'R10_rk': {
        'init': 1/lib.T1(3.0, 'kidney'),
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Right kidney tissue precontrast R1',
        'unit': 'Hz',
    },
    'S0_rk': {
        'init': 1.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Right kidney signal scaling factor',
        'unit': 'a.u.',
    },
    'vol_rk': {
        'init': None,
        'default_free': False,
        'bounds': None,
        'name': 'Right kidney volume',
        'unit': 'mL',
    },
}

PARAMS_DERIVED = {

    # Derived parameters

    'GFR': {
        'name': 'Glomerular Filtration Rate',
        'unit': 'mL/sec',
    },
    'RPF_lk': {
        'name': 'Left kidney plasma flow',
        'unit': 'mL/sec',
    },
    'GFR_lk': {
        'name': 'Left kidney glomerular filtration rate',
        'unit': 'mL/sec',
    },
    'Fp_lk': {
        'name': 'Left kidney plasma flow',
        'unit': 'mL/sec/cm3',
    },
    'Tp_lk': {
        'name': 'Left kidney plasma mean transit time',
        'unit': 'sec',
    },
    'Tv_lk': {
        'name': 'Left kidney vascular mean transit time',
        'unit': 'sec',
    },
    'Ft_lk': {
        'name': 'Left kidney tubular flow',
        'unit': 'mL/sec/cm3',
    },
    'FF_lk': {
        'name': 'Left kidney filtration fraction',
        'unit': '',
    },
    'E_lk': {
        'name': 'Left kidney extraction fraction',
        'unit': '',
    },
    'RPF_rk': {
        'name': 'Right kidney plasma flow',
        'unit': 'mL/sec',
    },
    'GFR_rk': {
        'name': 'Right kidney glomerular filtration rate',
        'unit': 'mL/sec',
    },
    'Fp_rk': {
        'name': 'Right kidney plasma flow',
        'unit': 'mL/sec/cm3',
    },
    'Tp_rk': {
        'name': 'Right kidney plasma mean transit time',
        'unit': 'sec',
    },
    'Tv_rk': {
        'name': 'Right kidney vascular mean transit time',
        'unit': 'sec',
    },
    'Ft_rk': {
        'name': 'Right kidney tubular flow',
        'unit': 'mL/sec/cm3',
    },
    'FF_rk': {
        'name': 'Right kidney filtration fraction',
        'unit': '',
    },
    'E_rk': {
        'name': 'Right kidney extraction fraction',
        'unit': '',
    },
}


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b == 0, 0, np.divide(a, b))
