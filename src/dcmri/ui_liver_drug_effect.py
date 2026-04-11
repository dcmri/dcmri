import matplotlib.pyplot as plt
import numpy as np

from dcmri import sig, utils, ui, pk
from dcmri.lexicon import LEXICON
from dcmri.pk import flux_aorta
from dcmri.utils import lib


LEXICON = LEXICON | {

    # Assay parameters
    'c_tmax': {'init': 4 * 60 * 60, 'name': 'Control visit - maximum acquisition time', 'unit': 'sec'},
    'c_dose': {'init': 0.05, 'name': 'Control visit - first contrast agent dose', 'unit': 'mL/kg'},
    'c_BAT': {'init': 120, 'bounds': [-60, 60], 'name': 'Control visit - first bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},
 
    'd_tmax': {'init': 4 * 60 * 60, 'name': 'Drug visit - maximum acquisition time', 'unit': 'sec'},
    'd_dose': {'init': 0.05, 'name': 'Drug visit - first contrast agent dose', 'unit': 'mL/kg'},
    'd_BAT': {'init': 120, 'bounds': [-60, 60], 'name': 'Drug visit - first bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},
 
    # MRI signal parameters - control visit
    'c_R10_a': {'init': 1/lib.T1(3.0, 'blood'), 'name': 'Control visit - aorta first baseline R1', 'unit': 'Hz'},
    'c_R10_l': {'init': 1/lib.T1(3.0, 'liver'), 'name': 'Control visit - liver first baseline R1', 'unit': 'Hz'},
    'c_S0_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},

    # MRI signal parameters - drug visit
    'd_R10_a': {'init': 1/lib.T1(3.0, 'blood'), 'name': 'Drug visit - aorta first baseline R1', 'unit': 'Hz'},
    'd_R10_l': {'init': 1/lib.T1(3.0, 'liver'), 'name': 'Drug visit - liver first baseline R1', 'unit': 'Hz'},
    'd_S0_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},

    'c_B1corr_a': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Arterial B1-correction factor', 'unit': ''},
    'c_B1corr_l': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Liver B1-correction factor', 'unit': ''},
    'd_B1corr_a': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Arterial B1-correction factor', 'unit': ''},
    'd_B1corr_l': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Liver B1-correction factor', 'unit': ''},

    # Kinetics - control visit
    'c_vol': {'init': 1000, 'name': 'Control visit - liver volume', 'unit': 'cm3'},
    'c_khe': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_kbh': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Control visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'c_CL': {'name': 'Control visit - liver plasma clearance', 'unit': 'mL/sec'},

    # Kinetics - drug visit
    'd_vol': {'init': 1000, 'name': 'Drug visit - liver volume', 'unit': 'cm3'},
    'd_khe': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_kbh': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_CL': {'name': 'Drug visit - liver plasma clearance', 'unit': 'mL/sec'},

    # Kinetics - common
    'r_khe': {'name': 'Relative effect in hepatocellular uptake rate', 'unit': ''},
    'r_kbh': {'name': 'Relative effect in biliary excretion rate', 'unit': ''},
    'r_CL': {'name': 'Relative effect in liver plasma clearance', 'unit': ''},
    'a_khe': {'name': 'Absolute effect in hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'a_kbh': {'name': 'Absolute effect in biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'a_CL': {'name': 'Absolute effect in liver plasma clearance', 'unit': 'mL/sec'},
}


class LiverDrugEffect(ui.SuperModel):
    """Joint model for aorta and liver signals measured over two scans.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta and liver, measured over two separate scans.

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
        kinetics (str, optional): Tracer-kinetic liver model. See table 
          :ref:`table-liver-models` for options - only single-inlet models 
          are allowed. Defaults to '1I-IC_HFD'.
        stationary (str, optional): For intracellular tracers - stationarity 
          regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
          For more detail see :ref:`liver-tissues`. Defaults to 'UE'.
        stationary (str, optional): Stationarity regime of the hepatocytes. 
          The options are 'UE', 'E', 'U' or None. For more detail 
          see :ref:`liver-tissues`. Defaults to 'UE'.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
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

        >>> time = (time[0], time[1], time[0], time[1])
        >>> signal = (aif[0], aif[1], roi[0], roi[1])

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

        >>> model.train(time, signal, n0=10, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare against 
        the experimentally derived data:

        >>> model.plot(time, signal)

        We can also have a look at the model parameters after training:

        >>> model.print_params(round_to=3)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        Aorta second signal scale factor (S02a): 195.824 (2.025) a.u.
        Liver second signal scale factor (S02l): 297.854 (4.9) a.u.
        Second bolus arrival time (BAT_2): 254.512 (0.137) sec
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
            * - BAT, BAT_2, CO, Thl, Dhl, To, Eo, Tie, Eb
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

    configs = {'sequence': ['3D-SPGR-SS', '3D-SPGR-SSI']}
    
    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._version = '1.0'
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(lexicon=LEXICON, **params)

    def _params(self, select=None):
        if select is None:
            select = 'all'
        seq = self._cnfg['sequence']
        inflow_pars = ['TF'] if seq == '3D-SPGR-SSI' else []

        pars_list = {
            'all': inflow_pars + [
                # _time
                'dt', 'c_tmax', 'd_tmax', 
                # _conc_aorta
                'dose_tolerance', 'agent', 'weight', 'rate', 
                'GFR', 'H', 'CO', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                'c_khe', 'c_vol',
                'd_khe', 'd_vol', 
                'c_dose', 'c_BAT',
                'd_dose', 'd_BAT',
                # _conc_liver
                'Tg', 'Dg', 've', 'c_kbh', 'd_kbh',
                # _relax_aorta
                'field_strength', 'c_R10_a', 'd_R10_a', 
                # _relax_liver
                'c_R10_l', 'd_R10_l',
                # Signal
                'c_S0_a', 'c_S0_l',
                'd_S0_a', 'd_S0_l',
                'c_B1corr_l', 'c_B1corr_a',
                'd_B1corr_l', 'd_B1corr_a',
                # Predict
                'TS',
            ],
            'free': inflow_pars + [
                # _conc_aorta 
                'CO', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                'c_khe',
                'd_khe', 
                'c_BAT',
                'd_BAT',
                # _conc_liver
                'Tg', 'Dg', 've', 'c_kbh', 'd_kbh',
            ],
            'free_control': [
                # _conc_aorta 
                'CO', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                'c_khe',
                'c_BAT', 
                # _conc_liver
                'Tg', 'Dg', 've', 'c_kbh',
            ],
            'free_drug': [
                # _conc_aorta 
                'd_khe', 
                'd_BAT',
                # _conc_liver
                'd_kbh',
            ],
            'free_drug_1': [
                # _conc_aorta 
                'd_khe', 
                'd_BAT',
                # _conc_liver
                'd_kbh',
            ],
        }
        return pars_list[select]
    
    # ==========================================
    # Forward Model: Helpers
    # ==========================================

    def _time(self, visit):
        p = self._pars
        return np.arange(0, p[f'{visit}_tmax'], p['dt'])

    def _conc_aorta(self, visit):
        p = self._pars
        t = self._time(visit)

        # Source
        conc = lib.ca_conc(p['agent'])
        J = lib.ca_injection(
            t, p['weight'], conc, p[f'{visit}_dose'], p['rate'], 
            p[f'{visit}_BAT'],
        )

        # Body extraction fraction
        khe = p[f'{visit}_khe']
        CL = khe * p[f'{visit}_vol'] + p['GFR']
        Eb = CL / (CL + p['CO'] * (1 - p['H']))
        
        # Compute aorta flux
        Jb = flux_aorta(
            J, E=Eb, dt=p['dt'], tol=p['dose_tolerance'],
            heartlung=['pfcomp', (p['Thl'], p['Dhl'])],
            organs=['2cxm', ([p['To'], p['To_e']], p['Eo'])],
        )
        return Jb / p['CO']

    def _conc_liver(self, cb, visit):
        p = self._pars

        cp = cb / (1 - p['H'])
        cp = pk.flux_pfcomp(cp, p['Tg'], p['Dg'], dt=p['dt'])  
        Ce = p['ve'] * cp
    
        khe = p[f'{visit}_khe']
        vh = 1 - p['ve'] / (1 - p['H'])
        Th = vh / p[f'{visit}_kbh']
        Ch = pk.conc(khe * cp, Th, dt=p['dt'], model="comp")

        return np.stack((Ce, Ch))

    def _relax_aorta(self, ca, visit):
        p = self._pars
        rb = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        R1a = p[f'{visit}_R10_a'] + rb * ca
        return R1a

    def _relax_liver(self, Cl, visit):
        p = self._pars
        rp = lib.relaxivity(p['field_strength'], 'plasma', p['agent'])
        rh = lib.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])
        R1l = p[f'{visit}_R10_l'] + rp * Cl[0, :] + rh * Cl[1, :]
        return R1l

    def _signal(self, R1, visit, roi):
        p = self._pars
        seq = self._cnfg['sequence']
        roi_seq = {
            'a': seq,
            'l': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
        }[roi]
        return sig.Signal(roi_seq, **p)(
            R1=R1, 
            S0=p[f'{visit}_S0_{roi}'], 
            B1corr=p[f'{visit}_B1corr_{roi}'],
            TE=0, PA=0,
        )
    
    # ==========================================
    # Forward Model: Times
    # ==========================================

    def _set_time_control(self):
        self._t_control = self._time('c')

    def _set_time_drug(self):
        self._t_drug = self._time('d')

    def _set_time(self):
        self._t_control = self._time('c')
        self._t_drug = self._time('d')

    # ==========================================
    # Forward Model: Aorta Control
    # ==========================================

    def _compute_conc_aorta_control(self):
        self._ca_control = self._conc_aorta('c')

    def _compute_relax_aorta_control(self):
        self._compute_conc_aorta_control()
        self._R1a_control = self._relax_aorta(self._ca_control, 'c')

    def _compute_signal_aorta_control(self):
        self._compute_relax_aorta_control()
        self._Sa_control = self._signal(self._R1a_control, 'c', 'a')

    def _predict_aorta_control(self, time):
        self._set_time_control()
        self._compute_signal_aorta_control()
        return utils.sample(time, self._t_control, self._Sa_control, self._pars['TS'])

    # ==========================================
    # Forward Model: Aorta Drug
    # ==========================================

    def _compute_conc_aorta_drug(self):
        self._ca_drug = self._conc_aorta('d')

    def _compute_relax_aorta_drug(self):
        self._compute_conc_aorta_drug()
        self._R1a_drug = self._relax_aorta(self._ca_drug, 'd')

    def _compute_signal_aorta_drug(self):
        self._compute_relax_aorta_drug()
        self._Sa_drug = self._signal(self._R1a_drug, 'd', 'a')

    def _predict_aorta_drug(self, time):
        self._set_time_drug()
        self._compute_signal_aorta_drug()
        return utils.sample(time, self._t_drug, self._Sa_drug, self._pars['TS'])

    # ==========================================
    # Forward Model: Liver Control
    # ==========================================

    def _compute_conc_liver_control(self):
        self._Cl_control = self._conc_liver(self._ca_control, 'c')

    def _compute_relax_liver_control(self):
        self._compute_conc_liver_control()
        self._R1l_control = self._relax_liver(self._Cl_control, 'c')

    def _compute_signal_liver_control(self):
        self._compute_relax_liver_control()
        self._Sl_control = self._signal(self._R1l_control, 'c', 'l')

    def _predict_liver_control(self, time):
        self._set_time_control()
        self._compute_signal_liver_control()
        return utils.sample(time, self._t_control, self._Sl_control, self._pars['TS'])
    
    # ==========================================
    # Forward Model: Liver Drug
    # ==========================================

    def _compute_conc_liver_drug(self):
        self._Cl_drug = self._conc_liver(self._ca_drug, 'd')

    def _compute_relax_liver_drug(self):
        self._compute_conc_liver_drug()
        self._R1l_drug = self._relax_liver(self._Cl_drug, 'd')

    def _compute_signal_liver_drug(self):
        self._compute_relax_liver_drug()
        self._Sl_drug = self._signal(self._R1l_drug, 'd', 'l')

    def _predict_liver_drug(self, time):
        self._set_time_drug()
        self._compute_signal_liver_drug()
        return utils.sample(time, self._t_drug, self._Sl_drug, self._pars['TS'])

    # ==========================================
    # Forward Model: All scans
    # ==========================================

    def _predict_control(self, time):
        Sa = self._predict_aorta_control(time[0])
        Sl = self._predict_liver_control(time[1])
        return Sa, Sl
    
    def _predict_drug(self, time):
        Sa = self._predict_aorta_drug(time[0])
        Sl = self._predict_liver_drug(time[1])
        return Sa, Sl
    
    def _predict(self, time):
        Sc = self._predict_control(time[:2])
        Sd = self._predict_drug(time[2:])
        return Sc + Sd
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================  

    def _estimate_parameters(self, time, signal, n0):
        p = self._pars
        seq = self._cnfg['sequence']
        roi_seq = {
            'a': seq,
            'l': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
        }

        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[2 * i: 2 * i + 2]))

            # Estimate BAT and BAT_2
            t_hl, d_hl = p['Thl'], p['Dhl']
            bat = time[2 * i][np.argmax(signal[2 * i])] - (1 - d_hl) * t_hl
            p[f'{visit}_BAT'] = max(bat, 0)

            def estimate_s0(roi, i0):
                B1 = p[f'{visit}_B1corr_{roi}']
                R10 = p[f'{visit}_R10_{roi}']
                s_ref = sig.Signal(roi_seq[roi], **p)(R1=R10, S0=1, B1corr=B1, TE=0, PA=0)
                p[f'{visit}_S0_{roi}'] = np.mean(signal[i0 + 2 * i][:n0[i]]) / s_ref if s_ref > 0 else 0

            estimate_s0('a', 0)
            estimate_s0('l', 1)


    def _train(
            self, time: tuple, signal: tuple, free=None, 
            bounds:dict=None, n0=[1, 1], staged=False, **kwargs,
        ):
        p = self._pars
        self._estimate_parameters(time, signal, n0)
        free = self._set_free_pars(free, bounds, LEXICON)

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == '3D-SPGR-SSI':
            for par in ['c_S0_a', 'd_S0_a']:
                if par not in free:
                    raise ValueError(f"For SSI sequence, '{par}' must be a free parameter.")    

        if staged:

            # Train control data
            v = 0
            t, s = time[v: v + 2], signal[v: v + 2]
            free_stage = {k: v for k, v in free.items() if k in self._params('free_control')}
            utils.train(self._predict_control, t, s, p, free_stage, **kwargs)
            
            # Train drug data
            v = 2
            t, s = time[v: v + 2], signal[v: v + 2]
            free_stage = {k: v for k, v in free.items() if k in self._params('free_drug')}
            utils.train(self._predict_drug, t, s, p, free_stage, **kwargs)

        # Train all parameters on all data
        return utils.train(self._predict, time, signal, p, free, **kwargs)

    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _plot(self, time: tuple, signal: tuple, xlim=None, fname=None, show=True):
        
        self._set_time()
        self._compute_signal_aorta_control()
        self._compute_signal_liver_control()
        self._compute_signal_aorta_drug()
        self._compute_signal_liver_drug()
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.3)

        ax1.set_title('First visit')
        ax2.set_title('Second visit')
        ax3.set_title('First visit')
        ax4.set_title('Second visit')

        def plot_data2scan(t, s, ti, si, ax, xl, color):
            if xl is None: xl = [0, t[-1]]
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xl)/60)
            ax.plot(ti / 60, si, marker='o', color=color[0], label='fitted data', linestyle='None')
            ax.plot(t / 60, s, linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        plot_data2scan(self._t_control, self._Sa_control, time[0], signal[0], ax1, xlim, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_control, self._Sl_control, time[1], signal[1], ax5, xlim, ['cornflowerblue', 'darkblue'])
        plot_data2scan(self._t_drug, self._Sa_drug, time[2], signal[2], ax2, xlim, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_drug, self._Sl_drug, time[3], signal[3], ax6, xlim, ['cornflowerblue', 'darkblue'])

        def plot_conc_aorta(t, c, ax, xl):
            if xl is None: xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xl)/60)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * c, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
            ax.legend()
        
        plot_conc_aorta(self._t_control, self._ca_control, ax3, xlim)
        plot_conc_aorta(self._t_drug, self._ca_drug, ax4, xlim)

        def plot_conc_liver(t, C, ax, xl):
            if xl is None: xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xl)/60)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * C[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax.plot(t / 60, 1000 * C[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax.plot(t / 60, 1000 * C.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')       
            ax.legend()

        plot_conc_liver(self._t_control, self._Cl_control, ax7, xlim)
        plot_conc_liver(self._t_drug, self._Cl_drug, ax8, xlim)

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    
    # ==========================================
    # Public API: dict Extraction
    # ==========================================

    def time(self) -> dict:
        """Time points in aorta and liver for the two visits"""
        self._set_time()
        tc, td = self._t_control, self._t_drug
        return {
            ('ctrl', 'aorta'): tc,
            ('ctrl', 'liver'): tc,
            ('drug', 'aorta'): td,
            ('drug', 'liver'): td,
        }

    def conc(self) -> dict:
        """Concentrations in aorta and liver.

        Returns:
            tuple: aorta blood concentrations, liver concentrations.
        """
        self._compute_conc_aorta_control()
        self._compute_conc_liver_control()
        self._compute_conc_aorta_drug()
        self._compute_conc_liver_drug()

        return {
            ('ctrl', 'aorta'): self._ca_control,
            ('ctrl', 'liver'): self._Cl_control,
            ('drug', 'aorta'): self._ca_drug,
            ('drug', 'liver'): self._Cl_drug,
        }
    
    def relax(self) -> dict:
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: aorta blood R1, liver R1.
        """
        self._compute_relax_aorta_control()
        self._compute_relax_liver_control()
        self._compute_relax_aorta_drug()
        self._compute_relax_liver_drug()

        return {
            ('ctrl', 'aorta'): self._R1a_control,
            ('ctrl', 'liver'): self._R1l_control,
            ('drug', 'aorta'): self._R1a_drug,
            ('drug', 'liver'): self._R1l_drug,
        }
    
    def signal(self) -> dict:
        """Signal in aorta and liver.

        Returns:
            tuple: aorta blood signal, liver 
              signal.
        """
        self._compute_signal_aorta_control()
        self._compute_signal_liver_control()
        self._compute_signal_aorta_drug()
        self._compute_signal_liver_drug()

        return {
            ('ctrl', 'aorta'): self._Sa_control,
            ('ctrl', 'liver'): self._Sl_control,
            ('drug', 'aorta'): self._Sa_drug,
            ('drug', 'liver'): self._Sl_drug,
        }
    
    def predict(self, time: dict) -> tuple:
        """Predict the data at given time points

        Args:
            time (tuple): tuple of 8 arrays with time points. The first 
              four are from the control visit: aorta in 
              the first scan, aorta in the second scan, liver in the first 
              scan, and liver in the second scan, in that order. 
              The second group of 4 is the same data for the treatment visit.

        Returns:
            tuple: tuple of 8 arrays with signals corresponding to time.
        """
        if isinstance(time, dict):
            time = (
                time['ctrl', 'aorta'], 
                time['ctrl', 'liver'], 
                time['drug', 'aorta'], 
                time['drug', 'liver'], 
            )
        else:
            time = tuple(4 * [time])
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[2 * i: 2 * i + 2]))
        S = self._predict(time)
        return {
            ('ctrl', 'aorta'): S[0],
            ('ctrl', 'liver'): S[1],
            ('drug', 'aorta'): S[0],
            ('drug', 'liver'): S[1],
        } 

    def train(
        self, time: dict, signal: dict, free=None, 
        bounds:dict=None, n0=[1, 1], staged=False, **kwargs,
    ) -> tuple:
        """Train the free parameters

        Args:
            time (tuple): (time_1_aorta, time_2_aorta, time_1_liver, time_2_liver)
            signal (tuple): (signal_1_aorta, signal_2_aorta, signal_1_liver, signal_2_liver).
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            R102a (float, optional): R1 value in arterial blood before the second injection. 
            R102l (float, optional): R1 value in liver before the second injection. 
            staged (bool, optional): If True, the training is performed in stages
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        if isinstance(time, dict):
            time = (
                time['ctrl', 'aorta'], 
                time['ctrl', 'liver'], 
                time['drug', 'aorta'], 
                time['drug', 'liver'], 
            )
        else:
            time = tuple(4 * [time])
        signal = (
            signal['ctrl', 'aorta'], 
            signal['ctrl', 'liver'], 
            signal['drug', 'aorta'], 
            signal['drug', 'liver'], 
        )
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[2 * i: 2 * i + 2]))
        return self._train(time, signal, free, bounds, n0, staged, **kwargs)


    def plot(self, time: dict, signal: dict, xlim=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 4 arrays with time points for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The four arrays can 
              be different in length and value.
            signal (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        if isinstance(time, dict):
            time = (
                time['ctrl', 'aorta'], 
                time['ctrl', 'liver'], 
                time['drug', 'aorta'], 
                time['drug', 'liver'], 
            )
        else:
            time = tuple(4 * [time])
        signal = (
            signal['ctrl', 'aorta'], 
            signal['ctrl', 'liver'], 
            signal['drug', 'aorta'], 
            signal['drug', 'liver'], 
        )
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[2 * i: 2 * i + 2]))
        self._plot(time, signal, xlim, fname, show)

    def cost(self, time: dict, signal: dict, metric: str = 'NRMS', nfree=None) -> float:
        """Return the goodness-of-fit

        Args:
            time (np.ndarray): array with time points
            signal (array-like): array with signal data for all pixels.
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
        if isinstance(time, dict):
            time = (
                time['ctrl', 'aorta'], 
                time['ctrl', 'liver'], 
                time['drug', 'aorta'], 
                time['drug', 'liver'], 
            )
        else:
            time = tuple(4 * [time])
        signal = np.concatenate((
            signal['ctrl', 'aorta'], 
            signal['ctrl', 'liver'], 
            signal['drug', 'aorta'], 
            signal['drug', 'liver'], 
        ))
        signal_pred = np.concatenate(self._predict(time))
        cost = utils.loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]



# def sdev_effect(v, dv):
#     # z = (x-y)/y = x/y-1 
#     # dz = sqrt((dy * dz/dy)**2 + (dx * dz/dx)**2)
#     # dz/dx = 1/y
#     # dz/dy = -x/y**2
#     x, y = v[0], v[1]
#     dx, dy = dv[0], dv[1]
#     dz_dx = 1 / y if y != 0 else 0
#     dz_dy = - x / y**2 if y != 0 else 0
#     return np.sqrt((dy * dz_dy)**2 + (dx * dz_dx)**2)
    
# def _deriv_params(pars):
    
#     vh = 1 - pars['v(e)'] / (1 - pars['H'])
#     C_khe = pars['C-k(he)']
#     C_kbh = pars['C-k(bh)']
#     D_khe = pars['D-k(he)'] 
#     D_kbh = pars['D-k(bh)']
#     C_CL = C_khe * pars['C-vol']
#     D_CL = D_khe * pars['D-vol']
#     pars_deriv = {
#         'v(h)': vh,
#         'C-CL': C_CL,
#         'D-CL': D_CL,
#         'RE-k(he)': (D_khe - C_khe) / C_khe,
#         'RE-k(bh)': (D_kbh - C_kbh) / C_kbh,
#         'RE-CL': (D_CL - C_CL) / C_CL,
#         'AE-k(he)': D_khe - C_khe,
#         'AE-k(bh)': D_kbh - C_kbh,
#         'AE-CL': D_CL - C_CL,
#     }
#     return pars_deriv

# def _div(a, b):
#     with np.errstate(divide='ignore'):
#         return np.divide(a, b)
