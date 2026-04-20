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

    Use `fake.tissue` to generate synthetic test data from 
    experimentally-derived concentrations:

    >>> time, aif, roi, gt = dc.fake.tissue2scan(R10=1/dc.const.T1(3.0,'liver'))

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
"""


import matplotlib.pyplot as plt
import numpy as np

import dcmri.kinetics.lib as pk
from dcmri import const
from dcmri.lexicon import QUANTITIES
from dcmri.bloch import Signal
from dcmri.utils.misc import sample, interp
from dcmri.utils.fit import train, loss
from dcmri.core import SuperModel


QUANTITIES = QUANTITIES | {

    # Assay parameters
    'c_tmax': {'init': 4 * 60 * 60, 'name': 'Control visit - maximum acquisition time', 'unit': 'sec'},
    'c_t_scan2': {'init': 2 * 60 * 60, 'name': 'Control visit - start of second scan', 'unit': 'sec'},
    'c_dose_1': {'init': 0.05, 'name': 'Control visit - first contrast agent dose', 'unit': 'mL/kg'},
    'c_dose_2': {'init': 0.05, 'name': 'Control - second contrast agent dose', 'unit': 'mL/kg'},
    'c_BAT_1': {'init': 120, 'bounds': [-60, 60], 'name': 'Control visit - first bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},
    'c_BAT_2': {'init': 7200 + 900, 'bounds': [-60, 60], 'name': 'Control visit - second bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},
 
    'd_tmax': {'init': 4 * 60 * 60, 'name': 'Drug visit - maximum acquisition time', 'unit': 'sec'},
    'd_tmax_1': {'init': 2 * 60 * 60, 'name': 'Drug visit - first scan acquisition time', 'unit': 'sec'},
    'd_t_scan2': {'init': 2 * 60 * 60, 'name': 'Drug visit - start of second scan', 'unit': 'sec'},
    'd_dose_1': {'init': 0.05, 'name': 'Drug visit - first contrast agent dose', 'unit': 'mL/kg'},
    'd_dose_2': {'init': 0.05, 'name': 'Drug - second contrast agent dose', 'unit': 'mL/kg'},
    'd_BAT_1': {'init': 120, 'bounds': [-60, 60], 'name': 'Drug visit - first bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},
    'd_BAT_2': {'init': 7200 + 900, 'bounds': [-60, 60], 'name': 'Drug visit - second bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},

    # MRI signal parameters - control visit
    'c_R10_a': {'init': 1/const.T1(3.0, 'blood'), 'name': 'Control visit - aorta first baseline R1', 'unit': 'Hz'},
    'c_R10_l': {'init': 1/const.T1(3.0, 'liver'), 'name': 'Control visit - liver first baseline R1', 'unit': 'Hz'},
    'c_R20s_a': {'init': 20, 'name': 'Control visit - aorta first baseline R2*', 'unit': 'Hz'},
    'c_R20s_l': {'init': 20, 'name': 'Control visit - liver first baseline R2*', 'unit': 'Hz'},
    'c_S0_1_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_1_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_2_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_2_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},

    # MRI signal parameters - drug visit
    'd_R10_a': {'init': 1/const.T1(3.0, 'blood'), 'name': 'Drug visit - aorta first baseline R1', 'unit': 'Hz'},
    'd_R10_l': {'init': 1/const.T1(3.0, 'liver'), 'name': 'Drug visit - liver first baseline R1', 'unit': 'Hz'},
    'd_R20s_a': {'init': 20, 'name': 'Drug visit - aorta first baseline R2*', 'unit': 'Hz'},
    'd_R20s_l': {'init': 20, 'name': 'Drug visit - liver first baseline R2*', 'unit': 'Hz'},
    'd_S0_1_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_1_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_2_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_2_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},

    'c_B1corr_1_a': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Arterial B1-correction factor', 'unit': ''},
    'c_B1corr_1_l': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Liver B1-correction factor', 'unit': ''},
    'c_B1corr_2_a': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Arterial B1-correction factor of a second scan', 'unit': ''},
    'c_B1corr_2_l': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Liver B1-correction factor of a second scan', 'unit': ''},
    'd_B1corr_1_a': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Arterial B1-correction factor', 'unit': ''},
    'd_B1corr_1_l': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Liver B1-correction factor', 'unit': ''},
    'd_B1corr_2_a': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Arterial B1-correction factor of a second scan', 'unit': ''},
    'd_B1corr_2_l': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Liver B1-correction factor of a second scan', 'unit': ''},

    # Kinetics - control visit
    'c_vol': {'init': 1000, 'name': 'Control visit - liver volume', 'unit': 'cm3'},
    'c_khe_i': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_khe_f': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_khe': {'name': 'Control visit - Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_dkhe': {'name': 'Control visit - change in hepatocellular uptake rate', 'unit': ''},
    'c_kbh': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Control visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'c_CL': {'name': 'Control visit - liver plasma clearance', 'unit': 'mL/sec'},

    # Kinetics - drug visit
    'd_vol': {'init': 1000, 'name': 'Drug visit - liver volume', 'unit': 'cm3'},
    'd_khe_i': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_khe_f': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_khe': {'name': 'Drug visit - Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_dkhe': {'name': 'Drug visit - Change in hepatocellular uptake rate', 'unit': ''},
    'd_kbh_i': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_kbh_f': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - final biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_kbh': {'name': 'Drug visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_dkbh': {'name': 'Drug visit - change in biliary excretion rate', 'unit': ''},
    'd_CL': {'name': 'Drug visit - liver plasma clearance', 'unit': 'mL/sec'},

    # Kinetics - common
    'r_khe': {'name': 'Relative effect in hepatocellular uptake rate', 'unit': ''},
    'r_kbh': {'name': 'Relative effect in biliary excretion rate', 'unit': ''},
    'r_CL': {'name': 'Relative effect in liver plasma clearance', 'unit': ''},
    'a_khe': {'name': 'Absolute effect in hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'a_kbh': {'name': 'Absolute effect in biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'a_CL': {'name': 'Absolute effect in liver plasma clearance', 'unit': 'mL/sec'},
}


def _sample_signal(time, t, S, TS) -> tuple:
    if isinstance(time, np.ndarray):
        return sample(time, t, S, TS)
    else:
        return tuple([sample(ti, t, S, TS) for ti in time])


class AortaLiverDynamicDrug(SuperModel):
    """Aorta and liver signals over two vists with two scans each.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta and liver, measured over two separate scans. 

    Args:
        sequence (str, optional): imaging sequence. 
        params (dict, optional): override parameter defaults. 
    See Also:
        `AortaLiver`
    """

    configs = {'sequence': ['3D-SPGR-SS', '3D-SPGR-SSI']}

    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._version = '1.0'
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(lexicon=QUANTITIES, **params)
            
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
                'c_khe_i', 'c_khe_f', 'c_vol',
                'd_khe_i', 'd_khe_f', 'd_vol', 
                'c_dose_1', 'c_BAT_1',  'c_dose_2', 'c_BAT_2',
                'd_dose_1', 'd_BAT_1',  'd_dose_2', 'd_BAT_2',
                # _conc_liver
                'Tg', 'Dg', 've', 'c_kbh', 'd_kbh_i', 'd_kbh_f',
                # _relax_aorta
                'field_strength', 
                'c_R10_a', 'd_R10_a', 
                'c_R20s_a', 'd_R20s_a', 
                # _relax_liver
                'c_R10_l', 'd_R10_l',
                'c_R20s_l', 'd_R20s_l', 
                # sequence
                'TR', 'FA', 
                # Signal
                'c_t_scan2', 'c_S0_1_a', 'c_S0_2_a', 'c_S0_1_l', 'c_S0_2_l',
                'd_t_scan2', 'd_S0_1_a', 'd_S0_2_a', 'd_S0_1_l', 'd_S0_2_l',
                'c_B1corr_1_l', 'c_B1corr_1_a', 'c_B1corr_2_l', 'c_B1corr_2_a',
                'd_B1corr_1_l', 'd_B1corr_1_a', 'd_B1corr_2_l', 'd_B1corr_2_a',
                # Predict
                'TS',
            ],
            'free': inflow_pars + [
                # _conc_aorta 
                'CO', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                'c_khe_i', 'c_khe_f',
                'd_khe_i', 'd_khe_f', 
                'c_BAT_1',  'c_BAT_2',
                'd_BAT_1',  'd_BAT_2',
                # _conc_liver
                'Tg', 'Dg', 've', 'c_kbh', 'd_kbh_i', 'd_kbh_f',
                # Signal
                'c_S0_2_a', 'c_S0_2_l',
                'd_S0_2_a', 'd_S0_2_l',
            ],
            'free_control': [
                # _conc_aorta 
                'CO', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                'c_khe_i', 'c_khe_f',
                'c_BAT_1',  'c_BAT_2',
                # _conc_liver
                'Tg', 'Dg', 've', 'c_kbh',
                # Signal
                'c_S0_2_a', 'c_S0_2_l',
            ],
            'free_control_1': [
                # _conc_aorta 
                'CO', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                'c_khe_i',
                'c_BAT_1',
                # _conc_liver
                'Tg', 'Dg', 've', 'c_kbh',
            ],
            'free_control_2': [
                # _conc_aorta 
                'c_khe_f',
                'c_BAT_2',
                # Signal
                'c_S0_2_a', 'c_S0_2_l',
            ],
            'free_drug': [
                # _conc_aorta 
                'd_khe_i', 'd_khe_f', 
                'd_BAT_1',  'd_BAT_2',
                # _conc_liver
                'd_kbh_i', 'd_kbh_f',
                # Signal
                'd_S0_2_a', 'd_S0_2_l',
            ],
            'free_drug_1': [
                # _conc_aorta 
                'd_khe_i', 
                'd_BAT_1',
                # _conc_liver
                'd_kbh_i',
            ],
            'free_drug_2': [
                # _conc_aorta 
                'd_khe_f', 
                'd_BAT_2',
                # _conc_liver
                'd_kbh_f',
                # Signal
                'd_S0_2_a', 'd_S0_2_l',
            ],
        }
        return pars_list[select]
    
    # ==========================================
    # Forward Model: Helpers
    # ==========================================

    def _time(self, visit):
        p = self._pars
        return np.arange(0, p[f'{visit}_tmax'], p['dt'])

    def _conc_aorta(self, visit, scans):
        p = self._pars
        t = self._time(visit)
        
        # Source
        conc = const.ca_conc(p['agent'])
        J = pk.ca_injection(
            t, p['weight'], conc, p[f'{visit}_dose_1'], p['rate'], 
            p[f'{visit}_BAT_1'],
        )
        if scans==2:
            J += pk.ca_injection(
                t, p['weight'], conc, p[f'{visit}_dose_2'], p['rate'], 
                p[f'{visit}_BAT_2'],
            )

        # Body extraction fraction
        if scans==1:
            khe = p[f'{visit}_khe_i']
        elif scans==2:
            khe = interp([p[f'{visit}_khe_i'], p[f'{visit}_khe_f']], t)

        CL = khe * p[f'{visit}_vol'] + p['GFR']
        Eb = CL / (CL + p['CO'] * (1 - p['H']))

        # Compute aorta flux
        Jb = pk.flux_aorta(
            J, E=Eb, dt=p['dt'], tol=p['dose_tolerance'],
            heartlung=['pfcomp', (p['Thl'], p['Dhl'])],
            organs=['2cxm', ([p['To'], p['To_e']], p['Eo'])],
        )
        return Jb / p['CO']

    def _conc_liver(self, cb, visit, scans):
        p = self._pars
        t = self._time(visit)

        if scans==1:
            khe = p[f'{visit}_khe_i']
        elif scans==2:
            khe = interp([p[f'{visit}_khe_i'], p[f'{visit}_khe_f']], t)    

        cp = cb / (1 - p['H'])
        cp = pk.flux_pfcomp(cp, p['Tg'], p['Dg'], dt=p['dt'])  
        Ce = p['ve'] * cp
    
        vh = 1 - p['ve'] / (1 - p['H'])
        if visit == 'c':
            Th = vh / p[f'{visit}_kbh']
            Ch = pk.conc(khe * cp, Th, dt=p['dt'], model="comp")
        elif visit == 'd':
            if scans==1:
                Th = vh / p[f'{visit}_kbh_i']
                Ch = pk.conc(khe * cp, Th, dt=p['dt'], model="comp")
            else:
                Th_i = vh / p[f'{visit}_kbh_i']
                Th_f = vh / p[f'{visit}_kbh_f']
                Th = interp([Th_i, Th_f], t)
                Ch = pk.conc(khe * cp, Th, dt=p['dt'], model="nscomp")

        return np.stack((Ce, Ch))

    def _relax_aorta(self, ca, visit):
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        R1a = p[f'{visit}_R10_a'] + rb * ca
        r2s = const.r2s(p['field_strength'], 'blood', p['agent'])
        R2sa = p[f'{visit}_R20s_a'] + r2s * ca
        return R1a, R2sa

    def _relax_liver(self, Cl, visit):
        p = self._pars
        rp = const.r1(p['field_strength'], 'plasma', p['agent'])
        rh = const.r1(p['field_strength'], 'hepatocytes', p['agent'])
        R1l = p[f'{visit}_R10_l'] + rp * Cl[0, :] + rh * Cl[1, :]
        r2s = const.r2s(p['field_strength'], 'tissue', p['agent'])
        R2sl = p[f'{visit}_R20s_l'] + r2s * Cl.sum(axis=0) 
        return R1l, R2sl

    def _signal(self, R1, R2s, visit, scans, roi):
        p = self._pars
        t = self._time(visit)

        seq = self._cnfg['sequence']
        roi_seq = {
            'a': seq,
            'l': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
        }[roi]

        def roi_signal(scan, R1roi, R2sroi):
            return Signal(roi_seq, **p)(
                R1=R1roi, 
                R2s=R2sroi,
                S0=p[f'{visit}_S0_{scan}_{roi}'], 
                B1corr=p[f'{visit}_B1corr_{scan}_{roi}'],
            )

        if scans==1:
            S = roi_signal(1, R1, R2s)

        elif scans==2:
            S = np.zeros_like(t)

            # First scan signal
            ts = t < p[f'{visit}_t_scan2']
            S[ts] = roi_signal(1, R1[ts], R2s[ts])

            # Second scan signal
            ts = t >= p[f'{visit}_t_scan2']
            S[ts] = roi_signal(2, R1[ts], R2s[ts])

        return S
    
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

    def _compute_conc_aorta_control(self, scans=2):
        self._ca_control = self._conc_aorta('c', scans)

    def _compute_relax_aorta_control(self, scans=2):
        self._compute_conc_aorta_control(scans)
        self._R1a_control, self._R2sa_control = self._relax_aorta(self._ca_control, 'c')

    def _compute_signal_aorta_control(self, scans=2):
        self._compute_relax_aorta_control(scans)
        self._Sa_control = self._signal(self._R1a_control, self._R2sa_control, 'c', scans, 'a')

    def _predict_aorta_control(self, time, scans=2):
        self._set_time_control()
        self._compute_signal_aorta_control(scans)
        return _sample_signal(time, self._t_control, self._Sa_control, self._pars['TS'])

    # ==========================================
    # Forward Model: Aorta Drug
    # ==========================================

    def _compute_conc_aorta_drug(self, scans=2):
        self._ca_drug = self._conc_aorta('d', scans)

    def _compute_relax_aorta_drug(self, scans=2):
        self._compute_conc_aorta_drug(scans)
        self._R1a_drug, self._R2sa_drug = self._relax_aorta(self._ca_drug, 'd')

    def _compute_signal_aorta_drug(self, scans=2):
        self._compute_relax_aorta_drug(scans)
        self._Sa_drug = self._signal(self._R1a_drug, self._R2sa_drug, 'd', scans, 'a')

    def _predict_aorta_drug(self, time, scans=2):
        self._set_time_drug()
        self._compute_signal_aorta_drug(scans)
        return _sample_signal(time, self._t_drug, self._Sa_drug, self._pars['TS'])

    # ==========================================
    # Forward Model: Liver Control
    # ==========================================

    def _compute_conc_liver_control(self, scans=2):
        self._Cl_control = self._conc_liver(self._ca_control, 'c', scans)

    def _compute_relax_liver_control(self, scans=2):
        self._compute_conc_liver_control(scans)
        self._R1l_control, self._R2sl_control = self._relax_liver(self._Cl_control, 'c')

    def _compute_signal_liver_control(self, scans=2):
        self._compute_relax_liver_control(scans)
        self._Sl_control = self._signal(self._R1l_control, self._R2sl_control, 'c', scans, 'l')

    def _predict_liver_control(self, time, scans=2):
        self._set_time_control()
        self._compute_signal_liver_control(scans)
        return _sample_signal(time, self._t_control, self._Sl_control, self._pars['TS'])
    
    # ==========================================
    # Forward Model: Liver Drug
    # ==========================================

    def _compute_conc_liver_drug(self, scans=2):
        self._Cl_drug = self._conc_liver(self._ca_drug, 'd', scans)

    def _compute_relax_liver_drug(self, scans=2):
        self._compute_conc_liver_drug(scans)
        self._R1l_drug, self._R2sl_drug = self._relax_liver(self._Cl_drug, 'd')

    def _compute_signal_liver_drug(self, scans=2):
        self._compute_relax_liver_drug(scans)
        self._Sl_drug = self._signal(self._R1l_drug, self._R2sl_drug, 'd', scans, 'l')

    def _predict_liver_drug(self, time, scans=2):
        self._set_time_drug()
        self._compute_signal_liver_drug(scans)
        return _sample_signal(time, self._t_drug, self._Sl_drug, self._pars['TS'])

    # ==========================================
    # Forward Model: All scans
    # ==========================================

    def _predict_control_1(self, time):
        Sa = self._predict_aorta_control(time[0], scans=1)
        Sl = self._predict_liver_control(time[1], scans=1)
        return Sa, Sl
    
    def _predict_control_2(self, time):
        Sa = self._predict_aorta_control(time[0], scans=2)
        Sl = self._predict_liver_control(time[1], scans=2)
        return Sa, Sl

    def _predict_control(self, time):
        Sa = self._predict_aorta_control(time[:2])
        Sl = self._predict_liver_control(time[2:])
        return Sa + Sl
    
    def _predict_drug_1(self, time):
        Sa = self._predict_aorta_drug(time[0], scans=1)
        Sl = self._predict_liver_drug(time[1], scans=1)
        return Sa, Sl
    
    def _predict_drug_2(self, time):
        Sa = self._predict_aorta_drug(time[0], scans=2)
        Sl = self._predict_liver_drug(time[1], scans=2)
        return Sa, Sl
    
    def _predict_drug(self, time):
        Sa = self._predict_aorta_drug(time[:2])
        Sl = self._predict_liver_drug(time[2:])
        return Sa + Sl
    
    def _predict(self, time):
        Sc = self._predict_control(time[:4])
        Sd = self._predict_drug(time[4:])
        return Sc + Sd
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================  

    def _estimate_parameters(
        self, time, signal, n0, R102a, R102l
    ):
        p = self._pars
        seq = self._cnfg['sequence']
        roi_seq = {
            'a': seq,
            'l': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
        }

        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[4 * i: 4 * i + 4]))

            # Estimate BAT and BAT_2
            t_hl, d_hl = p['Thl'], p['Dhl']
            bat1 = time[0 + 4 * i][np.argmax(signal[0 + 4 * i])] - (1 - d_hl) * t_hl
            bat2 = time[1 + 4 * i][np.argmax(signal[1 + 4 * i])] - (1 - d_hl) * t_hl
            p[f'{visit}_BAT_1'] = max(bat1, 0)
            p[f'{visit}_BAT_2'] = max(bat2, 0)

            def estimate_s0(scan, roi, i0, R10):
                B1=p[f'{visit}_B1corr_{scan}_{roi}']
                R20s=p[f'{visit}_R20s_{roi}']
                s_ref = Signal(roi_seq[roi], **p)(R1=R10, R2s=R20s, S0=1, B1corr=B1)
                p[f'{visit}_S0_{scan}_{roi}'] = np.mean(signal[i0 + 4 * i][:n0[i]]) / s_ref if s_ref > 0 else 0

            estimate_s0(1, 'a', 0, p[f'{visit}_R10_a'])
            estimate_s0(1, 'l', 2, p[f'{visit}_R10_l'])

            def estimate_s02(roi, i0, R10):
                if R10 is None:
                    p[f'{visit}_S0_2_{roi}'] = p[f'{visit}_S0_1_{roi}']
                else:
                    estimate_s0(2, roi, i0, R10[i])

            estimate_s02('a', 1, R102a)
            estimate_s02('l', 3, R102l)


    def _train(
            self, time: tuple, signal: tuple, free=None, 
            bounds:dict=None, R102a=None, R102l=None, n0=[1, 1], 
            staged=False, **kwargs,
        ):
        p = self._pars
        self._estimate_parameters(time, signal, n0, R102a, R102l)
        free = self._set_free_pars(free, bounds, QUANTITIES)

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == '3D-SPGR-SSI':
            for par in ['c_S0_1_a', 'd_S0_1_a', 'c_S0_2_a', 'd_S0_2_a']:
                if par not in free:
                    raise ValueError(f"For SSI sequence, '{par}' must be a free parameter.")     

        if staged:

            # Train control data
            v = 0

            t, s = (time[v + 0], time[v + 2]), (signal[v + 0], signal[v + 2])
            free_stage = {k: v for k, v in free.items() if k in self._params('free_control_1')}
            train(self._predict_control_1, t, s, p, free_stage, **kwargs)

            t, s = (time[v + 1], time[v + 3]), (signal[v + 1], signal[v + 3])
            free_stage = {k: v for k, v in free.items() if k in self._params('free_control_2')}
            train(self._predict_control_2, t, s, p, free_stage, **kwargs)

            t, s = time[v: v + 4], signal[v: v + 4]
            free_stage = {k: v for k, v in free.items() if k in self._params('free_control')}
            train(self._predict_control, t, s, p, free_stage, **kwargs)
            
            # Train drug data
            v = 4

            t, s = (time[v + 0], time[v + 2]), (signal[v + 0], signal[v + 2])
            free_stage = {k: v for k, v in free.items() if k in self._params('free_drug_1')}
            train(self._predict_drug_1, t, s, p, free_stage, **kwargs)

            t, s = (time[v + 1], time[v + 3]), (signal[v + 1], signal[v + 3])
            free_stage = {k: v for k, v in free.items() if k in self._params('free_drug_2')}
            train(self._predict_drug_2, t, s, p, free_stage, **kwargs)

            t, s = time[v: v + 4], signal[v: v + 4]
            free_stage = {k: v for k, v in free.items() if k in self._params('free_drug')}
            train(self._predict_drug, t, s, p, free_stage, **kwargs)

        # Train all parameters on all data
        return train(self._predict, time, signal, p, free, **kwargs)

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
            ax.plot(np.concatenate(ti) / 60, np.concatenate(si), marker='o', color=color[0], label='fitted data', linestyle='None')
            ax.plot(t / 60, s, linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        plot_data2scan(self._t_control, self._Sa_control, time[0:2], signal[0:2], ax1, xlim, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_control, self._Sl_control, time[2:4], signal[2:4], ax5, xlim, ['cornflowerblue', 'darkblue'])
        plot_data2scan(self._t_drug, self._Sa_drug, time[4:6], signal[4:6], ax2, xlim, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_drug, self._Sl_drug, time[6:8], signal[6:8], ax6, xlim, ['cornflowerblue', 'darkblue'])

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
    # Public API: Data Extraction TODO provided option of 1 time array, use dict for signals
    # ==========================================

    def time(self) -> dict:
        """Time points in aorta and liver for the two visits"""
        self._set_time()
        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['c_t_scan2'], self._pars['d_t_scan2']
        return {
            ('ctrl', 'aorta', 1): tc[tc < t2c],
            ('ctrl', 'aorta', 2): tc[tc >= t2c],
            ('ctrl', 'liver', 1): tc[tc < t2c],
            ('ctrl', 'liver', 2): tc[tc >= t2c],
            ('drug', 'aorta', 1): td[td < t2d],
            ('drug', 'aorta', 2): td[td >= t2d],
            ('drug', 'liver', 1): td[td < t2d],
            ('drug', 'liver', 2): td[td >= t2d],
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

        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['c_t_scan2'], self._pars['d_t_scan2']  
        return {
            ('ctrl', 'aorta', 1): self._ca_control[tc < t2c],
            ('ctrl', 'aorta', 2): self._ca_control[tc >= t2c],
            ('ctrl', 'liver', 1): self._Cl_control[:, tc < t2c],
            ('ctrl', 'liver', 2): self._Cl_control[:, tc >= t2c],
            ('drug', 'aorta', 1): self._ca_drug[td < t2d],
            ('drug', 'aorta', 2): self._ca_drug[td >= t2d],
            ('drug', 'liver', 1): self._Cl_drug[:, td < t2d],
            ('drug', 'liver', 2): self._Cl_drug[:, td >= t2d] ,
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

        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['c_t_scan2'], self._pars['d_t_scan2']
        R1 = {
            ('ctrl', 'aorta', 1): self._R1a_control[tc < t2c],
            ('ctrl', 'aorta', 2): self._R1a_control[tc >= t2c],
            ('ctrl', 'liver', 1): self._R1l_control[tc < t2c],
            ('ctrl', 'liver', 2): self._R1l_control[tc >= t2c],
            ('drug', 'aorta', 1): self._R1a_drug[td < t2d],
            ('drug', 'aorta', 2): self._R1a_drug[td >= t2d],
            ('drug', 'liver', 1): self._R1l_drug[td < t2d],
            ('drug', 'liver', 2): self._R1l_drug[td >= t2d],
        } 
        R2s = {
            ('ctrl', 'aorta', 1): self._R2sa_control[tc < t2c],
            ('ctrl', 'aorta', 2): self._R2sa_control[tc >= t2c],
            ('ctrl', 'liver', 1): self._R2sl_control[tc < t2c],
            ('ctrl', 'liver', 2): self._R2sl_control[tc >= t2c],
            ('drug', 'aorta', 1): self._R2sa_drug[td < t2d],
            ('drug', 'aorta', 2): self._R2sa_drug[td >= t2d],
            ('drug', 'liver', 1): self._R2sl_drug[td < t2d],
            ('drug', 'liver', 2): self._R2sl_drug[td >= t2d],
        }
        return R1, R2s
    
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

        tc, td = self._t_control, self._t_drug
        t2c, t2d = self._pars['c_t_scan2'], self._pars['d_t_scan2']
        return {
            ('ctrl', 'aorta', 1): self._Sa_control[tc < t2c],
            ('ctrl', 'aorta', 2): self._Sa_control[tc >= t2c],
            ('ctrl', 'liver', 1): self._Sl_control[tc < t2c],
            ('ctrl', 'liver', 2): self._Sl_control[tc >= t2c],
            ('drug', 'aorta', 1): self._Sa_drug[td < t2d],
            ('drug', 'aorta', 2): self._Sa_drug[td >= t2d],
            ('drug', 'liver', 1): self._Sl_drug[td < t2d],
            ('drug', 'liver', 2): self._Sl_drug[td >= t2d],
        } 
    
    def predict(self, time: dict) -> dict:
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
                time['ctrl', 'aorta', 1], time['ctrl', 'aorta', 2], 
                time['ctrl', 'liver', 1], time['ctrl', 'liver', 2], 
                time['drug', 'aorta', 1], time['drug', 'aorta', 2], 
                time['drug', 'liver', 1], time['drug', 'liver', 2], 
            )
        else:
            time = tuple(8 * [time])
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[4 * i: 4 * i + 4]))
        Sc = self._predict_control(time[:4])
        Sd = self._predict_drug(time[4:])
        return {
            ('ctrl', 'aorta', 1): Sc[0],
            ('ctrl', 'aorta', 2): Sc[1],
            ('ctrl', 'liver', 1): Sc[2],
            ('ctrl', 'liver', 2): Sc[3],
            ('drug', 'aorta', 1): Sd[0],
            ('drug', 'aorta', 2): Sd[1],
            ('drug', 'liver', 1): Sd[2],
            ('drug', 'liver', 2): Sd[3],
        } 

    def train(
            self, time: dict, signal: dict, free=None, 
            bounds:dict=None, R102a=None, R102l=None, n0=[1, 1], 
            staged=False, **kwargs,
        ):
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
                time['ctrl', 'aorta', 1], time['ctrl', 'aorta', 2], 
                time['ctrl', 'liver', 1], time['ctrl', 'liver', 2], 
                time['drug', 'aorta', 1], time['drug', 'aorta', 2], 
                time['drug', 'liver', 1], time['drug', 'liver', 2], 
            )
        else:
            time = tuple(8 * [time])
        signal = (
            signal['ctrl', 'aorta', 1], signal['ctrl', 'aorta', 2], 
            signal['ctrl', 'liver', 1], signal['ctrl', 'liver', 2], 
            signal['drug', 'aorta', 1], signal['drug', 'aorta', 2], 
            signal['drug', 'liver', 1], signal['drug', 'liver', 2], 
        )
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[4 * i: 4 * i + 4]))
        return self._train(time, signal, free, bounds, R102a, R102l, n0, staged, **kwargs)


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
                time['ctrl', 'aorta', 1], time['ctrl', 'aorta', 2], 
                time['ctrl', 'liver', 1], time['ctrl', 'liver', 2], 
                time['drug', 'aorta', 1], time['drug', 'aorta', 2], 
                time['drug', 'liver', 1], time['drug', 'liver', 2], 
            )
        else:
            time = tuple(8 * [time])
        signal = (
            signal['ctrl', 'aorta', 1], signal['ctrl', 'aorta', 2], 
            signal['ctrl', 'liver', 1], signal['ctrl', 'liver', 2], 
            signal['drug', 'aorta', 1], signal['drug', 'aorta', 2], 
            signal['drug', 'liver', 1], signal['drug', 'liver', 2], 
        )
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[4 * i: 4 * i + 4]))
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
                time['ctrl', 'aorta', 1], time['ctrl', 'aorta', 2], 
                time['ctrl', 'liver', 1], time['ctrl', 'liver', 2], 
                time['drug', 'aorta', 1], time['drug', 'aorta', 2], 
                time['drug', 'liver', 1], time['drug', 'liver', 2], 
            )
        else:
            time = tuple(8 * [time])
        signal = np.concatenate((
            signal['ctrl', 'aorta', 1], signal['ctrl', 'aorta', 2], 
            signal['ctrl', 'liver', 1], signal['ctrl', 'liver', 2], 
            signal['drug', 'aorta', 1], signal['drug', 'aorta', 2], 
            signal['drug', 'liver', 1], signal['drug', 'liver', 2], 
        ))
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
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

#     # TODO: return sdev of derived
    
#     vh = 1 - pars['ve'] / (1 - pars['H'])
#     C_khe = np.mean([pars['c_khe_i'], pars['c_khe_f']])
#     C_kbh = pars['c_kbh']
#     D_khe = pars['d_khe_i'] 

#     t = np.arange(0, pars[f'd_tmax'], pars['dt'])
#     Th_i = _div(vh, pars['d_kbh_i'])
#     Th_f = _div(vh, pars['d_kbh_f'])
#     Th = utils.interp([Th_i, Th_f], t)
#     D_kbh = _div(vh, Th.mean())
    
#     C_CL = C_khe * pars['c_vol']
#     D_CL = D_khe * pars['d_vol']
#     pars_deriv = {
#         'vh': vh,
#         'c_khe': C_khe,
#         'd_khe': D_khe,
#         'd_kbh': D_kbh,
#         'c_CL': C_CL,
#         'd_CL': D_CL,
#         'r_khe': _div(D_khe - C_khe, C_khe),
#         'r_kbh': _div(D_kbh - C_kbh, C_kbh),
#         'r_CL': _div(D_CL - C_CL, C_CL),
#         'a_khe': D_khe - C_khe,
#         'a_kbh': D_kbh - C_kbh,
#         'a_CL': D_CL - C_CL,
#         'c_dkhe': _div(pars['c_khe_f'] - pars['c_khe_i'], pars['c_khe_i']),
#         'd_dkhe': _div(pars['d_khe_f'] - pars['d_khe_i'], pars['d_khe_i']),
#         'd_dkbh': _div(pars['d_kbh_f'] - pars['d_kbh_i'], pars['d_kbh_i']),
#     }
#     return pars_deriv

# def _div(a, b):
#     with np.errstate(divide='ignore'):
#         return np.divide(a, b)
