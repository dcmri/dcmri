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

    >>> time, aif, roi, gt = dc.fake.tissue2scan(R1b=1/dc.const.T1(3.0,'liver'))

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
    Aorta first baseline R1 (R1ba): 0.614 Hz
    Aorta first signal scale factor (S0a): 100.117 a.u.
    Liver first baseline R1 (R1bl): 1.33 Hz
    Liver first signal scale factor (S0l): 150.003 a.u.
    Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
    Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec

"""

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.quantities import QUANTITIES
from dcmri.core.sequences import SEQUENCES
from dcmri.core.tools import export_params
from dcmri.utils import const
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.kinetics.functions_input import ca_injection
from dcmri.kinetics.functions_aorta import flux_aorta
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.signal.modules_tissue import Signal



QUANTITIES = QUANTITIES | {

    # Seq Params
    'c_FA': {'init': 15, 'bounds': [0, 180], 'name': 'Control visit - Flip angle', 'unit': 'deg'},
    'd_FA': {'init': 15, 'bounds': [0, 180], 'name': 'Drug visit - Flip angle', 'unit': 'deg'},

    # Assay parameters
    'c_tmax': {'init': 4 * 60 * 60, 'name': 'Control visit - maximum acquisition time', 'unit': 'sec'},
    'c_dose': {'init': 0.05, 'name': 'Control visit - first contrast agent dose', 'unit': 'mL/kg'},
    'c_BAT': {'init': 120, 'bounds': [-60, 60], 'name': 'Control visit - first bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},
 
    'd_tmax': {'init': 4 * 60 * 60, 'name': 'Drug visit - maximum acquisition time', 'unit': 'sec'},
    'd_dose': {'init': 0.05, 'name': 'Drug visit - first contrast agent dose', 'unit': 'mL/kg'},
    'd_BAT': {'init': 120, 'bounds': [-60, 60], 'name': 'Drug visit - first bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},
 
    # MRI signal parameters - control visit
    'c_R1b_a': {'init': 1/const.T1(3.0, 'blood'), 'name': 'Control visit - aorta first baseline R1', 'unit': 'Hz'},
    'c_R1b_l': {'init': 1/const.T1(3.0, 'liver'), 'name': 'Control visit - liver first baseline R1', 'unit': 'Hz'},
    'c_R2sb_a': {'init': 20, 'name': 'Control visit - aorta first baseline R2*', 'unit': 'Hz'},
    'c_R2sb_l': {'init': 20, 'name': 'Control visit - liver first baseline R2*', 'unit': 'Hz'},
    'c_S0_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_Si_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_Si_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},

    # MRI signal parameters - drug visit
    'd_R1b_a': {'init': 1/const.T1(3.0, 'blood'), 'name': 'Drug visit - aorta first baseline R1', 'unit': 'Hz'},
    'd_R1b_l': {'init': 1/const.T1(3.0, 'liver'), 'name': 'Drug visit - liver first baseline R1', 'unit': 'Hz'},
    'd_R2sb_a': {'init': 20, 'name': 'Drug visit - aorta first baseline R2*', 'unit': 'Hz'},
    'd_R2sb_l': {'init': 20, 'name': 'Drug visit - liver first baseline R2*', 'unit': 'Hz'},
    'd_S0_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_Si_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_Si_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},

    'c_B1corr_a': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Arterial B1-correction factor', 'unit': ''},
    'c_B1corr_l': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Liver B1-correction factor', 'unit': ''},
    'd_B1corr_a': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Arterial B1-correction factor', 'unit': ''},
    'd_B1corr_l': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Liver B1-correction factor', 'unit': ''},

    # Aorta Kinetics - control visit
    'c_fCO_l': {'init': 0.25, 'bounds': [0.01, 0.99], 'name': 'Control visit - Liver flow fraction', 'unit': ''},
    'c_Eb': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Control visit - Body extraction', 'unit': ''},
    'c_CO': {'init': 100, 'bounds': [0, 500], 'name': 'Control visit - Cardiac output', 'unit': 'mL/s'},
    'c_GFR': {'init': 2, 'bounds': [0.5, 3], 'name': 'Control visit - glomerular filtration rate', 'unit': 'mL/sec'},
    'c_Thl': {'init': 10, 'bounds': [0, 30], 'name': 'Control visit - Heart-lung MTT', 'unit': 's'},
    'c_Dhl': {'init': 0.2, 'bounds': [0.01, 0.99], 'name': 'Control visit - Heart-lung dispersion', 'unit': ''},
    'c_To': {'init': 20, 'bounds': [0, 60], 'name': 'Control visit - Organ blood MTT', 'unit': 's'},
    'c_Eo': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Control visit - Organ extraction', 'unit': ''},
    'c_To_e': {'init': 120, 'bounds': [0, 800], 'name': 'Control visit - Organ EES MTT', 'unit': 's'},

    # Liver Kinetics - control visit
    'c_vol': {'init': 1000, 'name': 'Control visit - liver volume', 'unit': 'cm3'},
    'c_khe': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_kbh': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Control visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'c_Kbh': {'init': 0.00035, 'bounds': [0, 0.0007], 'name': 'Control visit - biliary tissue excretion rate', 'unit': '1/sec'},
    'c_Th': {'init': 3000, 'bounds': [0, 6 * 3600], 'name': 'Control visit - hepatocellular transit time', 'unit': 'sec'},
    'c_Tg': {'init': 30, 'bounds': [15, 120], 'name': 'Control visit - Gut transit time', 'unit': 'sec'},
    'c_Dg': {'init': 0.5, 'bounds': [0.01, 0.99], 'name': 'Control visit - Gut dispersion', 'unit': ''},
    'c_CL': {'init': None, 'name': 'Control visit - liver plasma clearance', 'unit': 'mL/sec'},
    'c_ve': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Control visit - Extracellular volume fraction', 'unit': 'mL/cm3'},
    'c_El': {'init': 0.05, 'bounds': [0, 1], 'name': 'Control visit - Liver extraction fraction', 'unit': ''},

    # Aorta Kinetics - drug visit
    'd_fCO_l': {'init': 0.25, 'bounds': [0.01, 0.99], 'name': 'Drug visit - Liver flow fraction', 'unit': ''},
    'd_Eb': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Drug visit - Body extraction', 'unit': ''},
    'd_CO': {'init': 100, 'bounds': [0, 500], 'name': 'Drug visit - Cardiac output', 'unit': 'mL/s'},
    'd_GFR': {'init': 2, 'bounds': [0.5, 3], 'name': 'Drug visit - glomerular filtration rate', 'unit': 'mL/sec'},
    'd_Thl': {'init': 10, 'bounds': [0, 30], 'name': 'Drug visit - Heart-lung MTT', 'unit': 's'},
    'd_Dhl': {'init': 0.2, 'bounds': [0.01, 0.99], 'name': 'Drug visit - Heart-lung dispersion', 'unit': ''},
    'd_To': {'init': 20, 'bounds': [0, 60], 'name': 'Drug visit - Organ blood MTT', 'unit': 's'},
    'd_Eo': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Drug visit - Organ extraction', 'unit': ''},
    'd_To_e': {'init': 120, 'bounds': [0, 800], 'name': 'Drug visit - Organ EES MTT', 'unit': 's'},    'd_vol': {'init': 1000, 'name': 'Drug visit - liver volume', 'unit': 'cm3'},

    # Liver Kinetics - drug visit
    'd_vol': {'init': 1000, 'name': 'Drug visit - liver volume', 'unit': 'cm3'},
    'd_khe': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_kbh': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_Kbh': {'init': 0.00035, 'bounds': [0, 0.0007], 'name': 'Drug visit - biliary tissue excretion rate', 'unit': '1/sec'},
    'd_Th': {'init': 3000, 'bounds': [0, 6 * 3600], 'name': 'Drug visit - hepatocellular transit time', 'unit': 'sec'},
    'd_Tg': {'init': 30, 'bounds': [15, 120], 'name': 'Drug visit - Gut transit time', 'unit': 'sec'},
    'd_Dg': {'init': 0.5, 'bounds': [0.01, 0.99], 'name': 'Drug visit - Gut dispersion', 'unit': ''},
    'd_CL': {'init': None, 'name': 'Drug visit - liver plasma clearance', 'unit': 'mL/sec'},
    'd_ve': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Drug visit -Extracellular volume fraction', 'unit': 'mL/cm3'},
    'd_El': {'init': 0.05, 'bounds': [0, 1], 'name': 'Control visit - Liver extraction fraction', 'unit': ''},

    # Kinetics - common
    'GFR': {'init': 2, 'bounds': [0.5, 3], 'name': 'Glomerular filtration rate', 'unit': 'mL/sec'},
    'a_Eb': {'init': None, 'name': 'Absolute effect in Body extraction', 'unit': ''},
    'r_Eb': {'init': None, 'name': 'Relative effect in Body extraction', 'unit': ''},
    'r_khe': {'init': None, 'name': 'Relative effect in hepatocellular uptake rate', 'unit': ''},
    'r_kbh': {'init': None, 'name': 'Relative effect in biliary excretion rate', 'unit': ''},
    'r_CL': {'init': None, 'name': 'Relative effect in liver plasma clearance', 'unit': ''},
    'a_khe': {'init': None, 'name': 'Absolute effect in hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'a_kbh': {'init': None, 'name': 'Absolute effect in biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'a_CL': {'init': None, 'name': 'Absolute effect in liver plasma clearance', 'unit': 'mL/sec'},

    # Derived AUC and RE
    'c_AUC_Cb': {'init': None, 'name': 'Control visit AUC for Cb (0-inf)', 'unit': 'M*sec'},
    'c_AUC_Cl': {'init': None, 'name': 'Control visit AUC for Cl (0-inf)', 'unit': 'M*sec'}, 
    'c_AUC35_Cb': {'init': None, 'name': 'Control visit AUC for Cb (0-35min)', 'unit': 'M*sec'},
    'c_AUC35_Cl': {'init': None, 'name': 'Control visit AUC for Cl (0-35min)', 'unit': 'M*sec'}, 
    'c_RE_R1b': {'init': None, 'name': 'Control visit RE for R1b at 20min', 'unit': ''},
    'c_RE_R1l': {'init': None, 'name': 'Control visit RE for R1l at 20min', 'unit': ''},
    'c_RE_Sb': {'init': None, 'name': 'Control visit RE for Sb at 20min', 'unit': ''},
    'c_RE_Sl': {'init': None, 'name': 'Control visit RE for Sl at 20min', 'unit': ''},

    'd_AUC_Cb': {'init': None, 'name': 'Drug visit AUC for Cb (0-inf)', 'unit': 'M*sec'},
    'd_AUC_Cl': {'init': None, 'name': 'Drug visit AUC for Cl (0-inf)', 'unit': 'M*sec'}, 
    'd_AUC35_Cb': {'init': None, 'name': 'Drug visit AUC for Cb (0-35min)', 'unit': 'M*sec'},
    'd_AUC35_Cl': {'init': None, 'name': 'Drug visit AUC for Cl (0-35min)', 'unit': 'M*sec'}, 
    'd_RE_R1b': {'init': None, 'name': 'Drug visit RE for R1b at 20min', 'unit': ''},
    'd_RE_R1l': {'init': None, 'name': 'Drug visit RE for R1l at 20min', 'unit': ''},
    'd_RE_Sb': {'init': None, 'name': 'Drug visit RE for Sb at 20min', 'unit': ''},
    'd_RE_Sl': {'init': None, 'name': 'Drug visit RE for Sl at 20min', 'unit': ''},
}

QVALUES = {k: v['init'] for k, v in QUANTITIES.items()}

def _div(a, b):
    with np.errstate(divide='ignore'):
        return np.divide(a, b)
    
def _deriv_params(p, sdev=None):

    fCO_l = p[f'fCO_l']
    Fb = fCO_l * p[f'CO'] / p[f'c_vol']

    Fpl = Fb * (1 - p['H'])
    Te = p[f've'] / Fpl

    c_El = p[f'c_khe'] / (p[f'c_khe'] + Fpl)
    d_El = p[f'd_khe'] / (p[f'd_khe'] + Fpl)

    CL = p[f'GFR']
    Fpk = (1 - fCO_l) * p[f'CO'] * (1 - p['H'])
    Eg = CL / (CL + Fpk)

    # c_Eb
    CL = p['c_khe'] * p[f'c_vol'] + p[f'GFR']
    Eb = CL / (CL + p[f'CO'] * (1 - p['H']))
    c_Eb = np.mean(Eb)
    # c_Eb = p[f'c_Eb']

    # d_Eb
    CL = p['d_khe'] * p[f'd_vol'] + p[f'GFR']
    Eb = CL / (CL + p[f'CO'] * (1 - p['H']))
    d_Eb = np.mean(Eb)
    # d_Eb = p[f'd_Eb']
    
    vh = 1 - p[f've'] / (1 - p['H'])
    c_khe = p['c_khe']
    #c_kbh = vh * p['c_Kbh']
    c_kbh = p['c_kbh']

    d_khe = p['d_khe'] 
    d_kbh = p['d_kbh']
    #d_kbh = vh * p['d_Kbh']
    c_CL = c_khe * p['c_vol']
    d_CL = d_khe * p['d_vol']
    p_deriv = {
        'c_Th': _div(vh, c_kbh),
        'd_Th': _div(vh, d_kbh),
        'c_El': c_El,
        'd_El': d_El,
        'Eg': Eg,
        'Te': Te,
        'Fb': Fb,
        'c_Eb': c_Eb,
        'd_Eb': d_Eb,
        'r_Eb': _div(d_Eb - c_Eb, c_Eb),
        'a_Eb': d_Eb - c_Eb,
        'vh': vh,
        'c_kbh': c_kbh,
        'd_kbh': d_kbh,
        'c_CL': c_CL,
        'd_CL': d_CL,
        'r_khe': _div(d_khe - c_khe, c_khe),
        'r_kbh': _div(d_kbh - c_kbh, c_kbh),
        'r_CL': _div(d_CL - c_CL, c_CL),
        'a_khe': d_khe - c_khe,
        'a_kbh': d_kbh - c_kbh,
        'a_CL': d_CL - c_CL,
    }
    sd_deriv = {}
    if sdev is not None:
        pass
    return p_deriv, sd_deriv



class AortaLiverDrug(SuperRoiModel):
    """Aorta and liver signals over control and treatment visits.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta and liver, measured over two days with and without administration 
    of a drug.

    Args:
        sequence (str, optional): imaging sequence.
        params (dict, optional): override parameter defaults.

    See Also:
        `AortaLiver`
    """

    configs = {
        'heartlung': ConcAortaLiver.configs['heartlung'],
        'organs': ConcAortaLiver.configs['organs'],
        'liver': ConcAortaLiver.configs['liver'],
        'non_stationary': ConcAortaLiver.configs['non_stationary'],
        'sequence': ['3D-SPGR-SS', '3D-SPGR-SSI', 'ZTE-3D-SPGR-SS'],
    }
    
    def __init__(
            self, 
            heartlung='pfcomp', 
            organs='comp', 
            liver='1I-IC', 
            non_stationary=None, 
            sequence='3D-SPGR-SS', 
            **params,
        ):
        self._version = '1.0'
        cnfg = {
            'heartlung': heartlung, 
            'organs': organs, 
            'liver': liver, 
            'non_stationary': non_stationary, 
            'sequence': sequence,
        }
        self._set_config(cnfg)
        self._conc = {
            'c': ConcAortaLiver(**self._cnfg).connect_inputs(self._map('c')),
            'd': ConcAortaLiver(**self._cnfg).connect_inputs(self._map('d')),
        }
        self._set_params(QVALUES | params)

        # Set multi-channel baseline if not done by the user
        if sequence in ['Eq-DE-EPI', 'DE-EPI']:
            for roi in ['a', 'l']:
                Sb = f"Sb_{roi}"
                if Sb not in params:
                    self._pars[Sb] = np.full(2, self._pars[Sb])

    # ==========================================
    # Backend
    # ==========================================

    # Helper function
    def _sequence(self, roi):
        seq = self._cnfg['sequence']
        if roi in ['l'] and seq == '3D-SPGR-SSI':
            return '3D-SPGR-SS'
        return seq
    
    # Helper function
    def _tissue_props(self, roi):
        return set(SEQUENCES[self._sequence(roi)]['parameters']['tissue'])
    
    # ==========================================
    # Model Parameters
    # ==========================================

    def _map(self, visit):
        visit_pars = [
            # _conc_aorta
            'tmax',
            'khe', 'vol_l',
            'dose',
            'BAT',
            # _conc_liver
            'kbh', 
            'FA', 
            # Signal
            'Si_a', 'Si_l',
            'B1corr_l', 'B1corr_a',
        ]
        return {f'{visit}_{p}': p  for p in visit_pars}

    def _params(self, select=None):
        seq = self._cnfg['sequence']
        inflow_pars = ['TF'] if seq == '3D-SPGR-SSI' else []

        if select is None:
            pars = inflow_pars + [
                # _conc_aorta
                'dt', 'c_tmax', 'd_tmax', 
                'dose_tolerance', 'agent', 'weight', 'rate',  
                'H', 'Thl', 'Dhl', 'To', 'To_e', 'Eo', 
                'fCO_l', 'CO', 'GFR',
                'c_khe', 'c_vol',
                'd_khe', 'd_vol', 
                'c_dose', 'd_dose',
                'c_BAT', 'd_BAT',
                # _conc_liver
                'Tg', 've', 
                'c_kbh', 'd_kbh', 
                #'c_Kbh', 'd_Kbh', 
                # _relax_aorta
                'field_strength', 
                'R1b_a', 'R2sb_a', 
                # _relax_liver
                'R1b_l', 'R2sb_l',  
                # sequence
                'TR', 'TE',
                'c_FA', 'd_FA', 
                # Signal
                'c_Si_a', 'c_Si_l',
                'd_Si_a', 'd_Si_l',
                'c_B1corr_l', 'c_B1corr_a',
                'd_B1corr_l', 'd_B1corr_a',
                # Predict
                'TS',
            ]

        if select=='free':
            pars = [
                # _conc_aorta 
                'c_BAT', 'd_BAT',
                'fCO_l', 'CO', 'GFR', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                # _conc_liver
                'Tg', 've', 
                'c_khe', 'c_kbh', #'c_Kbh',
                'd_khe', 'd_kbh', #'d_Kbh',
            ]
        
        return pars
    
    # ==========================================
    # Forward Model: Helpers
    # ==========================================

    def _compute_conc(self):
        p = self._pars 
        self._C = {visit: self._conc[visit](**p) for visit in ['c', 'd']}

    def _relax_aorta(self, ca):
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        r2s = const.r2s(p['field_strength'], 'blood', p['agent'])
        R1a = p[f'R1b_a'] + rb * ca
        R2sa = p[f'R2sb_a'] + r2s * ca
        return R1a, R2sa

    def _relax_liver(self, Cl):
        p = self._pars
        rp = const.r1(p['field_strength'], 'plasma', p['agent'])
        rh = const.r1(p['field_strength'], 'hepatocytes', p['agent'])
        r2s = const.r2s(p['field_strength'], 'tissue', p['agent'])
        R1l = p[f'R1b_l'] + rp * Cl[0, :] + rh * Cl[1, :]
        R2sl = p[f'R2sb_l'] + r2s * Cl.sum(axis=0) 
        return R1l, R2sl

    def _compute_relax(self):
        self._compute_conc()
        self._R1a_control, self._R2sa_control = self._relax_aorta(self._ca_control)
        self._R1l_control, self._R2sl_control = self._relax_liver(self._Cl_control)
        self._R1a_drug, self._R2sa_drug = self._relax_aorta(self._ca_drug)
        self._R1l_drug, self._R2sl_drug = self._relax_liver(self._Cl_drug)


    def _signal(self, R1, R2s, visit, roi):
        p =self._pars

        # Signal model
        seq = self._cnfg['sequence']
        roi_seq = {
            'a': seq,
            'l': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
        }[roi]
        FA = p[f'{visit}_FA']
        B1 = p[f'{visit}_B1corr_{roi}']
        const = {'Fw': 0, 'v': 1, 'me': 1, 'noise_sdev':0, 'FA': FA, 'B1corr': B1}
        signal = Signal(roi_seq, defaults=p)
    
        # Derive S0 from the baseline signal
        S_ref = signal(R1=R1[0], R2s=R2s[0], S0=1, **const)
        S0 = p[f'{visit}_Si_{roi}'] / S_ref if S_ref > 0 else 0

        return signal(R1=R1, R2s=R2s, S0=S0, **const)

    def _compute_signal(self):
        self._compute_relax()
        self._Sa_control = self._signal(self._R1a_control, self._R2sa_control, 'c', 'a')
        self._Sa_drug = self._signal(self._R1a_drug, self._R2sa_drug, 'd', 'a')
        self._Sl_control = self._signal(self._R1l_control, self._R2sl_control, 'c', 'l')
        self._Sl_drug = self._signal(self._R1l_drug, self._R2sl_drug, 'd', 'l')

    def _time(self, visit):
        p = self._pars
        return np.arange(0, p[f'{visit}_tmax'], p['dt'])
    
    def _predict(self, time):
        self._compute_signal()
        tc, td = self._time('c'), self._time('d')

        Sac = sample(time[0], tc, self._Sa_control, self._pars['TS'])   
        Slc = sample(time[1], tc, self._Sl_control, self._pars['TS'])
        Sad = sample(time[2], td, self._Sa_drug, self._pars['TS'])
        Sld = sample(time[3], td, self._Sl_drug, self._pars['TS'])
        return Sac, Slc, Sad,  Sld
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================  

    def _estimate_bat(self, time, signal):
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[2 * i: 2 * i + 2]))

            t_hl, d_hl = p[f'Thl'], p[f'Dhl']
            bat = time[2 * i][np.argmax(signal[2 * i])] - (1 - d_hl) * t_hl
            p[f'{visit}_BAT'] = max(bat, 0)

    def _estimate_baseline(self, signal, n0):
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_Si_a'] = np.mean(signal[0 + 2 * i][:n0[i]]) 
            p[f'{visit}_Si_l'] = np.mean(signal[1 + 2 * i][:n0[i]])


    def _train(
            self, time: tuple, signal: tuple, free: dict, 
            bounds: dict, n0: list, n_runs: list, sigma: tuple=None, **kwargs,
        ):

        # Normalize signal
        scl = [1 / np.mean(si) if np.mean(si) != 0 else 1 for si in signal]
        signal = tuple([si * scl[i] for i, si in enumerate(signal)])
        
        p = self._pars
        self._estimate_bat(time, signal)
        self._estimate_baseline(signal, n0)
        free = self._set_free_pars(free, bounds, QUANTITIES)

        # train(self._predict, time, signal, p, free, **kwargs)

        # Train with multiple random initializations
        best_loss = np.inf
        best_vals = None
        best_sdev = None
        best_pcov = None
        for run in range(n_runs):
            # Set initial values for free parameters
            self._estimate_bat(time, signal)
            if run > 0:
                for param_name, bounds in free.items():
                    if param_name not in ['c_BAT', 'd_BAT']:
                        lower, upper = bounds[0], bounds[1]
                        p[param_name] = np.random.uniform(lower, upper)

            # Perform training (joint)
            vals, sdev, pcov = train(self._predict, time, signal, p, free, sigma=sigma, **kwargs)

            # Evaluate loss
            current_loss = self._cost(time, signal, 'RMS', None)
            if current_loss < best_loss:
                best_loss = current_loss
                best_vals = vals
                best_sdev = sdev
                best_pcov = pcov

        # Update state with optimal values
        for k, v in best_vals.items():
            p[k] = v
        
        # Renormalize signal amplitudes
        for i, visit in enumerate(['c', 'd']): 
            p[f'{visit}_Si_a'] /= scl[0 + 2 * i]
            p[f'{visit}_Si_l'] /= scl[1 + 2 * i]

        return best_vals, best_sdev, best_pcov


    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _plot(self, time: tuple, signal: tuple, xlim=None, clim=None, fname=None, show=True):
        
        self._set_time()
        self._compute_signal_aorta_control()
        self._compute_signal_liver_control()
        self._compute_signal_aorta_drug()
        self._compute_signal_liver_drug()
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.3)

        ax1.set_title('Control visit')
        ax2.set_title('Treatment visit')
        ax3.set_title('Control visit')
        ax4.set_title('Treatment visit')

        def plot_data2scan(t, s, ti, si, ax, xl, yl, color):
            if xl is None: xl = [0, t[-1]]
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(ti / 60, si, marker='o', color=color[0], label='fitted data', linestyle='None')
            ax.plot(t / 60, s, linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        ylim_a = [
            0.9 * min(self._Sa_control.min(), self._Sa_drug.min(), signal[0].min(), signal[2].min()), 
            1.1 * max(self._Sa_control.max(), self._Sa_drug.max(), signal[0].max(), signal[2].max()),
        ]
        ylim_l = [
            0.9 * min(self._Sl_control.min(), self._Sl_drug.min(), signal[1].min(), signal[3].min()), 
            1.1 * max(self._Sl_control.max(), self._Sl_drug.max(), signal[1].max(), signal[3].max()),
        ]

        plot_data2scan(self._t_control, self._Sa_control, time[0], signal[0], ax1, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_control, self._Sl_control, time[1], signal[1], ax5, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(self._t_drug, self._Sa_drug, time[2], signal[2], ax2, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_drug, self._Sl_drug, time[3], signal[3], ax6, xlim, ylim_l, ['cornflowerblue', 'darkblue'])

        def plot_conc_aorta(t, c, ax, xl, yl):
            if xl is None: xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * c, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
            ax.legend()

        if clim is None:
            ylim = [0, 1.1 * 1000 * max(self._ca_control.max(), self._ca_drug.max())]
        else:
            ylim = [0, clim[0] * 1000]

        plot_conc_aorta(self._t_control, self._ca_control, ax3, xlim, ylim)
        plot_conc_aorta(self._t_drug, self._ca_drug, ax4, xlim, ylim)

        def plot_conc_liver(t, C, ax, xl, yl):
            if xl is None: xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * C[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax.plot(t / 60, 1000 * C[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax.plot(t / 60, 1000 * C.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')       
            ax.legend()

        if clim is None:
            ylim = [0, 1.1 * 1000 * max(self._Cl_control.max(), self._Cl_drug.max())]
        else:
            ylim = [0, clim[1] * 1000]

        plot_conc_liver(self._t_control, self._Cl_control, ax7, xlim, ylim)
        plot_conc_liver(self._t_drug, self._Cl_drug, ax8, xlim, ylim)

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    def _cost(self, time, signal, metric, nfree) -> float:
        signal_pred = np.concatenate(self._predict(time))
        signal = np.concatenate(signal)
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]
    
    def _time_dict(self) -> dict:
        self._set_time()
        tc, td = self._t_control, self._t_drug
        return {
            ('ctrl', 'aorta'): tc,
            ('ctrl', 'liver'): tc,
            ('drug', 'aorta'): td,
            ('drug', 'liver'): td,
        }
    
    def _conc_dict(self) -> dict:
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
    
    def _relax_dict(self) -> dict:
        self._compute_relax_aorta_control()
        self._compute_relax_liver_control()
        self._compute_relax_aorta_drug()
        self._compute_relax_liver_drug()

        R1 = {
            ('ctrl', 'aorta'): self._R1a_control,
            ('ctrl', 'liver'): self._R1l_control,
            ('drug', 'aorta'): self._R1a_drug,
            ('drug', 'liver'): self._R1l_drug,
        }
        R2s = {
            ('ctrl', 'aorta'): self._R2sa_control,
            ('ctrl', 'liver'): self._R2sl_control,
            ('drug', 'aorta'): self._R2sa_drug,
            ('drug', 'liver'): self._R2sl_drug,
        }
        return R1, R2s
    
    def _signal_dict(self) -> dict:
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
    
    def _desc(self):
        t = self._time_dict() 
        C = self._conc_dict()
        R1, _ = self._relax_dict()
        S = self._signal_dict()

        pars = {}

        for visit in ['ctrl', 'drug']:

            # Compute AUC over 3hrs
            BAT = self._pars[f'{visit[0]}_BAT']

            tAUCb = (BAT < t[visit, 'aorta']) & (t[visit, 'aorta'] < BAT + 180 * 60)
            tAUCl = (BAT < t[visit, 'liver']) & (t[visit, 'liver'] < BAT + 180 * 60)
            AUC_Cb = np.trapezoid(C[visit, 'aorta'][tAUCb], t[visit, 'aorta'][tAUCb]) 
            AUC_Cl = np.trapezoid(C[visit, 'liver'].sum(axis=0)[tAUCl], t[visit, 'liver'][tAUCl])

            # Compute AUC over 35min
            tAUCb = (BAT < t[visit, 'aorta']) & (t[visit, 'aorta'] < BAT + 35 * 60)
            tAUCl = (BAT < t[visit, 'liver']) & (t[visit, 'liver'] < BAT + 35 * 60)
            AUC35_Cb = np.trapezoid(C[visit, 'aorta'][tAUCb], t[visit, 'aorta'][tAUCb]) 
            AUC35_Cl = np.trapezoid(C[visit, 'liver'].sum(axis=0)[tAUCl], t[visit, 'liver'][tAUCl])

            # Compute relative enhancement at 20mins
            tRE = BAT + 20*60
            R1b = R1[visit, 'aorta']
            R1l = R1[visit, 'liver']
            RE_R1b = (R1b[t[visit, 'aorta'] < tRE][-1] - R1b[0])/R1b[0]
            RE_R1l = (R1l[t[visit, 'liver'] < tRE][-1] - R1l[0])/R1l[0]

            S0b = np.mean(S[visit, 'aorta'][t[visit, 'aorta'] < BAT - 30])
            S0l = np.mean(S[visit, 'liver'][t[visit, 'liver'] < BAT - 30])
            RE_Sb = (S[visit, 'aorta'][t[visit, 'aorta'] < tRE][-1] - S0b)/S0b
            RE_Sl = (S[visit, 'liver'][t[visit, 'liver'] < tRE][-1] - S0l)/S0l

            pars = pars | {
                f'{visit[0]}_AUC_Cb' : AUC_Cb, 
                f'{visit[0]}_AUC_Cl': AUC_Cl,
                f'{visit[0]}_AUC35_Cb': AUC35_Cb,
                f'{visit[0]}_AUC35_Cl': AUC35_Cl, 
                f'{visit[0]}_RE_R1b': RE_R1b,
                f'{visit[0]}_RE_R1l': RE_R1l,
                f'{visit[0]}_RE_Sb': RE_Sb, 
                f'{visit[0]}_RE_Sl': RE_Sl,
            } 

        return pars

    
    # ==========================================
    # Public API: dict Extraction
    # ==========================================

    def export_params(self, sdev=None, desc=False) -> dict:
        """Parameters with values, definition and units"""
        pars_deriv, sdev_deriv = _deriv_params(self._pars, sdev)
        if desc:
            pars_deriv = pars_deriv | self._desc()
        pars = self._pars | pars_deriv
        sdev = sdev | sdev_deriv if sdev is not None else sdev_deriv
        return export_params(pars, sdev=sdev, lexicon=QUANTITIES)

    def time(self) -> dict:
        """Time points in aorta and liver for the two visits"""
        return self._time_dict()

    def conc(self) -> dict:
        """Concentrations in aorta and liver.

        Returns:
            tuple: aorta blood concentrations, liver concentrations.
        """
        return self._conc_dict()
    
    def relax(self) -> dict:
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: aorta blood R1, liver R1.
        """
        return self._relax_dict()
    
    def signal(self) -> dict:
        """Signal in aorta and liver.

        Returns:
            tuple: aorta blood signal, liver 
              signal.
        """
        return self._signal_dict()
    
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
        elif isinstance(time, np.ndarray):
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
        bounds:dict=None, n0=[1, 1], n_runs=1, **kwargs,
    ) -> tuple:
        """Train the free parameters

        Args:
            time (tuple): (time_1_aorta, time_2_aorta, time_1_liver, time_2_liver)
            signal (tuple): (signal_1_aorta, signal_2_aorta, signal_1_liver, signal_2_liver).
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            n_runs (int, optional): Number of fits to run. A different set of initial values is chosen each time.
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
        elif isinstance(time, np.ndarray):
            time = tuple(4 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['ctrl', 'aorta'], 
                signal['ctrl', 'liver'], 
                signal['drug', 'aorta'], 
                signal['drug', 'liver'], 
            )
        return self._train(time, signal, free, bounds, n0, n_runs, **kwargs)


    def plot(self, time: dict, signal: dict, xlim=None, clim=None, fname=None, show=True):
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
        elif isinstance(time, np.ndarray):
            time = tuple(4 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['ctrl', 'aorta'], 
                signal['ctrl', 'liver'], 
                signal['drug', 'aorta'], 
                signal['drug', 'liver'], 
            )
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[2 * i: 2 * i + 2]))
        self._plot(time, signal, xlim, clim, fname, show)

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
        elif isinstance(time, np.ndarray):
            time = tuple(4 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['ctrl', 'aorta'], 
                signal['ctrl', 'liver'], 
                signal['drug', 'aorta'], 
                signal['drug', 'liver'], 
            )
        return self._cost(time, signal, metric, nfree)