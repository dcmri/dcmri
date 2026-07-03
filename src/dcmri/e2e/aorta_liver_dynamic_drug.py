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

from dcmri.kinetics.functions_input import ca_injection
from dcmri.kinetics.functions_aorta import flux_aorta
from dcmri.utils import const
from dcmri.core.quantities import QUANTITIES
from dcmri.core.tools import export_params
from dcmri.signal.modules_tissue import Signal
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.core.model import SuperModel
from dcmri.kinetics.modules_conc import ConcLiver


QUANTITIES = QUANTITIES | {

    # Seq Params
    'c_FA_1': {'init': 15, 'bounds': [0, 180], 'name': 'Control visit - first flip angle', 'unit': 'deg'},
    'c_FA_2': {'init': 15, 'bounds': [0, 180], 'name': 'Control visit - second flip angle', 'unit': 'deg'},
    'd_FA_1': {'init': 15, 'bounds': [0, 180], 'name': 'Drug visit - first flip angle', 'unit': 'deg'},
    'd_FA_2': {'init': 15, 'bounds': [0, 180], 'name': 'Drug visit - second flip angle', 'unit': 'deg'},

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

    'FFl': {'init': 0.25, 'bounds': [0.01, 0.99], 'name': 'Liver flow fraction', 'unit': ''},

    # MRI signal parameters - control visit
    'c_R1b_a': {'init': 1/const.T1(3.0, 'blood'), 'name': 'Control visit - aorta first baseline R1', 'unit': 'Hz'},
    'c_R1b_l': {'init': 1/const.T1(3.0, 'liver'), 'name': 'Control visit - liver first baseline R1', 'unit': 'Hz'},
    'c_R2sb_a': {'init': 20, 'name': 'Control visit - aorta first baseline R2*', 'unit': 'Hz'},
    'c_R2sb_l': {'init': 20, 'name': 'Control visit - liver first baseline R2*', 'unit': 'Hz'},
    'c_S0_1_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_1_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_2_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_S0_2_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_Si_1_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta first baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_Si_1_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver first baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_Si_2_a': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - aorta second baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'c_Si_2_l': {'init': 1, 'bounds': [0, 2], 'name': 'Control visit - liver second baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},

    # MRI signal parameters - drug visit
    'd_R1b_a': {'init': 1/const.T1(3.0, 'blood'), 'name': 'Drug visit - aorta first baseline R1', 'unit': 'Hz'},
    'd_R1b_l': {'init': 1/const.T1(3.0, 'liver'), 'name': 'Drug visit - liver first baseline R1', 'unit': 'Hz'},
    'd_R2sb_a': {'init': 20, 'name': 'Drug visit - aorta first baseline R2*', 'unit': 'Hz'},
    'd_R2sb_l': {'init': 20, 'name': 'Drug visit - liver first baseline R2*', 'unit': 'Hz'},
    'd_S0_1_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_1_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver first signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_2_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_S0_2_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_Si_1_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta first baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_Si_1_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver first baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_Si_2_a': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - aorta second baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'd_Si_2_l': {'init': 1, 'bounds': [0, 2], 'name': 'Drug visit - liver second baseline signal', 'unit': 'a.u.', 'bounds_type': 'mult'},

    'c_B1corr_1_a': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Arterial B1-correction factor', 'unit': ''},
    'c_B1corr_1_l': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Liver B1-correction factor', 'unit': ''},
    'c_B1corr_2_a': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Arterial B1-correction factor of a second scan', 'unit': ''},
    'c_B1corr_2_l': {'init': 1, 'bounds': [0, 5], 'name': 'Control visit - Liver B1-correction factor of a second scan', 'unit': ''},
    'd_B1corr_1_a': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Arterial B1-correction factor', 'unit': ''},
    'd_B1corr_1_l': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Liver B1-correction factor', 'unit': ''},
    'd_B1corr_2_a': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Arterial B1-correction factor of a second scan', 'unit': ''},
    'd_B1corr_2_l': {'init': 1, 'bounds': [0, 5], 'name': 'Drug visit - Liver B1-correction factor of a second scan', 'unit': ''},

    # Aorta Kinetics - control visit
    'c_FFl': {'init': 0.25, 'bounds': [0.01, 0.99], 'name': 'Control visit - Liver flow fraction', 'unit': ''},
    'c_Eb': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Control visit - Body extraction', 'unit': ''},
    'c_Eb_i': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Control visit - Initial Body extraction', 'unit': ''},
    'c_Eb_f': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Control visit - Final Body extraction', 'unit': ''},
    'c_CO': {'init': 100, 'bounds': [0, 500], 'name': 'Control visit - Cardiac output', 'unit': 'mL/s'},
    'c_GFR': {'init': 2, 'bounds': [0.5, 3], 'name': 'Control visit - glomerular filtration rate', 'unit': 'mL/sec'},
    'c_Thl': {'init': 10, 'bounds': [0, 30], 'name': 'Control visit - Heart-lung MTT', 'unit': 's'},
    'c_Dhl': {'init': 0.2, 'bounds': [0.01, 0.99], 'name': 'Control visit - Heart-lung dispersion', 'unit': ''},
    'c_To': {'init': 20, 'bounds': [0, 60], 'name': 'Control visit - Organ blood MTT', 'unit': 's'},
    'c_Eo': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Control visit - Organ extraction', 'unit': ''},
    'c_To_e': {'init': 120, 'bounds': [0, 800], 'name': 'Control visit - Organ EES MTT', 'unit': 's'},
    
    # Liver Kinetics - control visit
    'c_vol': {'init': 1000, 'name': 'Control visit - liver volume', 'unit': 'cm3'},
    'c_khe_i': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_khe_f': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_khe': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Control visit - Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'c_dkhe': {'name': 'Control visit - change in hepatocellular uptake rate', 'unit': ''},
    'c_kbh': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Control visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'c_Kbh': {'init': 0.00035, 'bounds': [0, 0.0007], 'name': 'Control visit - biliary tissue excretion rate', 'unit': '1/sec'},
    'c_Th': {'init': 3000, 'bounds': [0, 6 * 3600], 'name': 'Control visit - hepatocellular transit time', 'unit': 'sec'},
    'c_Tg': {'init': 30, 'bounds': [15, 120], 'name': 'Control visit - Gut transit time', 'unit': 'sec'},
    'c_Dg': {'init': 0.5, 'bounds': [0.01, 0.99], 'name': 'Control visit - Gut dispersion', 'unit': ''},
    'c_CL': {'name': 'Control visit - liver plasma clearance', 'unit': 'mL/sec'},
    'c_ve': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Control visit - Extracellular volume fraction', 'unit': 'mL/cm3'},
    'c_El': {'init': 0.05, 'bounds': [0, 1], 'name': 'Control visit - Liver extraction fraction', 'unit': ''},

    # Aorta Kinetics - drug visit
    'd_FFl': {'init': 0.25, 'bounds': [0.01, 0.99], 'name': 'Drug visit - Liver flow fraction', 'unit': ''},
    'd_Eb': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Drug visit - Body extraction', 'unit': ''},
    'd_Eb_i': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Drug visit - Initial Body extraction', 'unit': ''},
    'd_Eb_f': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Drug visit - Final Body extraction', 'unit': ''},
    'd_CO': {'init': 100, 'bounds': [0, 500], 'name': 'Drug visit - Cardiac output', 'unit': 'mL/s'},
    'd_GFR': {'init': 2, 'bounds': [0.5, 3], 'name': 'Drug visit - glomerular filtration rate', 'unit': 'mL/sec'},
    'd_Thl': {'init': 10, 'bounds': [0, 30], 'name': 'Drug visit - Heart-lung MTT', 'unit': 's'},
    'd_Dhl': {'init': 0.2, 'bounds': [0.01, 0.99], 'name': 'Drug visit - Heart-lung dispersion', 'unit': ''},
    'd_To': {'init': 20, 'bounds': [0, 60], 'name': 'Drug visit - Organ blood MTT', 'unit': 's'},
    'd_Eo': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Drug visit - Organ extraction', 'unit': ''},
    'd_To_e': {'init': 120, 'bounds': [0, 800], 'name': 'Drug visit - Organ EES MTT', 'unit': 's'},    'd_vol': {'init': 1000, 'name': 'Drug visit - liver volume', 'unit': 'cm3'},

    # Liver Kinetics - drug visit
    'd_khe_i': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_khe_f': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_khe': {'init': 0.0025, 'bounds': [0.0, 0.005], 'name': 'Drug visit - Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'd_dkhe': {'name': 'Drug visit - Change in hepatocellular uptake rate', 'unit': ''},
    'd_kbh_i': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_kbh_f': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - final biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_kbh': {'init': 0.00025, 'bounds': [0, 0.0005], 'name': 'Drug visit - biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'd_dkbh': {'name': 'Drug visit - change in biliary excretion rate', 'unit': ''},
    'd_Kbh': {'init': 0.00035, 'bounds': [0, 0.0007], 'name': 'Drug visit - biliary tissue excretion rate', 'unit': '1/sec'},
    'd_Th': {'init': 3000, 'bounds': [0, 6 * 3600], 'name': 'Drug visit - hepatocellular transit time', 'unit': 'sec'},
    'd_Tg': {'init': 30, 'bounds': [15, 120], 'name': 'Drug visit - Gut transit time', 'unit': 'sec'},
    'd_Dg': {'init': 0.5, 'bounds': [0.01, 0.99], 'name': 'Drug visit - Gut dispersion', 'unit': ''},
    'd_CL': {'name': 'Drug visit - liver plasma clearance', 'unit': 'mL/sec'},
    'd_ve': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Drug visit -Extracellular volume fraction', 'unit': 'mL/cm3'},
    'd_El': {'init': 0.05, 'bounds': [0, 1], 'name': 'Control visit - Liver extraction fraction', 'unit': ''},

    # Kinetics - common
    'a_Eb': {'name': 'Absolute effect in Body extraction', 'unit': ''},
    'r_Eb': {'name': 'Relative effect in Body extraction', 'unit': ''},
    'r_khe': {'name': 'Relative effect in hepatocellular uptake rate', 'unit': ''},
    'r_kbh': {'name': 'Relative effect in biliary excretion rate', 'unit': ''},
    'r_CL': {'name': 'Relative effect in liver plasma clearance', 'unit': ''},
    'a_khe': {'name': 'Absolute effect in hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'a_kbh': {'name': 'Absolute effect in biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'a_CL': {'name': 'Absolute effect in liver plasma clearance', 'unit': 'mL/sec'},

    # Derived AUC and RE
    'c_AUC_Cb': {'name': 'Control visit AUC for Cb (0-inf)', 'unit': 'M*sec'},
    'c_AUC_Cl': {'name': 'Control visit AUC for Cl (0-inf)', 'unit': 'M*sec'}, 
    'c_AUC35_Cb': {'name': 'Control visit AUC for Cb (0-35min)', 'unit': 'M*sec'},
    'c_AUC35_Cl': {'name': 'Control visit AUC for Cl (0-35min)', 'unit': 'M*sec'}, 
    'c_RE_R1b': {'name': 'Control visit RE for R1b at 20min', 'unit': ''},
    'c_RE_R1l': {'name': 'Control visit RE for R1l at 20min', 'unit': ''},
    'c_RE_Sb': {'name': 'Control visit RE for Sb at 20min', 'unit': ''},
    'c_RE_Sl': {'name': 'Control visit RE for Sl at 20min', 'unit': ''},

    'd_AUC_Cb': {'name': 'Drug visit AUC for Cb (0-inf)', 'unit': 'M*sec'},
    'd_AUC_Cl': {'name': 'Drug visit AUC for Cl (0-inf)', 'unit': 'M*sec'}, 
    'd_AUC35_Cb': {'name': 'Drug visit AUC for Cb (0-35min)', 'unit': 'M*sec'},
    'd_AUC35_Cl': {'name': 'Drug visit AUC for Cl (0-35min)', 'unit': 'M*sec'}, 
    'd_RE_R1b': {'name': 'Drug visit RE for R1b at 20min', 'unit': ''},
    'd_RE_R1l': {'name': 'Drug visit RE for R1l at 20min', 'unit': ''},
    'd_RE_Sb': {'name': 'Drug visit RE for Sb at 20min', 'unit': ''},
    'd_RE_Sl': {'name': 'Drug visit RE for Sl at 20min', 'unit': ''},
}


def _div(a, b):
    with np.errstate(divide='ignore'):
        return np.divide(a, b)
    
def _deriv_params(p, c_t, d_t, sdev=None):

    FFl = p[f'FFl']
    Fb = FFl * p[f'CO'] / p[f'c_vol']

    Fpl = Fb * (1 - p['H'])
    Te = p[f've'] / Fpl

    CL = p[f'GFR']
    Fpk = (1 - FFl) * p[f'CO'] * (1 - p['H'])
    Eg = CL / (CL + Fpk)

    # c_Eb
    khe_i, khe_f = p[f'c_khe_i'], p[f'c_khe_f']
    Eli = khe_i / (khe_i + Fpl)
    Elf = khe_f / (khe_f + Fpl)
    c_El = np.mean([Eli, Elf])
    khe = khe_i + (khe_f - khe_i) * c_t / c_t.max()
    CL = khe * p[f'c_vol'] + p['GFR']
    Eb = CL / (CL + p['CO'] * (1 - p['H']))
    c_Eb = np.mean(Eb)

    # d_Eb
    khe_i, khe_f = p[f'd_khe_i'], p[f'd_khe_f']
    Eli = khe_i / (khe_i + Fpl)
    Elf = khe_f / (khe_f + Fpl)
    d_El = np.mean([Eli, Elf])
    khe = khe_i + (khe_f - khe_i) * d_t / d_t.max()
    CL = khe * p[f'd_vol'] + p['GFR']
    Eb = CL / (CL + p['CO'] * (1 - p['H']))
    d_Eb = np.mean(Eb)
    
    vh = 1 - p['ve'] / (1 - p['H'])
    c_khe = np.mean([p['c_khe_i'], p['c_khe_f']])
    #c_kbh = vh * p['c_Kbh']
    c_kbh = p['c_kbh']

    d_khe = p['d_khe_i'] 
    #d_kbh = vh * p['d_Kbh']
    d_kbh = p['d_kbh']
    
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
        'c_khe': c_khe,
        'd_khe': d_khe,
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
        'c_dkhe': _div(p['c_khe_f'] - p['c_khe_i'], p['c_khe_i']),
        'd_dkhe': _div(p['d_khe_f'] - p['d_khe_i'], p['d_khe_i']),
        #'d_dkbh': _div(p['d_kbh_f'] - p['d_kbh_i'], p['d_kbh_i']),
    }
    sd_deriv = {}
    if sdev is not None:
        sd_deriv['c_khe'] = 0.5 * np.sqrt(sdev['c_khe_i'] ** 2 + sdev['c_khe_f'] ** 2)
        sd_deriv['d_khe'] = 0.5 * np.sqrt(sdev['d_khe_i'] ** 2 + sdev['d_khe_f'] ** 2)
    return p_deriv, sd_deriv


def _sample_signal(time, t, S, TS) -> tuple:
    return tuple([sample(ti, t, S, TS) for ti in time])
    # if isinstance(time, np.ndarray):
    #     return sample(time, t, S, TS)
    # else:
    #     return tuple([sample(ti, t, S, TS) for ti in time])

CONSTANTS = {'Fw': 0, 'v': 1, 'me': 1, 'noise_sdev':0}

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
    
    # ==========================================
    # User Interface
    # ==========================================

    configs = {'sequence': ['ZTE-3D-SPGR-SS', '3D-SPGR-SS', '3D-SPGR-SSI']}

    def __init__(self, sequence='ZTE-3D-SPGR-SS', **params):
        self._version = '1.0'
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(lexicon=QUANTITIES, **params)

    def export_params(self, sdev=None, desc=False) -> dict:
        """Parameters with values, definition and units"""
        self._set_time()
        tc, td = self._t_control, self._t_drug
        pars_deriv, sdev_deriv = _deriv_params(self._pars, tc, td, sdev)
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
            time = tuple([time[0], time[1], time[0], time[1], time[0], time[1], time[0], time[1]])
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
            bounds:dict=None, n0=[1, 1], n_runs=1, **kwargs,
        ):
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
                time['ctrl', 'aorta', 1], time['ctrl', 'aorta', 2], 
                time['ctrl', 'liver', 1], time['ctrl', 'liver', 2], 
                time['drug', 'aorta', 1], time['drug', 'aorta', 2], 
                time['drug', 'liver', 1], time['drug', 'liver', 2], 
            )
        else:
            time = tuple([time[0], time[1], time[0], time[1], time[0], time[1], time[0], time[1]])
        if isinstance(signal, dict):
            signal = (
                signal['ctrl', 'aorta', 1], signal['ctrl', 'aorta', 2], 
                signal['ctrl', 'liver', 1], signal['ctrl', 'liver', 2], 
                signal['drug', 'aorta', 1], signal['drug', 'aorta', 2], 
                signal['drug', 'liver', 1], signal['drug', 'liver', 2], 
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
                time['ctrl', 'aorta', 1], time['ctrl', 'aorta', 2], 
                time['ctrl', 'liver', 1], time['ctrl', 'liver', 2], 
                time['drug', 'aorta', 1], time['drug', 'aorta', 2], 
                time['drug', 'liver', 1], time['drug', 'liver', 2], 
            )
        else:
            time = tuple([time[0], time[1], time[0], time[1], time[0], time[1], time[0], time[1]])
        if isinstance(signal, dict):
            signal = (
                signal['ctrl', 'aorta', 1], signal['ctrl', 'aorta', 2], 
                signal['ctrl', 'liver', 1], signal['ctrl', 'liver', 2], 
                signal['drug', 'aorta', 1], signal['drug', 'aorta', 2], 
                signal['drug', 'liver', 1], signal['drug', 'liver', 2], 
            )
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[4 * i: 4 * i + 4]))
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
                time['ctrl', 'aorta', 1], time['ctrl', 'aorta', 2], 
                time['ctrl', 'liver', 1], time['ctrl', 'liver', 2], 
                time['drug', 'aorta', 1], time['drug', 'aorta', 2], 
                time['drug', 'liver', 1], time['drug', 'liver', 2], 
            )
        else:
            time = tuple([time[0], time[1], time[0], time[1], time[0], time[1], time[0], time[1]])
        if isinstance(signal, dict):
            signal = (
                signal['ctrl', 'aorta', 1], signal['ctrl', 'aorta', 2], 
                signal['ctrl', 'liver', 1], signal['ctrl', 'liver', 2], 
                signal['drug', 'aorta', 1], signal['drug', 'aorta', 2], 
                signal['drug', 'liver', 1], signal['drug', 'liver', 2], 
            )
        return self._cost(time, signal, metric, nfree)



    # ==========================================
    # Backend
    # ==========================================



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
                'H', 
                'FFl', 'CO', 'GFR', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                'c_khe_i', 'c_khe_f', 'c_vol',
                'd_khe_i', 'd_khe_f', 'd_vol', 
                'c_dose_1', 'c_BAT_1',  'c_dose_2', 'c_BAT_2',
                'd_dose_1', 'd_BAT_1',  'd_dose_2', 'd_BAT_2',
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
                'TR', 
                'c_FA_1', 'c_FA_2', 'd_FA_1', 'd_FA_2', 
                # Signal
                'c_t_scan2', 'c_Si_1_a', 'c_Si_2_a', 'c_Si_1_l', 'c_Si_2_l',
                'd_t_scan2', 'd_Si_1_a', 'd_Si_2_a', 'd_Si_1_l', 'd_Si_2_l',
                'c_B1corr_1_l', 'c_B1corr_1_a', 'c_B1corr_2_l', 'c_B1corr_2_a',
                'd_B1corr_1_l', 'd_B1corr_1_a', 'd_B1corr_2_l', 'd_B1corr_2_a',
                # Predict
                'TS',
            ],
            'free': inflow_pars + [
                # _conc_aorta 
                'c_BAT_1',  'c_BAT_2',
                'd_BAT_1',  'd_BAT_2',
                'FFl', 'CO', 'GFR', 'Thl', 'Dhl', 'To', 'To_e', 'Eo',
                # _conc_liver
                'Tg', 've', 
                'c_khe_i', 'c_khe_f', 'c_kbh', #'c_Kbh', 
                'd_khe_i', 'd_khe_f', 'd_kbh', #'d_Kbh',
            ],
        }
        return pars_list[select]
    
    
    # ==========================================
    # Forward Model: Helpers
    # ==========================================

    def _time(self, visit) -> np.ndarray:
        p = self._pars
        return np.arange(0, p[f'{visit}_tmax'], p['dt'])

    def _conc_aorta(self, visit, scans):
        p = self._pars
        t = self._time(visit)
        
        # Source
        conc = const.ca_conc(p['agent'])
        J = ca_injection(
            t, p['weight'], conc, p[f'{visit}_dose_1'], p['rate'], 
            p[f'{visit}_BAT_1'],
        )
        if scans==2:
            J += ca_injection(
                t, p['weight'], conc, p[f'{visit}_dose_2'], p['rate'], 
                p[f'{visit}_BAT_2'],
            )

        FFl = p[f'FFl']
        Fpl = FFl * p[f'CO'] * (1 - p['H']) / p[f'{visit}_vol']

        Eli = p[f'{visit}_khe_i'] / (p[f'{visit}_khe_i'] + Fpl)
        Elf = p[f'{visit}_khe_f'] / (p[f'{visit}_khe_f'] + Fpl)
        El = Eli + (Elf - Eli) * t / t.max()
        Te = p[f've'] / Fpl

        CL = p[f'GFR']
        Fpk = (1 - FFl) * p[f'CO'] * (1 - p['H'])
        Ek = CL / (CL + Fpk)

        Rl = FFl * (1 - El)
        Ro = (1 - FFl) * (1 - Ek)

        Jb = flux_aorta(
            J, dt=p['dt'], tol=p['dose_tolerance'],
            heartlung = {'model': 'pfcomp', 'params': {'T':p[f'Thl'], 'D':p[f'Dhl']}},
            organs = [
                # Liver
                {'vr': Rl, 'model': 'bicomp', 'params': {'T':[p[f'Tg'], Te]}},
                # Other organs
                {'vr': Ro, 'model': '2cxm', 'params': {'T':[p[f'To'], p[f'To_e']], 'E':p[f'Eo']}},
            ]
        )
        return Jb / p[f'CO']

    def _conc_liver(self, cb, visit, scans):
        p = self._pars
        
        # cb = flux_comp(cb, p[f'Tg'], dt=p['dt'])
        cp = cb / (1 - p['H'])

        vh = 1 - p[f've'] / (1 - p['H'])
        Th = vh / p[f'{visit}_kbh'] 

        FFl = p[f'FFl']
        Fpl = FFl * p[f'CO'] * (1 - p['H']) / p[f'{visit}_vol']
        Eli = p[f'{visit}_khe_i'] / (p[f'{visit}_khe_i'] + Fpl)
        Elf = p[f'{visit}_khe_f'] / (p[f'{visit}_khe_f'] + Fpl)

        return ConcLiver('1I-IC', 'U')(
            ci=cp, dt=p['dt'], 
            Ta = 0,
            Tg = p['Tg'],
            ve = p[f've'],
            Fp = Fpl,
            E_i = Eli,
            E_f = Elf,
            Th = Th,
        )

    def _relax_aorta(self, ca, visit):
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        R1a = p[f'R1b_a'] + rb * ca
        r2s = const.r2s(p['field_strength'], 'blood', p['agent'])
        R2sa = p[f'R2sb_a'] + r2s * ca
        return R1a, R2sa

    def _relax_liver(self, Cl, visit):
        p = self._pars
        rp = const.r1(p['field_strength'], 'plasma', p['agent'])
        rh = const.r1(p['field_strength'], 'hepatocytes', p['agent'])
        R1l = p[f'R1b_l'] + rp * Cl[0, :] + rh * Cl[1, :]
        r2s = const.r2s(p['field_strength'], 'tissue', p['agent'])
        R2sl = p[f'R2sb_l'] + r2s * Cl.sum(axis=0) 
        return R1l, R2sl


    def _signal(self, R1, R2s, visit, scans, roi):
        p = self._pars
        
        # Signal model
        seq = self._cnfg['sequence']
        roi_seq = {
            'a': seq,
            'l': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
        }[roi]
        
        def scan_signal(scan, R1_scan, R2s_scan):
            FA = p[f'{visit}_FA_{scan}']
            B1 = p[f'{visit}_B1corr_{scan}_{roi}']
            signal = Signal(roi_seq, defaults=p)
            S_ref = signal(R1=R1_scan[0], R2s=R2s_scan[0], S0=1, B1corr=B1, FA=FA, **CONSTANTS)
            S0 = p[f'{visit}_Si_{scan}_{roi}'] / S_ref if S_ref > 0 else 0
            return signal(R1=R1_scan, R2s=R2s_scan, S0=S0, B1corr=B1, FA=FA, **CONSTANTS)

        # if scans==1:
        #     S = scan_signal(1, R1, R2s)

        # TODO: This is the only option used so can remove the scans keyword.
        if scans==2:
            t = self._time(visit)
            S = np.zeros_like(t)

            # First scan signal
            ts = t < p[f'{visit}_t_scan2']
            S[ts] = scan_signal(1, R1[ts], R2s[ts])

            # Second scan signal
            ts = t >= p[f'{visit}_t_scan2']
            S[ts] = scan_signal(2, R1[ts], R2s[ts])

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
    
    def _predict_control(self, time):
        Sa = self._predict_aorta_control(time[:2])
        Sl = self._predict_liver_control(time[2:])
        return Sa + Sl
    
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

    def _estimate_bat(self, time, signal):
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            i0 = 4 * i
            p[f'{visit}_tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time[i0: i0 + 4]))

            # Estimate BAT
            t_hl, d_hl = p[f'Thl'], p[f'Dhl']
            bat1 = time[0 + i0][np.argmax(signal[0 + i0])] - (1 - d_hl) * t_hl
            bat2 = time[1 + i0][np.argmax(signal[1 + i0])] - (1 - d_hl) * t_hl
            p[f'{visit}_BAT_1'] = max(bat1, 0)
            p[f'{visit}_BAT_2'] = max(bat2, 0)

    def _estimate_baseline(self, signal, n0):
        p = self._pars
        for i, visit in enumerate(['c', 'd']):
            i0 = 4 * i
            p[f'{visit}_Si_1_a'] = np.mean(signal[0 + i0][:n0[i]]) 
            p[f'{visit}_Si_2_a'] = np.mean(signal[1 + i0][:n0[i]]) 
            p[f'{visit}_Si_1_l'] = np.mean(signal[2 + i0][:n0[i]])
            p[f'{visit}_Si_2_l'] = np.mean(signal[3 + i0][:n0[i]])   


    def _train(
            self, time: tuple, signal: tuple, free: dict, 
            bounds:dict, n0: list, n_runs: int, sigma: tuple=None, **kwargs,
        ):

        # Normalize signal
        scl = [1 / np.mean(si) if np.mean(si) != 0 else 1 for si in signal]
        signal = tuple([si * scl[i] for i, si in enumerate(signal)])

        p = self._pars
        self._estimate_bat(time, signal)
        self._estimate_baseline(signal, n0)
        free = self._set_free_pars(free, bounds, QUANTITIES)  

        # train(self._predict, time, signal, p, free, sigma=sigma, **kwargs)

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
                    if param_name not in ['c_BAT_1', 'c_BAT_2', 'd_BAT_1', 'd_BAT_2']:
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

        for i, visit in enumerate(['c', 'd']):
            p[f'{visit}_Si_1_a'] /= scl[0 + 4 * i]
            p[f'{visit}_Si_2_a'] /= scl[1 + 4 * i]
            p[f'{visit}_Si_1_l'] /= scl[2 + 4 * i]
            p[f'{visit}_Si_2_l'] /= scl[3 + 4 * i]

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
            ax.plot(np.concatenate(ti) / 60, np.concatenate(si), marker='o', color=color[0], label='fitted data', linestyle='None')
            ax.plot(t / 60, s, linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        ylim_a = [
            0.9 * min(self._Sa_control.min(), self._Sa_drug.min(), signal[0].min(), signal[1].min(), signal[4].min(), signal[5].min()), 
            1.1 * max(self._Sa_control.max(), self._Sa_drug.max(), signal[0].max(), signal[1].max(), signal[4].max(), signal[5].max())
        ]
        ylim_l = [
            0.9 * min(self._Sl_control.min(), self._Sl_drug.min(), signal[2].min(), signal[3].min(), signal[6].min(), signal[7].min()), 
            1.1 * max(self._Sl_control.max(), self._Sl_drug.max(), signal[2].max(), signal[3].max(), signal[6].max(), signal[7].max()),
        ]

        plot_data2scan(self._t_control, self._Sa_control, time[0:2], signal[0:2], ax1, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_control, self._Sl_control, time[2:4], signal[2:4], ax5, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(self._t_drug, self._Sa_drug, time[4:6], signal[4:6], ax2, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(self._t_drug, self._Sl_drug, time[6:8], signal[6:8], ax6, xlim, ylim_l, ['cornflowerblue', 'darkblue'])

        def plot_conc_aorta(t, c, ax, xl, yl):
            if xl is None: xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * c, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
            ax.plot(t / 60, 10 * 1000 * c, linestyle='--', color='darkred', linewidth=2.0, label='Aorta (x10)')
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

    def _conc_dict(self) -> dict:
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
    
    def _relax_dict(self) -> dict:
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
    
    def _signal_dict(self) -> dict:
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
    
    def _desc(self):
        t = self._time_dict() 
        C = self._conc_dict()
        R1, _ = self._relax_dict()
        S = self._signal_dict()

        pars = {}

        for visit in ['ctrl', 'drug']:

            # Compute AUC over 3hrs
            BAT = self._pars[f'{visit[0]}_BAT_1']

            tAUCb = (BAT < t[visit, 'aorta', 1]) & (t[visit, 'aorta', 1] < BAT + 180 * 60)
            tAUCl = (BAT < t[visit, 'liver', 1]) & (t[visit, 'liver', 1] < BAT + 180 * 60)
            AUC_Cb = np.trapezoid(C[visit, 'aorta', 1][tAUCb], t[visit, 'aorta', 1][tAUCb]) 
            AUC_Cl = np.trapezoid(C[visit, 'liver', 1].sum(axis=0)[tAUCl], t[visit, 'liver', 1][tAUCl])

            # Compute relative enhancement at 20mins
            tRE = BAT + 20*60
            R1b = R1[visit, 'aorta', 1]
            R1l = R1[visit, 'liver', 1]
            RE_R1b = (R1b[t[visit, 'aorta', 1] < tRE][-1] - R1b[0])/R1b[0]
            RE_R1l = (R1l[t[visit, 'liver', 1] < tRE][-1] - R1l[0])/R1l[0]

            S0b = np.mean(S[visit, 'aorta', 1][t[visit, 'aorta', 1]< BAT - 30])
            S0l = np.mean(S[visit, 'liver', 1][t[visit, 'liver', 1] < BAT - 30])
            RE_Sb = (S[visit, 'aorta', 1][t[visit, 'aorta', 1] < tRE][-1] - S0b)/S0b
            RE_Sl = (S[visit, 'liver', 1][t[visit, 'liver', 1] < tRE][-1] - S0l)/S0l

            # Compute AUC over 35min
            tAUCb = (BAT < t[visit, 'aorta', 1]) & (t[visit, 'aorta', 1] < BAT + 35 * 60)
            tAUCl = (BAT < t[visit, 'liver', 1]) & (t[visit, 'liver', 1] < BAT + 35 * 60)
            AUC35_Cb = np.trapezoid(C[visit, 'aorta', 1][tAUCb], t[visit, 'aorta', 1][tAUCb]) 
            AUC35_Cl = np.trapezoid(C[visit, 'liver', 1].sum(axis=0)[tAUCl], t[visit, 'liver', 1][tAUCl])

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




