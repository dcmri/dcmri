from types import MappingProxyType

import numpy as np





SEQUENCES = {
    'ZTE-3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1'],
            'read': ['S0', 'FA', 'B1corr', 'noise_sdev'],
        },
    },
    'ZTE-3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1'],
            'read': ['S0', 'FA', 'B1corr', 'noise_sdev'],
        },
    },
    '3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-SR-SPGR-SS': {
        'mz_prep_tissue': 'SR-SPGR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-PR-SPGR-SS': {
        'mz_prep_tissue': 'PR-SPGR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '2D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '2D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    }, # Non-selective preparation, so inflow is freely recovering
    '2D-SR-SPGR-SS': {
        'mz_prep_tissue': 'SR-SPGR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '2D-PR-SPGR-SS': {
        'mz_prep_tissue': 'PR-SPGR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-IR-SPGR': {
        'mz_prep_tissue': 'IR-SPGR',
        'mz_prep_inflow': 'IR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SPGR',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-PR-SPGR': {
        'mz_prep_tissue': 'PR-SPGR',
        'mz_prep_inflow': 'PR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-PR-SS': {
        'mz_prep_tissue': 'PR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-IR-SS': {
        'mz_prep_tissue': 'IR-SS',
        'mz_prep_inflow': 'IR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-SR-SS': {
        'mz_prep_tissue': 'SR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '2D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'Eq',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '2D-IR-SPGR': {
        'mz_prep_tissue': 'IR-SPGR', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '2D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '2D-PR-SPGR': {
        'mz_prep_tissue': 'PR-SPGR',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    '3D-SPGR-SSI': {
        'mz_prep_tissue': 'SSI',
        'mz_prep_inflow': 'SSI',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    'GE-EPI': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    'SE-EPI': {
        'mz_prep_tissue': 'SE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    'DE-EPI': {
        'mz_prep_tissue': 'DE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE1', 'TE2', 'noise_sdev'],
        },
    },
    'Eq-GE-EPI': {
        'mz_prep_tissue': 'Eq',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    'Eq-SE-EPI': {
        'mz_prep_tissue': 'Eq',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R2'],
            'read': ['S0', 'FA', 'B1corr', 'TE', 'noise_sdev'],
        },
    },
    'Eq-DE-EPI': {
        'mz_prep_tissue': 'Eq',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R2', 'R2s'],
            'read': ['S0', 'FA', 'B1corr', 'TE1', 'TE2', 'noise_sdev'],
        },
    },
}


MZ_PREP_PARANS = { 
    'Eq': [],
    'IR-SS': ['TA'],
    'SR-SS': ['TA'],
    'PR-SS': ['TA', 'PA'],
    'SPGR': ['TC', 'TR', 'FA', 'B1corr', 'TA'],
    'SR-SPGR': ['TC', 'TR', 'FA', 'B1corr', 'TP', 'TA'],
    'IR-SPGR': ['TC', 'TR', 'FA', 'B1corr', 'TP', 'TA'],
    'PR-SPGR': ['TC', 'TR', 'FA', 'B1corr', 'TP', 'TA', 'PA'],
    'SPGR-SS': ['TR', 'FA', 'B1corr'],
    'SR-SPGR-SS': ['TC', 'TR', 'FA', 'B1corr', 'TP', 'TA'],
    'IR-SPGR-SS': ['TC', 'TR', 'FA', 'B1corr', 'TP', 'TA'],
    'PR-SPGR-SS': ['TC', 'TR', 'FA', 'B1corr', 'TP', 'TA', 'PA'],
    'SSI': ['TR', 'FA', 'B1corr', 'TF', 'SA'],
    'SE-SS': ['TE', 'TR', 'FA', 'B1corr'],
    'DE-SS': ['TE2', 'TR', 'FA', 'B1corr'],
}

# Add preparation module parameters
for seq, props in SEQUENCES.items():
    pars_tissue = MZ_PREP_PARANS[props['mz_prep_tissue']]
    pars_inflow = MZ_PREP_PARANS[props['mz_prep_inflow']]
    pars = list(set(pars_tissue + pars_inflow))
    pars.sort()
    props['parameters']['prep'] = pars


# ---- Initial Values ----
dt_init, tmax_init = 0.5, 240
t_init = np.arange(0, tmax_init, dt_init, dtype=float)
ca_init = 0.005 * np.exp(-t_init / 60)
ca_init = np.interp(t_init-30, t_init, ca_init, left=0)
cv_init = np.interp(t_init-10, t_init, ca_init, left=0)


QUANTITIES = MappingProxyType(  # This makes the dict immutable
    {
        
    # ---- Simulation parameters ---
    'dose_tolerance': {'init': 0.1, 'name': 'Dose tolerance', 'unit': ''},
    'tmax': {'init': tmax_init, 'name': 'Max time', 'unit': 's'},

    # --- Injection & Contrast Agent ---
    'r1': {'init': 3500, 'bounds': [0, 1e4], 'name': 'Longitudinal contrast agent relaxivity', 'unit': 'Hz/M'},
    'r2': {'init': 4000, 'bounds': [0, 1e4], 'name': 'Transverse contrast agent relaxivity', 'unit': 'Hz/M'},
    'r2s': {'init': 20000, 'bounds': [0, 1e5], 'name': 'Transverse contrast agent relaxivity', 'unit': 'Hz/M'},
    'r2s_quad': {'init': 1000, 'bounds': [0, 1e4], 'name': 'Quadratic transverse contrast agent relaxivity', 'unit': 'Hz/M^2'},
    'r2s_vasc': {'init': 20000, 'bounds': [0, 1e5], 'name': 'Vacular transverse contrast agent relaxivity', 'unit': 'Hz/M'},
    'r2s_ees': {'init': 20000, 'bounds': [0, 1e5], 'name': 'Extravascular, extracellular transverse contrast agent relaxivity', 'unit': 'Hz/M'},
    'agent': {'init': 'gadoterate', 'name': 'Contrast agent', 'unit': None},
    'weight': {'init': 70, 'bounds': [0, 300], 'name': 'Weight', 'unit': 'kg'},
    'dose': {'init': 0.1, 'bounds': [0, 0.2], 'name': 'Dose', 'unit': 'mL/kg'},
    'dose2': {'init': 0.05, 'bounds': [0, 0.2], 'name': 'Second contrast agent dose', 'unit': 'mL/kg'},
    'rate': {'init': 1, 'bounds': [0, 10], 'name': 'Injection rate', 'unit': 'mL/s'},
    'BAT': {'init': 60, 'bounds': [-30, 30], 'name': 'Bolus arrival time', 'unit': 's', 'bounds_type': 'add'},
    'BAT2': {'init': 120 + 60, 'bounds': [-60.0, 60.0], 'name': 'Second bolus arrival time', 'unit': 'sec', 'bounds_type': 'add'},

    # --- Input Function ---
    'c_a': {'init': ca_init, 'name': 'Arterial blood concentration', 'unit': 'M'},
    'c_v': {'init': cv_init, 'name': 'Venous blood concentration', 'unit': 'M'},
    'dt': {'init': dt_init, 'name': 'Forward model time step', 'unit': 'sec'},
    'irf': {'init': 0.02 * np.ones_like(t_init), 'name': 'Impulse response function', 'unit': 'mL/sec/cm3'},

    # --- Experimental Setup ---
    'field_strength': {'init': 3, 'name': 'Magnetic field strength', 'unit': 'T'},
    't_scan2': {'init': 120, 'name': 'Start of second scan', 'unit': 'sec'},
    'noise_sdev': {'init': 0.0, 'name': 'Standard deviation of the signal noise', 'unit': 'a.u.'},

    # --- MRI Sequence & Signal Parameters ---
    'B1corr': {'init': 1, 'bounds': [0, 5], 'name': 'Tissue B1-correction factor', 'unit': ''},
    'B1corr_a': {'init': 1, 'bounds': [0, 5], 'name': 'Arterial B1-correction factor', 'unit': ''},
    'B1corr_v': {'init': 1, 'bounds': [0, 5], 'name': 'Venous B1-correction factor', 'unit': ''},
    'B1corr_l': {'init': 1, 'bounds': [0, 5], 'name': 'Liver B1-correction factor', 'unit': ''},
    'B1corr_lk': {'init': 1, 'bounds': [0, 5], 'name': 'Left kidney B1-correction factor', 'unit': ''},
    'B1corr_rk': {'init': 1, 'bounds': [0, 5], 'name': 'Right kidney B1-correction factor', 'unit': ''},
    'B1corr_2': {'init': 1, 'bounds': [0, 5], 'name': 'Tissue B1-correction factor of a second scan', 'unit': ''},
    'B1corr_2_a': {'init': 1, 'bounds': [0, 5], 'name': 'Arterial B1-correction factor of a second scan', 'unit': ''},
    'B1corr_2_l': {'init': 1, 'bounds': [0, 5], 'name': 'Liver B1-correction factor of a second scan', 'unit': ''},
    'FAcorr': {'name': 'B1-corrected Flip Angle', 'unit': 'deg'},
    'SA': {'init': 0, 'bounds': [0, 180], 'name': 'Saturation Slab Flip Angle', 'unit': 'deg'},
    'PA': {'init': 90, 'bounds': [0, 180], 'name': 'Preparation Pulse Flip Angle', 'unit': 'deg'},
    'FA': {'init': 15, 'bounds': [0, 180], 'name': 'Flip angle', 'unit': 'deg'},
    'FAR': {'init': 15, 'bounds': [0, 180], 'name': 'Readout flip angle', 'unit': 'deg'},
    'FA2': {'init': 15.0, 'bounds': [0.0, 180], 'name': 'Second flip angle', 'unit': 'deg'},
    'FA_2': {'init': 15.0, 'bounds': [0.0, 180], 'name': 'Second flip angle', 'unit': 'deg'},
    'TR': {'init': 0.005, 'name': 'Repetition time', 'unit': 'sec'},
    'TC': {'init': 0.2, 'name': 'Time to k-space center', 'unit': 'sec'},
    'TP': {'init': 0.05, 'name': 'Preparation delay', 'unit': 'sec'},
    'TE': {'init': 0.005, 'name': 'Echo time', 'unit': 'sec'},
    'TE1': {'init': 0.005, 'name': 'First echo time in a multi-echo sequence', 'unit': 'sec'},
    'TE2': {'init': 0.005, 'name': 'Second echo time in a multi-echo sequence', 'unit': 'sec'},
    'TA': {'init': 2.0, 'name': 'Acquisition time', 'unit': 'sec'},
    'TS': {'init': 0, 'name': 'Sampling time', 'unit': 'sec'},
    'n_init': {'init': 1, 'name': 'Initial relative magnetization', 'unit': ''},
    'n0': {'init': 1, 'name': 'Number of baseline dynamics', 'unit': ''},

    # --- Magnetization and flow ---
    'TF': {'init': 0.5, 'bounds': [0, 10], 'name': 'Inflow time', 'unit': 's'},
    'Fi': {'init': 0.02, 'bounds': [0, 1], 'name': 'Inflow', 'unit': 'mL/sec/cm3'},
    'R1i': {'init': 1/1.5, 'bounds': [0, 5], 'name': 'Inflow R1', 'unit': 'Hz'},
    'me': {'init': 1, 'bounds': [0, 5], 'name': 'Equilibrium magnetization', 'unit': 'A/m'},
    'v': {'init': 1, 'bounds': [0, 1], 'name': 'Water volume fraction', 'unit': 'mL/cm3'},
    'Fw': {'init': 0, 'bounds': [0, 1], 'name': 'Water exchange matrix', 'unit': 'mL/sec/cm3'},
    'Mz': {'init': 1, 'bounds': [0, 5], 'name': 'Longitudinal magnetization', 'unit': 'A/m'},

    # --- Relaxation ---
    'R1': {'init': 0.65, 'bounds': [0, 5], 'name': 'Tissue R1', 'unit': 'Hz'},
    'R2': {'init': 0.0, 'bounds': [0, 5], 'name': 'Tissue R2', 'unit': 'Hz'}, 
    'R2s': {'init': 0.0, 'bounds': [0, 5], 'name': 'Tissue R2*', 'unit': 'Hz'},
    'R10': {'init': 0.65, 'bounds': [0, 5], 'name': 'Tissue precontrast R1', 'unit': 'Hz'},
    'R10_c': {'init': 0.65, 'bounds': [0, 5], 'name': 'Cortex precontrast R1', 'unit': 'Hz'},
    'R10_m': {'init': 0.65, 'bounds': [0, 5], 'name': 'Medulla precontrast R1', 'unit': 'Hz'},
    'R10_a': {'init': 0.65, 'bounds': [0, 5], 'name': 'Arterial precontrast R1', 'unit': 'Hz'},
    'R10_l': {'init': 0.65, 'bounds': [0, 5], 'name': 'Liver precontrast R1', 'unit': 'Hz'},
    'R10_v': {'init': 1/0.8, 'bounds': [0, 5], 'name': 'Portal baseline R1', 'unit': 'Hz'},
    'R10_lk': {'init': 1/1.5, 'bounds': [0, 5], 'name': 'Left kidney tissue precontrast R1', 'unit': 'Hz'},
    'R10_rk': {'init': 1/1.5, 'bounds': [0, 5], 'name': 'Right kidney tissue precontrast R1', 'unit': 'Hz'},
    'R20': {'init': 15, 'bounds': [0, 100], 'name': 'Tissue precontrast R2', 'unit': 'Hz'}, # Initialized to zero as R20/R20s factors typically absorbed in scaling factor S0
    'R20s': {'init': 25, 'bounds': [0, 100], 'name': 'Tissue precontrast R2*', 'unit': 'Hz'}, # Idem
    'R20s_c': {'init': 20, 'bounds': [0, 100], 'name': 'Cortex precontrast R2*', 'unit': 'Hz'},
    'R20s_m': {'init': 20, 'bounds': [0, 100], 'name': 'Medulla precontrast R2*', 'unit': 'Hz'},
    'R20s_a': {'init': 20, 'bounds': [0, 100], 'name': 'Arterial precontrast R2*', 'unit': 'Hz'}, # Idem
    'R20s_l': {'init': 20, 'bounds': [0, 100], 'name': 'Liver precontrast R2*', 'unit': 'Hz'},
    'R20s_v': {'init': 20, 'bounds': [0, 100], 'name': 'Portal precontrast R2*', 'unit': 'Hz'},
    'R20s_lk': {'init': 20, 'bounds': [0, 100], 'name': 'Left kidney precontrast R2*', 'unit': 'Hz'},
    'R20s_rk': {'init': 20, 'bounds': [0, 100], 'name': 'Right kidney precontrast R2*', 'unit': 'Hz'},
    # --- Scaling ---
    'S0': {'init': 1.0, 'bounds': [0, 5], 'name': 'Signal scaling factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'S0_c': {'init': 1.0, 'bounds': [0, 5], 'name': 'Cortex signal scaling factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'S0_m': {'init': 1.0, 'bounds': [0, 5], 'name': 'Medulla signal scaling factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'S0_a': {'init': 1.0, 'bounds': [0, 5], 'name': 'Arterial signal scaling factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'S0_l': {'init': 1.0, 'bounds': [0, 5], 'name': 'Liver signal scaling factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'S0_v': {'init': 1.0, 'bounds': [0, 5], 'name': 'Portal venous signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'S0_lk': {'init': 1.0, 'bounds': [0, 5], 'name': 'Left kidney signal scaling factor', 'unit': 'a.u.'},
    'S0_rk': {'init': 1.0, 'bounds': [0, 5], 'name': 'Right kidney signal scaling factor', 'unit': 'a.u.'},

    # Water kinetics
    'PSe': {'init': 0.03, 'bounds': [0, 100], 'name': 'Transendothelial water PS', 'unit': 'mL/sec/cm3'},
    'PSc': {'init': 0.03, 'bounds': [0, 100], 'name': 'Transcytolemmal water PS', 'unit': 'mL/sec/cm3'},
    'Twc': {'name': 'Intracellular water mean transit time', 'unit': 'sec'},
    'Twi': {'name': 'Interstitial water mean transit time', 'unit': 'sec'},
    'Twb': {'name': 'Intravascular water mean transit time', 'unit': 'sec'},

    # --- Blood Kinetics ---
    'H': {'init': 0.45, 'name': 'Tissue Hematocrit', 'unit': ''},
    'T_a': {'init': 0, 'bounds': [0, 10], 'name': 'Arterial delay', 'unit': 'sec'},
    'CO': {'init': 100, 'bounds': [0, 500], 'name': 'Cardiac output', 'unit': 'mL/s'},
    'Thl': {'init': 10, 'bounds': [0, 30], 'name': 'Heart-lung MTT', 'unit': 's'},
    'Dhl': {'init': 0.2, 'bounds': [0.01, 0.99], 'name': 'Heart-lung dispersion', 'unit': ''},
    'To': {'init': 20, 'bounds': [0, 60], 'name': 'Organ blood MTT', 'unit': 's'},
    'Eo': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Organ extraction', 'unit': ''},
    'To_e': {'init': 120, 'bounds': [0, 800], 'name': 'Organ EES MTT', 'unit': 's'},
    'Eb': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Body extraction', 'unit': ''},

    # --- Tissue Kinetics ---
    'Fb': {'init': 0.02, 'bounds': [0, 1], 'name': 'Blood flow', 'unit': 'mL/sec/cm3'},
    'PS': {'init': 0.003, 'bounds': [0, 1], 'name': 'Permeability-surface area product', 'unit': 'mL/sec/cm3'},
    'vi': {'init': 0.3, 'bounds': [1e-3, 1 - 1e-3], 'name': 'Interstitial volume', 'unit': 'mL/cm3'},
    'vb': {'init': 0.1, 'bounds': [1e-3, 1 - 1e-3], 'name': 'Blood volume', 'unit': 'mL/cm3'},
    'vc': {'init': 0.6, 'bounds': [1e-3, 1 - 1e-3], 'name': 'Intracellular volume', 'unit': 'mL/cm3'},
    'Fp': {'init': 0.02, 'bounds': [0, 0.05], 'name': 'Plasma flow', 'unit': 'mL/sec/cm3'},
    'vp': {'init': 0.15, 'bounds': [0, 0.3], 'name': 'Plasma volume', 'unit': 'mL/cm3'},
    'Te': {'init': 30.0, 'bounds': [0.1, 60], 'name': 'Extracellular mean transit time', 'unit': 'sec'},
    'De': {'init': 0.85, 'bounds': [0, 1], 'name': 'Extracellular dispersion', 'unit': ''},
    've': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Extracellular volume fraction', 'unit': 'mL/cm3'},
    've_app': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Apparent extracellular volume fraction', 'unit': 'mL/cm3'},
    'Ktrans': {'init': 0.015, 'bounds': [0.0, 0.1], 'name': 'Plasma clearance', 'unit': 'mL/sec/cm3'},
    'Ktrans_i': {'init': 0.015, 'bounds': [0.0, 0.1], 'name': 'Initial plasma clearance', 'unit': 'mL/sec/cm3'},
    'Ktrans_f': {'init': 0.015, 'bounds': [0.0, 0.1], 'name': 'Final plasma clearance', 'unit': 'mL/sec/cm3'},
    'E': {'init': 0.1, 'bounds': [0.0, 1.0], 'name': 'Extraction fraction', 'unit': ''},
    'E_i': {'init': 0.1, 'bounds': [0.0, 1.0], 'name': 'Initial extraction fraction', 'unit': ''},
    'E_f': {'init': 0.1, 'bounds': [0.0, 1.0], 'name': 'Final extraction fraction', 'unit': ''},
    'El': {'init': 0.05, 'bounds': [0, 1], 'name': 'Liver extraction fraction', 'unit': ''},

    'Ti': {'name': 'Interstitial mean transit time', 'unit': 'sec'},
    'Tp': {'init': 5, 'bounds': [0, 30], 'name': 'Plasma mean transit time', 'unit': 'sec'},
    'Tb': {'init': 5, 'bounds': [0, 30], 'name': 'Blood mean transit time', 'unit': 'sec'},
    
    # --- Kidney Kinetics ---
    'FF': {'init': 0.1, 'bounds': [0, 0.3], 'name': 'Filtration fraction', 'unit': ''},
    'Tt': {'init': 120, 'bounds': [0, 60 * 60], 'name': 'Tubular mean transit time', 'unit': 'sec'},
    'Ft': {'init': 0.005, 'bounds': [0, 0.05], 'name': 'Tubular flow', 'unit': 'mL/sec/cm3'},
    'Eg': {'init': 0.15, 'bounds': [0, 1], 'name': 'Glomerular extraction fraction', 'unit': ''},
    'fc': {'init': 0.8, 'bounds': [0, 1], 'name': 'Cortical flow fraction', 'unit': ''},
    'Tglom': {'init': 4, 'bounds': [0, 30], 'name': 'Glomerular mean transit time', 'unit': 'sec'},
    'Tv': {'init': 10, 'bounds': [0, 30], 'name': 'Peritubular & venous mean transit time', 'unit': 'sec'},
    'Tpt': {'init': 60, 'bounds': [0, 180], 'name': 'Proximal tubuli mean transit time', 'unit': 'sec'},
    'Tlh': {'init': 60, 'bounds': [0, 180], 'name': 'Lis of Henle mean transit time', 'unit': 'sec'},
    'Tdt': {'init': 30, 'bounds': [0, 180], 'name': 'Distal tubuli mean transit time', 'unit': 'sec'},
    'Tcd': {'init': 30, 'bounds': [0, 180], 'name': 'Collecting duct mean transit time', 'unit': 'sec'},
    'GFR': {'init': 2, 'bounds': [0, 10], 'name': 'Glomerular filtration rate', 'unit': 'mL/sec'},
    'CBF': {'init': 0.04, 'bounds': [0, 0.1], 'name': 'Cortical blood flow', 'unit': 'mL/sec/cm3'},
    'MBF': {'init': 0.004, 'bounds': [0, 0.1], 'name': 'Medullary blood flow', 'unit': 'mL/sec/cm3'},
    'RPF': {'init': 20, 'bounds': [0, 100], 'name': 'Renal plasma flow', 'unit': 'mL/sec'},
    'DRPF': {'init': 0.5, 'bounds': [0, 1], 'name': 'Differential renal plasma flow', 'unit': ''},
    'DRF': {'init': 0.5, 'bounds': [0, 1.0], 'name': 'Differential renal function', 'unit': ''},

    # --- Single Kidney Kinetics ---
    'SKGFR': {'init': 2, 'bounds': [0, 10], 'name': 'Single-kidney glomerular filtration rate', 'unit': 'mL/sec'},
    'SKBF': {'init': 20, 'bounds': [0, 100], 'name': 'Single-kidney blood flow', 'unit': 'mL/sec'},
    'SKMBF': {'init': 2, 'bounds': [0, 10], 'name': 'Single-kidney medullary blood flow', 'unit': 'mL/sec'},
    # Left kidney
    'T_a_lk': {'init': 0, 'bounds': [0, 3], 'name': 'Left kidney arterial mean transit time', 'unit': 'sec'},
    'vp_lk': {'init': 0.15, 'bounds': [0, 0.3], 'name': 'Left kidney plasma volume', 'unit': 'mL/cm3'},
    'Tt_lk': {'init': 120, 'bounds': [0, 600], 'name': 'Left kidney tubular mean transit time', 'unit': 'sec'},
    'RPF_lk': {'name': 'Left kidney plasma flow', 'unit': 'mL/sec'},
    'GFR_lk': {'name': 'Left kidney glomerular filtration rate', 'unit': 'mL/sec'},
    'Fp_lk': {'name': 'Left kidney plasma flow', 'unit': 'mL/sec/cm3'},
    'Tp_lk': {'name': 'Left kidney plasma mean transit time', 'unit': 'sec'},
    'Tv_lk': {'name': 'Left kidney vascular mean transit time', 'unit': 'sec'},
    'Ft_lk': {'name': 'Left kidney tubular flow', 'unit': 'mL/sec/cm3'},
    'FF_lk': {'name': 'Left kidney filtration fraction', 'unit': ''},
    'E_lk': {'name': 'Left kidney extraction fraction', 'unit': ''},
    # Right kidney
    'T_a_rk': {'init': 0, 'bounds': [0, 3], 'name': 'Right kidney arterial mean transit time', 'unit': 'sec'},
    'vp_rk': {'init': 0.15, 'bounds': [0, 0.3], 'name': 'Right kidney plasma volume', 'unit': 'mL/cm3'},
    'Tt_rk': {'init': 120, 'bounds': [0, 600], 'name': 'Right kidney tubular mean transit time', 'unit': 'sec'},
    'RPF_rk': {'name': 'Right kidney plasma flow', 'unit': 'mL/sec'},
    'GFR_rk': {'name': 'Right kidney glomerular filtration rate', 'unit': 'mL/sec'},
    'Fp_rk': {'name': 'Right kidney plasma flow', 'unit': 'mL/sec/cm3'},
    'Tp_rk': {'name': 'Right kidney plasma mean transit time', 'unit': 'sec'},
    'Tv_rk': {'name': 'Right kidney vascular mean transit time', 'unit': 'sec'},
    'Ft_rk': {'name': 'Right kidney tubular flow', 'unit': 'mL/sec/cm3'},
    'FF_rk': {'name': 'Right kidney filtration fraction', 'unit': ''},
    'E_rk': {'name': 'Right kidney extraction fraction', 'unit': ''},

    # --- Liver Kinetics ---
    'vh': {'init': 0.6, 'bounds': [0.1, 1.0], 'name': 'Hepatocellular volume fraction', 'unit': 'mL/cm3'},
    'fa': {'init': 0.2, 'bounds': [0, 1], 'name': 'Arterial flow fraction', 'unit': ''},
    'khe': {'init': 0.003, 'bounds': [0.0, 0.1], 'name': 'Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'Dkhe': {'init': 0.000, 'bounds': [-1.5e-7, +1.5e-7], 'name': 'Rate of change in hepatocellular uptake rate', 'unit': 'mL/sec/cm3/sec'},
    'khe_i': {'init': 0.002, 'bounds': [0.0, 0.1], 'name': 'Initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'khe_f': {'init': 0.002, 'bounds': [0.0, 0.1], 'name': 'Final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'kbh': {'init': 0.0004, 'bounds': [0.0, 0.001], 'name': 'Biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'kbh_i': {'init': 0.0004, 'bounds': [0.0, 0.001], 'name': 'Initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'kbh_f': {'init': 0.0004, 'bounds': [0.0, 0.001], 'name': 'Final biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'Th': {'init': 1800, 'bounds': [600, 36000], 'name': 'Hepatocellular mean transit time', 'unit': 'sec'},
    'DTh': {'init': 0, 'bounds': [-0.25, 0.25], 'name': 'Rate of change in hepatocellular mean transit time', 'unit': ''},
    'Th_i': {'init': 1800, 'bounds': [600, 36000], 'name': 'Initial hepatocellular mean transit time', 'unit': 'sec'},
    'Th_f': {'init': 1800, 'bounds': [600, 36000], 'name': 'Final hepatocellular mean transit time', 'unit': 'sec'},
    'Kbh': {'init': 0.0001, 'bounds': [0.0, 0.001], 'name': 'Biliary tissue excretion rate', 'unit': '/sec'},
    'Kbh_i': {'init': 0.0001, 'bounds': [0.0, 0.001], 'name': 'Initial biliary tissue excretion rate', 'unit': '/sec'},
    'Kbh_f': {'init': 0.0001, 'bounds': [0.0, 0.001], 'name': 'Final biliary tissue excretion rate', 'unit': '/sec'},
    'Fa': {'name': 'Arterial plasma flow', 'unit': 'mL/sec/cm3'},
    'Fv': {'name': 'Venous plasma flow', 'unit': 'mL/sec/cm3'},
    'Khe': {'name': 'Hepatocellular tissue uptake rate', 'unit': '/sec'},
    'CL': {'name': 'Liver plasma clearance', 'unit': 'mL/sec'},

    # Gut and portal vein
    'Tg': {'init': 30, 'bounds': [0.1, 60], 'name': 'Gut mean transit time', 'unit': 'sec'},
    'Dg': {'init': 0.85, 'bounds': [0, 1], 'name': 'Gut dispersion', 'unit': ''},
    'uv': {'init': 1.0,  'bounds': [0.0, 1.0], 'name': 'Portal vein volume fraction', 'unit': ''},
 
    # Organ volumes
    'vol_k': {'init': 150, 'name': 'Single-kidney volume', 'unit': 'cm3'},
    'vol_l': {'init': 1000, 'bounds': [0, 10000], 'name': 'Liver volume', 'unit': 'cm3'},
    'vol_lk': {'init': 150, 'name': 'Left kidney volume', 'unit': 'mL'},
    'vol_rk': {'init': 150, 'name': 'Right kidney volume', 'unit': 'mL'},

    }
)