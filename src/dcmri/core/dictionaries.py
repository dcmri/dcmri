# -----------------------------------------------------------------------------
# Contrast agent generic names
# -----------------------------------------------------------------------------

#: List of approved contrast agents
AGENTS = [
    'gadoxetate',
    'gadobutrol',
    'gadopentetate',
    'gadobenate',
    'gadodiamide',
    'gadoterate',
    'gadoteridol',
    'gadopiclenol',
    'gadobenade',
    'mangafodipir',
    'gadoversetamide',
    'ferucarbotran',
    'ferumoxide',
    'gadoxetate',
]

#: List of quantity groups
GROUPS = {
    'indicator',
    'signal',
    'sequence',
    'electromagnetic',
    'physiological',
    'hyperparameters',
    'whole-body',
    'inverse problem',
}

#: Notations for tissue ROIs
ROIS = {
    'ab': 'arterial blood',
    'ar': 'artery',
    'ao': 'aorta',
    'b': 'blood', # can be both ROI and compartment
    'bm': 'bone marrow',
    'ca': 'cartilage',
    'csf': 'cerebrospinal fluid',
    'gm': 'grey matter',
    'gu': 'gut',
    'he': 'heart',
    'hl': 'heart and Lungs',
    'kc': 'kidney cortex',
    'ki': 'kidney',
    'km': 'kidney medulla',
    'la': 'liver artery',
    'lag': 'liver artery and gut',
    'li': 'liver',
    'lk': 'left kidney',
    'lka': 'left kidney artery',
    'mu': 'muscle',
    'on': 'optic nerve',
    'or': 'organs',
    'pv': 'portal vein',
    'rk': 'right kidney',
    'rka': 'right kidney artery',
    'sc': 'spinal cord',
    'sk': 'skin',
    'ti': 'tissue',
    'vb': 'venous blood',
    'vc': 'vena cava',
    've': 'vein', 
    'wm': 'white matter',
}

#Note: No shared keys between ROIs and comps except when the same value (eg. blood)

#: Notations for subvoxel compartments
COMPS = { 
    'b': 'blood',
    'bc': 'blood and cells',
    'bi': 'blood and interstitium',
    'bl': 'bile',
    'bu': 'blood and tubuli',
    'c': 'cells',
    'cd': 'collecting ducts',
    'dt': 'distal tubuli',
    'e': 'extracellular space',
    'gc': 'glomerular capillaries',
    'h': 'hepatocytes',
    'i': 'interstitium',
    'ic': 'interstitium and cells',
    'lh': 'lis-of-Henle',
    'p': 'plasma',
    'pcv': 'peritubular capillaries and veins',
    'pt': 'proximal tubuli',
    't': 'tissue',
    'u': 'tubuli',
    'uc': 'tubuli and cells',
    'vb': 'venous blood',
}

#: Dictionary of quantities
QUANTITIES = {

    # Generic hyperparameters
    'dt': {'init': 0.5, 'bounds': None, 'name': 'pseudo-continuous time step', 'unit': 'sec', 'group': 'hyperparameters', 'dicom_key': None, 'osipi_key': None},
    'tmax': {'init': 240, 'bounds': None, 'name': 'maximum time point', 'unit': 'sec', 'group': 'hyperparameters', 'dicom_key': None, 'osipi_key': None},

    # Generic indicator quantities
    'agent': {'init': 'gadoterate', 'bounds': None, 'name': 'contrast agent generic name', 'unit': None, 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'BAT': {'init': 30, 'bounds': (-30, 30), 'name': 'bolus arrival time', 'unit': 'sec', 'bounds_type': 'add', 'group': 'indicator', 'dicom_key': None, 'osipi_key': 'Q.BA1.001'},
    'bdel': {'init': 30, 'bounds': (-30, 30), 'name': 'delay in a double injection', 'unit': 'sec', 'bounds_type': 'add', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'dose': {'init': 0.1, 'bounds': (0, 0.2), 'name': 'contrast agent dose', 'unit': 'mL/kg', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'rate': {'init': 1, 'bounds': (0, 10), 'name': 'injection rate', 'unit': 'mL/sec', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'duration': {'init': 5, 'bounds': (0, 120), 'name': 'injection duration', 'unit': 'sec', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'Tb': {'init': 30, 'bounds': (0.0, 60), 'name': 'bolus mean transit time', 'unit': 'sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Db': {'init': 0.2, 'bounds': (0, 1), 'name': 'bolus transit time dispersion', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    'tC': {'init': 0.0, 'bounds': None, 'name': 'concentration time points', 'unit': 'sec', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'J': {'init': 1, 'bounds': (0, 10), 'name': 'indicator flux', 'unit': 'mmol/sec', 'bounds_type': 'abs', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'Jinj': {'init': 1, 'bounds': (0, 10), 'name': 'indicator flux at the injection site', 'unit': 'mmol/sec', 'bounds_type': 'abs', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'c': {'init': 0.005, 'bounds': (0, 1), 'name': 'concentration', 'unit': 'mmol/mL', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'ci': {'init': 0.005, 'bounds': None, 'name': 'inlet concentration', 'unit': 'mmol/mL', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'C': {'init': 0.005, 'bounds':(0, 1), 'name': 'tissue concentration', 'unit': 'mmol/cm3', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'AUCC': {'init': 1, 'bounds': (0, 1e12), 'name': 'area under the concentration time curve', 'unit': 'M*sec', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},

    # Whole body properties
    'weight': {'init': 70, 'bounds': (0, 300), 'name': 'body weight', 'unit': 'kg', 'group': 'whole-body', 'dicom_key': 'PatientWeight', 'osipi_key': None},
    'vol': {'init': 150, 'bounds': (0.0, 1000), 'name': 'ROI volume', 'unit': 'cm3', 'group': 'whole-body', 'dicom_key': None, 'osipi_key': None},

    # ---- Kinetic hyperparameters ---
    'dose_tolerance': {'init': 0.1, 'bounds': None, 'name': 'dose tolerance', 'unit': '', 'group': 'hyperparameters', 'dicom_key': None, 'osipi_key': None},

    # Generic kinetics
    'F': {'init': 0.02, 'bounds': (0, 0.1), 'name': 'flow per unit tissue', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'T': {'init': 30, 'bounds': (0.0, 60), 'name': 'mean transit time', 'unit': 'sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': 'Q.PH1.006'},
    'Ti': {'init': 30, 'bounds': (0.0, 60), 'name': 'initial mean transit time', 'unit': 'sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Tf': {'init': 30, 'bounds': (0.0, 60), 'name': 'final mean transit time', 'unit': 'sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'D': {'init': 0.85, 'bounds': (0, 1), 'name': 'transit time dispersion', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'h': {'init': 1.0, 'bounds': (0.1, 60), 'name': 'transit time distribution', 'unit': 'Hz', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'TT': {'init': 1.0, 'bounds': (0.1, 60), 'name': 'boundaries of transit time histogram bins', 'unit': 'sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'E': {'init': 0.1, 'bounds': (0.0, 1.0), 'name': 'extraction fraction', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': 'Q.PH1.005'},
    'Ei': {'init': 0.1, 'bounds': (0.0, 1.0), 'name': 'initial extraction fraction', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Ef': {'init': 0.1, 'bounds': (0.0, 1.0), 'name': 'final extraction fraction', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Vmax': {'init': 1, 'bounds': (0.0, 10), 'name': 'limiting rate in a Michaelis-Menten compartment', 'unit': 'mmol/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Km': {'init': 1, 'bounds': (0.0, 10), 'name': 'Michaelis-Menten constant', 'unit': 'mmol/mL', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'ffp': {'init': 0.2, 'bounds': (0, 1), 'name': 'plug-flow fraction in a plucom system', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'vr': {'init': 0.15, 'bounds': (0, 1), 'name': 'Venous return', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'v': {'init': 1, 'bounds': (0, 1), 'name': 'volume fraction', 'unit': 'mL/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': 'Q.PH1.001'},
    'k': {'init': 0.003, 'bounds': (0.0, 0.1), 'name': 'tissue transfer rate', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'ki': {'init': 0.003, 'bounds': (0.0, 0.1), 'name': 'initial tissue transfer rate', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'kf': {'init': 0.003, 'bounds': (0.0, 0.1), 'name': 'final tissue transfer rate', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'K': {'init': 0.003, 'bounds': (0.0, 0.1), 'name': 'transfer rate', 'unit': '1/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': 'Q.PH1.009'},
    'CL': {'init': 10, 'bounds': (0.0, 100), 'name': 'Contrast agent clearance', 'unit': 'mL/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'BF': {'init': 2, 'bounds': (0, 100), 'name': 'blood flow', 'unit': 'mL/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'PF': {'init': 1, 'bounds': (0, 100), 'name': 'plasma flow', 'unit': 'mL/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'PV': {'init': 1e4, 'bounds': (0, 100), 'name': 'plasma volume', 'unit': 'mL', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'EV': {'init': 1e4, 'bounds': (0, 100), 'name': 'plasma volume', 'unit': 'mL', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # --- Liver Kinetics ---
    'ffa': {'init': 0.2, 'bounds': (0, 1), 'name': 'arterial flow fraction', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # --- Kidney Kinetics ---
    'GFR': {'init': 2, 'bounds': (0, 10), 'name': 'glomerular filtration rate', 'unit': 'mL/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'nmGFR': {'init': 2, 'bounds': (0, 10), 'name': 'nuclear medicine glomerular filtration rate', 'unit': 'mL/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'FF': {'init': 0.1, 'bounds': (0, 0.5), 'name': 'filtration fraction', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    # This can be ff_kc
    'ffc': {'init': 0.8, 'bounds': (0, 1), 'name': 'cortical flow fraction', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'DRPF': {'init': 0.5, 'bounds': (0, 1), 'name': 'Differential renal plasma flow', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'DRF': {'init': 0.5, 'bounds': (0, 1), 'name': 'Differential renal function', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # --- Whole Body Kinetics ---
    'CO': {'init': 100, 'bounds': (0, 500), 'name': 'cardiac output', 'unit': 'mL/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'fCO': {'init': 0.1, 'bounds': (0, 0.5), 'name': 'fraction of the cardiac output', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # --- Tissue Compartment Kinetics ---
    'H': {'init': 0.45, 'bounds': (0, 1), 'name': 'hematocrit', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': 'Q.PH1.012'},
    'P': {'init': 0.3, 'bounds': (0, 1), 'name': 'porosity', 'unit': '', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Ktrans': {'init': 0.015, 'bounds': (0.0, 0.1), 'name': 'plasma clearance', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': 'Q.PH1.008'},
    'kep': {'init': 0.015, 'bounds': (0.0, 0.1), 'name': 'interstitial excretion rate', 'unit': '1/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'PS': {'init': 0.003, 'bounds': (0, 1), 'name': 'permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': 'Q.PH1.004'},
    'PSB': {'init': 0.003, 'bounds': (0, 1), 'name': 'whole-body permeability-surface area product', 'unit': 'mL/sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # LS
    'irf': {'init': 0.02, 'bounds': (0, 10), 'name': 'Impulse response function', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # Water exchange
    'RM': {'init': 1, 'bounds': None, 'name': 'relaxivity mapping', 'unit': None, 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'inlets': {'init': (0,), 'bounds': None, 'name': 'water inlet compartments', 'unit': None, 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Fwi': {'init': 0.02, 'bounds': (0, 1), 'name': 'inflow in all water compartments', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    #'wx': {'init': '', 'bounds': None, 'name': 'indicator-to-water compartment map', 'unit': None, 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'vw': {'init': 1, 'bounds': (0, 1), 'name': 'water volume fraction', 'unit': 'mL/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'Kw': {'init': 0, 'bounds': (0, 1), 'name': 'water exchange matrix', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'PSw': {'init': 0.03, 'bounds': (0, 100), 'name': 'water permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # TissueX water exchange
    'PSe': {'init': 0.03, 'bounds': (0, 100), 'name': 'transendothelial water permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'PSc': {'init': 0.03, 'bounds': (0, 100), 'name': 'transcytolemmal water permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # --- Relaxation ---
    'R1': {'init': 0.65, 'bounds': (0, 5), 'name': 'tissue R1', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.001'},
    'R2': {'init': 2.0, 'bounds': (0, 5), 'name': 'tissue R2', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.004'},
    'R2s': {'init': 20, 'bounds': (0, 5), 'name': 'tissue R2*', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.007'},
    'R1i': {'init': 0.65, 'bounds': (0, 5), 'name': 'inlet R1', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},

    'R1b': {'init': 0.65, 'bounds': (0, 5), 'name': 'precontrast tissue R1', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.002'},
    'R2b': {'init': 20, 'bounds': (0, 100), 'name': 'precontrast tissue R2', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.005'}, 
    'R2sb': {'init': 20, 'bounds': (0, 100), 'name': 'precontrast tissue R2*', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.008'},
    'R1ib': {'init': 0.65, 'bounds': (0, 5), 'name': 'precontrast inlet R1', 'unit': 'Hz', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},

    # --- Relaxivity ---
    # 'comps': {'init': ('ti',), 'bounds': None, 'name': 'tissue compartments', 'unit': None, 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},
    'r1': {'init': 3500, 'bounds': (0, 1e4), 'name': 'longitudinal contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.015'},
    'r2': {'init': 4000, 'bounds': (0, 1e4), 'name': 'transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.016'},
    'r2s': {'init': 20000, 'bounds': (0, 1e5), 'name': 'transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.017'},
    'r2sq': {'init': 1000, 'bounds': (0, 1e4), 'name': 'quadratic transverse contrast agent relaxivity', 'unit': 'Hz/M^2', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': 'Q.EL1.020'},
    'r2sv': {'init': 20000, 'bounds': (0, 1e5), 'name': 'vascular transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'r2se': {'init': 20000, 'bounds': (0, 1e5), 'name': 'extravascular, extracellular transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'r1i': {'init': 3500, 'bounds': (0, 1e4), 'name': 'inlet longitudinal contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},

    # --- Magnetization and flow ---
    'tR': {'init': 0.0, 'bounds': None, 'name': 'relaxation rate time points', 'unit': 'sec', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'me': {'init': 1, 'bounds': (0, 5), 'name': 'equilibrium magnetization', 'unit': 'A cm2/mL', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'tM': {'init': 0.0, 'bounds': None, 'name': 'magnetization time points', 'unit': 'sec', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'Mz': {'init': 1, 'bounds': (0, 5), 'name': 'longitudinal magnetization', 'unit': 'A/cm', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'tMz': {'init': 0.0, 'bounds': None, 'name': 'longitudinal magnetization time points', 'unit': 'sec', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'JMz': {'init': 1, 'bounds': (0, 5), 'name': 'longitudinal magnetization inflow', 'unit': 'A/cm/sec', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'tJMz': {'init': 0.0, 'bounds': None, 'name': 'longitudinal magnetization inflow time points', 'unit': 'sec', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'Mzi': {'init': 1, 'bounds': (0, 5), 'name': 'longitudinal inlet magnetization', 'unit': 'A/cm', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'tMi': {'init': 0.0, 'bounds': None, 'name': 'inlet magnetization time points', 'unit': 'sec', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'Mxy': {'init': 1, 'bounds': (0, 5), 'name': 'transverse magnetization', 'unit': 'A/cm', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'M': {'init': 1, 'bounds': (0, 5), 'name': 'magnetization', 'unit': 'A/cm', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},
    'TF': {'init': 0.25, 'bounds': (0, 1), 'name': 'inflow time', 'unit': 'sec', 'group': 'physiological', 'dicom_key': None, 'osipi_key': None},

    # --- MRI Sequence Parameters ---
    'field_strength': {'init': 3, 'bounds': (0, 20), 'name': 'magnetic field strength', 'unit': 'T', 'group': 'sequence', 'dicom_key': 'MagneticFieldStrength', 'osipi_key': None},
    'tstart': {'init': 0, 'bounds': (0, 1e4), 'name': 'start of the acquisition', 'unit': 'sec', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'tacq': {'init': 240, 'bounds': (0, 1e4), 'name': 'acquisition duration', 'unit': 'sec', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'FA': {'init': 15, 'bounds': (0, 180), 'name': 'flip angle', 'unit': 'deg', 'group': 'sequence', 'dicom_key': 'FlipAngle', 'osipi_key': 'Q.MS1.007'},
    'Nk0': {'init': 64, 'bounds': (0, 1000), 'name': 'number of acquired phase lines to the center of k-space', 'unit': '', 'group': 'sequence', 'dicom_key': None, 'osipi_key': 'Q.MS1.011'},
    'Nph': {'init': 128, 'bounds': (0, 1000), 'name': 'number of acquired phase lines in k-space', 'unit': '', 'group': 'sequence', 'dicom_key': None, 'osipi_key': 'Q.MS1.012'},
    'Nz': {'init': 64, 'bounds': (0, 1000), 'name': 'number of slices in a multi-slice acquisition', 'unit': '', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'PA': {'init': 90, 'bounds': (0, 180), 'name': 'preparation pulse flip angle', 'unit': 'deg', 'group': 'sequence', 'dicom_key': None, 'osipi_key': 'Q.MS1.009'},
    'SA': {'init': 0, 'bounds': (0, 180), 'name': 'saturation Slab Flip Angle', 'unit': 'deg', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'TA': {'init': 2.0, 'bounds': (0, 30), 'name': 'acquisition time', 'unit': 'sec', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'TD': {'init': 0.05, 'bounds': (0, 1), 'name': 'prepulse delay', 'unit': 'sec', 'group': 'sequence', 'dicom_key': None, 'osipi_key': 'Q.MS1.008'},
    'TE': {'init': 0.001, 'bounds': (0, 10), 'name': 'echo time', 'unit': 'sec', 'group': 'sequence', 'dicom_key': 'EchoTime', 'osipi_key': 'Q.MS1.005'},
    'TE1': {'init': 0.001, 'bounds': (0, 1), 'name': 'first echo time in a multi-echo sequence', 'unit': 'sec', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'TE2': {'init': 0.005, 'bounds': (0, 1), 'name': 'second echo time in a multi-echo sequence', 'unit': 'sec', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'TP': {'init': 0.05, 'bounds': (0, 1), 'name': 'preparation delay', 'unit': 'sec', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},
    'TR': {'init': 0.005, 'bounds': (0, 1), 'name': 'repetition time', 'unit': 'sec', 'group': 'sequence', 'dicom_key': 'RepetitionTime', 'osipi_key': 'Q.MS1.006'},
    'iz': {'init': 0, 'bounds': (0, 1000), 'name': 'slice number in a multi-slice acquisition', 'unit': '', 'group': 'sequence', 'dicom_key': None, 'osipi_key': None},

    # --- Magnetic tissue properties ---
    'B1corr': {'init': 1, 'bounds': (0, 5), 'name': 'B1-correction factor', 'unit': '', 'group': 'electromagnetic', 'dicom_key': None, 'osipi_key': None},

    # Signal
    'tS': {'init': 0.0, 'bounds': None, 'name': 'signal time points', 'unit': 'sec', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'S': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.001'},
    'Sb': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal baseline', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.002'},
    'nb': {'init': 1, 'bounds': None, 'name': 'number of baseline time points', 'unit': 'a.u.', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'S0': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal scaling factor', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.010'},
    'Scal': {'init': 1.0, 'bounds': (0, 5), 'name': 'calibration signal', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.002'},
    'iScal': {'init': 0, 'bounds': None, 'name': 'indices of calibration signal', 'unit': '', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'iStrig': {'init': None, 'bounds': None, 'name': 'indices of the signal trigger', 'unit': '', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'Strig': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal trigger', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.002'},
    'NSR': {'init': 0.0, 'bounds': (0, 1e5), 'name': 'noise-to-signal ratio', 'unit': '', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'RE': {'init': 1, 'bounds': (0, 1000), 'name': 'relative signal enhancement', 'unit': '', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},

    # Inverse signal
    'pfree': {'init': None, 'bounds': None, 'name': 'set of free parameters', 'unit': 'a.u.', 'group': 'inverse problem', 'dicom_key': None, 'osipi_key': None},
    'popt': {'init': None, 'bounds': None, 'name': 'dictionary of optimized free parameter values', 'unit': 'a.u.', 'group': 'inverse problem', 'dicom_key': None, 'osipi_key': None},
    'pcov': {'init': None, 'bounds': None, 'name': 'dictionary with covariances of free parameters', 'unit': 'a.u.', 'group': 'inverse problem', 'dicom_key': None, 'osipi_key': None},
    'psdev': {'init': None, 'bounds': None, 'name': 'dictionary with parameter standard deviations', 'unit': 'a.u.', 'group': 'inverse problem', 'dicom_key': None, 'osipi_key': None},
    'pars': {'init': None, 'bounds': None, 'name': 'dictionary with source- and derived parameters', 'unit': 'a.u.', 'group': 'inverse problem', 'dicom_key': None, 'osipi_key': None},
    #'pars': {'init': None, 'bounds': None, 'name': 'dictionary with source- and derived parameters', 'unit': 'a.u.', 'group': 'inverse problem', 'dicom_key': None, 'osipi_key': None},
    'loss': {'init': None, 'bounds': None, 'name': 'loss value of optimized model', 'unit': 'a.u.', 'group': 'inverse problem', 'dicom_key': None, 'osipi_key': None},

}

QVALUES = {k: v['init'] for k, v in QUANTITIES.items()}


SEQUENCES = {
    'ZTE-3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1'},
            'read': {'FA', 'B1corr'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0'},
        },
    },
    'ZTE-3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1'},
            'read': {'FA', 'B1corr'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TP', 'TD'},
        },
    },
    '3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0'},
        },
    },
    '3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TP', 'TD'},
        },
    },
    '3D-SR-SPGR-SS': {
        'mz_prep_tissue': 'SR-SPGR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TP', 'TD'},
        },
    },
    '3D-PR-SPGR-SS': {
        'mz_prep_tissue': 'PR-SPGR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TP', 'PA', 'TD'},
        },
    },
    '2D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0'},
        },
    },
    '3D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0'},
        },
    },
    '3D-IR-SPGR': {
        'mz_prep_tissue': 'IR-SPGR',
        'mz_prep_inflow': 'IR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TP', 'TD'},
        },
    },
    '3D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SPGR',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TP', 'TD'},
        },
    },
    '3D-PR-SPGR': {
        'mz_prep_tissue': 'PR-SPGR',
        'mz_prep_inflow': 'PR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'PA', 'TP', 'TD'},
        },
    },
    '3D-PR-SS': {
        'mz_prep_tissue': 'PR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TA', 'PA'},
        },
    },
    '3D-IR-SS': {
        'mz_prep_tissue': 'IR-SS',
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TA'},
        },
    },
    '3D-SR-SS': {
        'mz_prep_tissue': 'SR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TA'},
        },
    },
    '2D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'Eq',
        'steady-state': False,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0'},
        },
    },
    '2D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TP', 'TD', 'iz'},
        },
    },
    '3D-SPGR-SSI': {
        'mz_prep_tissue': 'SSI',
        'mz_prep_inflow': 'SSI',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TF', 'SA'},
        },
    },
    'ZTE-3D-SPGR-SSI': {
        'mz_prep_tissue': 'SSI',
        'mz_prep_inflow': 'SSI',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr'},
            'prep': {'TR', 'FA', 'B1corr', 'Nph', 'Nk0', 'TF', 'SA'},
        },
    },
    '2D-GE-EPI': {
        'mz_prep_tissue': 'GE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr', 'Nz', 'iz'},
        },
    },
    '2D-SE-EPI': {
        'mz_prep_tissue': 'SE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TE', 'TR', 'FA', 'B1corr', 'Nz', 'iz'},
        },
    },
    '2D-DE-EPI': {
        'mz_prep_tissue': 'DE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2', 'R2s'},
            'read': {'FA', 'B1corr', 'TE1', 'TE2'},
            'prep': {'TE2', 'TR', 'FA', 'B1corr', 'Nz', 'iz'},
        },
    },
    '3D-GE-EPI': {
        'mz_prep_tissue': 'GE-SS',
        'mz_prep_inflow': 'GE-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2s'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TR', 'FA', 'B1corr'},
        },
    },
    '3D-SE-EPI': {
        'mz_prep_tissue': 'SE-SS',
        'mz_prep_inflow': 'SE-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2'},
            'read': {'FA', 'B1corr', 'TE'},
            'prep': {'TE', 'TR', 'FA', 'B1corr'},
        },
    },
    '3D-DE-EPI': {
        'mz_prep_tissue': 'DE-SS',
        'mz_prep_inflow': 'DE-SS',
        'steady-state': True,
        'parameters': {
            'tissue': {'R1', 'R2', 'R2s'},
            'read': {'FA', 'B1corr', 'TE1', 'TE2'},
            'prep': {'TE2', 'TR', 'FA', 'B1corr'},
        },
    },
}
