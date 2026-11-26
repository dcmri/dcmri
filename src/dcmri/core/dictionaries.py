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






QDATA_ROIS = {
    # 'Fb': {'init': 0.02, 'bounds': (0, 1)},
    # 'Fp': {'init': 0.02, 'bounds': (0, 0.05)},
    'T_b': {'init': 5, 'bounds': (0.1, 60)},
    'T_p': {'init': 5, 'bounds': (0, 30)},
    'T_e': {'init': 30.0, 'bounds': (0.1, 60)},
    'T_i': {'init': 30, 'bounds': (0, 600)},
    'v_b': {'init': 0.1, 'bounds': (1e-3, 1 - 1e-3)},
    'v_p': {'init': 0.15, 'bounds': (0, 0.3)},
    'v_e': {'init': 0.3, 'bounds': (0.01, 0.6)},
    'v_i': {'init': 0.3, 'bounds': (1e-3, 1 - 1e-3)},
    'v_c': {'init': 0.6, 'bounds': (1e-3, 1 - 1e-3)},
    # hl
    'T_hl': {'init': 10, 'bounds': (0, 30)},
    'D_hl': {'init': 0.2, 'bounds': (0.01, 0.99)},
    # or
    'T_b_or': {'init': 20, 'bounds': (0, 60)},
    'T_e_or': {'init': 120, 'bounds': (0, 800)},  
    'E_or': {'init': 0.15, 'bounds': (0, 0.5)},
    'vr_or': {'init': 0.9, 'bounds': (0, 1)},
    # li
    'v_e_li': {'init': 0.25, 'bounds': (0.01, 0.5)},
    'T_e_li': {'init': 30.0, 'bounds': (0.1, 60)},
    'E_li': {'init': 0.1, 'bounds': (0.0, 1.0)},
    'Ei_li': {'init': 0.1, 'bounds': (0.0, 1.0)},
    'Ef_li': {'init': 0.1, 'bounds': (0.0, 1.0)},
    'F_p_li': {'init': 0.01, 'bounds': (0, 0.05)},
    'T_h': {'init': 4 * 60 * 60, 'bounds': (600, 6 * 60 * 60)},
    'Ti_h': {'init': 4 * 60 * 60, 'bounds': (600, 6 * 60 * 60)},
    'Tf_h': {'init': 4 * 60 * 60, 'bounds': (600, 6 * 60 * 60)},
    'k_e2h': {'init': 0.0025, 'bounds': (0, 0.006)},
    'ki_e2h': {'init': 0.0025, 'bounds': (0, 0.006)},
    'kf_e2h': {'init': 0.0025, 'bounds': (0, 0.006)},

    # gu
    'T_gu': {'init': 30, 'bounds': (0.1, 60)},
    'D_gu': {'init': 0.85, 'bounds': (0, 1)},
    # ao
    'vol_ao': {'init': 1, 'bounds': (0.0, 1000)},
    'vol_li': {'init': 1000, 'bounds': (0, 10000)},
    'vol_ki': {'init': 300, 'bounds': (0.0, 1000)},
    'vol_lk': {'init': 150, 'bounds': (0.0, 1000)},
    'vol_rk': {'init': 150, 'bounds': (0.0, 1000)},
    # pv
    'R1b_pv': {'init': 1.25, 'bounds': (0, 5)},
    # lk / rk
    'PF_lk': {'init': 10, 'bounds': (0, 100)},
    'PF_rk': {'init': 10, 'bounds': (0, 100)},
    'GFR_lk': {'init': 1, 'bounds': (0, 20)},
    'GFR_rk': {'init': 1, 'bounds': (0, 20)},
    'T_lka': {'init': 0, 'bounds': (0, 5)},
    'T_rka': {'init': 0, 'bounds': (0, 5)},
    # ki
    'F_u': {'init': 0.005, 'bounds': (0, 0.05)},
    'T_u': {'init': 120, 'bounds': (30, 10 * 60)},
    'E_ki': {'init': 0.15, 'bounds': (0, 1)},
    'E_lk': {'init': 0.15, 'bounds': (0, 1)},
    'E_rk': {'init': 0.15, 'bounds': (0, 1)},
    'T_pt': {'init': 60, 'bounds': (0, 180)},
    'T_lh': {'init': 60, 'bounds': (0, 180)},
    'T_dt': {'init': 30, 'bounds': (0, 180)},
    'T_cd': {'init': 30, 'bounds': (0, 180)},
    'T_gc': {'init': 4, 'bounds': (0, 30)},
    'T_pcv': {'init': 10, 'bounds': (0, 30)},
}



# -----------------------------------------------------------------------------
# Contrast agent properties
# -----------------------------------------------------------------------------

CA_CONC = { # mmol/mL
    'gadoxetate': 0.25,
    'gadobutrol': 1.0,
    'gadopentetate': 0.5,
    'gadobenate': 0.5,
    'gadodiamide': 0.5,
    'gadoterate': 0.5,
    'gadoteridol': 0.5,
    'gadopiclenol': 0.5,   
}

CA_DOSE = { # mL/kg
    'gadoxetate': 0.1,   # https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf
    'gadobutrol': 0.1,
    'gadopiclenol': 0.1,
    'gadopentetate': 0.2,
    'gadobenate': 0.2,
    'gadodiamide': 0.2,
    'gadoterate': 0.2,
    'gadoteridol': 0.2,
}

# -----------------------------------------------------------------------------
# Global Relaxivity Databases (Hz/mM)
# -----------------------------------------------------------------------------

R1_RELAXIVITY = {
    'plasma': {
        'gadopiclenol': {1.5: 12.8, 3.0: 11.6},
        'gadopentetate': {0.47: 3.8, 1.5: 4.1, 3.0: 3.7, 4.7: 3.8}, # Magnevist
        'gadobutrol': {0.47: 6.1, 1.5: 5.2, 3.0: 5.0, 4.7: 4.7}, # Gadovist
        'gadoteridol': {0.47: 4.8, 1.5: 4.1, 3.0: 3.7, 4.7: 3.7}, # Prohance
        'gadobenade': {0.47: 9.2, 1.5: 6.3, 3.0: 5.5, 4.7: 5.2}, # Multihance
        'gadoterate': {0.47: 4.3, 1.5: 3.6, 3.0: 3.5, 4.7: 3.3}, # Dotarem
        'gadodiamide': {0.47: 4.4, 1.0: 4.35, 1.5: 4.3, 3.0: 4.0, 4.7: 3.9}, # Omniscan
        'mangafodipir': {0.47: 3.6, 1.5: 3.6, 3.0: 2.7, 4.7: 2.2}, # Teslascan
        'gadoversetamide': {0.47: 5.7, 1.5: 4.7, 3.0: 4.5, 4.7: 4.4}, # Optimark
        'ferucarbotran': {0.47: 15.0, 1.5: 7.4, 3.0: 3.3, 4.7: 1.7},  # Resovist
        'ferumoxide': {1.5: 4.5, 3.0: 2.7, 4.7: 1.2}, # Feridex
        'gadoxetate': {0.47: 8.7, 1.5: 8.1, 3.0: 6.4, 4.7: 6.4, 7.0: 6.2, 9.0: 6.1}, # Primovist
    },
    'hepatocytes': {
        'gadoxetate': {1.5: 14.6, 3.0: 9.8, 4.7: 7.6, 7.0: 6.0, 9.0: 6.1}
    },
}


# Known literature value mapping for pure r2 (spin-echo / CPMG sequence data).
# Missing entries are entirely omitted to leave out unknown combinations.
R2_RELAXIVITY = {
    'plasma': {
        'gadopiclenol': {1.5: 13.2, 3.0: 15.4},
        'gadopentetate': {0.47: 4.6, 1.5: 4.6, 3.0: 4.8, 4.7: 5.0},
        'gadobutrol': {0.47: 7.3, 1.5: 6.1, 3.0: 7.4, 4.7: 6.1},
        'gadoteridol': {0.47: 5.6, 1.5: 5.0, 3.0: 4.9, 4.7: 5.1},
        'gadobenade': {0.47: 10.9, 1.5: 8.4, 3.0: 8.1, 4.7: 8.3},
        'gadoterate': {0.47: 5.1, 1.5: 4.4, 3.0: 4.7, 4.7: 4.5},
        'gadodiamide': {0.47: 5.2, 1.5: 5.1, 3.0: 5.0, 4.7: 5.2},
        'mangafodipir': {0.47: 4.2, 1.5: 4.4, 3.0: 3.6, 4.7: 3.1},
        'gadoversetamide': {0.47: 6.8, 1.5: 5.9, 3.0: 6.0, 4.7: 6.2},
        'gadoxetate': {0.47: 10.8, 1.5: 10.1, 3.0: 8.8, 4.7: 9.1},
    }
}


R2_STAR_RELAXIVITY = {
    'blood': 10.0,  # Estimated from the range [0, 5mM] in data by van Osch MJ, Vonken EJ, Viergever MA, van der Grond J, Bakker CJ. Measuring the arterial input function with gradient echo sequences. Magn Reson Med 2003;49:1067–1076
    'tissue': 20.0, # Guess - look for data
}

R2_STAR_RELAXIVITY_SQ = {
    'blood': 1000.0,      # Guess - see if there are values
}


_HEMATOCRIT = 0.45 # temporary - integrate in broader dictionary


# Proton density values from:

# H. M. Gach, C. Tanase and F. Boada, "2D & 3D Shepp-Logan Phantom
# Standards for MRI," 2008 19th International Conference on Systems
# Engineering, Las Vegas, NV, USA, 2008, pp. 521-526, doi:
# 10.1109/ICSEng.2008.15.

PROTON_DENSITY = {
    'skin': 0.8,
    'bone marrow': 0.12,
    'csf': 0.98,
    'white matter': 0.617,
    'gray matter': 0.745,
    'blood': 0.95, # guess not from literature
}


T1_RELAXATION_TIMES = {
    'skin': {  # Gach 2008 (scalp)
        1.5: 0.324 * (1.5**0.137),
        3.0: 0.324 * (3.0**0.137),
    },
    'bone marrow': {  # Gach 2008
        1.5: 0.533 * (1.5**0.088),
        3.0: 0.533 * (3.0**0.088),
    },
    'csf': {  # Gach 2008
        1.5: 4.20,
        3.0: 4.20,
    },
    'muscle': {
        1.5: 1.008,
        3.0: 1.412,
    },
    'heart': {
        1.5: 1.030,
        3.0: 1.471,
    },
    'cartilage': {
        1.5: 1.024,
        3.0: 1.168,
    },
    'white matter': {
        1.5: 0.884,
        3.0: 1.084,
    },
    'gray matter': {
        1.5: 1.124,
        3.0: 1.820,
    },
    'optic nerve': {
        1.5: 0.815,
        3.0: 1.083,
    },
    'spinal cord': {
        1.5: 0.745,
        3.0: 0.993,
    },
    'blood': {
        1.0: 1.378,  # Extrapolated
        1.5: 1.441,
        3.0: 1 / (0.52 * _HEMATOCRIT + 0.38),  # Lu MRM 2004
        4.7: 1 / 1.70,  # https://cds.ismrm.org/ismrm-2002/PDF4/1048.PDF
        7.0: 1 / 2.29,  # 10.1016/j.mri.2012.08.008
    },
    'spleen': {
        4.7: 1 / 0.631,
        7.0: 1 / 0.611,
        9.0: 1 / 0.600,
    },
    'liver': {
        1.5: 0.602,  # liver R1 in 1/sec (Waterton 2021)
        3.0: 0.752,  # liver R1 in 1/sec (Waterton 2021)
        # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
        4.7: 1 / 1.281,
        # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
        7.0: 1 / 1.109,
        # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)
        9.0: 1 / 0.920,
    },
    'kidney': {
        # Reference values average over cortex and medulla from Cox et al
        # https://academic.oup.com/ndt/article/33/suppl_2/ii41/5078406
        1.0: 1.017,  # Extrapolated
        1.5: (1.024 + 1.272) / 2,
        3.0: (1.399 + 1.685) / 2,
    },
}


T2_RELAXATION_TIMES = {
    'skin': {  # Gach 2008 (scalp)
        1.5: 0.07,
        3.0: 0.07,
    },
    'bone marrow': {  # Gach 2008
        1.5: 0.05,
        3.0: 0.05,
    },
    'csf': {  # Gach 2008
        1.5: 1.99,
        3.0: 1.99,
    },
    'white matter': {
        1.5: 0.08,
        3.0: 0.08,
    },
    'gray matter': {
        1.5: 0.1,
        3.0: 0.1,
    },
    'blood': {
        1.5: 0.275,
        3.0: 0.130,
    },
    'arterial blood': {
        1.5: 0.275,
        3.0: 0.130,
    },
    'venous blood': {
        1.5: 0.175,
        3.0: 0.055,
    },
}


T2_STAR_RELAXATION_TIMES = {
    'blood': {
        1.5: 0.100,
        3.0: 0.045,
    },
    'arterial blood': {
        1.5: 0.250,
        3.0: 0.110,
    },
    'venous blood': {
        1.5: 0.040,
        3.0: 0.015,
    },
}


PERFUSION = {
    'Fb': {
        'skin': 0.005,  # 5 kg/s/m**3 = 5000mL/sec/100*100*100 mL = 0.005mL/sec/mL
        'bone marrow': 0.0013,  # 0.08 ml/ml/min
        'csf': 0.0,
        'white matter': 0.0033,
        'gray matter': 0.01,
    },
    'vb': {
        'skin': 0.03,
        'bone marrow': 0.25,
        'csf': 0.0,
        'white matter': 0.02,
        'gray matter': 0.05,
    },
    'PS': { # Needs verification
        'skin': 0.001,
        'bone marrow': 0.0002,
        'csf': 0.0,
        'white matter': 0.0,
        'gray matter': 0.0,
    },
    'vi':{ # Needs verification
        'skin': 0.03,
        'bone marrow': 0.2,
        'csf': 0.0,
        'white matter': 0.3,
        'gray matter': 0.35,
    },
}



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


