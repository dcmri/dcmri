# -----------------------------------------------------------------------------
# Contrast agent generic names
# -----------------------------------------------------------------------------


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


GROUPS = {
    'indicator': 'Indicator',
    'signal': 'Signal',
    'seq': 'Sequence',
    'EM': 'Electromagnetic',
    'phys': 'Physiological',
    'hyper': 'Hyperparameters',
    'body': 'Whole-body',
}

ROIS = {
    'ab': 'arterial blood',
    'ar': 'artery',
    'ao': 'aorta',
    'b': 'blood', # can be both ROI and compartment
    'bm': 'bone marrow',
    'ca': 'cartilage',
    'csf': 'cerebro-spinal fluid',
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
    'mu': 'muscle',
    'on': 'optic nerve',
    'or': 'organs',
    'pv': 'portal vein',
    'rk': 'right kidney',
    'sc': 'spinal cord',
    'sk': 'skin',
    'ti': 'tissue',
    'vb': 'venous blood',
    'vc': 'vena cava',
    've': 'vein', 
    'wm': 'white matter',
}

#Note: No shared keys between ROIs and comps except when the same value (eg. blood)

COMPS = { # subvoxel compartments
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



QUANTITIES = {

    # Generic hyperparameters
    'dt': {'init': 0.5, 'bounds': None, 'name': 'pseudo-continuous time step', 'unit': 'sec', 'group': 'hyper', 'dicom_key': None, 'osipi_key': None},
    'tmax': {'init': 240, 'bounds': None, 'name': 'maximum time point', 'unit': 'sec', 'group': 'hyper', 'dicom_key': None, 'osipi_key': None},

    # Generic indicator quantities
    'agent': {'init': 'gadoterate', 'bounds': None, 'name': 'contrast agent generic name', 'unit': None, 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'BAT': {'init': 30, 'bounds': (-30, 30), 'name': 'bolus arrival time', 'unit': 'sec', 'bounds_type': 'add', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'bdel': {'init': 30, 'bounds': (-30, 30), 'name': 'delay in a double injection', 'unit': 'sec', 'bounds_type': 'add', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    # 'BAT1': {'init': 30, 'bounds': (-60, 60), 'name': 'first bolus arrival time in a dual injection', 'unit': 'sec', 'group': 'indicator', 'bounds_type': 'add'},
    # 'BAT2': {'init': 90, 'bounds': (-60, 60), 'name': 'second bolus arrival time in a dual injection', 'unit': 'sec', 'group': 'indicator', 'bounds_type': 'add'},
    'dose': {'init': 0.1, 'bounds': (0, 0.2), 'name': 'contrast agent dose', 'unit': 'mL/kg', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    # 'dose1': {'init': 0.05, 'bounds': (0, 0.2), 'name': 'first contrast agent dose in a dual injection', 'unit': 'mL/kg', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    # 'dose2': {'init': 0.05, 'bounds': (0, 0.2), 'name': 'second contrast agent dose in a dual injection', 'unit': 'mL/kg', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'rate': {'init': 1, 'bounds': (0, 10), 'name': 'injection rate', 'unit': 'mL/s', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    # 'rate1': {'init': 1, 'bounds': (0, 10), 'name': 'first injection rate in a dual injection', 'unit': 'mL/s', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    # 'rate2': {'init': 1, 'bounds': (0, 10), 'name': 'second injection rate in a dual injection', 'unit': 'mL/s', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},

    'tC': {'init': 0.0, 'bounds': None, 'name': 'concentration time points', 'unit': 'sec', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'J': {'init': 1, 'bounds': (0, 10), 'name': 'indicator flux', 'unit': 'mmol/sec', 'bounds_type': 'abs', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'Jinj': {'init': 1, 'bounds': (0, 10), 'name': 'indicator flux at the injection site', 'unit': 'mmol/sec', 'bounds_type': 'abs', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'c': {'init': 0.005, 'bounds': (0, 1), 'name': 'concentration', 'unit': 'mmol/mL', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'ci': {'init': 0.005, 'bounds': None, 'name': 'inlet concentration', 'unit': 'mmol/mL', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},
    'C': {'init': 0.005, 'bounds':(0, 1), 'name': 'tissue concentration', 'unit': 'mmol/cm3', 'group': 'indicator', 'dicom_key': None, 'osipi_key': None},

    # Whole body properties
    'weight': {'init': 70, 'bounds': (0, 300), 'name': 'body weight', 'unit': 'kg', 'group': 'body', 'dicom_key': None, 'osipi_key': None},
    'vol': {'init': 150, 'bounds': (0.0, 1000), 'name': 'ROI volume', 'unit': 'cm3', 'group': 'body', 'dicom_key': None, 'osipi_key': None},

    # ---- Kinetic hyperparameters ---
    'dose_tolerance': {'init': 0.1, 'bounds': None, 'name': 'dose tolerance', 'unit': '', 'group': 'hyper', 'dicom_key': None, 'osipi_key': None},

    # Generic kinetics
    'F': {'init': 0.02, 'bounds': (0, 1), 'name': 'flow per unit tissue', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'T': {'init': 30, 'bounds': (0.1, 60), 'name': 'mean transit time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Ti': {'init': 30, 'bounds': (0.1, 60), 'name': 'initial mean transit time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Tf': {'init': 30, 'bounds': (0.1, 60), 'name': 'final mean transit time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'D': {'init': 0.85, 'bounds': (0, 1), 'name': 'transit time dispersion', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'h': {'init': 1.0, 'bounds': (0.1, 60), 'name': 'transit time distribution', 'unit': 'Hz', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'TT': {'init': 1.0, 'bounds': (0.1, 60), 'name': 'boundaries of transit time histogram bins', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'E': {'init': 0.1, 'bounds': (0.0, 1.0), 'name': 'extraction fraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Ei': {'init': 0.1, 'bounds': (0.0, 1.0), 'name': 'initial extraction fraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Ef': {'init': 0.1, 'bounds': (0.0, 1.0), 'name': 'final extraction fraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Vmax': {'init': 1, 'bounds': (0.0, 10), 'name': 'limiting rate in a Michaelis-Menten compartment', 'unit': 'mmol/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Km': {'init': 1, 'bounds': (0.0, 10), 'name': 'Michaelis-Menten constant', 'unit': 'mmol/mL', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'ffp': {'init': 0.2, 'bounds': (0, 1), 'name': 'plug-flow fraction in a plucom system', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'vr': {'init': 0.15, 'bounds': (0, 1), 'name': 'Venous return', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'v': {'init': 1, 'bounds': (0, 1), 'name': 'volume fraction', 'unit': 'mL/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'k': {'init': 0.003, 'bounds': (0.0, 0.1), 'name': 'tissue transfer rate', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'ki': {'init': 0.003, 'bounds': (0.0, 0.1), 'name': 'initial tissue transfer rate', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'kf': {'init': 0.003, 'bounds': (0.0, 0.1), 'name': 'final tissue transfer rate', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # --- Liver Kinetics ---
    'ffa': {'init': 0.2, 'bounds': (0, 1), 'name': 'arterial flow fraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # --- Kidney Kinetics ---
    'GFR': {'init': 2, 'bounds': (0, 10), 'name': 'glomerular filtration rate', 'unit': 'mL/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'FF': {'init': 0.1, 'bounds': (0, 0.5), 'name': 'filtration fraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'ffc': {'init': 0.8, 'bounds': (0, 1), 'name': 'cortical flow fraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'DRPF': {'init': 0.5, 'bounds': (0, 1), 'name': 'Differential renal plasma flow', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # --- Whole Body Kinetics ---
    'CO': {'init': 100, 'bounds': (0, 500), 'name': 'cardiac output', 'unit': 'mL/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'fCO': {'init': 0.1, 'bounds': (0, 0.5), 'name': 'fraction of the cardiac output', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # --- Tissue Compartment Kinetics ---
    'H': {'init': 0.45, 'bounds': (0, 1), 'name': 'hematocrit', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'P': {'init': 0.3, 'bounds': (0, 1), 'name': 'porosity', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Ktrans': {'init': 0.015, 'bounds': (0.0, 0.1), 'name': 'plasma clearance', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'PS': {'init': 0.003, 'bounds': (0, 1), 'name': 'permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # LS
    'irf': {'init': 0.02, 'bounds': (0, 10), 'name': 'Impulse response function', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # Water exchange
    'RM': {'init': '', 'bounds': None, 'name': 'relaxivity mapping', 'unit': None, 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'inlets': {'init': (0,), 'bounds': None, 'name': 'water inlet compartments', 'unit': None, 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Fwi': {'init': 0.02, 'bounds': (0, 1), 'name': 'inflow in all water compartments', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    #'wx': {'init': '', 'bounds': None, 'name': 'indicator-to-water compartment map', 'unit': None, 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'vw': {'init': 1, 'bounds': (0, 1), 'name': 'water volume fraction', 'unit': 'mL/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'Kw': {'init': 0, 'bounds': (0, 1), 'name': 'water exchange matrix', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'PSw': {'init': 0.03, 'bounds': (0, 100), 'name': 'water permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # TissueX water exchange
    'PSe': {'init': 0.03, 'bounds': (0, 100), 'name': 'transendothelial water permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'PSc': {'init': 0.03, 'bounds': (0, 100), 'name': 'transcytolemmal water permeability-surface area product', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # --- Relaxation ---
    'R1': {'init': 0.65, 'bounds': (0, 5), 'name': 'tissue R1', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'R2': {'init': 2.0, 'bounds': (0, 5), 'name': 'tissue R2', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'R2s': {'init': 20, 'bounds': (0, 5), 'name': 'tissue R2*', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'R1i': {'init': 0.65, 'bounds': (0, 5), 'name': 'inlet R1', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},

    'R1b': {'init': 0.65, 'bounds': (0, 5), 'name': 'precontrast tissue R1', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'R2b': {'init': 20, 'bounds': (0, 100), 'name': 'precontrast tissue R2', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None}, 
    'R2sb': {'init': 20, 'bounds': (0, 100), 'name': 'precontrast tissue R2*', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'R1ib': {'init': 0.65, 'bounds': (0, 5), 'name': 'precontrast inlet R1', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},

    # --- Relaxivity ---
    # 'comps': {'init': ('ti',), 'bounds': None, 'name': 'tissue compartments', 'unit': None, 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    'r1': {'init': 3500, 'bounds': (0, 1e4), 'name': 'longitudinal contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'r2': {'init': 4000, 'bounds': (0, 1e4), 'name': 'transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'r2s': {'init': 20000, 'bounds': (0, 1e5), 'name': 'transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'r2sq': {'init': 1000, 'bounds': (0, 1e4), 'name': 'quadratic transverse contrast agent relaxivity', 'unit': 'Hz/M^2', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'r2sv': {'init': 20000, 'bounds': (0, 1e5), 'name': 'vascular transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'r2se': {'init': 20000, 'bounds': (0, 1e5), 'name': 'extravascular, extracellular transverse contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'r1i': {'init': 3500, 'bounds': (0, 1e4), 'name': 'inlet longitudinal contrast agent relaxivity', 'unit': 'Hz/M', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},

    # --- Magnetization and flow ---
    'tR': {'init': 0.0, 'bounds': None, 'name': 'relaxation rate time points', 'unit': 'sec', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'me': {'init': 1, 'bounds': (0, 5), 'name': 'equilibrium magnetization', 'unit': 'A cm2/mL', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'tM': {'init': 0.0, 'bounds': None, 'name': 'magnetization time points', 'unit': 'sec', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'Mz': {'init': 1, 'bounds': (0, 5), 'name': 'longitudinal magnetization', 'unit': 'A/cm', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'tMz': {'init': 0.0, 'bounds': None, 'name': 'longitudinal magnetization time points', 'unit': 'sec', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'Mzi': {'init': 1, 'bounds': (0, 5), 'name': 'longitudinal inlet magnetization', 'unit': 'A/cm', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'tMi': {'init': 0.0, 'bounds': None, 'name': 'inlet magnetization time points', 'unit': 'sec', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'Mxy': {'init': 1, 'bounds': (0, 5), 'name': 'transverse magnetization', 'unit': 'A/cm', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'M': {'init': 1, 'bounds': (0, 5), 'name': 'magnetization', 'unit': 'A/cm', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},
    'TF': {'init': 0.5, 'bounds': (0, 10), 'name': 'inflow time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # --- MRI Sequence Parameters ---
    'field_strength': {'init': 3, 'bounds': (0, 20), 'name': 'magnetic field strength', 'unit': 'T', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'tstart': {'init': 0, 'bounds': (0, 1e4), 'name': 'start of the acquisition', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'tacq': {'init': 240, 'bounds': (0, 1e4), 'name': 'acquisition duration', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'FA': {'init': 15, 'bounds': (0, 180), 'name': 'flip angle', 'unit': 'deg', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'Nk0': {'init': 64, 'bounds': (0, 1000), 'name': 'number of acquired phase lines to the center of k-space', 'unit': '', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'Nph': {'init': 128, 'bounds': (0, 1000), 'name': 'number of acquired phase lines in k-space', 'unit': '', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'Nz': {'init': 64, 'bounds': (0, 1000), 'name': 'number of slices in a multi-slice acquisition', 'unit': '', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'PA': {'init': 90, 'bounds': (0, 180), 'name': 'preparation Pulse Flip Angle', 'unit': 'deg', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'SA': {'init': 0, 'bounds': (0, 180), 'name': 'saturation Slab Flip Angle', 'unit': 'deg', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'TA': {'init': 2.0, 'bounds': (0, 30), 'name': 'acquisition time', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'TD': {'init': 0.05, 'bounds': (0, 1), 'name': 'prepulse delay', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'TE': {'init': 0.001, 'bounds': (0, 10), 'name': 'echo time', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'TE1': {'init': 0.001, 'bounds': (0, 1), 'name': 'first echo time in a multi-echo sequence', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'TE2': {'init': 0.005, 'bounds': (0, 1), 'name': 'second echo time in a multi-echo sequence', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'TP': {'init': 0.05, 'bounds': (0, 1), 'name': 'preparation delay', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'TR': {'init': 0.005, 'bounds': (0, 1), 'name': 'repetition time', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    'iz': {'init': 0, 'bounds': (0, 1000), 'name': 'slice number in a multi-slice acquisition', 'unit': '', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},

    # --- Magnetic tissue properties ---
    'B1corr': {'init': 1, 'bounds': (0, 5), 'name': 'B1-correction factor', 'unit': '', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},

    # Signal
    'tS': {'init': 0.0, 'bounds': None, 'name': 'signal time points', 'unit': 'sec', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'S': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'Sb': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal baseline', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.002'},
    'nb': {'init': 1, 'bounds': None, 'name': 'number of baseline time points', 'unit': 'a.u.', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'S0': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal scaling factor', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.010'},
    'Scal': {'init': 1.0, 'bounds': (0, 5), 'name': 'calibration signal', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.002'},
    'iScal': {'init': 0, 'bounds': None, 'name': 'indices of calibration signal', 'unit': '', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'iStrig': {'init': None, 'bounds': None, 'name': 'indices of the signal trigger', 'unit': '', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'Strig': {'init': 1.0, 'bounds': (0, 5), 'name': 'signal trigger', 'unit': 'a.u.', 'bounds_type': 'mult', 'group': 'signal', 'dicom_key': None, 'osipi_key': 'Q.MS1.002'},
    'NSR': {'init': 0.0, 'bounds': (0, 1e5), 'name': 'noise-to-signal ratio', 'unit': '', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},

    # Inverse signal
    'pfree': {'init': 1, 'bounds': None, 'name': 'set of free parameters', 'unit': 'a.u.', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'popt': {'init': 1, 'bounds': None, 'name': 'dictionary of optimized free parameter values', 'unit': 'a.u.', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'pcov': {'init': 1, 'bounds': None, 'name': 'dictionary with covariances of free parameters', 'unit': 'a.u.', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'psdev': {'init': 1, 'bounds': None, 'name': 'dictionary with parameter standard deviations', 'unit': 'a.u.', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    'loss': {'init': 1, 'bounds': None, 'name': 'loss value of optimized model', 'unit': 'a.u.', 'group': 'signal', 'dicom_key': None, 'osipi_key': None},
    
    # 't_scan2': {'init': 120, 'bounds': None, 'name': 'Start of second scan', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'FAR': {'init': 15, 'bounds': (0, 180), 'name': 'Readout flip angle', 'unit': 'deg', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'FA2': {'init': 15.0, 'bounds': (0.0, 180), 'name': 'Second flip angle', 'unit': 'deg', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'FA_2': {'init': 15.0, 'bounds': (0.0, 180), 'name': 'Second flip angle', 'unit': 'deg', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'TC': {'init': 0.2, 'bounds': (0, 10), 'name': 'Time to k-space center', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'TS': {'init': 0, 'bounds': (0, 30), 'name': 'Sampling time', 'unit': 'sec', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'n_init': {'init': 1, 'bounds': (0, 1), 'name': 'Initial relative magnetization', 'unit': '', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'n0': {'init': 1, 'bounds': (0, 1000), 'name': 'Number of baseline dynamics', 'unit': '', 'group': 'seq', 'dicom_key': None, 'osipi_key': None},
    # 'FAcorr': {'init': 1, 'bounds': (0, 2), 'name': 'B1-corrected Flip Angle', 'unit': 'deg', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},

    # # --- Relaxation ---
    # 'R1b_pv': {'init': 1.25, 'bounds': (0, 5), 'name': 'Portal venous baseline R1', 'unit': 'Hz', 'group': 'EM', 'dicom_key': None, 'osipi_key': None},

    # # Water kinetics
    # 'Twc': {'init': 0.1, 'bounds': (0, 1), 'name': 'Intracellular water mean transit time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'Twi': {'init': 0.1, 'bounds': (0, 1), 'name': 'Interstitial water mean transit time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'Twb': {'init': 0.1, 'bounds': (0, 1), 'name': 'Intravascular water mean transit time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # # 'fvr_l': {'init': 0.1, 'bounds': (0, 1), 'name': 'Liver fraction of the venous return', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'Eb': {'init': 0.05, 'bounds': (0.01, 0.15), 'name': 'Body extraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # # --- Blood Kinetics ---
    # 'Tvc': {'init': 10, 'bounds': (0, 30), 'name': 'Vena cava MTT', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'Dvc': {'init': 0.2, 'bounds': (0.01, 0.99), 'name': 'Vena cava dispersion', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # # --- Kidney Kinetics ---
    # 'CBF': {'init': 0.04, 'bounds': (0, 0.1), 'name': 'Cortical blood flow', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'MBF': {'init': 0.004, 'bounds': (0, 0.1), 'name': 'Medullary blood flow', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'RPF': {'init': 20, 'bounds': (0, 100), 'name': 'Renal plasma flow', 'unit': 'mL/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'DRF': {'init': 0.5, 'bounds': (0, 1.0), 'name': 'Differential renal function', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # # --- Liver Kinetics ---
    # # TODO: no underscores in lexicon - just key names. Handle override naming some other way.
    # 'k_h2bi': {'init': 0.0004, 'bounds': (0.0, 0.001), 'name': 'Biliary excretion rate', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'ki_h2bi': {'init': 0.0004, 'bounds': (0.0, 0.001), 'name': 'Initial biliary excretion rate', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'kf_h2bi': {'init': 0.0004, 'bounds': (0.0, 0.001), 'name': 'Final biliary excretion rate', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'K_h2bi': {'init': 0.0001, 'bounds': (0.0, 0.001), 'name': 'Biliary tissue excretion rate', 'unit': '/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'K_e2h': {'init': 0.02, 'bounds': (0.0, 0.1), 'name': 'Hepatocellular tissue uptake rate', 'unit': '/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'Ki_h2bi': {'init': 0.0001, 'bounds': (0.0, 0.001), 'name': 'Initial biliary tissue excretion rate', 'unit': '/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'Kf_h2bi': {'init': 0.0001, 'bounds': (0.0, 0.001), 'name': 'Final biliary tissue excretion rate', 'unit': '/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'F_ar': {'init': 0.002, 'bounds': (0, 0.05), 'name': 'Arterial plasma flow', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'F_ve': {'init': 0.008, 'bounds': (0, 0.05), 'name': 'Venous plasma flow', 'unit': 'mL/sec/cm3', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    # 'CL': {'init': 10, 'bounds': (0.0, 100), 'name': 'Liver plasma clearance', 'unit': 'mL/sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},

    # # Portal vein
    # 'uv': {'init': 1.0,  'bounds': (0.0, 1.0), 'name': 'Portal vein volume fraction', 'unit': '', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
}

QVALUES = {k: v['init'] for k, v in QUANTITIES.items()}