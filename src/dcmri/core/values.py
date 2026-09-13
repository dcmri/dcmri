


QDATA_ROIS = {
    'Fb': {'init': 0.02, 'bounds': (0, 1)},
    'Fp': {'init': 0.02, 'bounds': (0, 0.05)},
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
    'T_e_li': {'init': 30.0, 'bounds': (0.1, 60)},
    'E_li': {'init': 0.1, 'bounds': (0.0, 1.0)},
    'Ei_li': {'init': 0.1, 'bounds': (0.0, 1.0)},
    'Ef_li': {'init': 0.1, 'bounds': (0.0, 1.0)},
    'F_p_li': {'init': 0.01, 'bounds': (0, 0.05)},
    'T_h': {'init': 1800, 'bounds': (600, 36000)},
    'Ti_h': {'init': 1800, 'bounds': (600, 36000)},
    'Tf_h': {'init': 1800, 'bounds': (600, 36000)},
    'k_e2h': {'init': 0.003, 'bounds': (0.0, 0.1)},
    'ki_e2h': {'init': 0.003, 'bounds': (0.0, 0.1)},
    'kf_e2h': {'init': 0.003, 'bounds': (0.0, 0.1)},

    # gu
    'T_gu': {'init': 30, 'bounds': (0.1, 60)},
    'D_gu': {'init': 0.85, 'bounds': (0, 1)},
    # ao
    'vol_ao': {'init': 10, 'bounds': (0.0, 1000)},
    'vol_li': {'init': 1000, 'bounds': (0, 10000)},
    'vol_ki': {'init': 300, 'bounds': (0.0, 1000)},
    'vol_lk': {'init': 150, 'bounds': (0.0, 1000)},
    'vol_rk': {'init': 150, 'bounds': (0.0, 1000)},
    # pv
    'R1b_pv': {'init': 1.25, 'bounds': (0, 5)},
    # lk / rk
    'RPF_lk': {'init': 10, 'bounds': (0, 100)},
    'RPF_rk': {'init': 10, 'bounds': (0, 100)},
    'GFR_lk': {'init': 1, 'bounds': (0, 20)},
    'GFR_rk': {'init': 1, 'bounds': (0, 20)},
    # ki
    'F_u': {'init': 0.005, 'bounds': (0, 0.05)},
    'T_u': {'init': 120, 'bounds': (0, 10 * 60)},
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