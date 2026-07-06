# TODO: Tissue parameters need to be in prep and read rather than a separate entry.


SEQUENCES = {
    'ZTE-3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1'],
            'read': ['FA', 'B1corr'],
        },
    },
    'ZTE-3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1'],
            'read': ['FA', 'B1corr'],
        },
    },
    '3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-SR-SPGR-SS': {
        'mz_prep_tissue': 'SR-SPGR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-PR-SPGR-SS': {
        'mz_prep_tissue': 'PR-SPGR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '2D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '2D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    }, # Non-selective preparation, so inflow is freely recovering
    '2D-SR-SPGR-SS': {
        'mz_prep_tissue': 'SR-SPGR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '2D-PR-SPGR-SS': {
        'mz_prep_tissue': 'PR-SPGR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-IR-SPGR': {
        'mz_prep_tissue': 'IR-SPGR',
        'mz_prep_inflow': 'IR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SPGR',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-PR-SPGR': {
        'mz_prep_tissue': 'PR-SPGR',
        'mz_prep_inflow': 'PR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-PR-SS': {
        'mz_prep_tissue': 'PR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-IR-SS': {
        'mz_prep_tissue': 'IR-SS',
        'mz_prep_inflow': 'IR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-SR-SS': {
        'mz_prep_tissue': 'SR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '2D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'Eq',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '2D-IR-SPGR': {
        'mz_prep_tissue': 'IR-SPGR', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '2D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '2D-PR-SPGR': {
        'mz_prep_tissue': 'PR-SPGR',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    '3D-SPGR-SSI': {
        'mz_prep_tissue': 'SSI',
        'mz_prep_inflow': 'SSI',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    'GE-EPI': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    'SE-EPI': {
        'mz_prep_tissue': 'SE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    'DE-EPI': {
        'mz_prep_tissue': 'DE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2', 'R2s'],
            'read': ['FA', 'B1corr', 'TE1', 'TE2'],
        },
    },
    'Eq-GE-EPI': {
        'mz_prep_tissue': 'Eq',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R2s'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    'Eq-SE-EPI': {
        'mz_prep_tissue': 'Eq',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R2'],
            'read': ['FA', 'B1corr', 'TE'],
        },
    },
    'Eq-DE-EPI': {
        'mz_prep_tissue': 'Eq',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R2', 'R2s'],
            'read': ['FA', 'B1corr', 'TE1', 'TE2'],
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