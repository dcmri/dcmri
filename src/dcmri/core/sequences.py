
SEQUENCES = {
    'ZTE-3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1'],
            'read': ['FA', 'B1corr', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph'],
        },
    },
    'ZTE-3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1'],
            'read': ['FA', 'B1corr', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TP', 'TD'],
        },
    },
    '3D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'SPGR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph'],
        },
    },
    '3D-IR-SPGR-SS': {
        'mz_prep_tissue': 'IR-SPGR-SS', 
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TP', 'TD'],
        },
    },
    '3D-SR-SPGR-SS': {
        'mz_prep_tissue': 'SR-SPGR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TP', 'TD'],
        },
    },
    '3D-PR-SPGR-SS': {
        'mz_prep_tissue': 'PR-SPGR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TP', 'PA', 'TD'],
        },
    },
    '2D-SPGR-SS': {
        'mz_prep_tissue': 'SPGR-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph'],
        },
    },
    '3D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph'],
        },
    },
    '3D-IR-SPGR': {
        'mz_prep_tissue': 'IR-SPGR',
        'mz_prep_inflow': 'IR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TP', 'TD'],
        },
    },
    '3D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SPGR',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TP', 'TD'],
        },
    },
    '3D-PR-SPGR': {
        'mz_prep_tissue': 'PR-SPGR',
        'mz_prep_inflow': 'PR-SPGR',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'PA', 'TP', 'TD'],
        },
    },
    '3D-PR-SS': {
        'mz_prep_tissue': 'PR-SS',
        'mz_prep_inflow': 'PR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
            'prep': ['TA', 'PA'],
        },
    },
    '3D-IR-SS': {
        'mz_prep_tissue': 'IR-SS',
        'mz_prep_inflow': 'IR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
            'prep': ['TA'],
        },
    },
    '3D-SR-SS': {
        'mz_prep_tissue': 'SR-SS',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
            'prep': ['TA'],
        },
    },
    '2D-SPGR': {
        'mz_prep_tissue': 'SPGR',
        'mz_prep_inflow': 'Eq',
        'steady-state': False,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph'],
        },
    },
    '2D-SR-SPGR': {
        'mz_prep_tissue': 'SR-SPGR',
        'mz_prep_inflow': 'SR-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TP', 'TD', 'iz'],
        },
    },
    '3D-SPGR-SSI': {
        'mz_prep_tissue': 'SSI',
        'mz_prep_inflow': 'SSI',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE', 'Nk0'],
            'prep': ['TR', 'FA', 'B1corr', 'Nph', 'TF', 'SA'],
        },
    },
    '2D-GE-EPI': {
        'mz_prep_tissue': 'GE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
            'prep': ['TR', 'FA', 'B1corr', 'Nz', 'iz'],
        },
    },
    '2D-SE-EPI': {
        'mz_prep_tissue': 'SE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2'],
            'read': ['FA', 'B1corr', 'TE'],
            'prep': ['TE', 'TR', 'FA', 'B1corr', 'Nz', 'iz'],
        },
    },
    '2D-DE-EPI': {
        'mz_prep_tissue': 'DE-SS',
        'mz_prep_inflow': 'Eq',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2', 'R2s'],
            'read': ['FA', 'B1corr', 'TE1', 'TE2'],
            'prep': ['TE2', 'TR', 'FA', 'B1corr', 'Nz', 'iz'],
        },
    },
    '3D-GE-EPI': {
        'mz_prep_tissue': 'GE-SS',
        'mz_prep_inflow': 'GE-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2s'],
            'read': ['FA', 'B1corr', 'TE'],
            'prep': ['TR', 'FA', 'B1corr'],
        },
    },
    '3D-SE-EPI': {
        'mz_prep_tissue': 'SE-SS',
        'mz_prep_inflow': 'SE-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2'],
            'read': ['FA', 'B1corr', 'TE'],
            'prep': ['TE', 'TR', 'FA', 'B1corr'],
        },
    },
    '3D-DE-EPI': {
        'mz_prep_tissue': 'DE-SS',
        'mz_prep_inflow': 'DE-SS',
        'steady-state': True,
        'parameters': {
            'tissue': ['R1', 'R2', 'R2s'],
            'read': ['FA', 'B1corr', 'TE1', 'TE2'],
            'prep': ['TE2', 'TR', 'FA', 'B1corr'],
        },
    },
}
