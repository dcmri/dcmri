import os

import numpy as np
import dcmri as dc


def test_dmr():

    data_dict = {
        'time': ['Acquisition times', 'sec', 'float'],
        'signal': ['Acquired signals', 'sec', 'float'],
        'FA': ['Flip angle', 'deg', 'float'],
        'TR': ['Repetition time', 'msec', 'float'],
        'Checked': ['Have the data been checked', '', 'bool'],
        'Checker': ['Who has checked the data', '', 'str'],
        'n0': ['Baseline length', '', 'int'],
    }
    rois = {
        '001': {
            'Baseline': {
                'time': [1,2,3,4],
                'signal': [5,6,7,8],
            },
            'Followup': {
                'time': [9,10,11],
                'signal':[12,13,14],
            },
        },
        '002': {
            'Baseline': {
                'time': [10,20,30,40],
                'signal':[50,60,70,80],
            },
            'Followup': {
                'time': [90,100,110],
                'signal':[120,130,140],
            },
        },
    }
    pars = {
        '001': {
            'Baseline': {
                'FA': 50,
                'TR': 5.4,
            },
            'Followup': {
                'FA': 40,
                'TR': 5.4,
            },
        },
        '002': {
            'Baseline': {
                'FA': 45,
                'TR': 5.4,
                'Checked': True,
                'Checker': 'John Doe',
            },
            'Followup': {
                'FA': 50,
                'TR': 5.4,
                'n0': 10,
                'Checked': False,
            },
        },
    }

    file = os.path.join(os.getcwd(), 'test.dmr')
    dc.write_dmr(file, data_dict, rois, pars, nest=True)

    dict_read, rois_read, pars_read = dc.read_dmr(file, nest=True)

    assert np.array_equal(
        dict_read['FA'],
        ['Flip angle', 'deg', 'float'],
    )
    assert np.array_equal(
        rois_read['001']['Baseline']['signal'],
        [5, 6, 7, 8],
    )
    assert np.array_equal(
        rois_read['001']['Followup']['signal'],
        [12, 13, 14],
    )
    assert pars_read['001']['Followup']['FA'] == 40
    assert pars_read['002']['Followup']['FA'] == 50
    assert pars_read['002']['Baseline']['Checked'] is True
    assert pars_read['002']['Followup']['n0'] == 10
    assert pars_read['002']['Followup']['Checked'] is False

    # Cleanup
    os.remove(file + '.zip')



if __name__ == "__main__":

    test_dmr()

    print('All data tests passed!!')