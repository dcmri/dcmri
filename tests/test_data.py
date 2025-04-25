import os

import numpy as np
import dcmri as dc


def test_dmr():

    data = {
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

    dmr = {'data':data, 'pars':pars, 'rois':rois}
    file = os.path.join(os.getcwd(), 'test.dmr')
    dc.write_dmr(file, dmr, 'nest')

    dmr = dc.read_dmr(file, 'nest')

    assert np.array_equal(
        dmr['data']['FA'],
        ['Flip angle', 'deg', 'float'],
    )
    assert np.array_equal(
        dmr['rois']['001']['Baseline']['signal'],
        [5, 6, 7, 8],
    )
    assert np.array_equal(
        dmr['rois']['001']['Followup']['signal'],
        [12, 13, 14],
    )
    assert dmr['pars']['001']['Followup']['FA'] == 40
    assert dmr['pars']['002']['Followup']['FA'] == 50
    assert dmr['pars']['002']['Baseline']['Checked'] is True
    assert dmr['pars']['002']['Followup']['n0'] == 10
    assert dmr['pars']['002']['Followup']['Checked'] is False

    # Cleanup
    os.remove(file + '.zip')



if __name__ == "__main__":

    test_dmr()

    print('All data tests passed!!')