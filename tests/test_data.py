import os

import numpy as np
import dcmri as dc


def test_dmr():

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
                'FA': ['Flip angle', 50, 'deg'],
                'TR': ['Repetition time', 5.4, 'msec'],
            },
            'Followup': {
                'FA': ['Flip angle', 40, 'deg'],
                'TR': ['Repetition time', 5.4, 'msec'],
            },
        },
        '002': {
            'Baseline': {
                'FA': ['Flip angle', 45, 'deg'],
                'TR': ['Repetition time', 5.4, 'msec'],
            },
            'Followup': {
                'FA': ['Flip angle', 50, 'deg'],
                'TR': ['Repetition time', 5.4, 'msec'],
            },
        },
    }
    file = os.path.join(os.getcwd(), 'test.dmr')
    dc.write_dmr(file, rois, pars, nest=True)
    rois_read, pars_read = dc.read_dmr(file, nest=True)

    assert np.array_equal(
        rois_read['001']['Baseline']['signal'],
        [5, 6, 7, 8],
    )
    assert np.array_equal(
        rois_read['001']['Followup']['signal'],
        [12, 13, 14],
    )

    # Cleanup
    os.remove(file + '.zip')



if __name__ == "__main__":

    test_dmr()

    print('All data tests passed!!')