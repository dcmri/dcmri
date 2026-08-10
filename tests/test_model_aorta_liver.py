import time
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration


def test_aorta_liver():
    def _test_config(cnfg):
        # if cnfg != {'bolus': 'dual', 'heartlung': 'comp', 'organs': 'comp', 't2s_relaxation': None, 'sequence': 'Eq-SE-EPI', 'magnitude': False}:
        #     return
        try:
           model = dc.AortaLiverModel(**cnfg)
        except InvalidConfiguration:
            return
        print(cnfg)
        data = dc.QVALUES | model.map_lexicon(dc.QVALUES)
        t0 = time.perf_counter()
        results = model(data)
        elapsed = time.perf_counter() - t0
        print(f"  [Total model execution time: {elapsed:.4f}s]")
        assert results['S_l'].ndim == 3

    cnt = 0
    for cnfg in dc.AortaLiverModel.configurations():
        cnt += 1
        _test_config(cnfg)
        # if cnt==100:
        #     break

    print(f'Successfully covered {cnt} aorta configurations!')


def test_aorta_liver_function():
    cnfg = {
        'bolus': 'single', 
        'heartlung': 'chain', 
        'organs': 'comp', 
        'lagut': 'pass', 
        'liver': '1I-IC', 
        'non_stationary': None, 
        't2s_relaxation': None, 
        'sequence': 'ZTE-3D-SPGR-SS', 
        'magnitude': False, 
        'calibrate': True,
    }
    try:
        model = dc.AortaLiverModel(**cnfg)
    except InvalidConfiguration:
        return
    print(model.inputs())
    print(model.outputs())
    data = dc.QVALUES | model.map_lexicon(dc.QVALUES)
    results = model(data)
    plt.plot(results['tS_l'], results['S_l'][0, 0, :], 'ro')
    plt.show()

if __name__ == '__main__':
    test_aorta_liver()
    #test_aorta_liver_function()

    print('All AortaLiver model coverage tests passed!!')