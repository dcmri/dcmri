import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import AortaPortalLiverModel
from dcmri.core.exceptions import InvalidConfiguration


def test_aorta_portal_liver():
    def _test_config(cnfg):
        try:
           model = AortaPortalLiverModel(**cnfg)
        except InvalidConfiguration:
            return
        
        print(cnfg)
        data = model.lexicon_data()
        t0 = time.perf_counter()
        results = model(data)
        elapsed = time.perf_counter() - t0
        print(f"  [Total model execution time: {elapsed:.4f}s]")
        assert results['S_l'].ndim == 3

    configs = AortaPortalLiverModel.configurations()
    cnt = 0
    for cnfg in tqdm(list(configs)):
        cnt += 1
        _test_config(cnfg)
        # if cnt==100:
        #     break

    print(f'Successfully covered {cnt} AortaLiver configurations!')


def test_aorta_portal_liver_function():
    cnfg = {
        'bolus': 'dual', 
        'heartlung': 'pfcomp', 
        'organs': '2cxm', 
        'liver': '1I-EC', 
        'non_stationary': None, 
        'water_exchange': 'F',
        't1_relaxation': 'lin',
        't2_relaxation': None, 
        't2s_relaxation': None, 
        'inflow': False,
        'sequence': 'ZTE-3D-IR-SPGR-SS', 
        'magnitude': False, 
        'calibrate': True,
    }
    try:
        model = AortaPortalLiverModel(**cnfg)
    except InvalidConfiguration:
        return
    
    print(model.inputs())
    print(model.outputs())

    data = model.lexicon_data()
    results = model(data)

    plt.plot(results['tS_l'], results['S_l'][0, 0, :], 'ro')
    plt.show()

if __name__ == '__main__':
    # test_aorta_portal_liver_function()
    test_aorta_portal_liver()
    
    print('All AortaPortalLiver model coverage tests passed!!')