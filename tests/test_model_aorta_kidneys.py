import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import AortaKidneysModel
from dcmri.core.exceptions import InvalidConfiguration


def test_aorta_kidneys():
    def _test_config(cnfg):
        try:
           model = AortaKidneysModel(**cnfg)
        except InvalidConfiguration:
            return
        
        print(cnfg)
        data = model.lexicon_data()
        t0 = time.perf_counter()
        results = model(data)
        elapsed = time.perf_counter() - t0
        print(f"  [Total model execution time: {elapsed:.4f}s]")
        assert results['S_lk'].ndim == 3

    configs = AortaKidneysModel.configurations()
    cnt = 0
    for cnfg in tqdm(list(configs)):
        cnt += 1
        _test_config(cnfg)
        # if cnt==100:
        #     break

    print(f'Successfully covered {cnt} AortaLiver configurations!')


def test_aorta_kidneys_function():
    cnfg = {
        'bolus': 'dual', 
        'heartlung': 'pfcomp', 
        'organs': '2cxm', 
        'kidneys': '2CF', 
        'water_exchange': '(b, t, c)',
        't1_relaxation': 'lin',
        't2_relaxation': None, 
        't2s_relaxation': None, 
        'inflow': False,
        'sequence': 'ZTE-3D-IR-SPGR-SS', 
        'magnitude': False, 
        'calibrate': True,
    }
    try:
        model = AortaKidneysModel(**cnfg)
    except InvalidConfiguration:
        return
    
    print(model.inputs())
    print(model.outputs())

    data = model.lexicon_data()
    results = model(data)

    plt.plot(results['tS_lk'], results['S_lk'][0, 0, :], 'ro')
    plt.show()

if __name__ == '__main__':
    # test_aorta_kidneys_function()
    test_aorta_kidneys()
    
    print('All AortaKidneysModel coverage tests passed!!')