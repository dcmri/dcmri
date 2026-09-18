import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import AortaPortalLiverModel
from dcmri.core.exceptions import InvalidConfiguration


def test_aorta_portal_liver(cls=AortaPortalLiverModel):
    def _test_config(cnfg):
        # if cnfg['sequence'] != '3D-SPGR-SS':
        #     return
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return
    
        data = instance.dummy_data()

        # --- DIAGNOSTIC TIMING ---
        t0 = time.perf_counter()
        # print(cnfg)
        instance(data)

        elapsed = time.perf_counter() - t0
        # print(cnfg)
        # print(f"  [Total model execution time: {elapsed:.4f}s]")

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False, sample=1e4, seed=51)

    configs = cls.all_configs(sample=1e4, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_aorta_portal_liver_instance():
    cnfg = {
        'bolus': 'dual', 
        'heartlung': 'pfcomp', 
        'organs': '2cxm', 
        'liver': '1I-EC', 
        'non_stationary': None, 
        'water_exchange': 'F',
        't1_relaxation_ao': 'lin',
        't2_relaxation_ao': None, 
        't2s_relaxation_ao': None, 
        't1_relaxation_pv': 'lin',
        't2_relaxation_pv': None, 
        't2s_relaxation_pv': None, 
        't1_relaxation_li': 'lin',
        't2_relaxation_li': None, 
        't2s_relaxation_li': None, 
        'inflow': 'none',
        'sequence': 'ZTE-3D-IR-SPGR-SS', 
        'magnitude': False, 
        'calibrate': True,
    }
    try:
        model = AortaPortalLiverModel(**cnfg)
    except InvalidConfiguration as e:
        print(e)
        return
    
    model.print_inputs()
    model.print_outputs()

    data = model.dummy_data()
    results = model(data)

    plt.plot(results['tS_ao'], results['S_ao'][0, 0, :], 'ro')
    plt.plot(results['tS_pv'], results['S_pv'][0, 0, :], 'gx')
    plt.plot(results['tS_li'], results['S_li'][0, 0, :], 'bx')
    plt.show()

if __name__ == '__main__':
    # test_aorta_portal_liver_instance()
    test_aorta_portal_liver()
    
    print('All AortaPortalLiver model coverage tests passed!!')