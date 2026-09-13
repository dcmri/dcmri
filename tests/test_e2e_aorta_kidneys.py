import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import AortaKidneysModel
from dcmri.core.exceptions import InvalidConfiguration


def test_aorta_kidneys(cls=AortaKidneysModel):
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

    configs = cls.all_configs(sample=1000, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    cls.print_all_io(verbose=1, simple=True)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_aorta_kidneys_function():
    # print(AortaKidneysModel().config)
    # return
    cnfg = {'inflow': False, 'sequence': '3D-SPGR-SS', 'tof_corr': False, 'magnitude': True, 'trigger': False, 'calibrate': False, 'baseline': 'literature', 'compartments': ('ki',), 'heartlung': 'pfcomp', 'organs': 'comp', 'kidneys': '2CF', 'bolus': 'single', 't1_relaxation_ao': 'lin', 't1_relaxation_lk': 'lin', 't1_relaxation_rk': 'lin', 't2_relaxation_ao': None, 't2_relaxation_lk': None, 't2_relaxation_rk': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_lk': 'lin', 't2s_relaxation_rk': 'lin'}
    try:
        model = AortaKidneysModel(**cnfg)
    except InvalidConfiguration as e:
        print(e)
        return
    
    model.print_inputs()
    model.print_outputs()

    data = model.dummy_data()
    results = model(data)

    plt.plot(results['tS_lk'], results['S_lk'][0, 0, :], 'ro')
    plt.show()

if __name__ == '__main__':
    # test_aorta_kidneys_function()
    test_aorta_kidneys()
    
    print('All AortaPortalLiver model coverage tests passed!!')