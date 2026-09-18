import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import CortMedModel
from dcmri.core.exceptions import InvalidConfiguration


def test_cort_med(cls=CortMedModel):
    def _test_config(cnfg):
        # if cnfg['sequence'] != '3D-SPGR-SS':
        #     return
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return

        # print(cnfg)
        data = instance.dummy_data()

        # --- DIAGNOSTIC TIMING ---
        t0 = time.perf_counter()
        
        instance(data)

        elapsed = time.perf_counter() - t0
        
        # print(f"  [Total model execution time: {elapsed:.4f}s]")

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False, sample=None, seed=51)

    configs = cls.all_configs(sample=None, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_cort_med_instance():
    # model = CortMedModel()
    # print(model.config)
    # return
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'sequence': '3D-SPGR-SS', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'kinetics': '7C'}
    try:
        model = CortMedModel(**cnfg)
    except InvalidConfiguration as e:
        print(e)
        return
    model.print_inputs()
    model.print_outputs()

    data = model.dummy_data()
    results = model(data)

    plt.plot(results['tS_kc'], results['S_kc'][0, 0, :], 'ro')
    plt.plot(results['tS_km'], results['S_km'][0, 0, :], 'bo')
    # plt.plot(results['tC'], results['C'][0], 'ro')

    plt.show()


if __name__ == '__main__':
    test_cort_med()
    # test_cort_med_instance()

    print('All CortMedModel coverage tests passed!!')