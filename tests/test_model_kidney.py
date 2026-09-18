import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import KidneyModel
from dcmri.core.exceptions import InvalidConfiguration


def test_kidney(cls=KidneyModel):
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

    configs = cls.all_configs(sample=1000, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_kidney_instance():
    # model = KidneyModel()
    # print(model.config)
    # return
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': None, 'inflow': 'none', 'sequence': 'ZTE-3D-IR-SPGR-SS', 'magnitude': False, 'trigger': False, 'calibrate': False, 'compartments': ('bc', 'u'), 'baseline': 'literature', 'kinetics': '2PF'}
    try:
        model = KidneyModel(**cnfg)
    except InvalidConfiguration as e:
        print(e)
        return
    model.print_inputs()
    model.print_outputs()

    data = model.dummy_data()
    results = model(data)

    plt.plot(results['tS'], results['S'][0, 0, :], 'ro')
    # plt.plot(results['tC'], results['C'][0], 'ro')

    plt.show()

if __name__ == '__main__':
    test_kidney()
    # test_kidney_instance()

    print('All KidneyModel coverage tests passed!!')