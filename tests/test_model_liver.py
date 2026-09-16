import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import LiverModel
from dcmri.core.exceptions import InvalidConfiguration


def test_liver(cls=LiverModel):
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
    cls.print_all_io(verbose=1, simple=False, sample=1e3, seed=51)

    configs = cls.all_configs(sample=1e3, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_liver_instance():
    # model = LiverModel()
    # print(model.config)
    # return
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': None, 'inflow': True, 'sequence': 'ZTE-3D-SPGR-SS', 'magnitude': False, 'trigger': True, 'calibrate': False, 'water_exchange': 'N', 'baseline': 'measured', 'kinetics': '1I-IC', 'non_stationary': None}
    try:
        model = LiverModel(**cnfg)
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
    #test_liver()
    test_liver_instance()

    print('All LiverModel coverage tests passed!!')