import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import ForwardLiver
from dcmri.core.module import InvalidConfig


def test_liver(cls=ForwardLiver):
    def _test_config(cnfg):
        # if cnfg['sequence'] != '3D-SPGR-SS':
        #     return
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return
    
        data = instance.test_data()
        instance(data)

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False, sample=1e4, seed=51)

    configs = cls.all_configs(sample=1e5, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_liver_instance():
    # model = ForwardLiver()
    # print(model.config)
    # return
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': None, 'inflow': 'pool', 'sequence': 'ZTE-3D-SPGR-SS', 'magnitude': False, 'trigger': True, 'calibrate': False, 'water_exchange': 'N', 'baseline': 'measured', 'kinetics': '1I-IC', 'non_stationary': None}
    try:
        model = ForwardLiver(**cnfg)
    except InvalidConfig as e:
        print(e)
        return
    model.print_inputs()
    model.print_outputs()

    data = model.test_data()
    results = model(data)

    plt.plot(results['tS_li'], results['S_li'][0, 0, :], 'ro')
    # plt.plot(results['tC'], results['C'][0], 'ro')

    plt.show()


if __name__ == '__main__':
    # test_liver_instance()
    test_liver()
    
    print('All ForwardLiver coverage tests passed!!')