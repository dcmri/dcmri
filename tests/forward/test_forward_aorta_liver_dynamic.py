import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import ForwardAortaLiverSplit
from dcmri.core.module import InvalidConfig


def test_aorta_liver_dynamic(cls=ForwardAortaLiverSplit):
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

    configs = cls.all_configs(sample=1e4, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_aorta_liver_dynamic_instance():
    # model = ForwardAortaLiverSplit()
    # print(model.config)
    # return
    cnfg = {'sequence': '3D-SPGR-SS', 'tof_corr': False, 'inflow': 'none', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'bolus': 'dual', 'heartlung': 'pfcomp', 'organs': 'comp', 'lagut': 'comp', 'liver': '1I-EC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_li': 'lin'}
    try:
        model = ForwardAortaLiverSplit(**cnfg)
    except InvalidConfig as e:
        print(e)
        return
    model.print_inputs()
    model.print_outputs()

    data = model.test_data()
    results = model(data)

    plt.plot(results['tS_1_ao'], results['S_1_ao'][0, 0, :], 'ro')
    plt.plot(results['tS_2_ao'], results['S_2_ao'][0, 0, :], 'rx')
    plt.plot(results['tS_1_li'], results['S_1_li'][0, 0, :], 'bo')
    plt.plot(results['tS_2_li'], results['S_2_li'][0, 0, :], 'bx')
    plt.show()

if __name__ == '__main__':
    test_aorta_liver_dynamic_instance()
    test_aorta_liver_dynamic()
    

    print('All AortaLiverSplit model coverage tests passed!!')