import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import ForwardTissueX
from dcmri.core.module import InvalidConfig


def test_tissue_x(cls=ForwardTissueX):
    def _test_config(cnfg):
        # if cnfg['sequence'] != '3D-SPGR-SS':
        #     return
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return

        # print(cnfg)
        data = instance.test_data()
        instance(data)

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False, sample=1e4, seed=51)

    configs = cls.all_configs(sample=1e4, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_tissue_x_instance():
    # model = ForwardTissueX()
    # print(model.config)
    # return
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'inflow': 'none', 'sequence': '3D-SPGR-SS', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'FF', 'kinetics': '2CX', 'baseline': 'literature'}
    try:
        model = ForwardTissueX(**cnfg)
    except InvalidConfig as e:
        print(e)
        return
    
    model.print_inputs()
    model.print_outputs()

    data = model.test_data()
    results = model(data)

    plt.plot(results['tS'], results['S'][0, 0, :], 'ro')
    plt.show()


if __name__ == '__main__':
    #test_tissue_x_instance()
    test_tissue_x()
    

    print('All ForwardTissueX coverage tests passed!!')