import time
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri.core.module import Module
from dcmri import InverseAortaLiverSplit as InverseModel
from dcmri.core.module import InvalidConfig


def test_module(cls=Module, simple=False, io_sample=1e5, cnfg_sample=1e4, seed=51):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return

        data = instance.dummy_data()
        result = instance(data, n_bat=5)

        print(result['loss'])

        assert result['loss'] < 1e-3, f"Loss for config {cnfg} is {result['loss']}"

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=simple, sample=io_sample, seed=seed)

    configs = cls.all_configs(sample=cnfg_sample, seed=seed)
    # [
    #     _test_config(cnfg)
    #     for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}')
    # ]

    Parallel(n_jobs=-1)(
        delayed(_test_config)(cnfg) for cnfg in configs
    )

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_aorta_liver_dynamic_inverse_instance():
    # model = InverseModel()
    # print(model.config)
    # return
    cnfg = {'sequence': '2D-SR-SPGR', 'tof_corr': False, 'inflow': 'none', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'N', 'baseline': 'measured', 'bolus': 'dual', 'heartlung': 'pfcomp', 'organs': '2cxm', 'lagut': 'comp', 'liver': '1I-IC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_li': 'quad'}
    try:
        invert = InverseModel(**cnfg)
    except InvalidConfig as e:
        print(e)
        return
    
    invert.print_inputs()
    invert.print_outputs()

    # Direct from dummy data
    data = invert.dummy_data()
    result = invert(data, n_bat=5, verbose=2)
    # result = invert(data, verbose=2)

    truth = invert.forward.dummy_data()
    recon = invert.forward(truth | result['popt'])

    plt.plot(data['tS_1_ao'], data['S_1_ao'][0, 0, :], 'ro')
    plt.plot(recon['tS_1_ao'], recon['S_1_ao'][0, 0, :], 'r-')
    plt.plot(data['tS_2_ao'], data['S_2_ao'][0, 0, :], 'ro')
    plt.plot(recon['tS_2_ao'], recon['S_2_ao'][0, 0, :], 'r-')

    plt.plot(data['tS_1_li'], data['S_1_li'][0, 0, :], 'bo')
    plt.plot(recon['tS_1_li'], recon['S_1_li'][0, 0, :], 'b-')
    plt.plot(data['tS_2_li'], data['S_2_li'][0, 0, :], 'bo')
    plt.plot(recon['tS_2_li'], recon['S_2_li'][0, 0, :], 'b-')
    
    plt.show()    

    print('Loss (%): ', result['loss'])


if __name__ == '__main__':
    test_aorta_liver_dynamic_inverse_instance()
    test_module(InverseModel, simple=False, io_sample=1e5, cnfg_sample=1e4, seed=51)
    
    print('All aorta liver dynamic inverse coverage tests passed!!')