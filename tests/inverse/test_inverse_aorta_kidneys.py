import time
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri.core.module import Module, InvalidConfig
from dcmri import InverseAortaKidneys as InverseModel


def test_module(cls=Module, simple=False, io_sample=1e5, cnfg_sample=1e4, seed=51):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return

        data = instance.dummy_data()
        result = instance(data, n_bat=20, btol=1e-3)

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


def test_aorta_kidneys_inverse_instance():
    # model = InverseModel()
    # print(model.config)
    # return
    cnfg = {'inflow': 'none', 'sequence': '3D-SPGR-SS', 'tof_corr': False, 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'heartlung': 'pfcomp', 'organs': 'comp', 'kidneys': '2CF', 'bolus': 'single','t1_relaxation_ao': 'lin', 't1_relaxation_lk': 'lin', 't1_relaxation_rk': 'lin', 't2_relaxation_ao': None, 't2_relaxation_lk': None, 't2_relaxation_rk': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_lk': 'lin', 't2s_relaxation_rk': 'lin'}
    try:
        invert = InverseModel(**cnfg)
    except InvalidConfig as e:
        print(e)
        return
    
    invert.print_inputs()
    invert.print_outputs()

    # Direct from dummy data
    data = invert.dummy_data()
    result = invert(data, n_bat=20, btol=1e-3, verbose=2)

    truth = invert.forward.dummy_data()
    recon = invert.forward(truth | result['popt'])

    plt.plot(data['tS_ao'], data['S_ao'][0, 0, :], 'ro')
    plt.plot(recon['tS_ao'], recon['S_ao'][0, 0, :], 'r-')
    plt.plot(data['tS_lk'], data['S_lk'][0, 0, :], 'go')
    plt.plot(recon['tS_lk'], recon['S_lk'][0, 0, :], 'g-')
    plt.plot(data['tS_rk'], data['S_rk'][0, 0, :], 'bo')
    plt.plot(recon['tS_rk'], recon['S_rk'][0, 0, :], 'b-')
    plt.show()    

    print('Loss (%): ', result['loss'])


if __name__ == '__main__':
    test_aorta_kidneys_inverse_instance()
    test_module(InverseModel, simple=False, io_sample=1e5, cnfg_sample=1e5, seed=51)
    
    print('All aorta kidneys inverse coverage tests passed!!')