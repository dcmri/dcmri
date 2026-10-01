import time
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri.core.module import Module
from dcmri import InverseTissueLS as Inverse
from dcmri.core.module import InvalidConfig


def test_module(cls=Module, simple=False, io_sample=1e5, cnfg_sample=1e4, seed=51):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return

        data = instance.dummy_data()
        result = instance(data)

        print(np.sum(result['irf']))

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=simple, sample=io_sample, seed=seed)

    configs = cls.all_configs(sample=cnfg_sample, seed=seed)
    # [
    #     _test_config(cnfg)
    #     for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}')
    # ]

    Parallel(n_jobs=-1)(delayed(_test_config)(cnfg) for cnfg in configs)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_tissue_ls_inverse_instance():
    # model = Inverse()
    # print(model.config)
    # return
    cnfg = {'sequence': '3D-SPGR-SS', 'calibrate': True}
    try:
        invert = Inverse(**cnfg)
    except InvalidConfig as e:
        print(e)
        return
    
    invert.print_inputs()
    invert.print_outputs()

    # Direct from dummy data
    data = invert.dummy_data()
    result = invert(data, tol=1e-1)

    plt.plot(result['tC'], 1000 * result['C'][0], 'ro')
    plt.show() 

    # plt.plot(result['irf'], 'ro')
    # plt.plot(truth['irf'], 'b-')
    # plt.show()  

    #plt.plot(data['tS'], data['S'][0, 0, :], 'ro')
    # plt.plot(recon['tS'], recon['S'][0, 0, :], 'b-')
    #plt.show()    

    #print('Loss (%): ', result['loss'])


if __name__ == '__main__':
    # test_tissue_ls_inverse_instance()
    test_module(Inverse, simple=True, io_sample=1e4, cnfg_sample=1e4, seed=51)
    
    print('All inverse tissue ls model coverage tests passed!!')