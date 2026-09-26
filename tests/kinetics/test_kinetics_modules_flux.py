from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import dcmri as dc

from dcmri.core.module import InvalidConfig


def _test_class(cls):
    cls.print_all_inputs()
    cls.print_all_outputs()

    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return
        data = instance.dummy_data()
        instance(data)

    configs = cls.all_configs()
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_flux():
    for cls in [
        dc.Flux,
        dc.FluxInjection,
        dc.FluxTissueX,
        dc.FluxAorta,
    ]:
        _test_class(cls)


def test_flux_aorta():
    cnfg = {'heartlung': 'pfcomp', 'organs': 'comp', 'kidneys': 'pass', 'liver': None, 'lagut': None, 'bolus': 'single'}
    instance = dc.FluxAorta(**cnfg)
    data = instance.dummy_data()
    results = instance(data)
    print(instance.config)

    plt.plot(results['tC'], results['J_ao'], 'ro')
    plt.show()   


if __name__ == '__main__':
    # test_flux()
    test_flux_aorta()
    print('All flux tests passed!!')