from tqdm import tqdm

import numpy as np
import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration


def _test_class(cls):
    cls.print_all_inputs()
    cls.print_all_outputs()

    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return
        data = instance.dummy_data()
        instance(data)

    configs =cls.all_configs()
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


if __name__ == '__main__':
    test_flux()
    print('All flux tests passed!!')