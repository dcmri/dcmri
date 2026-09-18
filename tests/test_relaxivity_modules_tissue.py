from tqdm import tqdm

import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration
from dcmri.core.module import Module

def _test_class(cls: Module):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return
        data = instance.dummy_data(nc=2)
        instance(data)
        if 't2s_relaxation' in cnfg and cnfg['t2s_relaxation'] == 'leakage':
            return
        data = instance.dummy_data()
        instance(data)

    cls.print_configs()
    cls.print_all_io(verbose=0, simple=False)

    configs = cls.all_configs()
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_relax():
    for cls in [
        dc.R1,
        dc.R2,
        dc.R2s,
        dc.Relax,
        dc.ConcToRelax,
    ]:
        _test_class(cls)



if __name__ == "__main__":
    test_relax()
    
    print('All relaxivity models tests passing!')