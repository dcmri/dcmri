from tqdm import tqdm
import dcmri as dc

from dcmri.core.module import Module
from dcmri.core.module import InvalidConfig


def _test_class(cls:Module):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return
        data = instance.dummy_data()
        instance(data)

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False)

    configs = cls.all_configs(sample=None, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_conc():
    for cls in [
        dc.Conc,
        dc.ConcAorta,
        dc.ConcLiver,
        dc.ConcAortaLiver,
        dc.ConcAortaPortalLiver,
        dc.ConcKidney,
        dc.ConcAortaKidneys,
        dc.ConcCortMed,
        dc.ConcTissueX,
        dc.ConcTissueLS,
    ]:
        _test_class(cls)


if __name__ == '__main__':
    test_conc()
    print('All conc tests passed!!')