import numpy as np
from tqdm import tqdm

from dcmri.core.module import Module
from dcmri.core.module import InvalidConfig
from dcmri import Signal
from dcmri.signal.modules_tissue import signal_rice, RelaxToSignal, ConcToSignal 



def test_signal_rice():
    """Test signal_rice for clean math path and its fallback logic."""
    # Case 1: Zero noise standard deviation returns input directly
    res_zero_sigma = signal_rice(np.array([10.0]), sigma=0.0)
    np.testing.assert_array_almost_equal(res_zero_sigma, np.array([10.0]))
    
    # Case 2: Standard math path
    res_standard = signal_rice(np.array([2.0, 5.0]), sigma=1.0)
    assert res_standard.shape == (2,)
   
    # Case 3: Extreme inputs trigger the fallback logic cleanly
    res_fallback = signal_rice(np.array([1e10]), sigma=1e-10)
    np.testing.assert_array_almost_equal(res_fallback, np.array([1e10]))


def _test_class(cls: Module):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return
        
        data = instance.dummy_data(nc=3)
        instance(data)
        if 't2s_relaxation' in cnfg and cnfg['t2s_relaxation'] == 'leakage':
            return
        data = instance.dummy_data()
        instance(data)
        data = instance.dummy_data(nc=1)
        instance(data)

    cls.print_configs()
    cls.print_all_io(simple=False)

    configs = cls.all_configs()
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def _test_signal_class(cls: Module):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfig:
            return
        data = instance.dummy_data()
        instance(data)

    cls.print_configs()
    cls.print_all_io(simple=False)

    configs = cls.all_configs()
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_signal():
    _test_signal_class(Signal)
    for cls in [
        RelaxToSignal,
        ConcToSignal,
    ]:
        _test_class(cls)


if __name__ == "__main__":
    test_signal_rice()
    test_signal()
    
    print('All signal tests passing!')