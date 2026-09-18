from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

from dcmri.bloch.modules_tissue import MzPrep, MxyReadMz, Magnetization
from dcmri.core.exceptions import InvalidConfiguration


def _test_class(cls):
    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return
        # if cnfg != {'sequence': '3D-PR-SS', 'tof_corr': False, 'inflow': 'none'}:
        #     return
        # print(cnfg)
        data = instance.dummy_data(nc=2)
        instance(data)

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False)

    configs = cls.all_configs()
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_bloch():
    for cls in [
        MzPrep,
        MxyReadMz,
        Magnetization,
    ]:
        _test_class(cls)




def test_mzprep_exceptions():
    try:
        mz = MzPrep(sequence='3D-SPGR-SS')
        data = mz.dummy_data(nc=2)
        mz(data, Kw=np.ones((3,3)))
    except:
        pass
    else:
        assert False

    try:
        mz = MzPrep(sequence='3D-SPGR-SS')
        data = mz.dummy_data(nc=2)
        mz(data, R1=np.ones(3))
    except:
        pass
    else:
        assert False

    try:
        mz = MzPrep(sequence='3D-SPGR-SS', inflow='pool')
        data = mz.dummy_data(nc=2)
        mz(data, R1i=np.ones(3))
    except:
        pass
    else:
        assert False

    try:
        mz = MzPrep(sequence='3D-SPGR-SS', inflow='pool')
        data = mz.dummy_data(nc=2)
        mz(data, Fwi=np.ones(3))
    except:
        pass
    else:
        assert False



def test_mzprep_function():
    config = {'sequence': '3D-IR-SPGR', 'inflow': 'none'}
    mz = MzPrep(**config) # data = mz.dummy_data()
    nR = 50
    dt = 0.1
    data = mz.dummy_data() | {
        # Relaxation rates
        'tR': dt * np.arange(nR),
        'R1i': 0.65 * np.ones(nR),
        'R1': 0.65 * np.ones(nR),
        # Seq params
        'FA': 15,
        'TR': 0.005,
        'TD': 0.5,
        'TP': 0.001,
        'Nph': 128,
        # Tissue props
        'B1corr': 1,
        'vw': 1,
        'me': 1,
        'Fwi': 10,
        'Kw': 10,
    }
    result = mz(data)

    plt.plot(result['tMz'].flatten(), result['Mz'].flatten())
    plt.show()


if __name__ == "__main__":
    test_bloch()
    test_mzprep_exceptions()
    test_mzprep_function()

    print('All magnetization tests passing!')