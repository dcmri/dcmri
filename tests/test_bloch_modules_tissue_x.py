import numpy as np
import itertools
import time

import dcmri as dc
from dcmri.bloch.modules_tissue_x import MzPrepTissueX


def test_coverage_mzprep():
    values = MzPrepTissueX.configs.values()
    for cnfgs in itertools.product(*values):
        print(cnfgs)
        config = {k: cnfgs[i] for i, k in enumerate(MzPrepTissueX.configs)}
        mz = MzPrepTissueX(**config)
        if 'R1' in mz.inputs():
            mz.inputs()
            mz.outputs()
            p = dc.QVALUES
            v = dc.WaterVolumesTissueX(**config)(p)
            R1 = np.ones(v['v'].size)
            R1a = 1
            mz(p, R1=R1, R1a=R1a) 
            R1 = np.ones((v['v'].size, 5))
            R1a = np.ones(5)
            mz(p, R1=R1, R1a=R1a)

def test_coverage_mz():
    values = dc.MagnetizationTissueX.configs.values()
    for cnfgs in itertools.product(*values):
        print(cnfgs)
        config = {k: cnfgs[i] for i, k in enumerate(dc.MagnetizationTissueX.configs)}
        m = dc.MagnetizationTissueX(**config)
        m.inputs()
        m.outputs()
        p = dc.QVALUES
        v = dc.WaterVolumesTissueX(**config)(p)
        R1 = np.ones(v['v'].size)
        R2 = np.ones(v['v'].size)
        R1a = 1
        m(p, R1=R1, R2=R2, R1a=R1a) 
        R1 = np.ones((v['v'].size, 5))
        R2 = np.ones((v['v'].size, 5))
        R1a = np.ones(5)
        R2s = np.ones(5)
        m(p, R1=R1, R1a=R1a, R2=R2, R2s=R2s) 





if __name__ == "__main__":
    test_coverage_mzprep()
    test_coverage_mz()
    
    print('All tissue tests passing!')