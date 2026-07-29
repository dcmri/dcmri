import itertools
import numpy as np

from dcmri import Magnetization, QVALUES
from dcmri.bloch.modules_tissue import MzPrep, MxyReadMz



def test_coverage_readout():
    values = MxyReadMz.configs.values()
    for cnfgs in itertools.product(*values):
        print('Mxy', cnfgs)
        config = {k: cnfgs[i] for i, k in enumerate(MxyReadMz.configs)}
        read = MxyReadMz(**config)
        read.inputs()
        read.outputs()
        read(QVALUES | {'tacq': np.ones(3), 'Mz': np.ones((2,3)), 'tR': np.arange(5), 'R2':np.ones((2,5)), 'R2s':np.ones(5)})


def test_exceptions_readout():

    try:
        MxyReadMz('2D-GE-EPI')({'Mz':1})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('2D-SE-EPI')({'Mz':1})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('2D-GE-EPI')(QVALUES | {'Mz': np.ones((2,3))})
    except:
        pass
    else:
        assert False
    try:
        MxyReadMz('2D-GE-EPI')(QVALUES | {'Mz': np.ones((2,3)), 'R2s':np.ones(4)})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('2D-SE-EPI')(QVALUES | {'Mz': np.ones((2,3))})
    except:
        pass
    else:
        assert False
    try:
        MxyReadMz('2D-SE-EPI')(QVALUES | {'Mz': np.ones((2,3)), 'R2': np.ones(4)})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('2D-DE-EPI')(QVALUES | {'Mz': np.ones(3), 'R2': np.ones(3), 'R2s': np.ones(2)})
    except:
        pass
    else:
        assert False

def test_coverage_mzprep():

    # nc, nt
    nt = 10
    tR = np.arange(nt)
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = np.stack([np.full(nt, 0.6), np.full(nt, 0.8)])
    Fi = [0.008, 0.005]
    me = 2

    values = MzPrep.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(MzPrep.configs)}
        print('Mz', cnfgs)
        Mz = MzPrep(**config)
        if 'R1' in Mz.inputs():
            Mz(QVALUES, tR=tR, R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
        
    MzPrep(sequence='3D-SPGR-SS')(QVALUES, tR=tR, R1=R1, R1i=R1, Fi=Fi, v=v, Fw=Fw, me=me)



def test_exceptions_mzprep():
    try:
        MzPrep(sequence='3D-SPGR-SS')()
    except:
        pass
    else:
        assert False

    try:
        MzPrep(sequence='3D-SPGR-SS')(QVALUES, R1=np.ones((2, 3)), v=[0.5,0.5], Fw=np.ones((3,3)))
    except:
        pass
    else:
        assert False

    # try:
    #     MzPrep('3D-SPGR-SS', defaults=QVALUES)(R1=np.ones((2, 3)), R1i=np.ones((2, 3)), v=[0.5,0.5], Fw=np.ones((2,2)))
    # except:
    #     pass
    # else:
    #     assert False

    # try:
    #     MzPrep('3D-SPGR-SS', defaults=QVALUES)(R1=np.ones((2, 3)), v=[0.5,0.5], Fw=np.ones((2,2)), Fi=np.ones(3), R1i=np.ones((2, 3)))
    # except:
    #     pass
    # else:
    #     assert False

    try:
        MzPrep(sequence='3D-SPGR-SS')(QVALUES, R1=np.ones(3), v=[0.5, 0.5])
    except:
        pass
    else:
        assert False

    try:
        MzPrep(sequence='3D-SPGR-SS')(QVALUES, R1=np.ones((3, 3)), v=[0.5, 0.5])
    except:
        pass
    else:
        assert False

    try:
        MzPrep(sequence='3D-SPGR-SS', inflow=True)(QVALUES, R1i=[1,1])
    except:
        pass
    else:
        assert False

    try:
        MzPrep(sequence='3D-SPGR-SS', inflow=True)(QVALUES, R1i=1, Fi=np.ones((2,2)))
    except:
        pass
    else:
        assert False


def test_coverage_m():
    # nc, nt
    nt = 10
    tR = np.arange(nt)
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    R2 = R1
    R2s = np.full(nt, 1)
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = np.stack([np.full(nt, 0.6), np.full(nt, 0.8)])
    Fi = [0.008, 0.005]
    me = 2

    values = Magnetization.configs.values()
    for cnfgs in itertools.product(*values):
        print('M (nc, nt)', cnfgs)
        # if cnfgs != ('Eq-SE-EPI', False):
        #     continue
        config = {k: cnfgs[i] for i, k in enumerate(Magnetization.configs)}
        magn = Magnetization(**config)
        magn.inputs()
        magn.outputs()
        magn(QVALUES, tR=tR, R1=R1, R2=R2, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # scalar
    tR = 0
    R1 = 1
    R2 = 1
    R2s = 1
    v = 0.1
    Fw = 0.01
    R1i = 0.6
    Fi = 0.008
    me = 2

    values = Magnetization.configs.values()
    for cnfgs in itertools.product(*values):
        print('M scalar', cnfgs)
        # if cnfgs != ('Eq-SE-EPI', False):
        #     continue
        config = {k: cnfgs[i] for i, k in enumerate(Magnetization.configs)}
        magn = Magnetization(**config)
        magn.inputs()
        magn.outputs()
        magn(QVALUES, tR=tR, R1=R1, R2=R2, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)


if __name__ == "__main__":
    test_coverage_readout()
    test_exceptions_readout()
    test_coverage_mzprep()
    test_exceptions_mzprep()
    test_coverage_m()
    
    print('All magnetization tests passing!')