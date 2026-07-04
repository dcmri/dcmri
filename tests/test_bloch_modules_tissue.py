import itertools
import numpy as np

from dcmri import Magnetization, QVALUES
from dcmri.bloch.modules_tissue import MzPrep, MxyReadMz

def test_coverage_readout():
    values = MxyReadMz.configs.values()
    for cnfgs in itertools.product(*values):
        print(cnfgs)
        config = {k: cnfgs[i] for i, k in enumerate(MxyReadMz.configs)}
        read = MxyReadMz(**config)
        read.inputs()
        read.outputs()
        p = QVALUES
        # 1c
        read(p | {'Mz': 1, 'R2':1, 'R2s':1}) 
        # 1c array
        read(p | {'Mz': [1], 'R2':[1], 'R2s':[1]})
        # 2c array
        read(p | {'Mz': [1,1], 'R2':[1,1], 'R2s':[1,1]})
        # 2c + time
        read(p | {'Mz': np.ones((2,3)), 'R2':np.ones((2,3)), 'R2s':np.ones((2,3))})


def test_exceptions_readout():

    try:
        MxyReadMz('GE-EPI')({'Mz':1})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('SE-EPI')({'Mz':1})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('GE-EPI')(QVALUES | {'Mz': np.ones((2,3))})
    except:
        pass
    else:
        assert False
    try:
        MxyReadMz('GE-EPI')(QVALUES | {'Mz': np.ones((2,3)), 'R2s':np.ones(4)})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('SE-EPI')(QVALUES | {'Mz': np.ones((2,3))})
    except:
        pass
    else:
        assert False
    try:
        MxyReadMz('SE-EPI')(QVALUES | {'Mz': np.ones((2,3)), 'R2': np.ones(4)})
    except:
        pass
    else:
        assert False

    try:
        MxyReadMz('DE-EPI')(QVALUES | {'Mz': np.ones(3), 'R2': np.ones(3), 'R2s': np.ones(2)})
    except:
        pass
    else:
        assert False

def test_coverage_mzprep():

    # scalar
    R1 = 1
    v = 0.3
    Fw = 0.01
    R1i = 0.75
    Fi = 0.008
    me = 2
    values = MzPrep.configs.values()
    for cnfgs in itertools.product(*values):
        # if cnfgs != ('2D-SPGR-SS', True):
        #     continue
        print('scalar', cnfgs)
        config = {k: cnfgs[i] for i, k in enumerate(MzPrep.configs)}
        Mz = MzPrep(**config)
        if 'R1' in Mz.inputs():
            if config['inflow']:
                Mz(QVALUES, R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
                Mz(QVALUES, R1=[R1], R1i=[R1i], Fi=[Fi], v=[v], Fw=[Fw], me=me)
                Mz(QVALUES, R1=[R1], R1i=R1i, Fi=[Fi], v=[v], Fw=Fw, me=me)
            else:
                Mz(QVALUES, R1=R1, v=v, Fw=Fw, me=me)
                Mz(QVALUES, R1=[R1], v=[v], Fw=[Fw], me=me)
                Mz(QVALUES, R1=[R1], v=[v], Fw=Fw, me=me)

    # nc
    R1 = [1,0.5]
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = [0.5, 0.75]
    Fi = [0.008, 0.004]
    me = 2

    values = MzPrep.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(MzPrep.configs)}
        # if cnfgs[0] != '3D-IR-SPGR-SS':
        #     continue
        print('nc', cnfgs)
        Mz = MzPrep(**config)
        if 'R1' in Mz.inputs():
            Mz(QVALUES, R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # nt
    nt = 10
    R1 = np.full(nt, 1)
    v = 0.3
    Fw = 0.01
    R1i = np.full(nt, 0.06)
    Fi = 0.008
    me = 2

    values = MzPrep.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(MzPrep.configs)}
        print('nt', cnfgs)
        Mz = MzPrep(**config)
        if 'R1' in Mz.inputs():
            Mz(QVALUES, R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = np.stack([np.full(nt, 0.6), np.full(nt, 0.8)])
    Fi = [0.008, 0.005]
    me = 2

    values = MzPrep.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(MzPrep.configs)}
        print('(nc, nt)', cnfgs)
        Mz = MzPrep(**config)
        if 'R1' in Mz.inputs():
            Mz(QVALUES, R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
        
    MzPrep(sequence='3D-SPGR-SS')(QVALUES, R1=R1, R1i=R1, Fi=Fi, v=v, Fw=Fw, me=me)



def test_exceptions_mzprep():
    try:
        MzPrep(sequence='3D-SPGR-SS')()
    except:
        pass
    else:
        assert False

    try:
        MzPrep(sequence='3D-SPGR-SS')(QVALUES, R1=np.ones((2, 3)), v=1)
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
    values = Magnetization.configs.values()
    for cnfgs in itertools.product(*values):
        print(cnfgs)
        # if cnfgs != ('Eq-SE-EPI', False):
        #     continue
        config = {k: cnfgs[i] for i, k in enumerate(Magnetization.configs)}
        magn = Magnetization(**config)
        magn.inputs()
        magn.outputs()
        p = QVALUES
        M = magn(p)['M']


if __name__ == "__main__":
    test_coverage_readout()
    test_exceptions_readout()
    test_coverage_mzprep()
    test_exceptions_mzprep()
    test_coverage_m()
    
    print('All magnetization tests passing!')