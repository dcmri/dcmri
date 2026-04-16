import numpy as np
import itertools

from dcmri.kinetics import ConcTissueX
from dcmri.relaxivity import RelaxTissueX, R1TissueX
from dcmri.bloch import MzTissueX, SignalTissueX




def test_magn_tissue():
    nt = 10
    ca = np.ones(nt)
    kinetics='2CX'
    seq = '3D-SPGR-SS'

    p0 = {'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'R10_a': 1, 'R10': 1, 'r1': 0.005}
    C = ConcTissueX(kinetics)(ca, **p0)

    R1, R2, R2s, R1a = RelaxTissueX(kinetics, 'RR', sequence=seq)(C, ca, **p0)
    Mz = MzTissueX(kinetics, 'RR', sequence=seq)(R1, R1a, **p0)
    assert 0.01 < Mz[0,0] < 0.02

    R1, R2, R2s, R1a = RelaxTissueX(kinetics, 'FF', sequence=seq)(C, ca, **p0)
    Mz = MzTissueX(kinetics, 'FF', sequence=seq)(R1, R1a, **p0)
    assert 0.1 < Mz[0,0] < 0.2

    R1, R2, R2s, R1a = RelaxTissueX(kinetics, 'FR', sequence=seq)(C, ca, **p0)
    Mz = MzTissueX(kinetics, 'FR', sequence=seq)(R1, R1a, **p0)
    assert 0.04 < Mz[0,0] < 0.06
    


def test_signal_tissue():
    nt = 10
    ca = np.ones(nt)

    p0 = {'TE': 0, 'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'S0':10, 'R10':1, 'R10_a':1, 'r1':0.005}
    C = ConcTissueX('2CX')(ca, **p0)
    R1, R2, R2s, R1a = RelaxTissueX('2CX', 'RR', sequence='3D-SPGR-SS')(C, ca, **p0)
    S = SignalTissueX('2CX', 'RR', '3D-SPGR-SS')(R1, R2, R2s, R1a, **p0)
    assert 0.3 < S[0] < 0.4


def test_coverage():

    nt = 10
    ca = np.ones(nt)

    # Call options
    R1TissueX().params()

    # Run for coverage
    values = SignalTissueX.configs.values()
    for cnfgs in itertools.product(*values):
        kin, wex, seq = cnfgs
        print(kin, wex, seq)
        C = ConcTissueX(kin)(ca)
        R1, R2, R2s, R1a = RelaxTissueX(kin, wex, sequence=seq)(C, ca)
        S = SignalTissueX(kin, wex, seq)(R1, R2, R2s, R1a)


def test_exceptions():
    kin, wex, seq = '2CX', 'RR', '3D-SPGR-SS'
    nt = 10
    ca = np.ones(nt)

    try:
        MzTissueX(kin, 'FR', 'XX')
    except:
        pass
    else:
        assert False

    try:
        SignalTissueX('XXX', wex, seq).params()
    except:
        pass
    else:
        assert False
    try:
        SignalTissueX(kin, 'SSS', seq).params()
    except:
        pass
    else:
        assert False

    try:
        MzTissueX('XXX', wex, seq)
    except:
        pass
    else:
        assert False

    try:
        MzTissueX(kin, 'XXX', seq)
    except:
        pass
    else:
        assert False


if __name__ == "__main__":

    test_magn_tissue()
    test_signal_tissue()
    test_coverage()
    test_exceptions()
    
    print('All tissue tests passing!')