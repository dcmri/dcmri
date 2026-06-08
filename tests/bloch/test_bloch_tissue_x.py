import numpy as np
import itertools

from dcmri import ConcTissueX, Relax, RelaxTissueX, MzTissueX, SignalTissueX




def test_magn_tissue():
    nt = 10
    ca = np.ones(nt)
    kinetics='2CX'
    seq = '3D-SPGR-SS'

    pa = {'R10_a': 1, 'r1': 0.005}
    p0 = {'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'R10': 1, 'r1': 0.005}
    C = ConcTissueX(kinetics)(ca, **p0)
    R1a, _, _ = Relax(**pa)(ca, R10=pa['R10_a'])

    R1, _, _ = RelaxTissueX(kinetics, 'RR', **p0)(C)
    Mz = MzTissueX(kinetics, 'RR', sequence=seq)(R1, R1a, **p0)
    assert 0.01 < Mz[0,0] < 0.02

    R1, _, _ = RelaxTissueX(kinetics, 'FF', **p0)(C)
    Mz = MzTissueX(kinetics, 'FF', sequence=seq)(R1, R1a, **p0)
    assert 0.1 < Mz[0,0] < 0.2

    R1, _, _ = RelaxTissueX(kinetics, 'FR', **p0)(C)
    Mz = MzTissueX(kinetics, 'FR', sequence=seq)(R1, R1a, **p0)
    assert 0.04 < Mz[0,0] < 0.06

    Mz = MzTissueX(kinetics, 'FR', sequence='Eq-GE-EPI')(R1a=R1a, **p0)
    


def test_signal_tissue():
    nt = 10
    ca = np.ones(nt)

    pa = {'R10_a': 1, 'r1': 0.005}
    p0 = {'TE': 0, 'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'S0':10, 'R10':1, 'r1':0.005}
    C = ConcTissueX('2CX')(ca, **p0)
    R1a, _, _ = Relax(**pa)(ca, R10=pa['R10_a'])

    R1, R2, R2s = RelaxTissueX('2CX', 'RR', **p0)(C)
    S = SignalTissueX('2CX', 'RR', '3D-SPGR-SS')(R1, R2, R2s, R1a, **p0)
    assert 0.3 < S[0] < 0.4

    S = SignalTissueX('2CX', 'RR', '3D-SPGR-SS')(R2=R2, R2s=R2s, R1a=R1a, **p0)
    S = SignalTissueX('2CX', 'RR', 'Eq-GE-EPI')(R2s=R2s, R1a=R1a, **p0)


def test_coverage():

    nt = 10
    ca = np.ones(nt)

    # Run for coverage
    values = SignalTissueX.configs.values()
    for cnfgs in itertools.product(*values):
        kin, wex, seq = cnfgs
        print(kin, wex, seq)
        
        C = ConcTissueX(kin)(ca)
        R1, R2, R2s = RelaxTissueX(kin, wex)(C)
        R1a, _, _ = Relax()(ca)
        SignalTissueX(kin, wex, seq)(R1, R2, R2s, R1a)


def test_exceptions():
    try:
        MzTissueX(sequence='3D-SPGR-SS')(R1=None)
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