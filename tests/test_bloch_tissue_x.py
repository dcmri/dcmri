import numpy as np
import itertools
import time

from dcmri import ConcTissueX, Relax, RelaxTissueX, MzTissueX, SignalTissueX, init


DEFAULTS = init()

def test_magn_tissue():
    nt = 10
    ca = np.ones(nt)
    kinetics='2CX'
    seq = '3D-SPGR-SS'

    pa = {**DEFAULTS, 'R1b_a': 1, 'r1': 0.005}
    p0 = {'H':0.45, 'Ta':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'R1b': 1, 'r1': 0.005}
    C = ConcTissueX(kinetics, defaults=p0)(ca)
    Ra = Relax(defaults=DEFAULTS)(c=ca, R1b=pa['R1b_a'])

    R1, _, _ = RelaxTissueX(kinetics, 'RR', defaults=DEFAULTS)(C)
    Mz = MzTissueX(kinetics, 'RR', sequence=seq, defaults=DEFAULTS)(R1=R1, R1a=Ra['R1'], **p0)
    assert 0.004 < Mz[0,0] < 0.006

    R1, _, _ = RelaxTissueX(kinetics, 'FF', defaults=DEFAULTS)(C)
    Mz = MzTissueX(kinetics, 'FF', sequence=seq, defaults=DEFAULTS)(R1=R1, R1a=Ra['R1'], **p0)
    assert 0.08 < Mz[0,0] < 0.09

    R1, _, _ = RelaxTissueX(kinetics, 'FR', defaults=DEFAULTS)(C)
    Mz = MzTissueX(kinetics, 'FR', sequence=seq, defaults=DEFAULTS)(R1=R1, R1a=Ra['R1'], **p0)
    assert 0.02 < Mz[0,0] < 0.04

    Mz = MzTissueX(kinetics, 'FR', sequence='Eq-GE-EPI', defaults=DEFAULTS)(R1a=Ra['R1'], **p0)
    


def test_signal_tissue():
    nt = 10
    ca = np.ones(nt)

    pa = {**DEFAULTS, 'R1b_a': 1, 'r1': 0.005}
    p0 = {'TE': 0, 'H':0.45, 'Ta':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'S0':10, 'R1b':1, 'r1':0.005}
    C = ConcTissueX('2CX', defaults=p0)(ca)
    Ra = Relax(defaults=pa)(c=ca, R1b=pa['R1b_a'])

    R1, R2, R2s = RelaxTissueX('2CX', 'RR', defaults=DEFAULTS)(C, **p0)
    S = SignalTissueX('2CX', 'RR', '3D-SPGR-SS', defaults=DEFAULTS)(R1=R1, R2=R2, R2s=R2s, R1a=Ra['R1'], **p0)
    assert 0.3 < S[0] < 0.4

    S = SignalTissueX('2CX', 'RR', '3D-SPGR-SS', defaults=DEFAULTS)(R1=R1, R2=R2, R2s=R2s, R1a=Ra['R1'], **p0)
    S = SignalTissueX('2CX', 'RR', 'Eq-GE-EPI', defaults=DEFAULTS)(R2s=R2s, R1a=Ra['R1'], **p0)


def test_coverage():

    nt = 10
    ca = np.ones(nt)

    # Run for coverage
    values = SignalTissueX.configs.values()
    cnt=0
    for cnfgs in itertools.product(*values):
        cnt += 1
        start = time.perf_counter()
        kin, wex, seq, inflow = cnfgs
        
        C = ConcTissueX(kin, defaults=DEFAULTS)(ca)
        R1, R2, R2s = RelaxTissueX(kin, wex, defaults=DEFAULTS)(C)
        Ra = Relax(defaults=DEFAULTS)(c=ca)
        SignalTissueX(kin, wex, seq, inflow, defaults=DEFAULTS)(R1=R1, R2=R2, R2s=R2s, R1a=Ra['R1'])
        end = time.perf_counter()
        print(cnt, end-start, kin, wex, seq, inflow)


def test_exceptions():
    # try:
    #     MzTissueX(sequence='3D-SPGR-SS', defaults=DEFAULTS)()
    # except:
    #     pass
    # else:
    #     assert False
    pass


if __name__ == "__main__":

    test_magn_tissue()
    test_signal_tissue()
    test_exceptions()
    test_coverage()
    
    print('All tissue tests passing!')