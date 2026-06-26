import itertools
import numpy as np

import dcmri as dc
from dcmri import ConcTissueX, R1TissueX, R2TissueX, R2sTissueX, RelaxTissueX

DEFAULTS = dc.init()

def test_relax_tissue():

    t = np.arange(0, 300, 1.5)
    ca = dc.parker(t, BAT=20)
    H = 0.45

    # Test WV limit - exact
    p0 = {'H':H, 'Ta':0, 'vb':0.0, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX', defaults=p0)(ca, t)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR', defaults=p0)(C0)

    p1 = {'H':H, 'Ta':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R1b': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='WV', defaults=p1)(ca, t)
    R1_1 = R1TissueX(kinetics='WV', water_exchange='RR', defaults=p0)(C1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-9
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-9

    # Test WV limit - approx

    p0 = {'H':H, 'Ta':0, 'vb':0.5*1e-3, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX', defaults=p0)(ca, t)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR', defaults=p0)(C0)

    p = {'H':H, 'Ta':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R1b': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='WV', defaults=p1)(ca, t)
    R1_1 = R1TissueX(kinetics='WV', water_exchange='RR', defaults=p1)(C1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-3 * np.linalg.norm(C0[1:,:])
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-3 * np.linalg.norm(R1_0[1:,:])

    # Test HF limit - exact

    p0 = {'H':H, 'Ta':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX', defaults=p0)(ca, t)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR', defaults=p0)(C0)

    p1 = {'H':H, 'Ta':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HF', defaults=p0)(ca, t)
    R1_1 = R1TissueX(kinetics='HF', defaults=p0, water_exchange='RR')(C1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    p0 = {'H':H, 'Ta':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CU', defaults=p0)(ca, t)
    R1_0 = R1TissueX(kinetics='2CU', water_exchange='RR', defaults=p0)(C0)

    p1 = {'H':H, 'Ta':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HFU', defaults=p1)(ca, t)
    R1_1 = R1TissueX(kinetics='HFU', water_exchange='RR', defaults=p1)(C1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    # Test HF limit - approx

    p0 = {'H':H, 'Ta':0, 'vb':0.05, 'vi':0.3, 'Fb':10, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX', defaults=p0)(ca, t)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR', defaults=p0)(C0)

    p1 = {'H':H, 'Ta':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R1b': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HF', defaults=p1)(ca, t)
    R1_1 = R1TissueX(kinetics='HF', water_exchange='RR', defaults=p1)(C1)

    assert np.linalg.norm(C0-C1) < 1e-3 * np.linalg.norm(C0)
    assert np.linalg.norm(R1_0-R1_1) < 1e-3 * np.linalg.norm(R1_0)

    # Cover FX limit - ve = 0

    p0 = {'H':H, 'Ta':0, 've':1e-3, 'Fb':0.01, 'vb':0.0, 'R1b': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='FX', defaults=p0)(ca, t)
    R1_0 = R1TissueX(kinetics='FX', water_exchange='RR', defaults=p0)(C0)

    p1 = {'H':H, 'Ta':0, 've':0, 'Fb':0.01, 'vb':0.0, 'R1b': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='FX', defaults=p1)(ca, t)
    R1_1 = R1TissueX(kinetics='FX', water_exchange='RR', defaults=p1)(C1)

    assert np.linalg.norm(C0-C1) < 1e-3
    assert np.linalg.norm(R1_0-R1_1) < 1e-3



def test_coverage():

    nt = 10
    ca = np.ones(nt)

    # Run for coverage
    values = RelaxTissueX.configs.values()
    for cnfgs in itertools.product(*values):
        print(cnfgs)
        kin = cnfgs[0]
        C = ConcTissueX(kin, defaults=DEFAULTS)(ca)
        RelaxTissueX(*cnfgs, defaults=DEFAULTS)(C)

    R2sTissueX(kinetics='U', t2s_relaxation='leakage', defaults=DEFAULTS)(np.ones(5))
    R2TissueX(defaults=DEFAULTS)(np.ones(5))


def test_exceptions():
    kin, wex = '2CX', 'RR'
    nt = 10
    ca = np.ones(nt)

    C0 = ConcTissueX(kinetics=kin, defaults=DEFAULTS)(ca)

    try:
        R2sTissueX(t2s_relaxation='leakage', defaults=DEFAULTS)(C0)
    except:
        pass
    else:
        assert False



if __name__ == "__main__":
    test_relax_tissue()
    test_coverage()
    test_exceptions()
    
    print('All relaxivity TissueX tests passing!')