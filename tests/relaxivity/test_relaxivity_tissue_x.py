import itertools
import numpy as np

from dcmri import aif
from dcmri.kinetics import ConcTissueX
from dcmri.relaxivity import R1TissueX, R2TissueX, R2sTissueX, RelaxTissueX



def test_relax_tissue():

    t = np.arange(0, 300, 1.5)
    ca = aif.parker(t, BAT=20)
    H = 0.45

    # Test WV limit - exact
    p0 = {'H':H, 'T_a':0, 'vb':0.0, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='WV')(ca, t, **p1)
    R1_1 = R1TissueX(kinetics='WV', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-9
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-9

    # Test WV limit - approx

    p0 = {'H':H, 'T_a':0, 'vb':0.5*1e-3, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p = {'H':H, 'T_a':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='WV')(ca, t, **p1)
    R1_1 = R1TissueX(kinetics='WV', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-3 * np.linalg.norm(C0[1:,:])
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-3 * np.linalg.norm(R1_0[1:,:])

    # Test HF limit - exact

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HF')(ca, t, **p1)
    R1_1 = R1TissueX(kinetics='HF', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CU')(ca, t, **p0)
    R1_0 = R1TissueX(kinetics='2CU', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HFU')(ca, t, **p1)
    R1_1 = R1TissueX(kinetics='HFU', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    # Test HF limit - approx

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':10, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = R1TissueX(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HF')(ca, t, **p1)
    R1_1 = R1TissueX(kinetics='HF', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-3 * np.linalg.norm(C0)
    assert np.linalg.norm(R1_0-R1_1) < 1e-3 * np.linalg.norm(R1_0)

    # Cover FX limit - ve = 0

    p0 = {'H':H, 've':1e-3, 'Fb':0.01, 'vb':0.0, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='FX')(ca, t, **p0)
    R1_0 = R1TissueX(kinetics='FX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 've':0, 'Fb':0.01, 'vb':0.0, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='FX')(ca, t, **p1)
    R1_1 = R1TissueX(kinetics='FX', water_exchange='RR')(C1, **p1)

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
        C = ConcTissueX(kin)(ca)
        RelaxTissueX(*cnfgs)(C)

    R2sTissueX(kinetics='U', t2s_relaxation='leakage')(np.ones(5))
    R2TissueX()(np.ones(5))


def test_exceptions():
    kin, wex = '2CX', 'RR'
    nt = 10
    ca = np.ones(nt)

    C0 = ConcTissueX(kinetics=kin)(ca)

    try:
        R2sTissueX(t2s_relaxation='leakage')(C0)
    except:
        pass
    else:
        assert False



if __name__ == "__main__":
    test_relax_tissue()
    test_coverage()
    test_exceptions()
    
    print('All relaxivity TissueX tests passing!')