

import numpy as np
import dcmri as dc

from dcmri import tissue_x
from dcmri import relaxivity
from dcmri import aif
from dcmri.kinetics import ConcTissueX, FluxTissueX



def test_relax_tissue():

    t = np.arange(0, 300, 1.5)
    ca = aif.parker(t, BAT=20)
    H = 0.45

    # Test WV limit - exact
    p0 = {'H':H, 'T_a':0, 'vb':0.0, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue_x.R1(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='WV')(ca, t, **p1)
    R1_1 = tissue_x.R1(kinetics='WV', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-9
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-9

    # Test WV limit - approx

    p0 = {'H':H, 'T_a':0, 'vb':0.5*1e-3, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue_x.R1(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p = {'H':H, 'T_a':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='WV')(ca, t, **p1)
    R1_1 = tissue_x.R1(kinetics='WV', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-3 * np.linalg.norm(C0[1:,:])
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-3 * np.linalg.norm(R1_0[1:,:])

    # Test HF limit - exact

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue_x.R1(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HF')(ca, t, **p1)
    R1_1 = tissue_x.R1(kinetics='HF', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CU')(ca, t, **p0)
    R1_0 = tissue_x.R1(kinetics='2CU', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HFU')(ca, t, **p1)
    R1_1 = tissue_x.R1(kinetics='HFU', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    # Test HF limit - approx

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':10, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue_x.R1(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='HF')(ca, t, **p1)
    R1_1 = tissue_x.R1(kinetics='HF', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-3 * np.linalg.norm(C0)
    assert np.linalg.norm(R1_0-R1_1) < 1e-3 * np.linalg.norm(R1_0)

    # Cover FX limit - ve = 0

    p0 = {'H':H, 've':1e-3, 'Fb':0.01, 'vb':0.0, 'R10': 1, 'r1': 0.005}
    C0 = ConcTissueX(kinetics='FX')(ca, t, **p0)
    R1_0 = tissue_x.R1(kinetics='FX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 've':0, 'Fb':0.01, 'vb':0.0, 'R10': 1, 'r1': 0.005}
    C1 = ConcTissueX(kinetics='FX')(ca, t, **p1)
    R1_1 = tissue_x.R1(kinetics='FX', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-3
    assert np.linalg.norm(R1_0-R1_1) < 1e-3



def test_magn_tissue():
    nt = 10
    ca = np.ones(nt)
    kinetics='2CX'

    p0 = {'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'R10_a': 1, 'R10': 1, 'r1': 0.005}
    C = ConcTissueX(kinetics)(ca, **p0)
    R1a = relaxivity.relax_t1(ca, p0['R10_a'], p0['r1'])

    R1 = tissue_x.R1(kinetics, 'RR')(C, **p0)
    Mz = tissue_x.Mz(kinetics, 'RR', '3D-SPGR-SS')(R1, R1a, **p0)
    assert 0.01 < Mz[0,0] < 0.02

    R1 = tissue_x.R1(kinetics, 'FF')(C, **p0)
    Mz = tissue_x.Mz(kinetics, 'FF', '3D-SPGR-SS')(R1, R1a, **p0)
    assert 0.1 < Mz[0,0] < 0.2

    R1 = tissue_x.R1(kinetics, 'FR')(C, **p0)
    Mz = tissue_x.Mz(kinetics, 'FR', '3D-SPGR-SS')(R1, R1a, **p0)
    assert 0.04 < Mz[0,0] < 0.06
    
    try:
        Mz = tissue_x.Mz(kinetics, 'FR', 'XX')
    except:
        pass
    else:
        assert False


def test_signal_tissue():
    nt = 10
    ca = np.ones(nt)

    p0 = {'TE': 0, 'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'S0':10, 'R10':1, 'R10_a':1, 'r1':0.005}
    S = tissue_x.Signal('2CX', 'RR', '3D-SPGR-SS')(ca, **p0)
    assert 0.3 < S[0] < 0.4


def test_coverage():

    nt = 10
    ca = np.ones(nt)

    # Call options
    ConcTissueX()(ca)
    tissue_x.R1()(ca)
    FluxTissueX()(ca)
    tissue_x.R1().params()
    tissue_x.Mz().params()
    FluxTissueX().params()
    tissue_x.WaterVolumes().params()
    tissue_x.WaterVolumes()(vb=0.1)
    tissue_x.WaterFlows().params()
    tissue_x.WaterFlows()(Fb=0.01)

    signal = tissue_x.Signal('HF', 'FF', '3D-SPGR-SS')
    p = signal.params()
    p['TE'] = 0
    signal(ca, **p)

    # Run for coverage
    for kin in tissue_x.Signal.configs['kinetics']:
        for wex in tissue_x.Signal.configs['water_exchange']:
            for seq in tissue_x.Signal.configs['sequence']:
                for r2s in tissue_x.Signal.configs['transverse_relaxation']:
    # for kin in ['FX']:
    #     for wex in ['FF']:
    #         for seq in ['3D-SPGR-SS']:
    #             for r2s in tissue.Signal.configs['transverse_relaxation']:
                    print(kin, wex, seq, r2s)
                    signal = tissue_x.Signal(kin, wex, seq, r2s)
                    p = signal.params()
                    if seq in ['GE-EPI', 'SE-EPI', 'DE-EPI']:
                        pass
                    else:
                        p['TE'] = 0
                    S = signal(ca, **p)


def test_exceptions():
    kin, wex, seq = '2CX', 'RR', '3D-SPGR-SS'
    nt = 10
    ca = np.ones(nt)

    try:
        p = tissue_x.Signal('XXX', wex, seq).params()
    except:
        pass
    else:
        assert False
    try:
        p = tissue_x.Signal(kin, 'SSS', seq).params()
    except:
        pass
    else:
        assert False

    p = tissue_x.Signal(kin, wex, seq).params()

    try:
        J0 = FluxTissueX(kinetics='XXX')(ca, *p)
    except:
        pass
    else:
        assert False

    C0 = ConcTissueX(kinetics=kin)(ca, **p)

    try:
        R1 = tissue_x.R2s('leakage')(C0, **p) # kinetics must be specified
    except:
        pass
    else:
        assert False

    try:
        R1 = tissue_x.R1('XXX', wex)(C0, **p)
    except:
        pass
    else:
        assert False

    R1 = tissue_x.R1(kin, wex)(C0, **p)
    R1a = dc.relaxivity.relax_t1(ca, p['R10_a'], p['r1'])

    try:
        Mz = tissue_x.Mz('XXX', wex, seq)
    except:
        pass
    else:
        assert False

    try:
        Mz = tissue_x.Mz(kin, 'XXX', seq)
    except:
        pass
    else:
        assert False

    Mz = tissue_x.Mz(kin, wex, seq)(R1, R1a, **p)

    S = tissue_x.Signal(kin, wex, seq)(ca)
    S = tissue_x.Signal(kin, wex, seq)(ca, **p)

    try:
        kin = 'HF'
        C0 = ConcTissueX(kin)(ca)
        R1 = tissue_x.R1(kin, wex)(C0)
        R1a = dc.relaxivity.relax_t1(ca, p['R10_a'], p['r1'])
        Mz = tissue_x.Mz(kin, wex, seq)(R1, R1a) # No Fb with R1a provided
    except:
        pass
    else:
        assert False

if __name__ == "__main__":
    test_relax_tissue()
    test_magn_tissue()
    test_signal_tissue()

    test_coverage()
    test_exceptions()
    
    print('All tissue tests passing!')