import numpy as np


import dcmri as dc

def test_conc_block():
    dt = 1.5
    t = dt * np.arange(20)
    J = np.ones(20)
    C = dc.ConcBlock('comp')(J, t=t, T=20)
    assert round(C[-1]) == 15
    dc.ConcBlock()._param_names()

def test_conc_aorta():
    C = dc.ConcAorta('comp', 'comp')()
    C = dc.ConcAorta('pfcomp', 'comp')()
    C = dc.ConcAorta('chain', 'comp')()
    C = dc.ConcAorta('comp', '2cxm')()
    dc.ConcAorta()._params('body')

def test_conc_kidney():
    dt = 1.5
    t = dt*np.arange(20)
    ca = np.ones(20)
    p = {'Fp': 0.01, 'vp': 0.2, 'Ft': 0.005, 'Tt': 120}
    p['Tp'] = p['vp'] / (p['Fp'] + p['Ft'])
    C = dc.ConcKidney('2CF', **p)(ca, dt=dt)
    assert round(C[1,10], 2) == 0.0


def test_conc_cortmed():
    dt = 1.5
    ca = np.ones(20)
    Ccor, Cmed = dc.ConcCortMed('7C')(ca, dt=dt)
    assert round(Ccor[1,10], 2) == 0.09


def test_conc_liver():

    # EC

    tmax = 60
    nt = 10
    Ta = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Ta)/Ta

    p = {'ve': 0.1, 
         'Fp': 0.01, 
         'fa': 1.0, 
         'T_a': 0, 
         'Tg': 0,
        }
    C0 = dc.ConcLiver('1I-EC', **p)(ca, t)

    # p = {'ve': 0.1, 
    #      'Te': 0.1 / 0.01, 
    #      'De': 1.0,
    #     }
    # C1 = ConcLiver('1I-EC-D', **p)(ca, t)

    # assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-9

    # try:
    #     C1 = ConcLiver('1I-EC-D', 'U')
    # except:
    #     pass
    # else:
    #     assert False

    # IC

    tmax = 60
    nt = 30
    Taif = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Taif)/Taif

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
        }
    C0 = dc.ConcLiver('1I-IC-HF', **p)(ca, t)

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
        }
    C1 = dc.ConcLiver('1I-IC-HF', 'U', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    try:
        C1 = dc.ConcLiver('XX', 'U', **p)(ca, t)
    except:
        pass
    else:
        assert False


def test_conc_tissue():
    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.ConcTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0


if __name__ == '__main__':

    test_conc_block()
    test_conc_aorta()
    test_conc_cortmed()
    test_conc_kidney()
    test_conc_liver()
    test_conc_tissue()

    print('All conc tests passed!!')