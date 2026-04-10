import numpy as np
import dcmri as dc

from dcmri.liver import Conc


def test_ec():
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
    C0 = Conc('1I-EC', **p)(ca, t)

    p = {'ve': 0.1, 
         'Te': 0.1 / 0.01, 
         'De': 1.0,
        }
    C1 = Conc('1I-EC-D', **p)(ca, t)

    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-9


    p = {'ve': 0.1, 
         'Fp': 1000, 
         'fa': 1.0, 
         'T_a': 0, 
         'Tg': 0,
        }
    C0 = Conc('1I-EC', **p)(ca, t)

    p = {'ve': 0.1, 
         'fa': 1.0, 
         'T_a': 0, 
        }
    C1 = Conc('2I-EC-HF', **p)((ca, ca), t, **p)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3

    p = {'ve': 0.1, 
         'Fp': 1000,
         'fa': 1.0, 
         'T_a': 0, 
        }
    C1 = Conc('2I-EC', **p)((ca, ca), t)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3


def test_ic():

    tmax = 60
    nt = 30
    Taif = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Taif)/Taif

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
        }
    C0 = Conc('1I-IC-HF', **p)(ca, t)

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
        }
    C1 = Conc('1I-IC-HF', 'U', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = Conc('1I-IC-HF', 'E', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = Conc('1I-IC-HF', 'UE', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = Conc('1I-IC-HFD', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = Conc('1I-IC-HFD', 'U', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = Conc('1I-IC-HFD', 'E', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = Conc('1I-IC-HFD', 'UE', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 1500,
         'Tg': 10,
         'Dg': 0.5,
        }
    C0 = Conc('1I-IC-HFD', **p)(ca, t)

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Tg': 10,
         'Dg': 0.5,
        }
    C1 = Conc('1I-IC-HFDU', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Tg': 10,
         'Dg': 0.5,
        }
    C1 = Conc('1I-IC-HFDU', 'U', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
        }
    C0 = Conc('1I-IC-HF', **p)(ca, t)

    C = Conc('1I-IC-HF', **p)(ca, t)
    assert np.array_equal(C, C0)

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC-HF', **p)((ca, ca), t)

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC-HF', 'U', **p)((ca, ca), t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC-HF', 'E', **p)((ca, ca), t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005,
         'khe_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC-HF', 'UE', **p)((ca, ca), t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C0 = Conc('2I-IC-HF', **p)((ca, ca), t)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 0.1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,  
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC', 'U', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001,  
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC', 'E', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,  
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC', 'UE', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th': 1500,
         'T_a': 0,
         'fa': 1,
        }
    C0 = Conc('2I-IC', **p)((ca, ca), t)

    C = Conc('2I-IC', **p)((ca, ca), t)
    assert np.array_equal(C, C0)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC-U', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,
         'T_a': 0,
         'fa': 1,
        }
    C1 = Conc('2I-IC-U', 'U', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1


    # Test exceptions
    
    try:
        C1 = Conc('XX-YY', 'U', **p)((ca, ca), t)
    except:
        pass
    else:
        assert False


if __name__=='__main__':

    test_ec()
    test_ic()

    print('All liver tests passed!!')