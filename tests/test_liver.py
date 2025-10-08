import numpy as np
import dcmri as dc


def test_ec():
    tmax = 60
    nt = 10
    Ta = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Ta)/Ta

    p = {'ve': 0.1, 
         'Fp': 0.01, 
         'fa': 1.0, 
         'Ta': 0, 
         'Tg': 0,
        }
    C0 = dc.conc_liver(ca, t, kinetics='1I-EC', **p)

    p = {'ve': 0.1, 
         'Te': 0.1 / 0.01, 
         'De': 1.0,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-EC-D', **p)

    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-9


    p = {'ve': 0.1, 
         'Fp': 1000, 
         'fa': 1.0, 
         'Ta': 0, 
         'Tg': 0,
        }
    C0 = dc.conc_liver(ca, t, kinetics='1I-EC', **p)

    p = {'ve': 0.1, 
         'fa': 1.0, 
         'Ta': 0, 
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-EC-HF', **p)

    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-3

    p = {'ve': 0.1, 
         'Fp': 1000,
         'fa': 1.0, 
         'Ta': 0, 
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-EC', **p)

    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-3


def test_ic():

    tmax = 60
    nt = 30
    Taif = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Taif)/Taif

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th': 15,
        }
    C0 = dc.conc_liver(ca, t, kinetics='1I-IC-HF', **p)

    p = {'ve_app': 0.1, 
         'Ktrans_i': 0.005, 
         'Ktrans_f': 0.005, 
         'Th': 15,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-HF', non_stationary='U', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-HF', non_stationary='E', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans_i': 0.005, 
         'Ktrans_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-HF', non_stationary='UE', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th': 15,
         'Te': 0,
         'De': 1,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-D', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans_i': 0.005, 
         'Ktrans_f': 0.005, 
         'Th': 15,
         'Te': 0,
         'De': 1,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-D', non_stationary='U', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Te': 0,
         'De': 1,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-D', non_stationary='E', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans_i': 0.005, 
         'Ktrans_f': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Te': 0,
         'De': 1,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-D', non_stationary='UE', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th': 1500,
         'Te': 10,
         'De': 0.5,
        }
    C0 = dc.conc_liver(ca, t, kinetics='1I-IC-D', **p)

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Te': 10,
         'De': 0.5,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-DU', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans_i': 0.005, 
         'Ktrans_f': 0.005, 
         'Te': 10,
         'De': 0.5,
        }
    C1 = dc.conc_liver(ca, t, kinetics='1I-IC-DU', non_stationary='U', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th': 15,
        }
    C0 = dc.conc_liver(ca, t, kinetics='1I-IC-HF', **p)

    C = dc.conc_liver(ca, t, kinetics='1I-IC-HF', sum=False, **p)
    assert np.array_equal(C[0,:] + C[1,:], C0)

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-HF', **p)

    p = {'ve_app': 0.1, 
         'Ktrans_i': 0.005, 
         'Ktrans_f': 0.005, 
         'Th': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-HF', non_stationary='U', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-HF', non_stationary='E', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans_i': 0.005,
         'Ktrans_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-HF', non_stationary='UE', **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve_app': 0.1, 
         'Ktrans': 0.005, 
         'Th': 15,
         'Ta': 0,
         'fa': 1,
        }
    C0 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-HF', **p)

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'khe': 0.005, 
         'Th': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC', **p)
    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-1

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC', non_stationary='U', **p)
    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-1

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC', non_stationary='E', **p)
    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-1

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC', non_stationary='UE', **p)
    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-1

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'khe': 0.005, 
         'Th': 1500,
         'Ta': 0,
         'fa': 1,
        }
    C0 = dc.conc_liver((ca, ca), t, kinetics='2I-IC', **p)

    C = dc.conc_liver((ca, ca), t, kinetics='2I-IC', sum=False, **p)
    assert np.array_equal(C[0,:] + C[1,:], C0)

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'khe': 0.005, 
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-U', **p)
    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-1

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Ta': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-U', non_stationary='U', **p)
    assert np.linalg.norm(C0[1:]-C1[1:]) / np.linalg.norm(C0[1:]) < 1e-1


    # Test exceptions
    
    try:
        C1 = dc.conc_liver((ca, ca), t, kinetics='XX-YY', non_stationary='U', **p)
    except:
        assert True
    else:
        assert False

    try:
        p = {'ve': 0.1, 
            'Fp': 1000, 
            'khe_i': 0.005, 
            'khe_f': 0.005, 
            'Ta': 0,
            }
        C1 = dc.conc_liver((ca, ca), t, kinetics='2I-IC-U', non_stationary='U', **p)
    except:
        assert True
    else:
        assert False


if __name__=='__main__':

    test_ec()
    test_ic()

    print('All liver tests passed!!')