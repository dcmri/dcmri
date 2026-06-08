import numpy as np

import dcmri as dc
from dcmri.kinetics.lib.liver import dpars_liver


def test_liver():

    p = {
        've': 0.1, 
        'Fp': 0.01, 
        'fa': 1.0, 
        'T_a': 0, 
        'Tg': 0,
        'E_i': 0.5,
        'E_f': 0.5,
        'khe_i': 0.001,
        'khe_f': 0.001,
        'khe': 0.001,
        'Th_i': 300,
        'Th_f': 300,
        'Th': 300,
        'E': 0.5,
        'vol_l': 1000,
    }
    dpars_liver(p)   
    dpars_liver(p, '1I-EC')
    dpars_liver(p, '2I-EC')
    dpars_liver(p, '1I-IC')
    dpars_liver(p, '2I-IC')
    dpars_liver(p, '1I-IC-HF')
    dpars_liver(p, '1I-IC-D')
    dpars_liver(p, '2I-IC-HF')
    dpars_liver(p, '1I-IC-HFD')
    dpars_liver(p, '1I-IC-HFDU')
    dpars_liver(p, '2I-IC-U')

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
    C0 = dc.conc_liver_1i_ec(ca, t, **p)

    p = {'ve': 0.1, 
         'Te': 0.1 / 0.01, 
         'De': 1.0,
        }
    C1 = dc.conc_liver_1i_ec_d(ca, t, **p)

    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-9

    p = {'ve': 0.1, 
         'Fp': 1000, 
         'fa': 1.0, 
         'T_a': 0, 
         'Tg': 0,
        }
    C0 = dc.conc_liver_1i_ec(ca, t, **p)

    p = {'ve': 0.1, 
         'fa': 1.0, 
         'T_a': 0, 
        }
    C1 = dc.conc_liver_2i_ec_hf((ca, ca), t, **p)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3

    p = {'ve': 0.1, 
         'Fp': 1000,
         'fa': 1.0, 
         'T_a': 0, 
        }
    C1 = dc.conc_liver_2i_ec((ca, ca), t, **p)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3


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
    C0 = dc.conc_liver_1i_ic_hf(ca, t, **p)

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
        }
    C1 = dc.conc_liver_1i_ic_hf__u(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = dc.conc_liver_1i_ic_hf__e(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = dc.conc_liver_1i_ic_hf__ue(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = dc.conc_liver_1i_ic_hfd(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
         'Tg': 0, 
         'Dg': 1,
        }
    C1 = dc.conc_liver_1i_ic_hfd__u(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = dc.conc_liver_1i_ic_hfd__e(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = dc.conc_liver_1i_ic_hfd__ue(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 1500,
         'Tg': 10,
         'Dg': 0.5,
        }
    C0 = dc.conc_liver_1i_ic_hfd(ca, t, **p)

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Tg': 10,
         'Dg': 0.5,
        }
    C1 = dc.conc_liver_1i_ic_hfdu(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Tg': 10,
         'Dg': 0.5,
        }
    C1 = dc.conc_liver_1i_ic_hfdu__u(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
        }
    C0 = dc.conc_liver_1i_ic_hf(ca, t, **p)

    C = dc.conc_liver_1i_ic_hf(ca, t, **p)
    assert np.array_equal(C, C0)

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf((ca, ca), t, **p)

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf__u((ca, ca), t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf__e((ca, ca), t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005,
         'khe_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf__ue((ca, ca), t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C0 = dc.conc_liver_2i_ic_hf((ca, ca), t, **p)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 0.1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th': 15,
        }
    C1 = dc.conc_liver_1i_ic(ca, t, **p)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001, 
         'Th': 15,
        }
    C1 = dc.conc_liver_1i_ic__u(ca, t, **p)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = dc.conc_liver_1i_ic__e(ca, t, **p)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         
         'E_i': 0.001, 
         'E_f': 0.001, 
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = dc.conc_liver_1i_ic__ue(ca, t, **p)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,  
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic__u((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001,  
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic__e((ca, ca), t, **p)
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
    C1 = dc.conc_liver_2i_ic__ue((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th': 1500,
         'T_a': 0,
         'fa': 1,
        }
    C0 = dc.conc_liver_2i_ic((ca, ca), t, **p)

    C = dc.conc_liver_2i_ic((ca, ca), t, **p)
    assert np.array_equal(C, C0)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'T_a': 0,
         'fa': 1,
        }
    C1 = dc.conc_liver_2i_ic_u((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,
         'T_a': 0,
         'fa': 1,
        }
    C1 =  dc.conc_liver_2i_ic_u__u((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1




if __name__=='__main__':
    test_liver()