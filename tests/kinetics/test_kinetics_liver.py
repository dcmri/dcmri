import numpy as np

import dcmri as dc
from dcmri.kinetics.functions_liver import dpars_liver


def test_liver():

    p = {
        'v_e_li': 0.1, 
        'F_p_li': 0.01, 
        'ffa': 1.0, 
        'Ei_li': 0.5,
        'Ef_li': 0.5,
        'ki_e2h': 0.001,
        'kf_e2h': 0.001,
        'k_e2h': 0.001,
        'Ti_h': 300,
        'Tf_h': 300,
        'T_h': 300,
        'E_li': 0.5,
        'vol_l': 1000,
    }
    dpars_liver(p)  
    dpars_liver(p, '2I-EC')
    dpars_liver(p, '2I-EC-HF') 

    dpars_liver(p, '1I-EC')
    dpars_liver(p, '1I-EC-HF')

    dpars_liver(p, '2I-IC')
    dpars_liver(p, '2I-IC-HF')
    dpars_liver(p, '2I-IC-U')
    
    dpars_liver(p, '1I-IC')
    dpars_liver(p, '1I-IC-HF')
    # dpars_liver(p, '1I-IC-D')
    # dpars_liver(p, '1I-IC-D')
    # dpars_liver(p, '1I-IC-HFD')
    # dpars_liver(p, '1I-IC-HFDU')
    

    # EC

    tmax = 60
    nt = 10
    Ta = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Ta)/Ta

    p = {'v_e_li': 0.1, 
         'F_p_li': 1000, 
        }
    C0 = dc.conc_liver_1i_ec(ca, t, **p)

    p = {'v_e_li': 0.1, 
         'ffa': 1.0, 
        }
    C1 = dc.conc_liver_2i_ec_hf((ca, ca), t, **p)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3

    p = {'v_e_li': 0.1, 
         'F_p_li': 1000,
         'ffa': 1.0, 
        }
    C1 = dc.conc_liver_2i_ec((ca, ca), t, **p)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3

    p = {'v_e_li': 0.1, 
        }
    C1 = dc.conc_liver_1i_ec_hf(ca, t, **p)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3

    # IC

    tmax = 60
    nt = 30
    Taif = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Taif)/Taif

    p = {'v_e_li': 0.1, 
         'k_e2h': 0.005, 
         'T_h': 15,
        }
    C0 = dc.conc_liver_1i_ic_hf(ca, t, **p)

    p = {'v_e_li': 0.1, 
         'ki_e2h': 0.005, 
         'kf_e2h': 0.005, 
         'T_h': 15,
        }
    C1 = dc.conc_liver_1i_ic_hf_nsu(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'v_e_li': 0.1, 
         'k_e2h': 0.005, 
         'Ti_h': 15,
         'Tf_h': 15,
        }
    C1 = dc.conc_liver_1i_ic_hf_nse(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'v_e_li': 0.1, 
         'ki_e2h': 0.005, 
         'kf_e2h': 0.005,
         'Ti_h': 15,
         'Tf_h': 15,
        }
    C1 = dc.conc_liver_1i_ic_hf_nsue(ca, t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    # p = {'v_e_li': 0.1, 
    #      'k_e2h': 0.005, 
    #      'T_h': 15,
    #      'T_g': 0,
    #      'Dg': 1,
    #     }
    # C1 = dc.conc_liver_1i_ic_hfd(ca, t, **p)
    # assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    # p = {'v_e_li': 0.1, 
    #      'ki_e2h': 0.005, 
    #      'kf_e2h': 0.005, 
    #      'T_h': 15,
    #      'T_g': 0, 
    #      'Dg': 1,
    #     }
    # C1 = dc.conc_liver_1i_ic_hfd_nsu(ca, t, **p)
    # assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    # p = {'v_e_li': 0.1, 
    #      'k_e2h': 0.005, 
    #      'Ti_h': 15,
    #      'Tf_h': 15,
    #      'T_g': 0,
    #      'Dg': 1,
    #     }
    # C1 = dc.conc_liver_1i_ic_hfd_nse(ca, t, **p)
    # assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    # p = {'v_e_li': 0.1, 
    #      'ki_e2h': 0.005, 
    #      'kf_e2h': 0.005, 
    #      'Ti_h': 15,
    #      'Tf_h': 15,
    #      'T_g': 0,
    #      'Dg': 1,
    #     }
    # C1 = dc.conc_liver_1i_ic_hfd_nsue(ca, t, **p)
    # assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    # p = {'v_e_li': 0.1, 
    #      'k_e2h': 0.005, 
    #      'T_h': 1500,
    #      'T_g': 10,
    #      'Dg': 0.5,
    #     }
    # C0 = dc.conc_liver_1i_ic_hfd(ca, t, **p)

    # p = {'v_e_li': 0.1, 
    #      'k_e2h': 0.005, 
    #      'T_g': 10,
    #      'Dg': 0.5,
    #     }
    # C1 = dc.conc_liver_1i_ic_hfdu(ca, t, **p)
    # assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    # p = {'v_e_li': 0.1, 
    #      'ki_e2h': 0.005, 
    #      'kf_e2h': 0.005, 
    #      'T_g': 10,
    #      'Dg': 0.5,
    #     }
    # C1 = dc.conc_liver_1i_ic_hfdu_nsu(ca, t, **p)
    # assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'v_e_li': 0.1, 
         'k_e2h': 0.005, 
         'T_h': 15,
        }
    C0 = dc.conc_liver_1i_ic_hf(ca, t, **p)

    C = dc.conc_liver_1i_ic_hf(ca, t, **p)
    assert np.array_equal(C, C0)

    p = {'v_e_li': 0.1, 
         'k_e2h': 0.005, 
         'T_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf((ca, ca), t, **p)

    p = {'v_e_li': 0.1, 
         'ki_e2h': 0.005, 
         'kf_e2h': 0.005, 
         'T_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf_nsu((ca, ca), t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'v_e_li': 0.1, 
         'k_e2h': 0.005, 
         'Ti_h': 15,
         'Tf_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf_nse((ca, ca), t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'v_e_li': 0.1, 
         'ki_e2h': 0.005,
         'kf_e2h': 0.005,
         'Ti_h': 15,
         'Tf_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_hf_nsue((ca, ca), t, **p)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'v_e_li': 0.1, 
         'k_e2h': 0.005, 
         'T_h': 15,
         'ffa': 1,
        }
    C0 = dc.conc_liver_2i_ic_hf((ca, ca), t, **p)

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'E_li': 0.001, 
         'T_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 0.1

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'E_li': 0.001, 
         'T_h': 15,
        }
    C1 = dc.conc_liver_1i_ic(ca, t, **p)

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'Ei_li': 0.001, 
         'Ef_li': 0.001, 
         'T_h': 15,
        }
    C1 = dc.conc_liver_1i_ic_nsu(ca, t, **p)

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'E_li': 0.001, 
         'Ti_h': 15,
         'Tf_h': 15,
        }
    C1 = dc.conc_liver_1i_ic_nse(ca, t, **p)

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5,        
         'Ei_li': 0.001, 
         'Ef_li': 0.001, 
         'Ti_h': 15,
         'Tf_h': 15,
        }
    C1 = dc.conc_liver_1i_ic_nsue(ca, t, **p)

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'Ei_li': 0.001, 
         'Ef_li': 0.001,  
         'T_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_nsu((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'E_li': 0.001,  
         'Ti_h': 15,
         'Tf_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_nse((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'Ei_li': 0.001, 
         'Ef_li': 0.001,  
         'Ti_h': 15,
         'Tf_h': 15,
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_nsue((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'E_li': 0.001, 
         'T_h': 1500,
         'ffa': 1,
        }
    C0 = dc.conc_liver_2i_ic((ca, ca), t, **p)

    C = dc.conc_liver_2i_ic((ca, ca), t, **p)
    assert np.array_equal(C, C0)

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'E_li': 0.001, 
         'ffa': 1,
        }
    C1 = dc.conc_liver_2i_ic_u((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'Ei_li': 0.001, 
         'Ef_li': 0.001,
         'ffa': 1,
        }
    C1 =  dc.conc_liver_2i_ic_u_nsu((ca, ca), t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'E_li': 0.001, 
        }
    C1 = dc.conc_liver_1i_ic_u(ca, t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'v_e_li': 0.1 / (1 - 0.001), 
         'F_p_li': 5, 
         'Ei_li': 0.001, 
         'Ef_li': 0.001,
        }
    C1 =  dc.conc_liver_1i_ic_u_nsu(ca, t, **p)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1




if __name__=='__main__':
    test_liver()
    
    print("All liver tests passed!")