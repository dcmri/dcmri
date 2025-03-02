import numpy as np
import dcmri as dc

def test_params_kidney():
    dc.params_kidney('2CF')

def test_conc_kidney_cm():
    t = np.arange(0, 300, 1.5)
    ca = dc.aif_parker(t, BAT=20)
    params = 0.03, 0.15, 0.8, 4, 10, 60, 60, 30, 30
    Cc, Cm = dc.conc_kidney_cm(ca, *params, t=t, kinetics='7C', sum=True)
    assert round(Cm[100], 4) == 0.0003
    Cc, Cm = dc.conc_kidney_cm(ca, *params, t=t, kinetics='7C', sum=False)
    assert round(Cm[2,100], 5) == 7e-5
    try:
        dc.conc_kidney_cm(ca, kinetics='X')
    except:
        assert True
    else:
        assert False


def test_conc_kidney():
    dt = 1.5
    t = dt*np.arange(20)
    ca = np.ones(20)
    Fp, vp, Ft, Tt = 0.01, 0.2, 0.005, 120
    C = dc.conc_kidney(ca, Fp, vp, Ft, Tt, dt=dt, sum=True, kinetics='2CF')
    assert round(C[10], 1) == 0.1
    C = dc.conc_kidney(ca, Fp, vp, Ft, Tt, dt=dt, sum=False, kinetics='2CF')
    assert round(C[1,10], 2) == 0.02
    C = dc.conc_kidney(ca, vp, Ft, Tt, dt=dt, sum=True, kinetics='HF')
    assert round(C[10], 1) == 0.3
    C = dc.conc_kidney(ca, vp, Ft, Tt, dt=dt, sum=False, kinetics='HF')
    assert round(C[1,10], 2) == 0.07
    Tp = vp/(Fp+Ft)
    h = [1,2,3,4,3,2,1]
    C = dc.conc_kidney(ca, Fp, Tp, Ft, h, dt=dt, sum=True, kinetics='FN')
    assert round(C[10], 1) == 0.2
    C = dc.conc_kidney(ca, Fp, Tp, Ft, h, t=t, sum=True, kinetics='FN')
    assert round(C[10], 1) == 0.1
    C = dc.conc_kidney(ca, Fp, Tp, Ft, h, dt=dt, sum=False, kinetics='FN')
    assert round(C[1,10], 2) == 0.02

    try:
        dc.conc_kidney(ca, kinetics='X')
    except:
        assert True
    else:
        assert False


if __name__ == '__main__':

    test_params_kidney()
    test_conc_kidney()
    test_conc_kidney_cm()

    print('All kidney tests passed!!')