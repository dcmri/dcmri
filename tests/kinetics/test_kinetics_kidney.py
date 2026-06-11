import numpy as np

from dcmri.kinetics.kidney import dpars_kidney
import dcmri as dc


def test_kidney():
    dt = 1.5
    ca = np.ones(20)
    
    p = {
        'Fp': 0.01, 
        'vp': 0.2, 
        'Ft': 0.005, 
        'FF': 0.005 / 0.01,
        'Tt': 120, 
        'h': [1,2,3,4,3,2,1],
        'vol': 150,
        'fc': 0.8,
        'Tglom': 2, 
        'Tpt': 8, 
        'Tlh': 20, 
        'Tdt': 15, 
        'Tcd': 10,
    }
    p = p | dpars_kidney(p)

    pm = {k: v for k, v in p.items() if k in ['Fp', 'vp', 'FF', 'Tt']}
    C = dc.conc_kidney_2cf(ca, dt=dt, **pm)
    assert round(C[1, 10], 2) == 0.02

    pm = {k: v for k, v in p.items() if k in ['vp', 'Ft', 'Tt']}
    C = dc.conc_kidney_hf(ca, dt=dt, **pm)
    assert round(C[1, 10], 2) == 0.07

    pm = {k: v for k, v in p.items() if k in ['Fp', 'vp', 'FF', 'h']}
    C = dc.conc_kidney_fn(ca, dt=dt, **pm)
    assert round(C[1, 10], 2) == 0.02
    C = dc.conc_kidney_fn(ca, t=dt*np.arange(ca.size), **pm)
    assert round(C[1, 10], 2) == 0.02

    pm = {k: v for k, v in p.items() if k in ['Fp', 'Eg', 'fc', 'Tglom', 'Tv', 'Tpt', 'Tlh', 'Tdt', 'Tcd']}
    Ccor, Cmed = dc.conc_kidney_cm9(ca, dt=dt, **pm)
    assert round(Ccor[-1, -1], 3) == 0.015
    assert round(Cmed[-1, -1], 3) == 0.005

    # Exceptions
    dpars_kidney({'Fp': 1}, H=1)



if __name__=='__main__':
    test_kidney()