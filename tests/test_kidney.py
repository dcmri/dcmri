import numpy as np
import dcmri as dc

from dcmri.kidney import Conc



def test_conc_kidney():
    dt = 1.5
    t = dt*np.arange(20)
    ca = np.ones(20)
    p = {'Fp': 0.01, 'vp': 0.2, 'Ft': 0.005, 'Tt': 120}
    p['Tp'] = p['vp'] / (p['Fp'] + p['Ft'])
    C = Conc('2CF', **p)(ca, dt=dt).sum(axis=0)
    assert round(C[10], 1) == 0.1
    C = Conc('2CF', **p)(ca, dt=dt)
    assert round(C[1,10], 2) == 0.02
    C = Conc('HF', **p)(ca, dt=dt).sum(axis=0)
    assert round(C[10], 1) == 0.3
    C = Conc('HF', **p)(ca, dt=dt)
    assert round(C[1,10], 2) == 0.07
    h = [1,2,3,4,3,2,1]
    C = Conc('FN', **p)(ca, dt=dt, ht=h).sum(axis=0)
    assert round(C[10], 1) == 0.2
    C = Conc('FN', **p)(ca, dt=dt, ht=h)
    assert round(C[1,10], 2) == 0.02

    try:
        Conc('X')(ca)
    except:
        assert True
    else:
        assert False


if __name__ == '__main__':

    test_conc_kidney()

    print('All kidney tests passed!!')