import numpy as np
import dcmri as dc
from dcmri.cort_med import Conc



def test_conc_kidney_cm():
    t = np.arange(0, 300, 1.5)
    ca = dc.aif_parker(t, BAT=20)
    p = {'T_a':0, 'Fp':0.03, 'Eg':0.15, 'fc':0.8, 'Tglom':4, 'Tv':10, 'Tpt':60, 'Tlh':60, 'Tdt':30, 'Tcd':30}
    Cc, Cm = Conc('7C', **p)(ca, t=t)
    assert round(Cm[:,100].sum(), 4) == 0.0003
    Cc, Cm = Conc('7C', **p)(ca, t=t)
    assert round(Cm[2,100], 5) == 7e-5

    try:
        Conc('X')
    except:
        assert True
    else:
        assert False



if __name__ == '__main__':
    test_conc_kidney_cm()
    print('All cort_med tests passed!!')