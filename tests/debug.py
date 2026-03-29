import numpy as np

from dcmri import pk_aorta
from dcmri import pk

import dcmri as dc

from dcmri import tissue


def test_bug():

    dt_init, tmax_init = 0.5, 240
    t_init = np.arange(0, tmax_init, dt_init, dtype=float)
    ca = pk_aorta.aif_tristan(t_init, agent='gadodiamide', BAT=20)

    Mx = tissue.Mz('NX', 'RF', 'SS')   # its a 2-comp SS problewm

    M1 = Mx(ca) 
    M2 = Mx(ca * 0) 

    print('\n\n')
    print(M1[1,:5]-M2[1,:5])



if __name__ == "__main__":

    test_bug()
    
    print('Done!!')