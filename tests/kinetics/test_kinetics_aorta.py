import numpy as np

import dcmri as dc


def test_flux_aorta():
    Jv = np.arange(10)
    j = dc.flux_aorta(Jv, dt=3, heartlung={'model': 'pfcomp', 'params': {'T':2, 'D':0.2}})
    j = dc.flux_aorta(Jv, dt=3, heartlung={'model': 'pfcomp', 'params': {'T':2, 'D':0.2}}, max_it=1)


if __name__=='__main__':
    test_flux_aorta()