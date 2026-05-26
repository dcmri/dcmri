import numpy as np

import dcmri.kinetics.lib as pk


def test_flux_aorta():
    Jv = np.arange(10)
    j = pk.flux_aorta(Jv, dt=3, E=0.5, FFkl=0.5, heartlung=['pfcomp', (2, 0.2)])
    j = pk.flux_aorta(Jv, dt=3, E=0.5, FFkl=0.5, heartlung=['pfcomp', (2, 0.2)], max_it=1)
    j = pk.flux_aorta_hlo(Jv, dt=3, E=0.5, heartlung=['pfcomp', (2, 0.2)])
    j = pk.flux_aorta_hlo(Jv, dt=3, E=0.5, heartlung=['pfcomp', (2, 0.2)], max_it=1)
    j = pk.flux_aorta_hlol(Jv, dt=3, FFl=0.25, heartlung=['pfcomp', (2, 0.2)])
    j = pk.flux_aorta_hlol(Jv, dt=3, FFl=0.25, heartlung=['pfcomp', (2, 0.2)], max_it=1)


if __name__=='__main__':
    test_flux_aorta()