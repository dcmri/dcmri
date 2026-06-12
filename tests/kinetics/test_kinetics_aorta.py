import numpy as np

import dcmri as dc


def test_flux_aorta():
    Jv = np.arange(10)
    j = dc.conc_aorta(Jv, dt=3, heartlung=['pfcomp', {'T':2, 'D':0.2}])
    j = dc.conc_aorta(Jv, dt=3, heartlung=['pfcomp', {'T':2, 'D':0.2}], max_it=1)
    j = dc.flux_aorta(Jv, dt=3, E=0.5, FFkl=0.5, heartlung=['pfcomp', {'T':2, 'D':0.2}])
    j = dc.flux_aorta(Jv, dt=3, E=0.5, FFkl=0.5, heartlung=['pfcomp', {'T':2, 'D':0.2}], max_it=1)
    j = dc.flux_aorta_hlo(Jv, dt=3, E=0.5, heartlung=['pfcomp', {'T':2, 'D':0.2}])
    j = dc.flux_aorta_hlo(Jv, dt=3, E=0.5, heartlung=['pfcomp', {'T':2, 'D':0.2}], max_it=1)
    j = dc.flux_aorta_hlol(Jv, dt=3, FFl=0.25, heartlung=['pfcomp', {'T':2, 'D':0.2}])
    j = dc.flux_aorta_hlol(Jv, dt=3, FFl=0.25, heartlung=['pfcomp', {'T':2, 'D':0.2}], max_it=1)
    j = dc.flux_aorta_hlok(Jv, dt=3, FFk=0.25, heartlung=['pfcomp', {'T':2, 'D':0.2}])
    j = dc.flux_aorta_hlok(Jv, dt=3, FFk=0.25, heartlung=['pfcomp', {'T':2, 'D':0.2}], max_it=1)


if __name__=='__main__':
    test_flux_aorta()