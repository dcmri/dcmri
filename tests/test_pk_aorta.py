import numpy as np
import dcmri as dc


def test_aif_tristan():

    t = np.arange(0, 6*60, 1)
    ca = dc.aif_tristan(t)
    assert round(max(ca), 4) == 0.0035


def test_flux_aorta():
    
    J = np.ones(100)
    Ja = dc.flux_aorta(J, FFkl=0.2)
    assert round(max(Ja), 2) == 2.87



if __name__ == "__main__":

    test_aif_tristan()
    test_flux_aorta()

    print('All pk_aorta tests passed!!')
