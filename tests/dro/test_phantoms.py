import numpy as np
import dcmri as dc


def test_shepp_logan():
    n = 64
    roi = dc.phantoms.shepp_logan(n=n)
    im = dc.phantoms.shepp_logan('T1', 'T2', 'PD', 'Fb', 'vb', 'PS', 'vi', n=n)

    vals = im['Fb'][roi['CSF left']==1]
    assert 0 == np.amin(vals)
    assert 0 == np.amax(vals)
    vals = im['vb'][roi['CSF left']==1]
    assert 0 == np.amin(vals)
    assert 0 == np.amax(vals)
    vals = im['PS'][roi['CSF left']==1]
    assert 0 == np.amin(vals)
    assert 0 == np.amax(vals)
    vals = im['vi'][roi['CSF left']==1]
    assert 0 == np.amin(vals)
    assert 0 == np.amax(vals)

    # Special case - 1 parameter - does not return dict
    im = dc.phantoms.shepp_logan('Fb', n=n)
    vals = im[roi['CSF left']==1]
    assert 0 == np.amin(vals)
    assert 0 == np.amax(vals)


if __name__ == "__main__":

    test_shepp_logan()

    print('All phantoms tests passed!!')
