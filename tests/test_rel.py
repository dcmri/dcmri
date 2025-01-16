import numpy as np
import dcmri as dc



def test_c_lin():
    R1 = np.arange(10) + 5
    c = dc.c_lin(R1, 0.1)
    assert np.sum(c) == 450.0
    R1 = np.stack((R1, 2*R1))
    c = dc.c_lin(R1, [5,4])
    assert np.sum(c) == 31.5
    c = dc.c_lin(R1, 5)
    assert np.sum(c) == 27


def test_relax():
    c = np.arange(10)
    R1 = dc.relax(c, 1, 5)
    assert np.sum(R1) == 235
    c = np.stack((c, 2*c))
    R1 = dc.relax(c, [1,2], [5,4])
    assert np.sum(R1) == 615


if __name__ == "__main__":

    test_c_lin()
    test_relax()

    print('All rel tests passing!')