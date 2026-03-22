import numpy as np
import dcmri as dc


def test_signal():
    s = dc.signal(Mz=1, R2=10, S0=2, FA=15, TE=0.005, noise_sdev=0)
    assert 0.4 < s < 0.6
    s = dc.signal(Mz=[1], R2=10, S0=2, FA=15, TE=0.005, noise_sdev=0)
    assert 0.4 < s[0] < 0.6
    s = dc.signal(Mz=0.5*np.ones((2,6)), R2=10, S0=2, FA=15, TE=0.005, noise_sdev=0)
    assert 0.4 < s[0] < 0.6
    s = dc.signal(Mz=0.5*np.ones((2,6)), R2=10, S0=2, FA=15, TE=0.005, noise_sdev=1)
    assert 1 < s[0] < 1.5
    try:
        s = dc.signal(Mz=0.5*np.ones((2,6,7)), R2=10, S0=2, FA=15, TE=0.005, noise_sdev=1)
    except ValueError:
        pass
    else:
        assert False


if __name__ == "__main__":
    test_signal()
    print('All sig tests passing!')