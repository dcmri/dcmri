import numpy as np
import dcmri as dc


def test_signal():
    sequence = 'None'
    s = dc.signal(sequence, R1=1, R2=10, S0=2, FAR=15, TE=0.005, noise_sdev=0)
    assert 0.4 < s < 0.6
    s = dc.signal(sequence, R1=[1], R2=10, S0=2, FAR=15, TE=0.005, noise_sdev=0)
    assert 0.4 < s[0] < 0.6
    s = dc.signal(sequence, R1=np.ones((2,6)), R2=10, S0=2, FAR=15, TE=0.005, noise_sdev=0)
    assert 0.95 < s[0] < 1.05
    s = dc.signal(sequence, R1=np.ones((2,6)), R2=10, S0=2, FAR=15, TE=0.005, noise_sdev=1)
    assert 1 < s[0] < 2.0


if __name__ == "__main__":
    test_signal()
    print('All sig tests passing!')