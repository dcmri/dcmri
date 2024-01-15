import numpy as np
import dcmri

def test_signal_spgress():
    dcmri.signal_spgress(1,1,1,1)

def test_sample():
    dcmri.sample(np.arange(10), np.arange(10), np.arange(5), 1)


if __name__ == "__main__":

    test_signal_spgress()
    test_sample()

    print('All inv tests passed!!')