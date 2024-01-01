import numpy as np
import dcmri


def test_aif_parker():

    t = np.arange(0, 6*60, 1)
    ca = dcmri.aif_parker(t)

    # Test that this generates values in the right range
    assert np.round(1000*np.amax(ca)) == 6


if __name__ == "__main__":

    test_aif_parker()

    print('All AIF tests passed!!')
