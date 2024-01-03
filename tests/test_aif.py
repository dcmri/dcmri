import numpy as np
import dcmri


def test_aif_parker():

    t = np.arange(0, 6*60, 1)
    ca = dcmri.aif_parker(t)

    # Test that this generates values in the right range
    assert np.round(1000*np.amax(ca)) == 6

    # Add a delay and check that this produces the same maximum
    ca = dcmri.aif_parker(t, BAT=60)
    assert np.round(1000*np.amax(ca)) == 6

    # Try with list as input
    ca = dcmri.aif_parker([50, 100, 150])
    assert np.array_equal(np.round(1000*ca), [1, 1, 1]) 

    # Or just a single variable
    ca = dcmri.aif_parker(100)
    assert 1000*ca == 0.7929118932243691

    # Check that an error message is generated if BAT is not a scalar
    try:
        ca = dcmri.aif_parker(t, BAT=[60,120])
    except: 
        assert True
    else:
        assert False


if __name__ == "__main__":

    test_aif_parker()

    print('All AIF tests passed!!')
