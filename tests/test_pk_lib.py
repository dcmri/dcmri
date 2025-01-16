import numpy as np
import dcmri as dc


def test_aif_parker():

    t = np.arange(0, 6*60, 1)
    ca = dc.aif_parker(t)

    # Test that this generates values in the right range
    assert np.round(1000*np.amax(ca)) == 6

    # Add a delay and check that this produces the same maximum
    ca = dc.aif_parker(t, BAT=60)
    assert np.round(1000*np.amax(ca)) == 6

    # Try with list as input
    ca = dc.aif_parker([50, 100, 150])
    assert np.array_equal(np.round(1000*ca), [1, 1, 1]) 

    # Or just a single variable
    ca = dc.aif_parker(100)
    assert 1000*ca == 0.7929118932243691

    # Check that an error message is generated if BAT is not a scalar
    try:
        ca = dc.aif_parker(t, BAT=[60,120])
    except: 
        assert True
    else:
        assert False


def test_aif_tristan_rat():
    
    t = np.arange(0, 6*60, 1)
    ca = dc.aif_tristan_rat(t)
    assert np.round(1000*np.amax(ca), 1) == 0.3



if __name__ == "__main__":

    test_aif_parker()
    test_aif_tristan_rat()

    print('All pk_lib tests passed!!')
