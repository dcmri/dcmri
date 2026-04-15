import numpy as np
from dcmri import pk


def test_ca_injection():

    weight = 70
    conc = 0.5
    t0 = 5
    dose = 0.2
    rate = 3
    dt = 0.1

    t = np.arange(0, 20, dt)
    j = pk.ca_injection(t, weight, conc, dose, rate, t0)

    assert np.around(np.sum(j)*dt) == np.around(weight*dose*conc)

    # Test exceptions
    try:
        j = pk.ca_injection(t, 0*weight, conc, dose, rate, t0)
    except:
        assert True
    else:
        assert False

    try:
        j = pk.ca_injection(t, weight, conc, 0.01*dose, rate, t0)
    except:
        assert True
    else:
        assert False





if __name__ == "__main__":

    test_ca_injection()
    print('All lib tests passed!!')
