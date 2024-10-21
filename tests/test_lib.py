import numpy as np
import dcmri as dc

import matplotlib.pyplot as plt


def test_fetch():
    data = dc.fetch('tristan6drugs')
    assert 'FA' in data[0]


def test_influx_step():

    weight = 70
    conc = 0.5
    t0 = 5
    dose = 0.2
    rate = 3
    dt = 0.1

    t = np.arange(0, 20, dt)
    j = dc.influx_step(t, weight, conc, dose, rate, t0)

    assert np.around(np.sum(j)*dt) == np.around(weight*dose*conc)

    # Test exceptions
    try:
        j = dc.influx_step(t, 0*weight, conc, dose, rate, t0)
    except:
        assert True
    else:
        assert False

    try:
        j = dc.influx_step(t, weight, conc, 0.01*dose, rate, t0)
    except:
        assert True
    else:
        assert False


def test_ca_conc():

    assert dc.ca_conc('gadoxetate') == 0.25
    assert dc.ca_conc('gadobutrol') == 1.0
    assert dc.ca_conc('gadobenate') == 0.5
    try:
        dc.ca_conc('MyAgent') == 1.0
    except:
        assert True
    else:
        assert False

def test_ca_std_dose():

    assert dc.ca_std_dose('gadoxetate') == 0.1
    assert dc.ca_std_dose('gadobutrol') == 0.1
    assert dc.ca_std_dose('gadopiclenol') == 0.1
    assert dc.ca_std_dose('gadoterate') == 0.2
    try:
        dc.ca_std_dose('myagent') == 1.0
    except:
        assert True
    else:
        assert False
        
def test_relaxivity():
    assert dc.relaxivity(4.7, 'blood', 'gadobutrol') == 1000*4.7
    assert dc.relaxivity(4.7, 'plasma', 'gadobutrol') == 1000*4.7
    try:
        dc.relaxivity(4.7, 'water', 'gadobutrol')
    except:
        assert True
    else:
        assert False
    assert dc.relaxivity(3.0, 'hepatocytes', 'gadoxetate') == 9800
    assert dc.relaxivity(3.0, 'hepatocytes', 'gadodiamide') == 4000

def test_T1():
    assert dc.T1(4.7, 'liver') == 1/1.281
    try:
        dc.T1(4.7, 'hair')
    except:
        assert True
    else:
        assert False

def test_T2():
    assert dc.T2(1.5, 'csf') == 1.99
    try:
        dc.T2(4.7, 'hair')
    except:
        assert True
    else:
        assert False

def test_PD():
    assert dc.PD('csf') == 0.98
    try:
        dc.PD('hair')
    except:
        assert True
    else:
        assert False

def test_perfusion():
    assert dc.perfusion('Fb', 'csf') == 0.0
    assert dc.perfusion('vb', 'csf') == 0.0
    assert dc.perfusion('PS', 'csf') == 0.0
    assert dc.perfusion('vi', 'csf') == 0.0
    try:
        dc.perfusion('Fb', 'hair')
    except:
        assert True
    else:
        assert False

def test_shepp_logan():

    n=64
    roi = dc.shepp_logan(n=n)
    im = dc.shepp_logan('T1', 'T2', 'PD', 'Fb', 'vb', 'PS', 'vi', n=n)

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
    im = dc.shepp_logan('Fb', n=n)
    vals = im[roi['CSF left']==1]
    assert 0 == np.amin(vals)
    assert 0 == np.amax(vals)


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
    assert np.round(1000*np.amax(ca), 1) == 0.5



if __name__ == "__main__":

    test_fetch()
    test_influx_step()
    test_ca_conc()
    test_ca_std_dose()
    test_relaxivity()
    test_T1()
    test_T2()
    test_PD()
    test_perfusion()
    test_shepp_logan()
    test_aif_parker()
    test_aif_tristan_rat()

    print('All lib tests passed!!')
