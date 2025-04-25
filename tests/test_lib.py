import numpy as np
import dcmri as dc


def test_fetch():
    data = dc.fetch('tristan_rats_healthy_six_drugs')
    dmr = dc.read_dmr(data)
    assert 'FA' in dmr['data']


def test_ca_injection():

    weight = 70
    conc = 0.5
    t0 = 5
    dose = 0.2
    rate = 3
    dt = 0.1

    t = np.arange(0, 20, dt)
    j = dc.ca_injection(t, weight, conc, dose, rate, t0)

    assert np.around(np.sum(j)*dt) == np.around(weight*dose*conc)

    # Test exceptions
    try:
        j = dc.ca_injection(t, 0*weight, conc, dose, rate, t0)
    except:
        assert True
    else:
        assert False

    try:
        j = dc.ca_injection(t, weight, conc, 0.01*dose, rate, t0)
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




if __name__ == "__main__":

    test_fetch()
    test_ca_injection()
    test_ca_conc()
    test_ca_std_dose()
    test_relaxivity()
    test_T1()
    test_T2()
    test_PD()
    test_perfusion()
    test_shepp_logan()

    print('All lib tests passed!!')
