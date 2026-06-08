import dcmri as dc



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
        
def test_r1():
    assert dc.r1(4.7, 'blood', 'gadobutrol') == 1000*4.7
    assert dc.r1(4.7, 'plasma', 'gadobutrol') == 1000*4.7
    try:
        dc.r1(4.7, 'water', 'gadobutrol')
    except:
        assert True
    else:
        assert False
    assert dc.r1(3.0, 'hepatocytes', 'gadoxetate') == 9800
    assert dc.r1(3.0, 'hepatocytes', 'gadodiamide') == 4000
    

def test_r2s():
    assert dc.r2s(3.0, 'blood', 'gadobutrol') == 10e3
    try:
        dc.r2s(3.0, 'water', 'gadobutrol')
    except:
        pass
    else:
        assert False


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


if __name__ == "__main__":
    test_ca_conc()
    test_ca_std_dose()
    test_r1()
    test_r2s()
    test_T1()
    test_T2()
    test_PD()
    test_perfusion()


    print('All lib tests passed!!')
