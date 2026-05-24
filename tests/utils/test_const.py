from dcmri import const



def test_ca_conc():

    assert const.ca_conc('gadoxetate') == 0.25
    assert const.ca_conc('gadobutrol') == 1.0
    assert const.ca_conc('gadobenate') == 0.5
    try:
        const.ca_conc('MyAgent') == 1.0
    except:
        assert True
    else:
        assert False

def test_ca_std_dose():

    assert const.ca_std_dose('gadoxetate') == 0.1
    assert const.ca_std_dose('gadobutrol') == 0.1
    assert const.ca_std_dose('gadopiclenol') == 0.1
    assert const.ca_std_dose('gadoterate') == 0.2
    try:
        const.ca_std_dose('myagent') == 1.0
    except:
        assert True
    else:
        assert False
        
def test_r1():
    assert const.r1(4.7, 'blood', 'gadobutrol') == 1000*4.7
    assert const.r1(4.7, 'plasma', 'gadobutrol') == 1000*4.7
    try:
        const.r1(4.7, 'water', 'gadobutrol')
    except:
        assert True
    else:
        assert False
    assert const.r1(3.0, 'hepatocytes', 'gadoxetate') == 9800
    assert const.r1(3.0, 'hepatocytes', 'gadodiamide') == 4000
    

def test_r2s():
    assert const.r2s(3.0, 'blood', 'gadobutrol') == 10e3
    try:
        const.r2s(3.0, 'water', 'gadobutrol')
    except:
        pass
    else:
        assert False


def test_T1():
    assert const.T1(4.7, 'liver') == 1/1.281
    try:
        const.T1(4.7, 'hair')
    except:
        assert True
    else:
        assert False

def test_T2():
    assert const.T2(1.5, 'csf') == 1.99
    try:
        const.T2(4.7, 'hair')
    except:
        assert True
    else:
        assert False

def test_PD():
    assert const.PD('csf') == 0.98
    try:
        const.PD('hair')
    except:
        assert True
    else:
        assert False

def test_perfusion():
    assert const.perfusion('Fb', 'csf') == 0.0
    assert const.perfusion('vb', 'csf') == 0.0
    assert const.perfusion('PS', 'csf') == 0.0
    assert const.perfusion('vi', 'csf') == 0.0
    try:
        const.perfusion('Fb', 'hair')
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
