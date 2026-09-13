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
    assert dc.r1(4.7, 'blood', 'gadobutrol') == 1000 * 4.7

    # Special case hepatocytes
    assert dc.r1(4.7, 'hepatocytes', 'gadobutrol') == 1000 * 4.7
    assert dc.r1(4.7, 'hepatocytes', 'gadoxetate') == 1000 * 7.6

    # By default return blood values for unknown tissues
    assert dc.r1(4.7, 'concrete', 'gadobutrol') == dc.r1(4.7, 'plasma', 'gadobutrol')

    # Unless when force is set
    try:
        dc.r1(4.7, 'concrete', 'gadobutrol', force=True)
    except:
        pass
    else:
        assert False, 'Setting force should cause error for unknown tissues'

    # Error if field strength is unknown
    try:
        dc.r1(-1, 'concrete', 'gadobutrol')
    except:
        pass
    else:
        assert False, 'Setting unknown field strength should raise an error'

    # Error if agent is unknown
    try:
        dc.r1(3.0, 'concrete', 'gadobrol')
    except:
        pass
    else:
        assert False, 'Setting unknown agent should raise an error'


def test_r2():
    assert dc.r2(4.7, 'blood', 'gadobutrol') == 1000 * 6.1

    # By default return blood values for unknown tissues
    assert dc.r2(4.7, 'concrete', 'gadobutrol') == dc.r2(4.7, 'plasma', 'gadobutrol')

    # Unless when force is set
    try:
        dc.r2(4.7, 'concrete', 'gadobutrol', force=True)
    except:
        pass
    else:
        assert False, 'Setting force should cause error for unknown tissues'

    # Error if field strength is unknown
    try:
        dc.r2(-1, 'concrete', 'gadobutrol')
    except:
        pass
    else:
        assert False, 'Setting unknown field strength should raise an error'

    # Error if agent is unknown
    try:
        dc.r2(3.0, 'concrete', 'gadobrol')
    except:
        pass
    else:
        assert False, 'Setting unknown agent should raise an error'  


def test_r2s():
    assert dc.r2s(3.0, 'blood', 'gadobutrol') == 10e3
    assert dc.r2s(3.0, 'concrete', 'gadobutrol') == dc.r2s(3.0, 'blood', 'gadobutrol')

    # Unless when force is set
    try:
        dc.r2s(3.0, 'concrete', 'gadobutrol', force=True)
    except:
        pass
    else:
        assert False, 'Setting force should cause error for unknown tissues'


def test_T1():
    assert dc.T1(4.7, 'liver') == 1/1.281

    # By default return blood values for unknown tissues
    assert dc.T1(4.7, 'concrete') == dc.T1(4.7, 'blood')

    # Unless when force is set
    try:
        dc.T1(4.7, 'concrete', force=True)
    except:
        pass
    else:
        assert False, 'Setting force should cause error for unknown tissues'

    # Error if field strength is unknown
    try:
        dc.T1(-1, 'concrete')
    except:
        pass
    else:
        assert False, 'Setting unknown field strength should raise an error'


def test_T2():
    assert dc.T2(1.5, 'csf') == 1.99

    # By default return blood values for unknown tissues
    assert dc.T2(3.0, 'concrete') == dc.T2(3.0, 'blood')

    try:
        dc.T2(3.0, 'concrete', force=True)
    except:
        pass
    else:
        assert False, 'Setting force should cause error for unknown tissues'

    try:
        assert dc.T2(-1, 'concrete')
    except:
        pass
    else:
        assert False, 'Setting unknown field strength should raise an error'


def test_T2_star():
    assert dc.T2s(1.5, 'arterial blood') == 0.250

    # By default return blood values for unknown tissues
    assert dc.T2s(3.0, 'concrete') == dc.T2s(3.0, 'blood')

    try:
        dc.T2s(3.0, 'concrete', force=True)
    except:
        pass
    else:
        assert False, 'Setting force should cause error for unknown tissues'

    try:
        assert dc.T2s(-1, 'concrete')
    except:
        pass
    else:
        assert False, 'Setting unknown field strength should raise an error'


def test_PD():
    assert dc.PD('csf') == 0.98
    assert dc.PD('concrete') == dc.PD('blood')

    try:
        dc.PD('concrete', force=True)
    except:
        pass
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
    test_r2()
    test_r2s()
    test_T1()
    test_T2()
    test_T2_star()
    test_PD()
    test_perfusion()


    print('All lib tests passed!!')
