from dcmri.core.tools import get_quantity, get_sequence, build_varname, parse_varname, extend_varname

def test_get_sequence():
    print(get_sequence('name'))
    print(get_sequence('steady-state', '3D-SPGR-SS'))
    print(get_sequence('tissue_params', '3D-SPGR-SS'))
    print(get_sequence('mz_prep_inflow', '3D-SPGR-SS'))
    print(get_sequence('mz_prep_tissue', '3D-SPGR-SS'))
    print(get_sequence('prep_params', '3D-SPGR-SS'))

def test_get_quantity():
    print('ki_1_e2h: ', get_quantity('ki_1_e2h'))
    print('T_ao: ', get_quantity('T_ao'))
    print('T_b: ', get_quantity('T_b'))
    print('T_b_ao: ', get_quantity('T_b_ao'))
    print('T_b2i_ao: ', get_quantity('T_b2i_ao'))
    print('PSw_c2u: ', get_quantity('PSw_c2u'))
    print('PSw_5_c2u: ', get_quantity('PSw_5_c2u'))
    print('PSw_5: ', get_quantity('PSw_5'))
    print('PSw_5_b: ', get_quantity('PSw_5_b'))
    print('PSw_5_c2u_ki: ', get_quantity('PSw_5_c2u_ki'))

def test_parser():
    assert 'PS_e2h' == build_varname("PS", compartment="e2h")
    assert 'PS_1_e2h_li' == build_varname("PS", index=1, compartment="e2h", roi="li")
    assert parse_varname("PS_e2h") == {'name': 'PS', 'index': None, 'compartment': 'e2h', 'roi': None}
    assert parse_varname("PS_1_e2h_li") == {'name': 'PS', 'index': 1, 'compartment': 'e2h', 'roi': 'li'}
    assert parse_varname("PS_1_li") == {'name': 'PS', 'index': 1, 'compartment': None, 'roi': 'li'}
    assert parse_varname("PS_1") == {'name': 'PS', 'index': 1, 'compartment': None, 'roi': None}
    assert parse_varname("PS_e2h_li") == {'name': 'PS', 'index': None, 'compartment': 'e2h', 'roi': 'li'}
    assert parse_varname('c_ar') == {'name': 'c', 'index': None, 'compartment': None, 'roi': 'ar'}
    
    try: 
        build_varname("PS", compartment="e2x")   # x not in COMPS
    except:
        pass
    else:
        assert False

    assert extend_varname('E_ki', index=1) == 'E_1_ki'
    assert extend_varname('E_2_ki', index=1) == 'E_1_ki'


if __name__ == '__main__':
    test_get_sequence()
    test_get_quantity()
    test_parser()
    print('All tools tests passed!!')