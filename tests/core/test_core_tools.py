from dcmri.core.pseq_tools import get_sequence
from dcmri.core.quantity import Qty

def test_get_sequence():
    print(get_sequence('name'))
    print(get_sequence('steady-state', '3D-SPGR-SS'))
    print(get_sequence('tissue_params', '3D-SPGR-SS'))
    print(get_sequence('mz_prep_inflow', '3D-SPGR-SS'))
    print(get_sequence('mz_prep_tissue', '3D-SPGR-SS'))
    print(get_sequence('prep_params', '3D-SPGR-SS'))

def test_qty():
    print('ki_1_e2h: ', Qty('ki_1_e2h')._dct)
    print('T_ao: ', Qty('T_ao')._dct)
    print('T_b: ', Qty('T_b')._dct)
    print('T_b_ao: ', Qty('T_b_ao')._dct)
    print('T_b2i_ao: ', Qty('T_b2i_ao')._dct)
    print('PSw_c2u: ', Qty('PSw_c2u')._dct)
    print('PSw_5_c2u: ', Qty('PSw_5_c2u')._dct)
    print('PSw_5: ', Qty('PSw_5')._dct)
    print('PSw_5_b: ', Qty('PSw_5_b')._dct)
    print('PSw_5_c2u_ki: ', Qty('PSw_5_c2u_ki')._dct)

def test_parser():
    assert 'PS_e2h' == Qty(name="PS", compartment="e2h").key
    assert 'PS_1_e2h_li' == Qty(name="PS", index=1, compartment="e2h", roi="li").key
    assert "PS_e2h" == Qty(**{'name': 'PS', 'index': None, 'compartment': 'e2h', 'roi': None}).key
    assert "PS_1_e2h_li" == Qty(**{'name': 'PS', 'index': 1, 'compartment': 'e2h', 'roi': 'li'}).key
    assert "PS_1_li" == Qty(**{'name': 'PS', 'index': 1, 'compartment': None, 'roi': 'li'}).key
    assert "PS_1" == Qty(**{'name': 'PS', 'index': 1, 'compartment': None, 'roi': None}).key
    assert "PS_e2h_li" == Qty(**{'name': 'PS', 'index': None, 'compartment': 'e2h', 'roi': 'li'}).key
    assert 'c_ar' == Qty(**{'name': 'c', 'index': None, 'compartment': None, 'roi': 'ar'}).key
    
    try: 
        Qty(name="PS", compartment="e2x")   # x not in COMPS
    except:
        pass
    else:
        assert False

    assert Qty('E_ki', index=1).key == 'E_1_ki'
    assert Qty('E_2_ki', index=1).key == 'E_1_ki'


if __name__ == '__main__':
    test_get_sequence()
    test_qty()
    test_parser()
    print('All tools tests passed!!')