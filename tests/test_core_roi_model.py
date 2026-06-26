import os
import shutil

import numpy as np

from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.quantities import QUANTITIES
from dcmri.core.tools import init


CUSTOM = QUANTITIES | {
    'XX': {'init': 15, 'bounds': [0, 180], 'name': 'Custom quantity', 'unit': '', 'group': 'indicator'},
}

VALUES = init(lexicon=CUSTOM)
VALUES['S0'] = 100 + np.arange(10)
VALUES['R1'] = 1 + 0.1 * np.arange(10)

# --- Mock Concrete Class Implementation for testing ---
class MockModel(SuperRoiModel):
    configs = {'mode': ['linear', 'nonlinear']}

    def __init__(self, mode='linear', **params):
        cnfg = {
            'mode': mode, 
        }
        self._version = '1.0'
        self._set_config(cnfg)
        self._set_params(VALUES)

    def params(self, select='all') -> list:
        lin = self._cnfg['mode'] == 'linear'
        if select == 'all':
            if lin:
                return ['XX', 'S0', 'R1']
            else:
                return ['XX', 'S0', 'R1', 'BAT']
        if select == 'free':
            if lin:
                return ['S0', 'R1']
            else:
                return ['S0', 'R1']

    def _predict(self, t, x):
        if self._cnfg['mode'] == 'linear':
            return self._pars['XX'] + self._pars['S0'][x] * (1 + self._pars['R1'][x] * t)
        else:
            return self._pars['XX'] + self._pars['S0'][x] * (1 + self._pars['R1'][x] * t + self._pars['BAT'] * (t ** 2))


# =============================================================================
# 2. Test Cases (One per Method)
# =============================================================================

def test_supermodel_init_super():
    model = SuperRoiModel()
    assert model.params() == []
    assert model._predict() is None

def test_supermodel_init():
    model = MockModel()
    assert model._version == '1.0'
    try:
        MockModel(mode='quadratic')
    except:
        pass
    else:
        assert False
    try:
        MockModel()._set_params()
    except:
        pass
    else:
        assert False
    print("-> test_supermodel_init passed!")

def test_supermodel_coverage():
    MockModel.print_configs()
    print("-> test_supermodel_coverage passed!")

def test_supermodel_params():
    model = MockModel()
    assert isinstance(model.params(), list)
    print("-> test_supermodel_params_abstract passed!")

def test_supermodel_predict():
    model = MockModel()
    t = np.array([0.0, 1.0, 2.0])
    pred = model._predict(t, 3)
    print("-> test_supermodel_predict_abstract passed!")

def test_supermodel_print_params():
    model = MockModel()
    model.print_params(lexicon=CUSTOM)
    model.print_params('XX', lexicon=CUSTOM)
    model.print_params(lexicon=CUSTOM, fixed_only=True)
    model.print_params(lexicon=CUSTOM, free_only=True)
    print("-> test_supermodel_print_params passed!")

def test_supermodel_export_params():
    model = MockModel()
    res = model.export_params(CUSTOM)
    assert 'R1' in res
    assert res['R1']['value'][0] == 1
    try:
        model.export_params() # Without the custom lexicon cant find the detail
    except:
        pass
    else:
        assert False
    print("-> test_supermodel_export_params passed!")

def test_supermodel_save_and_load():
    # Use localized mock temporary directory folder path string string
    folder_path = "./tmp_test_model_store"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    model = MockModel()
    model.save(folder_path)
    assert os.path.isdir(folder_path)

    blank_model = MockModel()
    blank_model.load(folder_path)

    assert blank_model._version == '1.0'
    assert blank_model._cnfg['mode'] == 'linear'
    assert blank_model._pars['R1'][0] == 1

    try:
        MockModel().load('X')
    except:
        pass
    else:
        assert False

    try:
        SuperRoiModel().save(folder_path)
        MockModel().load(folder_path)
    except:
        pass
    else:
        assert False

    shutil.rmtree(folder_path)

    print("-> test_supermodel_save_and_load passed!")

def test_supermodel_set_free_pars():
    model = MockModel('nonlinear')

    free_default = model._set_free_pars(lexicon=CUSTOM)
    assert 'R1' in free_default

    custom_bounds = {'R1': [0.0, 4.0], 'S0': None, 'BAT':[-10, 10]}
    free_updated = model._set_free_pars(bounds=custom_bounds, lexicon=CUSTOM)
    assert free_updated['R1'] == [0.0, 4.0]
    assert 'S0' not in free_updated

    try:
        model._set_free_pars({'A': [0,1]}, lexicon=CUSTOM)
    except:
        pass
    else:
        assert False

    try:
        model = MockModel('nonlinear')
        model._set_free_pars({'BAT': [1, 2]}, lexicon=CUSTOM)
    except:
        pass
    else:
        assert False

    try:
        model = MockModel('nonlinear')
        model._set_free_pars({'S0': [2, 1]}, lexicon=CUSTOM)
    except:
        pass
    else:
        assert False

    try:
        model = MockModel('nonlinear')
        model._set_free_pars({'XX': [20, 30]}, lexicon=CUSTOM)
    except:
        pass
    else:
        assert False
    
    print("-> test_supermodel_set_free_pars passed!")


# =============================================================================
# 3. Execution Driver Block
# =============================================================================

if __name__ == '__main__':
    test_supermodel_coverage()
    test_supermodel_init_super()
    test_supermodel_init()
    test_supermodel_params()
    test_supermodel_predict()
    test_supermodel_export_params()
    test_supermodel_print_params()
    test_supermodel_save_and_load()
    test_supermodel_set_free_pars()
    
    print("==================================================")
    print(f"All SuperModel tests passing!")
    print("==================================================")