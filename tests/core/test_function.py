import numpy as np

from dcmri.core.function import Function


DEFAULTS = {
    'XX': 1,
    'S0': 100 * np.arange(50),
    'R1': 5 * np.arange(50),
    'BAT': 2,
    'ZZ': 10
}


# --- Mock Concrete Class Implementation for testing ---
class MockFunction(Function):
    configs = {'order': ['linear', 'nonlinear']}

    def __init__(self, order='linear', **defaults):
        cnfg = {
            'order': order, 
        }
        self._set_config(**cnfg)
        self._set_params(**defaults) 

    def __call__(self, t, x, **kwargs):
        p = self._update_params(**kwargs)
        if self._cnfg['order'] == 'linear':
            return p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t)
        else:
            return p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t + p['BAT'] * (t ** 2))

    def _param_names(self) -> list:
        lin = self._cnfg['order'] == 'linear'
        if lin:
            return ['XX', 'S0', 'R1']
        else:
            return ['XX', 'S0', 'R1', 'BAT']

# =============================================================================
# 2. Test Cases (One per Method)
# =============================================================================

def test_wrapper_function_init_super():
    model = Function(**DEFAULTS)
    assert model._param_names() == []
    assert model() is None

def test_wrapper_function_init():
    model = MockFunction(**DEFAULTS)
    try:
        MockFunction(order='quadratic')
    except:
        pass
    else:
        assert False
    print("-> test_wrapper_function_init passed!")

def test_wrapper_function_params():
    model = MockFunction(**DEFAULTS)
    assert model.params == model._pars
    assert isinstance(model.params, dict)
    print("-> test_wrapper_function_params passed!")

def test_wrapper_function_call():
    t = np.arange(3)
    model = MockFunction('linear', **DEFAULTS)
    assert model(t, 9)[0] == 901
    model = MockFunction('nonlinear', **DEFAULTS)
    assert model(t, 9, XX=10)[0] == 910
    print("-> test_layerfunction_call passed!")


# =============================================================================
# 3. Execution Driver Block
# =============================================================================

if __name__ == '__main__':
    test_wrapper_function_init_super()
    test_wrapper_function_init()
    test_wrapper_function_params()
    test_wrapper_function_call()

    print("==================================================")
    print(f"All Function tests passing!")
    print("==================================================")