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

    def __init__(self, order='linear', defaults=None, **kwargs):
        cnfg = {
            'order': order,
        }
        self._set_config(cnfg)
        self._set_params(defaults)

    def __call__(self, t, x, **kwargs):
        p = self._update_params(kwargs)
        if self._cnfg['order'] == 'linear':
            return p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t)
        else:
            return p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t + p['BAT'] * (t ** 2))

    def params(self) -> list:
        lin = self._cnfg['order'] == 'linear'
        if lin:
            return ['XX', 'S0', 'R1']
        else:
            return ['XX', 'S0', 'R1', 'BAT']

# =============================================================================
# 2. Test Cases (One per Method)
# =============================================================================

def test_wrapper_function_init_super():
    model = Function(defaults=DEFAULTS)
    assert model.params() == []
    assert model() is None

def test_wrapper_function_init():
    try:
        MockFunction(order='quadratic')
    except:
        pass
    else:
        assert False
    print("-> test_wrapper_function_init passed!")

def test_wrapper_function_params():
    model = MockFunction()
    assert isinstance(model.params(), list)
    print("-> test_wrapper_function_params passed!")

def test_wrapper_function_call():
    t = np.arange(3)
    model = MockFunction('linear')
    assert model(t, 9, **DEFAULTS)[0] == 901
    model = MockFunction('nonlinear')
    map = {'XX': 10}
    assert model(t, 9, **(DEFAULTS | map))[0] == 910
    try:
        MockFunction('linear')(t, 9)
    except:
        pass
    else:
        assert False
    print("-> test_wrapper_function_connect_inputs passed!")

    print("-> test_wrapper_function_call passed!")

def test_wrapper_function_connect_inputs():
    t = np.arange(3)
    model = MockFunction('linear')
    assert model(t, 9, **DEFAULTS)[0] == 901
    model = MockFunction('nonlinear').connect_inputs({'ZZ': 'XX'})
    assert model(t, 9, **DEFAULTS)[0] == 910
    print("-> test_wrapper_function_connect_inputs passed!")

# =============================================================================
# 3. Execution Driver Block
# =============================================================================

if __name__ == '__main__':
    test_wrapper_function_init_super()
    test_wrapper_function_init()
    test_wrapper_function_params()
    test_wrapper_function_call()
    test_wrapper_function_connect_inputs()

    print("==================================================")
    print(f"All Function tests passing!")
    print("==================================================")