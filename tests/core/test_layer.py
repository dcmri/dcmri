import numpy as np

from dcmri.core.layer import LayerFunction
from dcmri.lexicon.dicts import QUANTITIES


CUSTOM = QUANTITIES | {
    'XX': {'init': 15, 'bounds': [0, 180], 'name': 'Custom quantity', 'unit': ''},
}

# --- Mock Concrete Class Implementation for testing ---
class MockFunction(LayerFunction):
    configs = {'mode': ['linear', 'nonlinear']}

    def __init__(self, mode='linear', **params):
        cnfg = {
            'mode': mode, 
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(CUSTOM, S0=100 + np.arange(10), R1=1 + 0.1 * np.arange(10)) # overrule defaults
        self._override_pars(**params) # set user-defined parameters

    def _params(self, select='all') -> list:
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
        if select == 'pixel':
            return ['S0', 'R1']

    def __call__(self, t, x, **params):
        p = self._update_pars(**params)
        if self._cnfg['mode'] == 'linear':
            return p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t)
        else:
            return p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t + p['BAT'] * (t ** 2))


# =============================================================================
# 2. Test Cases (One per Method)
# =============================================================================

def test_layerfunction_init_super():
    model = LayerFunction()
    assert model._params() == []
    assert model() is None

def test_layerfunction_init():
    model = MockFunction()
    try:
        MockFunction(mode='quadratic')
    except:
        pass
    else:
        assert False
    print("-> test_layerfunction_init passed!")

def test_layerfunction_params():
    model = MockFunction()
    assert model.params() == model._pars
    assert isinstance(model._params(), list)
    print("-> test_layerfunction_params passed!")

def test_layerfunction_call():
    t = np.arange(3)
    model = MockFunction('linear')
    assert model(t, 9)[0] == 124
    model = MockFunction('nonlinear')
    assert model(t, 9, XX=10)[0] == 119
    print("-> test_layerfunction_call passed!")


# =============================================================================
# 3. Execution Driver Block
# =============================================================================

if __name__ == '__main__':
    test_layerfunction_init_super()
    test_layerfunction_init()
    test_layerfunction_params()
    test_layerfunction_call()

    print("==================================================")
    print(f"All LayerFunction tests passing!")
    print("==================================================")