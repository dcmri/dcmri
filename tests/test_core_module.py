import numpy as np

from dcmri.core.module import Module


def test_data():
    return {
        'XX': 1,
        'S0': 100 * np.arange(50),
        'R1': 5 * np.arange(50),
        'BAT': 2,
        'ZZ': 10
    }

def test_nested_data():
    return {
        'XX': 1,
        'S0_1': 100 * np.arange(50),
        'S0_2': 10 * np.arange(50),
        'R1_1': 5 * np.arange(50),
        'R1_2': 10 * np.arange(50),
        'BAT_1': 2,

        # Extra data not used as inputs
        'ZZ_1': 10,
        'ZZ_2': 10,
    }

class FaultyModule1(Module):
    configs = {
        'mode': {'A', 'B'}
    }
    defaults = {
        'mode': 'A',
    }   
    def outputs(self):
        if self.config['mode'] == 'A':
            return {'aa'}
        else:
            return {'bb'}
    def __call__(self, t, x, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)
        return 0

class FaultyModule2(Module):
    configs = {
        'mode': {'A', 'B'}
    }
    defaults = {
        'mode': 'A',
    }
    def inputs(self):
        if self.config['mode'] == 'A':
            return {'a'}
        else:
            return {'b'}   
    def __call__(self, t, x, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)
        return 0
    
class FaultyModule3(Module):
    configs = {
        'mode': {'A', 'B'}
    }
    defaults = {
        'mode': 'A',
    }
    def inputs(self):
        if self.config['mode'] == 'A':
            return {'a'}
        else:
            return {'b'}   
    def outputs(self):
        if self.config['mode'] == 'A':
            return {'aa'}
        else:
            return {'bb'}


# --- Mock Concrete Class Implementation for testing ---
class MockModule(Module):
    configs = {
        'order': {'linear', 'nonlinear'}
    }
    defaults = {
        'order': 'linear',
    }

    def inputs(self):
        if self.config['order'] == 'linear':
            return {'XX', 'S0', 'R1'}
        else:
            return {'XX', 'S0', 'R1', 'BAT'}   

    def outputs(self): 
        return {'YY'} 

    def __call__(self, t, x, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)
        if self.config['order'] == 'linear':
            return {'YY': p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t)}
        else:
            return {'YY': p['XX'] + p['S0'][x] * (1 + p['R1'][x] * t + p['BAT'] * ((1 + t) ** 2))}


class NestedModule(Module):
    configs = {
        'order_1': {'linear', 'nonlinear'},
        'order_2': {'linear', 'nonlinear'},
    }  
    defaults = {
        'order_1': 'linear',
        'order_2': 'nonlinear',
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        self._module_1 = MockModule(order=self.config['order_1'])
        self._module_2 = MockModule(order=self.config['order_2'])

        # XX is a shared input and not mapped
        imap_1 = {i: f'{i}_1' for i in self._module_1.inputs() if i not in {'XX'}}
        imap_2 = {i: f'{i}_2' for i in self._module_2.inputs() if i not in {'XX'}}

        self._module_1.map_inputs(imap_1)
        self._module_2.map_inputs(imap_2)

        self.map_io(imap, omap)

    def inputs(self):
        inputs = set()
        inputs |= self._module_1.mapped_inputs()
        inputs |= self._module_2.mapped_inputs()
        inputs -= {'BAT_2'}
        return inputs

    def outputs(self): 
        return {'YY', 'YY_1', 'YY_2'}

    def __call__(self, t, x, data: dict=None, **override) -> dict:
        p = self.map_data(data, override)

        p['BAT_2'] = 10

        result_1 = self._module_1(t, x, p)
        result_2 = self._module_2(t, x, p)
        return {
            'YY': result_1['YY'] + result_2['YY'],
            'YY_1': result_1['YY'], 
            'YY_2': result_2['YY'],
        }

def test_faulty_modules():
    try:
        FaultyModule1()
    except:
        pass
    else:
        assert False
    try:
        FaultyModule2().outputs()
    except:
        pass
    else:
        assert False
    try:
        FaultyModule3()()
    except:
        pass
    else:
        assert False
    print("-> test_super passed!")

def test_nested_module():
    t = np.arange(3)
    
    data = test_nested_data()
    mdl = NestedModule()
    mdl(t, 9, data)
    
    # Correct input mapping  
    mdl = NestedModule({'XX':'QQ'})

    # Error because QQ is not in data
    try:
        mdl(t, 9, data)
    except:
        pass
    else:
        assert False

    # Error because QQ is not in data
    try:
        mdl(t, 9)
    except:
        pass
    else:
        assert False

    data['QQ'] = 5

    # Error data is not a dict (wrong arg order)
    try:
        mdl(data, t, 9)
    except:
        pass
    else:
        assert False

    mdl(t, 9, data)
    mdl(t, 9, data, QQ=10)

    assert 'BAT_2' not in data # Internal derived data are not saved

    assert 'QQ' in mdl.mapped_inputs()
    assert 'XX' not in mdl.mapped_inputs()
    assert 'XX' in mdl.inputs()
    assert 'BAT_2' not in mdl.mapped_inputs() # BAT_2 = constant
    assert 'BAT_1' not in mdl.mapped_inputs() # default = linear

    # If order_1 = nonlinear, BAT_1 is an input
    mdl = NestedModule(order_1='nonlinear')
    assert 'BAT_1' in mdl.mapped_inputs()

    



def test_module():
    try:
        MockModule(order='quadratic') # false value
    except:
        pass
    else:
        assert False

    t = np.arange(3)

    # Intialize without arguments
    data = test_data()
    model = MockModule()
    result = model(t, 9, data) 
    assert result['YY'][0] == 901

    # Overwrite defaults
    data = test_data()
    model = MockModule(order='linear')
    result = model(t, 9, data) 
    assert result['YY'][0] == 901

    # Non-default config
    data = test_data()
    model = MockModule(order='nonlinear')
    result = model(t, 9, data) 
    assert result['YY'][0] == 2701

    # Use input map
    data = test_data()
    model = MockModule({'XX': 'ZZ'})
    result = model(t, 9, data) 
    assert result['YY'][0] == 910

    # Missing data
    data = test_data()
    model = MockModule()
    data.pop('XX')
    try:
        model(t, 9, data) 
    except:
        pass
    else:
        assert False


if __name__ == '__main__':
    test_faulty_modules()
    test_module()
    test_nested_module()

    print("==================================================")
    print(f"All Module tests passing!")
    print("==================================================")