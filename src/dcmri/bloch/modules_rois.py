import numpy as np

from dcmri.core.module import Module


# Helper functions
def div(C, v):
    if np.isscalar(v):
        if v==0:
            return C * 0 # In this case the result does not matter
        else:
            return C / v
    else:
        return np.vstack([div(C[k], v[k]) for k in range(v.size)])


def build_PSw(labels, i):
    n = len(labels)
    K = np.zeros((n, n))
    for row, dst in enumerate(labels):
        for col, src in enumerate(labels):
            if row == col:
                continue
            K[row, col] = i[f'PSw_{src}2{dst}']
    return K

def build_PSw_labels(labels):
    K = set()
    for row, dst in enumerate(labels):
        for col, src in enumerate(labels):
            if row == col:
                continue
            K |= {f'PSw_{src}2{dst}'}
    return K



class WaterExchangeArtery(Module):
    configs = { 
        'inflow': {True, False}
    }
    defaults = {
        'inflow': False,
    }
    _all_inputs = None
    _all_ouitputs = None

    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)

        o = {}
        o['vw'] = [1] # assume no pv effect
        o['Kw'] = np.zeros((1, 1))
        if self.config['inflow']:
            o['Kw'][0,0] = i['F_b_ar']
            o['inlets'] = [0]
            o['Fwi'] = [i['F_b_ar']]

        return self.map_results(o)
    
    def inputs(self) -> set:
        inputs = set()
        if self.config['inflow']:
            inputs |= {'F_b_ar'}
        return inputs
    
    def outputs(self) -> set:
        outputs = {'vw', 'Kw'}
        if self.config['inflow']:
            outputs |= {'inlets', 'Fwi'}

        return outputs





class WaterExchangeKidney(Module):
    configs = { 
        'water_exchange': {'F', 'R', 'N'}, 
        'inflow': {True, False}
    }
    defaults = {
        'water_exchange': 'F',
        'inflow': False,
    }
    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)

        wcomps = self._wcomps()

        o = {}
        o['vw'] = [i[f'v_{w}'] for w in wcomps]
        o['Kw'] = self._PSw(i)
        if self.config['inflow']:
            o['Kw'][0,0] = i['F_b_ki']
            o['inlets'] = [0]
            o['Fwi'] = [i['F_b_ki']]

        return self.map_results(o)
    
    def inputs(self) -> set:
        wcomps = self._wcomps()

        inputs = {f'v_{w}' for w in wcomps}
        inputs |= self._PSw_inputs()
        if self.config['inflow']:
            inputs |= {'F_b_ki'}

        return inputs

    def outputs(self) -> set:
        outputs = {'vw', 'Kw'}
        if self.config['inflow']:
            outputs |= {'inlets', 'Fwi'}

        return outputs

    def _wcomps(self):
        wex = self.config['water_exchange']
        wcomps = {
            'F': ('ki', ),
            'R': ('b', 'u', 'c'),
        }[wex.replace('N', 'R')]
        return wcomps    

    def _PSw(self, i):
        wcomps = self._wcomps()
        wex = self.config['water_exchange']
        n = len(wcomps)
        K = np.zeros((n, n))
        if wcomps == ('b', 'u', 'c'):
            if wex != 'N':
                K[0, 1] = i['F_u']  # b -> u Filtration
                K[1, 0] = 0        # u -> b No exchange
                K[0, 2] = i['PSw']  # b -> c bidrectional exchange
                K[2, 0] = i['PSw'] + i['F_u']  # c -> b bidrectional exchange + reabsorption
                K[1, 2] = i['F_u']  # u -> c reabsoprtion
                K[2, 1] = 0 # c -> u secretion ignored
        return K  

    def _PSw_inputs(self):
        wcomps = self._wcomps()
        wex = self.config['water_exchange']
        inputs = set()
        if wcomps == ('b', 'u', 'c'):
            if wex != 'N':
                inputs |= {'F_u', 'PSw'}
        return inputs    




class WaterExchangeLiver(Module):
    configs = { 
        'water_exchange': {'F', 'R', 'N'}, 
        'inflow': {True, False}
    }
    defaults = {
        'water_exchange': 'F',
        'inflow': False,
    }
    _all_inputs = None
    _all_outouts = None

    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)

        wcomps = self._wcomps()

        o = {}
        o['vw'] = [i[f'v_{w}'] for w in wcomps]
        o['Kw'] = self._PSw(i)
        if self.config['inflow']:
            o['Kw'][0,0] = i['F_b_li']
            o['inlets'] = [0]
            o['Fwi'] = [i['F_b_li']]

        return self.map_results(o)
    
    def inputs(self) -> set:
        wcomps = self._wcomps()

        inputs = {f'v_{w}' for w in wcomps}
        inputs |= self._PSw_inputs()
        if self.config['inflow']:
            inputs |= {'F_b_li'}

        return inputs
    
    def outputs(self) -> set:
        outputs = {'vw', 'Kw'}
        if self.config['inflow']:
            outputs |= {'inlets', 'Fwi'}

        return outputs

    def _wcomps(self):
        wex = self.config['water_exchange']
        wcomps = {
            'F': ('li', ),
            'R': ('e', 'h'),
        }[wex.replace('N', 'R')]
        return wcomps 

    def _PSw(self, i):
        wcomps = self._wcomps()
        wex = self.config['water_exchange']
        n = len(wcomps)
        K = np.zeros((n, n))
        if wcomps == ('e', 'h'):
            if wex != 'N':
                K[0, 1] = i['PSw']  
                K[1, 0] = i['PSw']
        return K  

    def _PSw_inputs(self):
        wcomps = self._wcomps()
        wex = self.config['water_exchange']
        inputs = set()
        if wcomps == ('e', 'h'):
            if wex != 'N':
                inputs |= {'PSw'}
        return inputs    


class WaterExchangeTissueX(Module):
    configs = { 
        'water_exchange': {'FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN'}, 
        'kinetics': {'2CX', 'HF', '2CU', 'HFU', 'WV', 'FX', 'NX', 'NXP', 'U'},
        'inflow': {True, False}
    }
    defaults = {
        'water_exchange': 'FF',
        'kinetics': '2CX',
        'inflow': False,
    }
    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)

        wcomps = self._wcomps()
        
        o = {}
        o['vw'] = [i[f'v_{w}'] for w in wcomps]
        o['Kw'] = self._PSw(i)
        if self.config['inflow']:
            if self.config['kinetics'] != 'U':
                o['Kw'][0,0] = i['F_b']
            o['inlets'] = [0]
            o['Fwi'] = [i['F_b']]

        return self.map_results(o)

    def inputs(self) -> set:
        wcomps = self._wcomps()

        inputs = {f'v_{w}' for w in wcomps}
        inputs |= self._PSw_inputs()
        if self.config['inflow']:
            inputs |= {'F_b'}

        return inputs
    
    def outputs(self) -> set:
        outputs = {'vw', 'Kw'}
        if self.config['inflow']:
            outputs |= {'inlets', 'Fwi'}

        return outputs

    def _wcomps(self):
        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        if wex=='FF':
            return ('ti', ) 
        if wex=='RF': 
            if kin in {'WV'}:
                return ('ic', )
            return ('b', 'ic')
        if wex=='FR': 
            if kin in {'WV'}:
                return ('i', 'c')
            return ('bi', 'c')
        if wex=='RR': 
            if kin in {'WV'}:
                return ('i', 'c')
            return ('b', 'i', 'c')

        raise ValueError(f"Unknown configuration {wex}, {kin}.") 

    def _PSw(self, i):
        wcomps = self._wcomps()
        wex = self.config['water_exchange']
        n = len(wcomps)
        K = np.zeros((n, n))
        if wcomps == ('b', 'ic'):
            if wex[0] != 'N':
                K[0, 1] = i['PSe']
                K[1, 0] = i['PSe']
        elif wcomps in [('i', 'c'), ('bi', 'c')]:
            if wex[1] != 'N':
                K[0, 1] = i['PSc']
                K[1, 0] = i['PSc']       
        elif wcomps == ('b', 'i', 'c'):
            if wex[0] != 'N':
                K[0, 1] = i['PSe'] 
                K[1, 0] = i['PSe']
            if wex[1] != 'N':
                K[1, 2] = i['PSc'] 
                K[2, 1] = i['PSc']
        return K  

    def _PSw_inputs(self):
        wcomps = self._wcomps()
        wex = self.config['water_exchange']
        inputs = set()
        if wcomps == ('b', 'ic'):
            if wex[0] != 'N':
                inputs |= {'PSe'}
        elif wcomps in [('i', 'c'), ('bi', 'c')]:
            if wex[1] != 'N':
                inputs |= {'PSc'}   
        elif wcomps == ('b', 'i', 'c'):
            if wex[0] != 'N':
                inputs |= {'PSe'}
            if wex[1] != 'N':
                inputs |= {'PSc'}  
        return inputs
        