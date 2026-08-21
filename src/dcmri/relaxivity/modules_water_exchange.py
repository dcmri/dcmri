import numpy as np

from dcmri.core.module import Module


class WaterExchangeFX(Module):
    def inputs(self) -> set:
        return {'Fi'}
    
    def outputs(self) -> set:
        return {'fx', 'Fw'}

    def __call__(self, data: dict=None, **kwargs) -> dict: 
        p = self.map_data(data, kwargs)

        fx = [list(np.arange(np.size(p['Fi'])))]
        Fw = np.array([np.sum(p['Fi'])]).reshape(1, 1)

        return self.map_results({'fx':fx, 'Fw':Fw})


class WaterExchangeLiver(Module):
    configs = {
        'water_exchange': {'F', 'N', 'R'},
    }
    defaults = {
        'water_exchange': 'F',
    }
    def inputs(self) -> set:
        inputs = {'Fi_l'}
        if self.config['water_exchange'] == 'R':
            inputs |= {'PSw'}
        return inputs
    
    def outputs(self) -> set:
        return {'fx_l', 'Fw_l'}

    def __call__(self, data: dict=None, **kwargs) -> dict: 
        p = self.map_data(data, kwargs)

        Fi_l = p['Fi_l'][0]
        if self.config['water_exchange']=='F':
            fx = [[0,1]]
            Fw_l = np.array([Fi_l])  

        elif self.config['water_exchange']=='N': 
            fx = []
            Fw_l = np.array([[Fi_l, 0], [0, 0]])

        elif self.config['water_exchange']=='R':
            fx = []
            Fw_l = np.array([[Fi_l, p['PSw']], [p['PSw'], 0]])

        return self.map_results({'fx_l':fx, 'Fw_l':Fw_l})


class WaterExchangeKidney(Module):
    configs = { 
        'water_exchange': {'(btc)', '(b, tc)', '(bc, t)', '(b, t, c)'},
    }
    defaults = {
        'water_exchange': '(btc)',
    }
    def inputs(self) -> set:
        inputs = {'Fi'}
        
        if self.config['water_exchange']=='(b, tc)': 
            inputs |= {'PSw_b_tc', 'PSw_tc_b'}

        elif self.config['water_exchange']=='(bc, t)':
            inputs |= {'PSw_bc_t', 'PSw_t_bc'}

        elif self.config['water_exchange']=='(bt, c)':
            inputs |= {'PSw_bt_c', 'PSw_c_bt'}

        elif self.config['water_exchange']=='(b, t, c)':
            inputs |= {'PSw_bt', 'PSw_bc', 'PSw_tb', 'PSw_tc', 'PSw_cb', 'PSw_ct'}

        return inputs
    
    def outputs(self) -> set:
        return {'fx', 'Fw'}

    def lexicon_data(self, q: dict=None): 
        PSw = {k: 0 for k in WaterExchangeKidney.all_inputs() if k[:3]=='PSw'}
        return self.update_data(q | PSw) 

    def __call__(self, data: dict=None, **kwargs) -> dict: 
        p = self.map_data(data, kwargs)

        # 0: blood
        # 1: tubuli
        # 2: tissue

        Fi = p['Fi'][0]

        if self.config['water_exchange']=='(btc)':
            fx = [[0, 1, 2]]
            Fw = np.array([Fi])  

        elif self.config['water_exchange']=='(b, tc)': 
            fx = [[0], [1, 2]] 
            Fw = np.array([
                [Fi,            p['PSw_b_tc']], 
                [p['PSw_tc_b'], 0            ],
            ])

        elif self.config['water_exchange']=='(bc, t)':
            fx = [[0, 2], [1]]
            Fw = np.array([
                [Fi,            p['PSw_bc_t']], 
                [p['PSw_t_bc'], 0            ],
            ])

        elif self.config['water_exchange']=='(bt, c)':
            fx = [[0, 1], [2]]
            Fw = np.array([
                [Fi,            p['PSw_bt_c']], 
                [p['PSw_c_bt'], 0            ],
            ])

        elif self.config['water_exchange']=='(b, t, c)':
            fx = [[0], [1], [2]] 
            Fw = np.array([
                [Fi,          p['PSw_bt'], p['PSw_bc']], 
                [p['PSw_tb'], 0,           p['PSw_tc']],
                [p['PSw_cb'], p['PSw_ct'], 0          ],
            ])

        results = {'fx':fx, 'Fw':Fw}

        return self.map_results(results)








_VOLUMES = {
    ('2CX', 'FF'): set(),
    ('2CU', 'FF'): set(),
    ('HF', 'FF'): set(),
    ('HFU', 'FF'): set(),
    ('NX', 'FF'): set(),
    ('NXP', 'FF'): set(),
    ('WV', 'FF'): set(),
    ('U', 'FF'): set(),
    ('FX', 'FF'): set(),

    ('2CX', 'FR'): {'vb', 'vi'},
    ('2CU', 'FR'): {'vb', 'vi'},
    ('HF', 'FR'): {'vb', 'vi'},
    ('HFU', 'FR'): {'vb', 'vi'},
    ('NX', 'FR'): {'vb', 'vi'},
    ('NXP', 'FR'): {'vb', 'vi'},
    ('WV', 'FR'): {'vi'},
    ('U', 'FR'): {'vb', 'vi'},
    ('FX', 'FR'): {'vb', 'vi'},

    ('2CX', 'RF'): {'vb'},
    ('2CU', 'RF'): {'vb'},
    ('HF', 'RF'): {'vb'},
    ('HFU', 'RF'): {'vb'},
    ('NX', 'RF'): {'vb'},
    ('NXP', 'RF'): {'vb'},
    ('WV', 'RF'): set(),
    ('U', 'RF'): {'vb'},
    ('FX', 'RF'): {'vb'},

    ('2CX', 'RR'): {'vb', 'vi'},
    ('2CU', 'RR'): {'vb', 'vi'},
    ('HF', 'RR'): {'vb', 'vi'},
    ('HFU', 'RR'): {'vb', 'vi'},
    ('NX', 'RR'): {'vb', 'vi'},
    ('NXP', 'RR'): {'vb', 'vi'},
    ('WV', 'RR'): {'vi'},
    ('U', 'RR'): {'vb', 'vi'},
    ('FX', 'RR'): {'vb', 'vi'},
}



class WaterVolumesTissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
    }
    defaults = {
        'kinetics': '2CX',
        'water_exchange': 'FF',
    }
    def inputs(self):
        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')
        return _VOLUMES[(kin, wex)]
    
    def outputs(self):
        return {'v'}

    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        # Add derived
        if {'vb', 'vi'}.issubset(p):
            p['vc'] = 1 - p['vb'] - p['vi']

        # Map water compartment volumes
        if (kin, wex) == ('2CX', 'FF'): v = np.array([1])
        if (kin, wex) == ('2CU', 'FF'): v = np.array([1])
        if (kin, wex) == ('HF', 'FF'): v = np.array([1])
        if (kin, wex) == ('HFU', 'FF'): v = np.array([1])
        if (kin, wex) == ('NX', 'FF'): v = np.array([1])
        if (kin, wex) == ('NXP', 'FF'): v = np.array([1])
        if (kin, wex) == ('WV', 'FF'): v = np.array([1])
        if (kin, wex) == ('U', 'FF'): v = np.array([1])
        if (kin, wex) == ('FX', 'FF'): v = np.array([1])

        if (kin, wex) == ('2CX', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('2CU', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('HF', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('HFU', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('NX', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('NXP', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('WV', 'FR'): v = np.array([p['vi'], 1-p['vi']])
        if (kin, wex) == ('U', 'FR'): v = np.array([1-p['vc'], p['vc']])
        if (kin, wex) == ('FX', 'FR'): v = np.array([1-p['vc'], p['vc']])

        if (kin, wex) == ('2CX', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('2CU', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('HF', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('HFU', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('NX', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('NXP', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('WV', 'RF'): v = np.array([1])
        if (kin, wex) == ('U', 'RF'): v = np.array([p['vb'], 1-p['vb']])
        if (kin, wex) == ('FX', 'RF'): v = np.array([p['vb'], 1-p['vb']])

        if (kin, wex) == ('2CX', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('2CU', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('HF', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('HFU', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('NX', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('NXP', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('WV', 'RR'): v = np.array([p['vi'], 1-p['vi']])
        if (kin, wex) == ('U', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])
        if (kin, wex) == ('FX', 'RR'): v = np.array([p['vb'], p['vi'], p['vc']])

        return {'v': v}


_FLOWS = {
    ('2CX', 'FF'): {'Fb'},
    ('2CU', 'FF'): {'Fb'},
    ('HF', 'FF'): set(),
    ('HFU', 'FF'): set(),
    ('NX', 'FF'): {'Fb'},
    ('NXP', 'FF'): {'Fb'},
    ('WV', 'FF'): set(),
    ('U', 'FF'): {'Fb'},
    ('FX', 'FF'): {'Fb'},

    ('2CX', 'FR'): {'Fb', 'PSc'},
    ('2CU', 'FR'): {'Fb', 'PSc'},
    ('HF', 'FR'): {'PSc'},
    ('HFU', 'FR'): {'PSc'},
    ('NX', 'FR'): {'Fb', 'PSc'},
    ('NXP', 'FR'): {'Fb', 'PSc'},
    ('WV', 'FR'): {'PSc'},
    ('U', 'FR'): {'Fb', 'PSc'},
    ('FX', 'FR'): {'Fb', 'PSc'},

    ('2CX', 'RF'): {'Fb', 'PSe'},
    ('2CU', 'RF'): {'Fb', 'PSe'},
    ('HF', 'RF'): {'PSe'},
    ('HFU', 'RF'): {'PSe'},
    ('NX', 'RF'): {'Fb', 'PSe'},
    ('NXP', 'RF'): {'Fb', 'PSe'},
    ('WV', 'RF'): set(),
    ('U', 'RF'): {'Fb', 'PSe'},
    ('FX', 'RF'): {'Fb', 'PSe'},

    ('2CX', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('2CU', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('HF', 'RR'): {'PSe', 'PSc'},
    ('HFU', 'RR'): {'PSe', 'PSc'},
    ('NX', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('NXP', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('WV', 'RR'): {'PSc'},
    ('U', 'RR'): {'Fb', 'PSe', 'PSc'},
    ('FX', 'RR'): {'Fb', 'PSe', 'PSc'},
}

class WaterFlowsTissueX(Module):
    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
    }
    defaults = {
        'kinetics': '2CX', 
        'water_exchange': 'FF',
    }
    def inputs(self) -> dict:
        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')
        inputs = _FLOWS[(kin, wex)]

        if wex[0] == 'N' and 'PSe' in inputs :
            inputs - {'PSe'}
        if wex[1] == 'N' and 'PSc' in inputs:
            inputs - {'PSc'}
        return inputs
    
    def outputs(self) -> dict:
        return {'Fw'}

    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        if wex[0] == 'N':
            p['PSe'] = 0
        if wex[1] == 'N':
            p['PSc'] = 0

        # Map water compartment flows
        if (kin, wex) == ('2CX', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('2CU', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('HF', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('HFU', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('NX', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('NXP', 'FF'): Fw = np.full((1, 1), p['Fb'])
        if (kin, wex) == ('WV', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('U', 'FF'): Fw = np.full((1, 1), 0)
        if (kin, wex) == ('FX', 'FF'): Fw = np.full((1, 1), p['Fb'])

        if (kin, wex) == ('2CX', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('2CU', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('HF', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('HFU', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('NX', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('NXP', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('WV', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('U', 'FR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('FX', 'FR'): Fw = np.array([[p['Fb'], p['PSc']], [p['PSc'], 0]])

        if (kin, wex) == ('2CX', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('2CU', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('HF', 'RF'): Fw = np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('HFU', 'RF'): Fw = np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('NX', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('NXP', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('WV', 'RF'): Fw = np.array([0])
        if (kin, wex) == ('U', 'RF'): Fw = np.array([[0, p['PSe']], [p['PSe'], 0]])
        if (kin, wex) == ('FX', 'RF'): Fw = np.array([[p['Fb'], p['PSe']], [p['PSe'], 0]])

        if (kin, wex) == ('2CX', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('2CU', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('HF', 'RR'): Fw = np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('HFU', 'RR'): Fw = np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('NX', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('NXP', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('WV', 'RR'): Fw = np.array([[0, p['PSc']], [p['PSc'], 0]])
        if (kin, wex) == ('U', 'RR'): Fw = np.array([[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])
        if (kin, wex) == ('FX', 'RR'): Fw = np.array([[p['Fb'], p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]])

        return {'Fw': Fw}


class ContrastConcTissueX(Module):
    # Convert tissue concentration in blood and interstitium to concentration.
    # For uptake models this introduces a new parameter

    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'}
    }
    defaults = {
        'kinetics': '2CX',
    }    
    def inputs(self) -> set:
        inputs = {'C'}
        if self.config['kinetics'] == 'FX':
            inputs |= {'H', 've'}
        elif self.config['kinetics'] in {'U', 'NX', 'NXP'}:
            inputs |= {'vb'}
        elif self.config['kinetics'] == 'WV':
            inputs |= {'vi'}
        elif self.config['kinetics'] in {'HFU', '2CU', 'HF', '2CX'}:
            inputs |= {'vb', 'vi'}
        return inputs
    
    def outputs(self) -> set:
        return {'c'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        kinetics = self.config['kinetics']

        if p['C'].ndim==1:
            C = np.array(p['C']).reshape(1, -1)
        else:
            C = np.array(p['C'])

        def div(Ci, vi):
            return Ci / vi if vi > 0 else Ci * 0

        c = np.zeros((2, C.shape[1]))

        if kinetics == 'FX':
            # vp = p['vb'] * (1 - p['H'])
            # vi = p['ve'] - vp
            # Cp = C[0,:] * vp / p['ve']
            # Ci = C[0,:] * vi / p['ve']
            # c[0,:] = div(Cp, p['vb'])
            # c[1,:] = div(Ci, vi)
            c[0,:] = div(C[0,:] * (1 - p['H']), p['ve'])
            c[1,:] = div(C[0,:], p['ve'])

        elif kinetics in ['U', 'NX', 'NXP']:
            c[0,:] = div(C[0,:], p['vb'])  
        
        elif kinetics == 'WV':
            # c[0,:] = ca but does not contribute to T2* at vb=0 so ignored here
            c[1,:] = div(C[0,:], p['vi']) 
        
        elif kinetics in ['HFU', '2CU', 'HF', '2CX']:
            c[0,:] = div(C[0,:], p['vb'])
            c[1,:] = div(C[1,:], p['vi']) # New parameter vi
        
        return {'c': c}


class WaterConcTissueX(Module):
    # Convert tissue concentration in kinetic compartments to concentration in water compartments.

    configs = {
        'kinetics': {'2CX', 'HF', 'WV', '2CU', 'HFU', 'FX', 'NX', 'NXP', 'U'},
        'water_exchange': {'FF','RF','NF','FR','RR','NR','FN','RN','NN'},
    }
    defaults = {
        'kinetics': '2CX',
        'water_exchange': 'FF',
    }
        
    def inputs(self) -> set:
        kinetics = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        inputs = {'C'}

        if kinetics == 'FX':
            if wex[0] != 'F':
                inputs |= {'ve', 'vb', 'H'}

        if wex == 'FF':
            pass

        elif wex == 'RF':
            if kinetics == 'WV':
                pass
            elif kinetics in ['U', 'NX', 'NXP', 'FX', 'HFU', 'HF', '2CU', '2CX']:
                inputs |= {'vb'}

        elif wex == 'FR':
            if kinetics == 'WV':
                inputs |= {'vi'}
            elif kinetics in ['U']:
                inputs |= {'vc'}
            elif kinetics in ['FX']:
                inputs |= {'vb', 'H', 've'}
            elif kinetics in ['NX', 'NXP', 'HFU', 'HF', '2CU', '2CX']:
                inputs |= {'vb', 'vi'}

        elif wex == 'RR':
            if kinetics == 'WV':
                inputs |= {'vi'}
            elif kinetics in ['NX', 'NXP', 'U']:
                inputs |= {'vb'}
            elif kinetics in ['FX']:
                inputs |= {'vb', 'H', 've'}
            elif kinetics in ['HF', 'HFU', '2CU', '2CX']:
                inputs |= {'vb', 'vi'}

        return inputs
    

    def outputs(self) -> set:
        return {'c'}


    def __call__(self, data: dict) -> dict: # (n_comp, n_times)
        p = self.map_data(data)

        C = np.array(p['C'])
        if C.ndim==1:
            C = C.reshape(1, -1)

        kinetics = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        # Define helper functions
        def div(Ci, vi):
            if vi==0:
                # In this case the result does not matter
                return Ci * 0
            else:
                return Ci / vi

        def mix_1_comp(C, v=None):
            C = C.sum(axis=0)
            if v is None:
                return C.reshape(1, -1)
            else:
                c = np.zeros((2, C.size))
                c[0,:] = div(C, v)
                return c

        def mix_2_comp(C, v):
            c = np.zeros((2, C.shape[1]))
            c[0,:] = div(C[0,:], v)
            c[1,:] = div(C[1,:], 1 - v)
            return c

        def mix_3_comp(C, v):
            c = np.zeros((3, C.shape[1]))
            for i in range(C.shape[0]):
                c[i, :] = div(C[i,:], v[i])
            return c

        # Separate well-mixed space if needed
        if kinetics == 'FX': # comp = 'e'
            if wex[0] != 'F':
                if p['ve'] == 0:
                    Cp = C[0,:] * 0
                    Ci = C[0,:] * 0
                else:
                    p['vp'] = (1 - p['H']) * p['vb']
                    p['vi'] = p['ve'] - p['vp']
                    Cp = C[0,:] * p['vp'] / p['ve']
                    Ci = C[0,:] * p['vi'] / p['ve']
                C = np.stack((Cp, Ci))

        # Map indicator compartments to water compartments
        if wex == 'FF':
            c = mix_1_comp(C)

        elif wex == 'RF':
            if kinetics == 'WV': # i
                c = mix_1_comp(C)
            elif kinetics in ['U', 'NX', 'NXP']: # b
                c = mix_1_comp(C, p['vb'])
            elif kinetics in ['FX', 'HFU', 'HF', '2CU', '2CX']: #bi
                c = mix_2_comp(C, p['vb'])

        elif wex == 'FR':
            if kinetics == 'WV':
                c = mix_1_comp(C, p['vi']) #i
            elif kinetics in ['U']:
                c = mix_1_comp(C, 1 - p['vc'])  #b
            elif kinetics == 'FX':
                p['vp'] = (1 - p['H']) * p['vb']
                p['vi'] = p['ve'] - p['vp']
                c = mix_1_comp(C, p['vb'] + p['vi'])
            elif kinetics in ['NX', 'NXP', 'HFU', 'HF', '2CU', '2CX']:
                c = mix_1_comp(C, p['vb'] + p['vi']) # 1-vc

        elif wex == 'RR':
            if kinetics == 'WV':
                c = mix_1_comp(C, p['vi'])
            elif kinetics in ['NX', 'NXP', 'U']:
                c = mix_3_comp(C, [p['vb']])
            elif kinetics == 'FX':
                p['vp'] = (1 - p['H']) * p['vb']
                p['vi'] = p['ve'] - p['vp']
                c = mix_3_comp(C, [p['vb'], p['vi']])
            elif kinetics in ['HF', 'HFU', '2CU', '2CX']:
                c = mix_3_comp(C, [p['vb'], p['vi']])

        return {'c': c}
