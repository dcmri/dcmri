import numpy as np

from dcmri.core.module import Module
from dcmri.utils import const


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






class RelaxivityArtery(Module):
    configs = { 
        'baseline': {'measured', 'literature'},
        't1_relaxation': {None, 'lin'},
        't2_relaxation': {None, 'lin'},
        't2s_relaxation': {None, 'lin', 'quad'},
        'inflow': {True, False}
    }
    defaults = {
        'baseline': 'literature',
        't1_relaxation': 'lin',
        't2_relaxation': None,
        't2s_relaxation': 'lin',
        'inflow': False,
    }
    _all_inputs = None
    _all_outputs = None

    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)

        # Get some notations
        wcomps = ('b', ) # water compartments
        icomps = ('b', ) # indicator compartments
        b0, cm = i['field_strength'], i['agent']

        o = {}

        o['RM'] = [[1]]

        if self.config['baseline']=='measured':
            R1b = [i[f"R1_{c}"] for c in wcomps]
        else:
            R1b = [1 / const.T1(b0, c) for c in wcomps]

        if self.config['inflow']:
            o['R1ib'] = [1 / const.T1(b0, 'blood')]
            o['r1i'] = [const.r1(b0, 'blood', cm)]

        if self.config['t1_relaxation']=='lin':
            o['R1b'] = R1b
            o['r1'] = [const.r1(b0, c, cm) for c in icomps]

        if self.config['t2_relaxation']=='lin':
            o['R2b'] = [1 / const.T2(b0, c) for c in wcomps]
            o['r2'] = [const.r2(b0, c, cm) for c in icomps]

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            o['R2sb'] = 1 / const.T2s(b0, 'arterial blood')
            o['r2s'] = const.r2s(b0, 'tissue', cm)

        if self.config['t2s_relaxation']=='quad':
            o['r2sq'] = const.r2sq(b0, 'tissue', cm)

        return self.map_results(o)
    
    def inputs(self) -> set:
        wcomps = ('b', )
        inputs = {'field_strength', 'agent'}

        if self.config['baseline']=='measured':
            inputs |= {f"R1_{c}" for c in wcomps}

        return inputs
    
    def outputs(self) -> set:
        outputs = {'RM'}

        if self.config['t1_relaxation']=='lin':
            outputs |= {'R1b', 'r1'}

        if self.config['t2_relaxation']=='lin':
            outputs |= {'R2b', 'r2'}

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            outputs |= {'R2sb', 'r2s'}

        if self.config['t2s_relaxation']=='quad':
            outputs |= {'r2sq'}

        if self.config['inflow']:
            outputs |= {'r1i', 'R1ib'}

        return outputs





class RelaxivityKidney(Module):
    configs = { 
        'water_exchange': {'F', 'R', 'N'}, 
        'baseline': {'measured', 'literature'},
        't1_relaxation': {None, 'lin'},
        't2_relaxation': {None, 'lin'},
        't2s_relaxation': {None, 'lin', 'quad'},
        'inflow': {True, False}
    }
    defaults = {
        'water_exchange': 'F',
        'baseline': 'literature',
        't1_relaxation': 'lin',
        't2_relaxation': None,
        't2s_relaxation': 'lin',
        'inflow': False,
    }
    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)

        wcomps = self._wcomps()
        icomps = ('b', 'u') # indicator compartments
        b0, cm = i['field_strength'], i['agent']
        
        o = {}

        if wcomps == ('ki', ): 
            o['RM'] = [ 
                [1 / i['v_ki'], 1 / i['v_ki']   ]
            ]
        if wcomps == ('b', 'uc'): 
            o['RM'] = [ 
                [1 / i['v_b'],  0               ], 
                [0,             1 / i['v_uc']   ] 
            ] 
        if wcomps == ('bc', 'u'): 
            o['RM'] = [
                [1 / i['v_bc'], 0               ],
                [0,             1 / i['v_u']    ]
            ]
        if wcomps == ('bu', 'c'): 
            o['RM'] = [ 
                [1 / i['v_bu'], 1 / i['v_bu']   ],
                [0,             0               ]
            ]
        if wcomps == ('b', 'u', 'c'): 
            o['RM'] = [
                [1 / i['v_b'],  0               ],
                [0,             1 / i['v_u']    ],
                [0,             0               ]
            ]
       
        if self.config['baseline']=='measured':
            R1b = [i[f"R1_{c}"] for c in wcomps]
        else:
            R1b = [1 / const.T1(b0, c) for c in wcomps]

        if self.config['inflow']:
            o['R1ib'] = [1 / const.T1(b0, 'blood')]
            o['r1i'] = [const.r1(b0, 'blood', cm)]

        if self.config['t1_relaxation']=='lin':
            o['R1b'] = R1b
            o['r1'] = [const.r1(b0, c, cm) for c in icomps]

        if self.config['t2_relaxation']=='lin':
            o['R2b'] = [1 / const.T2(b0, c) for c in wcomps]
            o['r2'] = [const.r2(b0, c, cm) for c in icomps]

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            o['R2sb'] = 1 / const.T2s(b0, 'arterial blood')
            o['r2s'] = const.r2s(b0, 'tissue', cm)

        if self.config['t2s_relaxation']=='quad':
            o['r2sq'] = const.r2sq(b0, 'tissue', cm)

        return self.map_results(o)
    

    def inputs(self) -> set:
        wcomps = self._wcomps()

        inputs = {'field_strength', 'agent'}
        inputs |= {f'v_{w}' for w in wcomps if w != 'c'}
        
        if self.config['baseline']=='measured':
            inputs |= {f"R1_{c}" for c in wcomps}

        return inputs

    
    def outputs(self) -> set:
        outputs = {'RM'}

        if self.config['t1_relaxation']=='lin':
            outputs |= {'R1b', 'r1'}

        if self.config['t2_relaxation']=='lin':
            outputs |= {'R2b', 'r2'}

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            outputs |= {'R2sb', 'r2s'}

        if self.config['t2s_relaxation']=='quad':
            outputs |= {'r2sq'}

        if self.config['inflow']:
            outputs |= {'r1i', 'R1ib'}

        return outputs

    def _wcomps(self):
        wex = self.config['water_exchange']
        wcomps = {
            'F': ('ki', ),
            'R': ('b', 'u', 'c'),
        }[wex.replace('N', 'R')]
        return wcomps    

 


class RelaxivityLiver(Module):
    configs = { 
        'water_exchange': {'F', 'R', 'N'}, 
        'baseline': {'measured', 'literature'},
        't1_relaxation': {None, 'lin'},
        't2_relaxation': {None, 'lin'},
        't2s_relaxation': {None, 'lin', 'quad'},
        'inflow': {True, False}
    }
    defaults = {
        'water_exchange': 'F',
        'baseline': 'literature',
        't1_relaxation': 'lin',
        't2_relaxation': None,
        't2s_relaxation': 'lin',
        'inflow': False,
    }
    _all_inputs = None
    _all_outouts = None


    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)

        wcomps = self._wcomps()
        icomps = ('e', 'h') # indicator compartments
        b0, cm = i['field_strength'], i['agent']

        o = {}

        if wcomps == ('li', ): 
            o['RM'] = [ 
                [1 / i['v_li'], 1 / i['v_li']   ]
            ]
        elif wcomps == ('e', 'h'): 
            o['RM'] = [ 
                [1 / i['v_e'],  0              ], 
                [0,             1 / i['v_h']   ] 
            ] 

        if self.config['baseline']=='measured':
            R1b = [i[f"R1_{c}"] for c in wcomps]
        else:
            R1b = [1 / const.T1(b0, c) for c in wcomps]

        if self.config['inflow']:
            o['R1ib'] = [1 / const.T1(b0, 'blood')]
            o['r1i'] = [const.r1(b0, 'blood', cm)]

        if self.config['t1_relaxation']=='lin':
            o['R1b'] = R1b
            o['r1'] = [const.r1(b0, c, cm) for c in icomps]

        if self.config['t2_relaxation']=='lin':
            o['R2b'] = [1 / const.T2(b0, c) for c in wcomps]
            o['r2'] = [const.r2(b0, c, cm) for c in icomps]

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            o['R2sb'] = 1 / const.T2s(b0, 'arterial blood')
            o['r2s'] = const.r2s(b0, 'tissue', cm)

        if self.config['t2s_relaxation']=='quad':
            o['r2sq'] = const.r2sq(b0, 'tissue', cm)

        return self.map_results(o)
    
    def inputs(self) -> set:
        wcomps = self._wcomps()

        inputs = {'field_strength', 'agent'}
        inputs |= {f'v_{w}' for w in wcomps}
        
        if self.config['baseline']=='measured':
            inputs |= {f"R1_{c}" for c in wcomps}

        return inputs
    
    def outputs(self) -> set:
        outputs = {'RM'}

        if self.config['t1_relaxation']=='lin':
            outputs |= {'R1b', 'r1'}

        if self.config['t2_relaxation']=='lin':
            outputs |= {'R2b', 'r2'}

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            outputs |= {'R2sb', 'r2s'}

        if self.config['t2s_relaxation']=='quad':
            outputs |= {'r2sq'}

        if self.config['inflow']:
            outputs |= {'r1i', 'R1ib'}

        return outputs

    def _wcomps(self):
        wex = self.config['water_exchange']
        wcomps = {
            'F': ('li', ),
            'R': ('e', 'h'),
        }[wex.replace('N', 'R')]
        return wcomps 

   

class RelaxivityTissueX(Module):
    configs = { 
        'water_exchange': {'FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN'}, 
        'kinetics': {'2CX', 'HF', '2CU', 'HFU', 'WV', 'FX', 'NX', 'NXP', 'U'},
        'baseline': {'measured', 'literature'},
        't1_relaxation': {None, 'lin'},
        't2_relaxation': {None, 'lin'},
        't2s_relaxation': {None, 'lin', 'quad'},
        'inflow': {True, False}
    }
    defaults = {
        'water_exchange': 'FF',
        'kinetics': '2CX',
        'baseline': 'literature',
        't1_relaxation': 'lin',
        't2_relaxation': None,
        't2s_relaxation': 'lin',
        'inflow': False,
    }
    def __call__(self, data: dict=None, **kwargs) -> dict: 
        i = self.map_data(data, kwargs)
        b0, cm = i['field_strength'], i['agent']

        wcomps = self._wcomps()
        icomps = self._icomps()

        o = {}

        o['RM'] = self._rm(i)

        # Below here is standard
        if self.config['baseline']=='measured':
            R1b = [i[f"R1_{c}"] for c in wcomps]
        else:
            R1b = [1 / const.T1(b0, c) for c in wcomps]

        if self.config['t1_relaxation']=='lin':
            o['R1b'] = R1b
            o['r1'] = [const.r1(b0, c, cm) for c in icomps]

        if self.config['t2_relaxation']=='lin':
            o['R2b'] = [1 / const.T2(b0, c) for c in wcomps]
            o['r2'] = [const.r2(b0, c, cm) for c in icomps]

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            o['R2sb'] = 1 / const.T2s(b0, 'arterial blood')
            o['r2s'] = const.r2s(b0, 'tissue', cm)

        if self.config['t2s_relaxation']=='quad':
            o['r2sq'] = const.r2sq(b0, 'tissue', cm)

        if self.config['inflow']:
            o['R1ib'] = [1 / const.T1(b0, 'blood')]
            o['r1i'] = [const.r1(b0, 'blood', cm)]

        return self.map_results(o)


    def inputs(self) -> set:
        inputs = {'field_strength', 'agent'}
        inputs |= self._rm_inputs()      

        wcomps = self._wcomps()
        if self.config['baseline']=='measured':
            inputs |= {f"R1_{c}" for c in wcomps}

        return inputs
    
    def outputs(self) -> set:
        outputs = {'RM'}

        if self.config['t1_relaxation']=='lin':
            outputs |= {'R1b', 'r1'}

        if self.config['t2_relaxation']=='lin':
            outputs |= {'R2b', 'r2'}

        if self.config['t2s_relaxation'] in ['lin', 'quad']:
            outputs |= {'R2sb', 'r2s'}

        if self.config['t2s_relaxation']=='quad':
            outputs |= {'r2sq'}

        if self.config['inflow']:
            outputs |= {'r1i', 'R1ib'}

        return outputs


    def _rm(self, i):
        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        if wex=='FF':
            if kin in {'2CX', 'HF', '2CU', 'HFU'}:
                return [ 
                    [1 / i['v_ti'], 1 / i['v_ti']]
                ]
            if kin in {'WV', 'FX', 'NX', 'NXP', 'U'}:
                return [
                    [1 / i['v_ti']]
                ]   

        if wex=='RF': 
            if kin in {'2CX', 'HF'}:
                return [ 
                    [1 / i['v_b'],  0                           ],
                    [0,             1 / (i['v_i'] + i['v_c'])   ],
                ]
            if kin in {'2CU', 'HFU'}:
                return [ 
                    [1 / i['v_b'],  0               ],
                    [0,             1 / i['v_ic']   ],
                ]
            if kin in {'WV'}:
                return [ 
                    [1 / (i['v_i'] + i['v_c'])],
                ]
            if kin in {'FX'}:
                # o['RM'] = [ 
                #     [i['v_p'] / i['v_e'] / i['v_b']],
                #     [i['v_i'] / i['v_e'] / i['v_ic']],
                # ]
                return [ 
                    [(1 - i['H']) / i['v_e']    ],
                    [i['P'] / i['v_e']          ], 
                ]
            if kin in {'NX', 'NXP', 'U'}: 
                return [ 
                    [1 / i['v_b']   ],
                    [0              ],
                ]

        if wex=='FR': 
            if kin in {'2CX', 'HF', '2CU', 'HFU'}:
                return [ 
                    [1 / (i['v_b'] + i['v_i']), 1 / (i['v_b'] + i['v_i'])   ],
                    [0,                         0                           ],
                ]                
            if kin in {'WV'}:
                return [ 
                    [1 / i['v_i']   ],
                    [0,             ],
                ]
            if kin in {'FX'}:
                return [ 
                    [1 / i['v_e']   ],
                    [0,             ],
                ]
            if kin in {'NX', 'NXP'}: 
                return [ 
                    [1 / (i['v_b'] + i['v_i'])  ],
                    [0                          ],
                ]   
            if kin in {'U'}:
                return [ 
                    [1 / i['v_bi']  ],
                    [0              ],
                ]  

        if wex=='RR': 
            if kin in {'2CX', 'HF', '2CU', 'HFU'}:
                return [ 
                    [1 / i['v_b'],  0               ],
                    [0,             1 / i['v_i']    ],
                    [0,             0               ]
                ]
            if kin in {'WV'}:
                return [
                    [1 / i['v_i']   ],
                    [0              ]
                ]  
            if kin in {'FX'}:
                return [
                    [(1 - i['H']) / i['v_e']],
                    [1 / i['v_e']           ],
                    [0                      ]
                ] 
            if kin in {'NX', 'NXP', 'U'}:
                return [
                    [1 / i['v_b']   ],
                    [0              ],
                    [0              ]
                ] 
            
        raise ValueError(f"Unknown configuration {wex}, {kin}.")      

    def _rm_inputs(self):
        kin = self.config['kinetics']
        wex = self.config['water_exchange'].replace('N','R')

        if wex=='FF':
            return {'v_ti'} 

        if wex=='RF': 
            if kin in {'2CX', 'HF'}:
                return {'v_b', 'v_i', 'v_c'} 
            if kin in {'2CU', 'HFU'}:
                return {'v_b', 'v_ic'}
            if kin in {'WV'}:
                return {'v_i', 'v_c'} 
            if kin in {'FX'}:
                return {'H', 'P', 'v_e'} 
            if kin in {'NX', 'NXP', 'U'}: 
                return {'v_b'} 

        if wex=='FR': 
            if kin in {'2CX', 'HF', '2CU', 'HFU'}:
                return {'v_b', 'v_i'}                 
            if kin in {'WV'}:
                return {'v_i'} 
            if kin in {'FX'}:
                return {'v_e'} 
            if kin in {'NX', 'NXP'}: 
                return {'v_b', 'v_i'}   
            if kin in {'U'}:
                return {'v_bi'}  

        if wex=='RR': 
            if kin in {'2CX', 'HF', '2CU', 'HFU'}:
                return {'v_b', 'v_i'} 
            if kin in {'WV'}:
                return {'v_i'}   
            if kin in {'FX'}:
                return {'H', 'v_e'} 
            if kin in {'NX', 'NXP', 'U'}:
                return {'v_b'}    

        raise ValueError(f"Unknown configuration {wex}, {kin}.") 

    def _icomps(self):
        kin = self.config['kinetics']

        if kin in {'2CX', 'HF', '2CU', 'HFU'}:
            return ('b', 'i')
        if kin in {'WV'}:
            return ('i', )
        if kin in {'FX'}:
            return ('e', )
        if kin in {'NX', 'NXP', 'U'}:
            return ('b', )

        raise ValueError(f"Unknown configuration {kin}.")     

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
      