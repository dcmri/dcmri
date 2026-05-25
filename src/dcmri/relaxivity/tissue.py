from itertools import combinations

import numpy as np

from dcmri.core import LayerFunction
import dcmri.relaxivity.lib as rel


class R1(LayerFunction):
    configs = {
        't1_relaxation': ['lin'],
    }
    def __init__(self, t1_relaxation='lin', **params):
        cnfg = {
            't1_relaxation': t1_relaxation,
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> list:
        t1 = self._cnfg['t1_relaxation']
        if t1 == 'lin':
            return ['R10', 'r1']

    def __call__(self, c: np.ndarray, **params):
        # Note this requires indicator c (not C) in the water 
        # compartments as input, dimensions (nc, nt) or (nt)
        p = self._update_pars(**params)
        t1r = self._cnfg['t1_relaxation']
        
        if t1r == 'lin':
            if c.ndim==1:
                return rel.relax_t1(c, p['R10'], p['r1'])
            else:
                if c.shape[0] != np.size(p['R10']):
                    R10 = np.full(c.shape[0], p['R10'])
                else:
                    R10 = p['R10']
                if c.shape[0] != np.size(p['r1']):
                    r1 = np.full(c.shape[0], p['r1'])
                else:
                    r1 = p['r1']
                # Compute R1 of water compartments
                R1_result = [rel.relax_t1(c[i,:], R10[i], r1[i]) for i in range(c.shape[0])]
                return np.stack(R1_result)


class R2(LayerFunction):
    configs = {
        't2_relaxation': ['lin'],
    }
    def __init__(self, t2_relaxation='lin', **params):
        cnfg = {
            't2_relaxation': t2_relaxation,
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> list:
        t2r = self._cnfg['t2_relaxation']
        if t2r == 'lin':
            return ['R20', 'r2']
    
    def __call__(self, c: np.ndarray, **params):
        # Note this requires indicator c (not C) in the water 
        # compartments as input, dimensions (nc, nt) or (nt)
        p = self._update_pars(**params)
        t2r = self._cnfg['t2_relaxation']
        
        if t2r == 'lin':
            if c.ndim==1:
                return rel.relax_t1(c, p['R20'], p['r2'])
            else:
                if c.shape[0] != np.size(p['R20']):
                    R20 = np.full(c.shape[0], p['R20'])
                else:
                    R20 = p['R20']
                if c.shape[0] != np.size(p['r2']):
                    r2 = np.full(c.shape[0], p['r2'])
                else:
                    r2 = p['r2']
                # Compute R2 of water compartments
                R2_result = [rel.relax_t1(c[i,:], R20[i], r2[i]) for i in range(c.shape[0])]
                return np.stack(R2_result)
        

class R2s(LayerFunction): 
    configs = {
        't2s_relaxation': ['lin', 'quad'],
    }
    def __init__(self, t2s_relaxation='lin', **params):
        cnfg = {
            't2s_relaxation': t2s_relaxation, 
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self) -> list:
        t2r = self._cnfg['t2s_relaxation']
        if t2r == 'lin':
            return ['R20s', 'r2s']
        if t2r == 'quad':
            return ['R20s', 'r2s', 'r2s_quad']
    
    def __call__(self, c, **params):
        p = self._update_pars(**params)
        t2r = self._cnfg['t2s_relaxation']
        
        c = np.array(c)
        # if c.ndim == 2:
        #     raise ValueError('T2*-relaxation requires a single concentration averaged over all compartments.')

        if t2r == 'lin':
            return rel.relax_t2s(c, p['R20s'], p['r2s'], model='lin')

        if t2r == 'quad':
            return rel.relax_t2s(c, p['R20s'], p['r2s'], p['r2s_quad'] , model='quad')


# Build all possible combinations of relaxation rates
weighting = ['R1', 'R2', 'R2s']
all_combinations = [
    set(combo)
    for r in range(1, len(weighting) + 1)
    for combo in combinations(weighting, r)
]    

class Relax(LayerFunction):
    configs = {
        't2s_relaxation': ['lin', 'quad'],
        't2_relaxation': ['lin'],
        't1_relaxation': ['lin'],
        'tissue_props': all_combinations,
    }
    def __init__(self, 
        t2s_relaxation='lin', 
        t2_relaxation='lin', 
        t1_relaxation='lin', 
        tissue_props={'R1', 'R2', 'R2s'},
        **params,
    ):
        cnfg = {
            't2s_relaxation': t2s_relaxation,
            't2_relaxation': t2_relaxation,
            't1_relaxation': t1_relaxation,
            'tissue_props': tissue_props,
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        props = self._cnfg['tissue_props']
        p = []
        if 'R1' in props:
            p += R1(**self._cnfg)._params()
        if 'R2' in props:
            p += R2(**self._cnfg)._params()
        if 'R2s' in props:
            p += R2s(**self._cnfg)._params()
        return p
    
    def __call__(self, c, **params):
        p = self._update_pars(**params)
        props = self._cnfg['tissue_props']

        R1_arr = R2_arr = R2s_arr = None

        if 'R1' in props:
            R1_arr = R1(**self._cnfg)(c, **p)
        if 'R2' in props:
            R2_arr = R2(**self._cnfg)(c, **p)
        if 'R2s' in props:
            R2s_arr = R2s(**self._cnfg)(c, **p)

        return R1_arr, R2_arr, R2s_arr