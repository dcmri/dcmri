from itertools import combinations

import numpy as np

from dcmri.core.function import Function
from dcmri.relaxivity.lib import relax_t1, relax_t2, relax_t2s


class R1(Function):
    configs = {
        't1_relaxation': ['lin'],
        'fast_water_exchange': [False, True],
    }
    def __init__(
            self, 
            t1_relaxation='lin', 
            fast_water_exchange=False,
            defaults=None, 
            **kwargs,
        ):
        cnfg = {
            't1_relaxation': t1_relaxation,
            'fast_water_exchange': fast_water_exchange,
        }
        self._set_config(cnfg)
        self._set_params(defaults)

    def params(self) -> list:
        pars = []
        if self._cnfg['t1_relaxation'] == 'lin':
            pars += ['R1b', 'r1']
        if self._cnfg['fast_water_exchange']:
            pars += ['C'] # Tissue concentration, dimensions (nc, nt) or (nt)
        else:
            pars += ['c'] # Concentration in water compartments, dimensions (nc, nt) or (nt)
        return pars

    def __call__(self, **params):
        p = self._update_params(params)

        if self._cnfg['fast_water_exchange']:
            conc = np.asarray(p['C'])
        else:
            conc = np.asarray(p['c'])
        
        if self._cnfg['t1_relaxation'] == 'lin':
            if conc.ndim==1:
                return relax_t1(conc, p['R1b'], p['r1'])
            else:
                if conc.shape[0] != np.size(p['R1b']):
                    R1b = np.full(conc.shape[0], p['R1b'])
                else:
                    R1b = p['R1b']
                if conc.shape[0] != np.size(p['r1']):
                    r1 = np.full(conc.shape[0], p['r1'])
                else:
                    r1 = p['r1']

                # Compute R1 of water compartments
                R1_result = [relax_t1(conc[i,:], R1b[i], r1[i]) for i in range(conc.shape[0])]
                R1_result = np.stack(R1_result)
                if self._cnfg['fast_water_exchange']:
                    R1_result = R1_result.sum(axis=0)
                return R1_result


class R2(Function):
    configs = {
        't2_relaxation': ['lin'],
        'fast_water_exchange': [False, True],
    }
    def __init__(
            self, 
            t2_relaxation='lin', 
            fast_water_exchange=False,
            defaults=None, 
            **kwargs,
        ):
        cnfg = {
            't2_relaxation': t2_relaxation,
            'fast_water_exchange': fast_water_exchange,
        }
        self._set_config(cnfg)
        self._set_params(defaults)

    def params(self) -> list:
        pars = []
        if self._cnfg['t2_relaxation'] == 'lin':
            pars += ['R2b', 'r2']
        if self._cnfg['fast_water_exchange']:
            pars += ['C'] # Tissue concentration, dimensions (nc, nt) or (nt)
        else:
            pars += ['c'] # Concentration in water compartments, dimensions (nc, nt) or (nt)
        return pars
    
    def __call__(self, **params):
        p = self._update_params(params)
        t2r = self._cnfg['t2_relaxation']

        if self._cnfg['fast_water_exchange']:
            conc = np.asarray(p['C'])
        else:
            conc = np.asarray(p['c'])
        
        if t2r == 'lin':
            if conc.ndim==1:
                return relax_t2(conc, p['R2b'], p['r2'])
            else:
                if conc.shape[0] != np.size(p['R2b']):
                    R2b = np.full(conc.shape[0], p['R2b'])
                else:
                    R2b = p['R2b']
                if conc.shape[0] != np.size(p['r2']):
                    r2 = np.full(conc.shape[0], p['r2'])
                else:
                    r2 = p['r2']

                # Compute R1 of water compartments
                R2_result = [relax_t2(conc[i,:], R2b[i], r2[i]) for i in range(conc.shape[0])]
                R2_result = np.stack(R2_result)
                if self._cnfg['fast_water_exchange']:
                    R2_result = R2_result.sum(axis=0)
                return np.stack(R2_result)
        

class R2s(Function): 
    configs = {
        't2s_relaxation': ['lin', 'quad', 'leakage'],
    }
    def __init__(
            self, 
            t2s_relaxation='lin', 
            defaults=None, 
            **kwargs,
        ):
        cnfg = {
            't2s_relaxation': t2s_relaxation, 
        }
        self._set_config(cnfg)
        self._set_params(defaults)

    def params(self) -> list:
        pars = ['c']
        if self._cnfg['t2s_relaxation'] == 'lin':
            pars += ['R2sb', 'r2s']
        if self._cnfg['t2s_relaxation'] == 'quad':
            pars += ['R2sb', 'r2s', 'r2s_quad']
        if self._cnfg['t2s_relaxation'] == 'leakage':
            pars += ['R2sb', 'r2s_vasc', 'r2s_ees']
        return pars
    
    def __call__(self, **params):
        p = self._update_params(params)
        t2r = self._cnfg['t2s_relaxation']
        # if c.ndim == 2:
        #     raise ValueError('T2*-relaxation requires a single concentration averaged over all compartments.')

        conc = np.array(p['c'])

        if t2r == 'lin':
            return relax_t2s(conc, p['R2sb'], p['r2s'], model='lin')

        if t2r == 'quad':
            return relax_t2s(conc, p['R2sb'], p['r2s'], p['r2s_quad'] , model='quad')
        
        if t2r == 'leakage':
            return relax_t2s(conc, p['R2sb'], r2s_vasc=p['r2s_vasc'], r2s_ees=p['r2s_ees'] , model='leakage')


# Build all possible combinations of relaxation rates
weighting = ['R1', 'R2', 'R2s']
all_combinations = [
    set(combo)
    for r in range(1, len(weighting) + 1)
    for combo in combinations(weighting, r)
]    

class Relax(Function):
    configs = {
        't2s_relaxation': ['lin', 'quad', 'leakage'],
        't2_relaxation': ['lin'],
        't1_relaxation': ['lin'],
        'tissue_props': all_combinations,
        'fast_water_exchange': [False, True],
    }
    def __init__(self, 
        t2s_relaxation='lin', 
        t2_relaxation='lin', 
        t1_relaxation='lin', 
        tissue_props={'R1', 'R2', 'R2s'},
        fast_water_exchange=False,
        defaults=None, 
        **kwargs
    ):
        cnfg = {
            't2s_relaxation': t2s_relaxation,
            't2_relaxation': t2_relaxation,
            't1_relaxation': t1_relaxation,
            'tissue_props': tissue_props,
            'fast_water_exchange': fast_water_exchange, 
        }
        self._set_config(cnfg)
        self._set_params(defaults)

    def params(self):
        props = self._cnfg['tissue_props']
        p = []
        if 'R1' in props:
            p += R1(**self._cnfg).params()
        if 'R2' in props:
            p += R2(**self._cnfg).params()
        if 'R2s' in props:
            ps = R2s(**self._cnfg).params()
            # In fast water exchange, if we don't need a leakage model, 
            # the input is C across all mechanisms
            if self._cnfg['fast_water_exchange']:
                if self._cnfg['t2s_relaxation'] != 'leakage':
                    ps = [k for k in ps if k != 'c']
                    ps.append('C')
            p += ps
        return p
    
    def __call__(self, **params):
        p = self._update_params(params)
        props = self._cnfg['tissue_props']

        R_arr = {}
        if 'R1' in props:
            R_arr['R1'] = R1(**self._cnfg, defaults=p)()
        if 'R2' in props:
            R_arr['R2'] = R2(**self._cnfg, defaults=p)()
        if 'R2s' in props:
            if self._cnfg['fast_water_exchange']:
                if self._cnfg['t2s_relaxation'] != 'leakage':
                    p['c'] = p['C']
                    if p['c'].ndim==2:
                        p['c'] = p['c'].sum(axis=0)
            R_arr['R2s'] = R2s(**self._cnfg, defaults=p)()

        return R_arr