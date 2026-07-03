from itertools import combinations

import numpy as np

from dcmri.core.module import Module
from dcmri.relaxivity.functions_relaxivity import relax_t1, relax_t2, relax_t2s


class R1(Module):
    configs = {
        't1_relaxation': {'lin'},
        'fast_water_exchange': {False, True},
    }
    defaults = {
        't1_relaxation': 'lin',
        'fast_water_exchange': True,
    }

    def inputs(self) -> set:
        inputs = set()
        if self.config['t1_relaxation'] == 'lin':
            inputs |= {'R1b', 'r1'}
        if self.config['fast_water_exchange']:
            inputs |= {'C'} # Tissue concentration, dimensions (nc, nt) or (nt)
        else:
            inputs |= {'c'} # Concentration in water compartments, dimensions (nc, nt) or (nt)
        return inputs
    
    def outputs(self) -> set:
        return {'R1'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # Possible input dimensions
        # ndim=2: conc (nc, nt), R1b (nc), r1 (nc, )
        # ndim=1: conc (nt, ), R1b (scalar), r1 (scalar)

        if self.config['t1_relaxation'] == 'lin':
            if self.config['fast_water_exchange']:
                conc = np.asarray(p['C'])
                if conc.ndim==2:
                    R1 = [relax_t1(conc[i,:], p['R1b'][i], p['r1'][i]) for i in range(conc.shape[0])]
                    R1 = np.stack(R1).sum(axis=0)
                else:
                    R1 = relax_t1(conc,  p['R1b'], p['r1'])
            else:
                conc = np.asarray(p['c'])
                if conc.ndim==2:
                    R1 = [relax_t1(conc[i,:], p['R1b'][i], p['r1'][i]) for i in range(conc.shape[0])]
                    R1 = np.stack(R1)
                else:
                    R1 = relax_t1(conc,  p['R1b'], p['r1'])

        return {'R1': R1}


class R2(Module):
    configs = {
        't2_relaxation': {'lin'},
        'fast_water_exchange': {False, True},
    }
    defaults = {
        't2_relaxation': 'lin',
        'fast_water_exchange': True,
    }
    def inputs(self) -> set:
        inputs = set()
        if self.config['t2_relaxation'] == 'lin':
            inputs |= {'R2b', 'r2'}
        if self.config['fast_water_exchange']:
            inputs |= {'C'} # Tissue concentration, dimensions (nc, nt) or (nt)
        else:
            inputs |= {'c'} # Concentration in water compartments, dimensions (nc, nt) or (nt)
        return inputs
    
    def outputs(self) -> set:
        return {'R2'}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # Possible input dimensions
        # ndim=2: conc (nc, nt), R1b (nc), r1 (nc, )
        # ndim=1: conc (nt, ), R1b (scalar), r1 (scalar)

        if self.config['t2_relaxation'] == 'lin':
            if self.config['fast_water_exchange']:
                conc = np.asarray(p['C'])
                if conc.ndim==2:
                    R2 = [relax_t2(conc[i,:], p['R2b'][i], p['r2'][i]) for i in range(conc.shape[0])]
                    R2 = np.stack(R2).sum(axis=0)
                else:
                    R2 = relax_t2(conc,  p['R2b'], p['r2'])
            else:
                conc = np.asarray(p['c'])
                if conc.ndim==2:
                    R2 = [relax_t2(conc[i,:],  p['R2b'][i], p['r2'][i]) for i in range(conc.shape[0])]
                    R2 = np.stack(R2)
                else:
                    R2 = relax_t2(conc,  p['R2b'], p['r2'])

        return {'R2': R2}
        

class R2s(Module): 
    configs = {
        't2s_relaxation': {'lin', 'quad', 'leakage'},
    }
    defaults = {
        't2s_relaxation': 'lin', 
    }
    def inputs(self) -> set:
        if self.config['t2s_relaxation'] == 'lin':
            inputs = {'C', 'R2sb', 'r2s'}
        if self.config['t2s_relaxation'] == 'quad':
            inputs = {'C', 'R2sb', 'r2s', 'r2s_quad'}
        if self.config['t2s_relaxation'] == 'leakage':
            inputs = {'c', 'R2sb', 'r2s_vasc', 'r2s_ees'} # C and c?
        return inputs
    
    def outputs(self) -> set:
        return {'R2s}'}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        t2r = self.config['t2s_relaxation']

        # Possible input dimensions
        # ndim=2: conc (nc, nt), R1b (nc), r1 (nc, )
        # ndim=1: conc (nt, ), R1b (scalar), r1 (scalar)

        # Output dim alwats (nt,)

        if t2r == 'lin':
            conc = np.array(p['C'])
            if conc.ndim==2:
                conc = conc.sum(axis=0)
            R2s = relax_t2s(conc, p['R2sb'], p['r2s'], model='lin')

        elif t2r == 'quad':
            conc = np.array(p['C'])
            if conc.ndim==2:
                conc = conc.sum(axis=0)
            R2s = relax_t2s(conc, p['R2sb'], p['r2s'], p['r2s_quad'] , model='quad')
        
        elif t2r == 'leakage':
            conc = np.array(p['c'])
            if conc.ndim==2:
                conc = conc.sum(axis=0)
            R2s = relax_t2s(conc, p['R2sb'], r2s_vasc=p['r2s_vasc'], r2s_ees=p['r2s_ees'] , model='leakage')

        return {'R2s': R2s}


# # Build all possible combinations of relaxation rates
# weighting = ['R1', 'R2', 'R2s']
# all_combinations = [
#     tuple(set(combo))
#     for r in range(1, len(weighting) + 1)
#     for combo in combinations(weighting, r)
# ]   

class Relax(Module):
    configs = {
        't2s_relaxation': {None} | R2s.configs['t2s_relaxation'],
        't2_relaxation': {None} | R2.configs['t2_relaxation'],
        't1_relaxation': {None} | R1.configs['t1_relaxation'],
        'fast_water_exchange': {False, True},
    }
    defaults = {
        't2s_relaxation': 'lin',
        't2_relaxation': 'lin',
        't1_relaxation': 'lin',
        'fast_water_exchange': True, 
    }
    def __init__(self, imap: dict=None, **config):
        self.set_config(config)

        if self.config['t1_relaxation']:
            self._R1 = R1(**config)
        if self.config['t2_relaxation']:
            self._R2 = R2(**config)
        if self.config['t2s_relaxation']:
            self._R2s = R2s(**config)

        self.map_inputs(imap)

    def inputs(self) -> set:
        inputs = set()
        if self.config['t1_relaxation']:
            inputs |= self._R1.mapped_inputs()
        if self.config['t2_relaxation']:
            inputs |= self._R2.mapped_inputs()
        if self.config['t2s_relaxation']:
            inputs |= self._R2s.mapped_inputs()
        return inputs
    
    def outputs(self) -> set:
        outputs = set()
        if self.config['t1_relaxation']:
            outputs |= self._R1.outputs()
        if self.config['t2_relaxation']:
            outputs |= self._R2.outputs()
        if self.config['t2s_relaxation']:
            outputs |= self._R2s.outputs()
        return outputs
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        R_arr = {}
        if self.config['t1_relaxation']:
            R_arr['R1'] = self._R1(p)
        if self.config['t2_relaxation']:
            R_arr['R2'] = self._R2(p)
        if self.config['t2s_relaxation']:
            R_arr['R2s'] = self._R2s(p)

        return R_arr