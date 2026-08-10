import numpy as np

from dcmri.core.module import Module
from dcmri.core.sequences import SEQUENCES
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.relaxivity.functions_relaxivity import relax_t1, relax_t2, relax_t2s, mix_fast_exchange


class R1(Module):
    configs = {
        't1_relaxation': {'lin'},
    }
    defaults = {
        't1_relaxation': 'lin',
    }
    def inputs(self) -> set:
        # c = Concentration in water compartments, dimensions (nc, nt) or (nt, )
        # v = volume fractions of water compartments, dimensions (nc, ) or scalar
        # fwx = list of compartments (indices) that are in fast water exchange
        inputs = {'c', 'v', 'fx'} 
        if self.config['t1_relaxation'] == 'lin':
            inputs |= {'R1b', 'r1'}
        return inputs
    
    def outputs(self) -> set:
        return {'R1', 'v'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # Possible input dimensions
        # ndim=2: conc (nc, nt), R1b (nc), r1 (nc, )
        # ndim=1: conc (nt, ), R1b (scalar), r1 (scalar)

        # First compute R1 of each compartment
        if self.config['t1_relaxation'] == 'lin':
            R1 = np.full_like(p['c'], np.nan)
            if p['c'].ndim==2:
                R1b = np.atleast_1d(p['R1b'])
                r1 = np.atleast_1d(p['r1'])

                for i in range(p['c'].shape[0]):
                    if not np.isnan(p['c'][i]).any():
                        R1[i] = relax_t1(p['c'][i], R1b[i], r1[i])

            elif not np.isnan(p['c']).any():
                R1 = relax_t1(p['c'], p['R1b'], p['r1']) # (nt, )

        v, R1 = mix_fast_exchange(p['v'], R1, p['fx'])
        return self.map_results({'v': v, 'R1': R1})


class R2(Module):
    configs = {
        't2_relaxation': {'lin'},
    }
    defaults = {
        't2_relaxation': 'lin',
    }
    def inputs(self) -> set:
        # c = Concentration in water compartments, dimensions (nc, nt) or (nt, )
        # v = volume fractions of water compartments, dimensions (nc, ) or scalar
        # fwx = list of compartments (indices) that are in fast water exchange
        inputs = {'c', 'v', 'fx'} 
        if self.config['t2_relaxation'] == 'lin':
            inputs |= {'R2b', 'r2'}
        return inputs
    
    def outputs(self) -> set:
        return {'R2'}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # Possible input dimensions
        # ndim=2: conc (nc, nt), R1b (nc), r1 (nc, )
        # ndim=1: conc (nt, ), R1b (scalar), r1 (scalar)

        if self.config['t2_relaxation'] == 'lin':
            R2 = np.full_like(p['c'], np.nan)
            if p['c'].ndim==2:
                R2b = np.atleast_1d(p['R2b'])
                r2 = np.atleast_1d(p['r2'])

                for i in range(p['c'].shape[0]):
                    if not np.isnan(p['c'][i]).any():
                        R2[i] = relax_t2(p['c'][i], R2b[i], r2[i])
                        
            elif not np.isnan(p['c']).any():
                R2 = relax_t2(p['c'], p['R2b'], p['r2']) # (nt, )

        v, R2 = mix_fast_exchange(p['v'], R2, p['fx'])
        return self.map_results({'v': v, 'R2': R2})


class R2s(Module): 
    configs = {
        't2s_relaxation': {'lin', 'quad', 'leakage'},
    }
    defaults = {
        't2s_relaxation': 'lin', 
    }
    def inputs(self) -> set:
        if self.config['t2s_relaxation'] == 'lin':
            inputs = {'v', 'c', 'R2sb', 'r2s'}
        if self.config['t2s_relaxation'] == 'quad':
            inputs = {'v', 'c', 'R2sb', 'r2s', 'r2s_quad'}
        if self.config['t2s_relaxation'] == 'leakage':
            inputs = {'c', 'R2sb', 'r2s_vasc', 'r2s_ees'} 
        return inputs
    
    def outputs(self) -> set:
        return {'R2s'}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        t2r = self.config['t2s_relaxation']

        # Possible input dimensions
        # ndim=2: conc (nc, nt), R1b (nc), r1 (nc, )
        # ndim=1: conc (nt, ), R1b (scalar), r1 (scalar)

        # Output dim always (nt,)

        if t2r == 'lin':
            if p['c'].ndim==1:
                C = p['v'] * p['c']
            else:
                C = np.atleast_1d(p['v']) @ p['c']
            R2s = relax_t2s(C, p['R2sb'], p['r2s'], model='lin')

        elif t2r == 'quad':
            if p['c'].ndim==1:
                C = p['v'] * p['c']
            else:
                C = np.atleast_1d(p['v']) @ p['c']
            R2s = relax_t2s(C, p['R2sb'], p['r2s'], p['r2s_quad'] , model='quad')
        
        elif t2r == 'leakage':
            R2s = relax_t2s(p['c'], p['R2sb'], r2s_vasc=p['r2s_vasc'], r2s_ees=p['r2s_ees'] , model='leakage')

        return self.map_results({'R2s': R2s})

 

class Relax(Module):
    configs = {
        't1_relaxation': {None} | R1.configs['t1_relaxation'],
        't2_relaxation': {None} | R2.configs['t2_relaxation'],
        't2s_relaxation': {None} | R2s.configs['t2s_relaxation'],
    }
    defaults = {
        't1_relaxation': 'lin',
        't2_relaxation': 'lin',
        't2s_relaxation': 'lin',
    }
    def __init__(self, sequence=None, imap:dict=None, omap:dict=None, **config):
        t1 = config['t1_relaxation'] if 't1_relaxation' in config else self.defaults['t1_relaxation']
        t2 = config['t2_relaxation'] if 't2_relaxation' in config else self.defaults['t2_relaxation']
        t2s = config['t2s_relaxation'] if 't2s_relaxation' in config else self.defaults['t2s_relaxation']

        # Make sure that the contrasts needed by the sequence are computed
        if sequence is None:
            props = set(SEQUENCES[sequence]['parameters']['tissue'])
            if 'R1' in props:
                if not t1:
                    raise InvalidConfiguration(f"The t1_relaxation option can't be None for T1-weighted sequences.")
            if 'R2' in props:
                if not t2:
                    raise InvalidConfiguration(f"The t2_relaxation option can't be None for T2-weighted sequences.")
            if 'R2s' in props:
                if not t2s:
                    raise InvalidConfiguration(f"The t2s_relaxation option can't be None for T2*-weighted sequences.")

        # Set configuration
        self.set_config(config)

        if self.config['t1_relaxation']:
            self._R1 = R1(t1_relaxation=t1)
        if self.config['t2_relaxation']:
            self._R2 = R2(t2_relaxation=t2)
        if self.config['t2s_relaxation']:
            self._R2s = R2s(t2s_relaxation=t2s)

        self.map_io(imap, omap)

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
            R_arr |= self._R1(p)
        if self.config['t2_relaxation']:
            R_arr |= self._R2(p)
        if self.config['t2s_relaxation']:
            R_arr |= self._R2s(p)

        return self.map_results(R_arr)