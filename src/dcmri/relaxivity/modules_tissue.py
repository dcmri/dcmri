import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import get_sequence
from dcmri.core.module import InvalidConfig
from dcmri.relaxivity.functions_relaxivity import relax_t2s


def div(C, v):
    if v==0:
        return C * 0 # In this case the result does not matter
    else:
        return C / v

# +--------------------------------------------------------------------------------------------------+
# |                                     R1 - all configs (n = 1)                                     |
# +---------------+------------------------------------------------------------------------+---------+
# | Key           | Values                                                                 | Default |
# +---------------+------------------------------------------------------------------------+---------+
# | t1_relaxation | lin                                                                    | lin     |
# +--------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------+
# |                                             R1 - all inputs (n = 4)                                              |
# +-----+----------+----------------------------------------+-----------------+-------+--------------+-------+-------+
# | Key | Unit     | Name                                   | Group           | Init  | Bounds       | DICOM | OSIPI |
# +-----+----------+----------------------------------------+-----------------+-------+--------------+-------+-------+
# | C   | mmol/cm3 | tissue concentration                   | Indicator       | 0.005 | (0, 1)       |       |       |
# +-----+----------+----------------------------------------+-----------------+-------+--------------+-------+-------+
# | R1b | Hz       | precontrast tissue R1                  | Electromagnetic | 0.65  | (0, 5)       |       |       |
# | r1  | Hz/M     | longitudinal contrast agent relaxivity | Electromagnetic | 3500  | (0, 10000.0) |       |       |
# +-----+----------+----------------------------------------+-----------------+-------+--------------+-------+-------+
# | RM  |          | relaxivity mapping                     | Physiological   |       |              |       |       |
# +------------------------------------------------------------------------------------------------------------------+

# +--------------------------------------------------------------------------+
# |                         R1 - all outputs (n = 1)                         |
# +-----+------+-----------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name      | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+-----------+-----------------+------+--------+-------+-------+
# | R1  | Hz   | tissue R1 | Electromagnetic | 0.65 | (0, 5) |       |       |
# +--------------------------------------------------------------------------+

class R1(Module):
    configs = {
        't1_relaxation': {'lin'},
    }
    defaults = {
        't1_relaxation': 'lin',
    }
    _all_inputs = {'C', 'r1', 'RM', 'R1b'}
    _all_outputs = {'R1'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # Input dimensions
        # conc (nc, nt), R1b (nc), r1 (nc, )

        # First compute R1 of each compartment
        if self.config['t1_relaxation'] == 'lin':
            R1b = np.array(p['R1b'])
            RM = np.array(p['RM'])
            r1 = np.array(p['r1'])

            R1 = R1b[:, None] + RM @ (r1[:, None] * p['C'])

            # shape = (len(p['wx']), p['C'].shape[-1])
            # R1 = np.zeros(shape)
            # for i, fx in enumerate(p['wx']):
            #     # Note division by vw[i] is intentional here
            #     R1[i] = p['R1b'][i] + np.sum([p['r1'][j] * div(p['C'][j], p['vw'][i]) for j in fx], axis=0)

        return self.map_results({'R1': R1})
    
    def inputs(self) -> set:
        inputs = {'C'} 
        if self.config['t1_relaxation'] == 'lin':
            inputs |= {'R1b', 'RM', 'r1'}
        return inputs
    
    def outputs(self) -> set:
        return {'R1'}

    def dummy_data(self, nc=2, nt=5):
        data = self.init_data()
        data |= {
            'C': np.ones((nc, nt)), 
            'R1b': np.ones(nc), 
            'r1': 1e3 * np.ones(nc), 
            'RM': np.eye(nc),

            # 'vw': np.ones(nc) / nc, 
            # 'wx': [[0]],
        }
        return data


# +--------------------------------------------------------------------------------------------------+
# |                                     R2 - all configs (n = 1)                                     |
# +---------------+------------------------------------------------------------------------+---------+
# | Key           | Values                                                                 | Default |
# +---------------+------------------------------------------------------------------------+---------+
# | t2_relaxation | lin                                                                    | lin     |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------+
# |                                            R2 - all inputs (n = 4)                                             |
# +-----+----------+--------------------------------------+-----------------+-------+--------------+-------+-------+
# | Key | Unit     | Name                                 | Group           | Init  | Bounds       | DICOM | OSIPI |
# +-----+----------+--------------------------------------+-----------------+-------+--------------+-------+-------+
# | C   | mmol/cm3 | tissue concentration                 | Indicator       | 0.005 | (0, 1)       |       |       |
# +-----+----------+--------------------------------------+-----------------+-------+--------------+-------+-------+
# | R2b | Hz       | precontrast tissue R2                | Electromagnetic | 20    | (0, 100)     |       |       |
# | r2  | Hz/M     | transverse contrast agent relaxivity | Electromagnetic | 4000  | (0, 10000.0) |       |       |
# +-----+----------+--------------------------------------+-----------------+-------+--------------+-------+-------+
# | RM  |          | relaxivity mapping                   | Physiological   |       |              |       |       |
# +----------------------------------------------------------------------------------------------------------------+

# +--------------------------------------------------------------------------+
# |                         R2 - all outputs (n = 1)                         |
# +-----+------+-----------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name      | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+-----------+-----------------+------+--------+-------+-------+
# | R2  | Hz   | tissue R2 | Electromagnetic | 2.0  | (0, 5) |       |       |
# +--------------------------------------------------------------------------+

class R2(Module):
    configs = {
        't2_relaxation': {'lin'},
    }
    defaults = {
        't2_relaxation': 'lin',
    }

    _all_inputs = {'r2', 'C', 'R2b', 'RM'}
    _all_outputs = {'R2'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        # Possible input dimensions
        # conc (nc, nt), R1b (nc), r1 (nc, )

        if self.config['t2_relaxation'] == 'lin':
            R2b = np.array(p['R2b'])
            RM = np.array(p['RM'])
            r2 = np.array(p['r2'])

            R2 = R2b[:, None] + RM @ (r2[:, None] * p['C'])

            # shape = (len(p['wx']), p['C'].shape[-1])
            # R2 = np.zeros(shape)
            # for i, fx in enumerate(p['wx']):
            #     R2[i] = p['R2b'][i] + np.sum([p['r2'][j] * div(p['C'][j], p['vw'][i]) for j in fx], axis=0)

        return self.map_results({'R2': R2})

    def inputs(self) -> set:
        inputs = {'C'} 
        if self.config['t2_relaxation'] == 'lin':
            inputs |= {'R2b', 'RM', 'r2'}
        return inputs
    
    def outputs(self) -> set:
        return {'R2'}

    def dummy_data(self, nc=2, nt=5):
        data = self.init_data()
        data |= {
            'C': np.ones((nc, nt)), 
            'R2b': np.ones(nc), 
            'r2': 1e3 * np.ones(nc), 
            'RM': np.eye(nc),

            # 'wx': [[0]],
            # 'vw': np.ones(nc) / nc, 
        }
        return data

# +--------------------------------------------------------------------------------------------------+
# |                                    R2s - all configs (n = 1)                                     |
# +----------------+-----------------------------------------------------------------------+---------+
# | Key            | Values                                                                | Default |
# +----------------+-----------------------------------------------------------------------+---------+
# | t2s_relaxation | leakage, lin, quad                                                    | lin     |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            R2s - all inputs (n = 7)                                                           |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | Key  | Unit     | Name                                                              | Group           | Init  | Bounds        | DICOM | OSIPI |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | C    | mmol/cm3 | tissue concentration                                              | Indicator       | 0.005 | (0, 1)        |       |       |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | R2sb | Hz       | precontrast tissue R2*                                            | Electromagnetic | 20    | (0, 100)      |       |       |
# | r2s  | Hz/M     | transverse contrast agent relaxivity                              | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# | r2se | Hz/M     | extravascular, extracellular transverse contrast agent relaxivity | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# | r2sq | Hz/M^2   | quadratic transverse contrast agent relaxivity                    | Electromagnetic | 1000  | (0, 10000.0)  |       |       |
# | r2sv | Hz/M     | vascular transverse contrast agent relaxivity                     | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | v    | mL/cm3   | volume fraction                                                   | Physiological   | 1     | (0, 1)        |       |       |
# +-----------------------------------------------------------------------------------------------------------------------------------------------+

# +---------------------------------------------------------------------------+
# |                         R2s - all outputs (n = 1)                         |
# +-----+------+------------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name       | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+------------+-----------------+------+--------+-------+-------+
# | R2s | Hz   | tissue R2* | Electromagnetic | 20   | (0, 5) |       |       |
# +---------------------------------------------------------------------------+


class R2s(Module): 
    configs = {
        't2s_relaxation': {'lin', 'quad', 'leakage'},
    }
    defaults = {
        't2s_relaxation': 'lin', 
    }

    _all_inputs = {'r2se', 'C', 'v', 'r2sv', 'R2sb', 'r2sq', 'r2s'}
    _all_outputs = {'R2s'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        t2r = self.config['t2s_relaxation']
        # Input dimensions
        # conc (nc, nt), R1b (nc), r1 (nc, )

        # Output dim always (nt,)

        if t2r == 'lin':
            C = p['C'].sum(axis=0)
            R2s = relax_t2s(C, p['R2sb'], p['r2s'], model='lin')

        elif t2r == 'quad':
            C = p['C'].sum(axis=0)
            R2s = relax_t2s(C, p['R2sb'], p['r2s'], p['r2sq'] , model='quad')
        
        elif t2r == 'leakage':
            c = np.array([
                div(p['C'][0], p['v'][0]), 
                div(p['C'][1], p['v'][1]),
            ])
            R2s = relax_t2s(c, p['R2sb'], r2s_vasc=p['r2sv'], r2s_ees=p['r2se'] , model='leakage')

        return self.map_results({'R2s': R2s})

    def inputs(self) -> set:
        if self.config['t2s_relaxation'] == 'lin':
            inputs = {'C', 'R2sb', 'r2s'}
        if self.config['t2s_relaxation'] == 'quad':
            inputs = {'C', 'R2sb', 'r2s', 'r2sq'}
        if self.config['t2s_relaxation'] == 'leakage':
            inputs = {'C', 'v', 'R2sb', 'r2sv', 'r2se'} 
        return inputs
    
    def outputs(self) -> set:
        return {'R2s'}
    
    def dummy_data(self, nc=2, nt=5):
        data = self.init_data()
        data |= {
            'v': np.ones(nc) / nc, 
            'C': np.ones((nc, nt)), 
        }
        return data

 
# +--------------------------------------------------------------------------------------------------+
# |                                   Relax - all configs (n = 3)                                    |
# +----------------+-----------------------------------------------------------------------+---------+
# | Key            | Values                                                                | Default |
# +----------------+-----------------------------------------------------------------------+---------+
# | t1_relaxation  | None, lin                                                             | lin     |
# | t2_relaxation  | None, lin                                                             | None    |
# | t2s_relaxation | None, leakage, lin, quad                                              | lin     |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                          Relax - all inputs (n = 12)                                                          |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | Key  | Unit     | Name                                                              | Group           | Init  | Bounds        | DICOM | OSIPI |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | C    | mmol/cm3 | tissue concentration                                              | Indicator       | 0.005 | (0, 1)        |       |       |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | R1b  | Hz       | precontrast tissue R1                                             | Electromagnetic | 0.65  | (0, 5)        |       |       |
# | R2b  | Hz       | precontrast tissue R2                                             | Electromagnetic | 20    | (0, 100)      |       |       |
# | R2sb | Hz       | precontrast tissue R2*                                            | Electromagnetic | 20    | (0, 100)      |       |       |
# | r1   | Hz/M     | longitudinal contrast agent relaxivity                            | Electromagnetic | 3500  | (0, 10000.0)  |       |       |
# | r2   | Hz/M     | transverse contrast agent relaxivity                              | Electromagnetic | 4000  | (0, 10000.0)  |       |       |
# | r2s  | Hz/M     | transverse contrast agent relaxivity                              | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# | r2se | Hz/M     | extravascular, extracellular transverse contrast agent relaxivity | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# | r2sq | Hz/M^2   | quadratic transverse contrast agent relaxivity                    | Electromagnetic | 1000  | (0, 10000.0)  |       |       |
# | r2sv | Hz/M     | vascular transverse contrast agent relaxivity                     | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | RM   |          | relaxivity mapping                                                | Physiological   |       |               |       |       |
# | v    | mL/cm3   | volume fraction                                                   | Physiological   | 1     | (0, 1)        |       |       |
# +-----------------------------------------------------------------------------------------------------------------------------------------------+

# +---------------------------------------------------------------------------+
# |                        Relax - all outputs (n = 3)                        |
# +-----+------+------------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name       | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+------------+-----------------+------+--------+-------+-------+
# | R1  | Hz   | tissue R1  | Electromagnetic | 0.65 | (0, 5) |       |       |
# | R2  | Hz   | tissue R2  | Electromagnetic | 2.0  | (0, 5) |       |       |
# | R2s | Hz   | tissue R2* | Electromagnetic | 20   | (0, 5) |       |       |
# +---------------------------------------------------------------------------+

class Relax(Module):
    configs = {
        't1_relaxation': {None} | R1.configs['t1_relaxation'],
        't2_relaxation': {None} | R2.configs['t2_relaxation'],
        't2s_relaxation': {None} | R2s.configs['t2s_relaxation'],
    }
    defaults = {
        't1_relaxation': 'lin',
        't2_relaxation': None, # default = DCE
        't2s_relaxation': 'lin',
    }

    _all_inputs = {'r2se', 'C', 'r2', 'r1', 'RM', 'r2sq', 'R2b', 'v', 'r2s', 'r2sv', 'R2sb', 'R1b'}
    _all_outputs = {'R1', 'R2', 'R2s'}

    def __init__(self, sequence=None, imap:dict=None, omap:dict=None, **config):
        t1 = config['t1_relaxation'] if 't1_relaxation' in config else self.defaults['t1_relaxation']
        t2 = config['t2_relaxation'] if 't2_relaxation' in config else self.defaults['t2_relaxation']
        t2s = config['t2s_relaxation'] if 't2s_relaxation' in config else self.defaults['t2s_relaxation']

        # Make sure that the contrasts needed by the sequence are computed
        if sequence is not None:
            props = get_sequence('tissue_params', sequence)
            if 'R1' in props:
                if t1 is None:
                    raise InvalidConfig(f"The t1_relaxation option can't be None for T1-weighted sequences.")
            if 'R2' in props:
                if t2 is None:
                    raise InvalidConfig(f"The t2_relaxation option can't be None for T2-weighted sequences.")
            if 'R2s' in props:
                if t2s is None:
                    raise InvalidConfig(f"The t2s_relaxation option can't be None for T2*-weighted sequences.")
            if 'R1' not in props:
                if t1 is not None:
                    raise InvalidConfig(f"The t1_relaxation option must be None for a sequence without T1-weighting.")
            if 'R2' not in props:
                if t2 is not None:
                    raise InvalidConfig(f"The t2_relaxation option must be None for a sequence without T2-weighting.")
            if 'R2s' not in props:
                if t2s is not None:
                    raise InvalidConfig(f"The t2s_relaxation option must be None for a sequence without T2s-weighting.")

        # Set configuration
        self.set_config(config)

        if self.config['t1_relaxation'] is not None:
            self._R1 = R1(t1_relaxation=t1)
        if self.config['t2_relaxation'] is not None:
            self._R2 = R2(t2_relaxation=t2)
        if self.config['t2s_relaxation'] is not None:
            self._R2s = R2s(t2s_relaxation=t2s)

        self.map_io(imap, omap)

    def inputs(self) -> set:
        inputs = set()
        if self.config['t1_relaxation'] is not None:
            inputs |= self._R1.mapped_inputs()
        if self.config['t2_relaxation'] is not None:
            inputs |= self._R2.mapped_inputs()
        if self.config['t2s_relaxation'] is not None:
            inputs |= self._R2s.mapped_inputs()
        return inputs
    
    def outputs(self) -> set:
        outputs = set()
        if self.config['t1_relaxation'] is not None:
            outputs |= self._R1.outputs()
        if self.config['t2_relaxation'] is not None:
            outputs |= self._R2.outputs()
        if self.config['t2s_relaxation'] is not None:
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

    def dummy_data(self, nc=2, nt=5):
        data = {}
        if self.config['t1_relaxation']:
            data |= self._R1.dummy_data(nc, nt)
        if self.config['t2_relaxation']:
            data |= self._R2.dummy_data(nc, nt)
        if self.config['t2s_relaxation']:
            data |= self._R2s.dummy_data(nc, nt)
        return data


# +--------------------------------------------------------------------------------------------------+
# |                                ConcToRelax - all configs (n = 4)                                 |
# +----------------+-----------------------------------------------------------------------+---------+
# | Key            | Values                                                                | Default |
# +----------------+-----------------------------------------------------------------------+---------+
# | t1_relaxation  | None, lin                                                             | lin     |
# | t2_relaxation  | None, lin                                                             | None    |
# | t2s_relaxation | None, leakage, lin, quad                                              | lin     |
# | inflow         | inlet, none, pool                                                     | none    |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                       ConcToRelax - all inputs (n = 16)                                                       |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | Key  | Unit     | Name                                                              | Group           | Init  | Bounds        | DICOM | OSIPI |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | C    | mmol/cm3 | tissue concentration                                              | Indicator       | 0.005 | (0, 1)        |       |       |
# | ci   | mmol/mL  | inlet concentration                                               | Indicator       | 0.005 |               |       |       |
# | tC   | sec      | concentration time points                                         | Indicator       | 0.0   |               |       |       |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | R1b  | Hz       | precontrast tissue R1                                             | Electromagnetic | 0.65  | (0, 5)        |       |       |
# | R1ib | Hz       | precontrast inlet R1                                              | Electromagnetic | 0.65  | (0, 5)        |       |       |
# | R2b  | Hz       | precontrast tissue R2                                             | Electromagnetic | 20    | (0, 100)      |       |       |
# | R2sb | Hz       | precontrast tissue R2*                                            | Electromagnetic | 20    | (0, 100)      |       |       |
# | r1   | Hz/M     | longitudinal contrast agent relaxivity                            | Electromagnetic | 3500  | (0, 10000.0)  |       |       |
# | r1i  | Hz/M     | inlet longitudinal contrast agent relaxivity                      | Electromagnetic | 3500  | (0, 10000.0)  |       |       |
# | r2   | Hz/M     | transverse contrast agent relaxivity                              | Electromagnetic | 4000  | (0, 10000.0)  |       |       |
# | r2s  | Hz/M     | transverse contrast agent relaxivity                              | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# | r2se | Hz/M     | extravascular, extracellular transverse contrast agent relaxivity | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# | r2sq | Hz/M^2   | quadratic transverse contrast agent relaxivity                    | Electromagnetic | 1000  | (0, 10000.0)  |       |       |
# | r2sv | Hz/M     | vascular transverse contrast agent relaxivity                     | Electromagnetic | 20000 | (0, 100000.0) |       |       |
# +------+----------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-------+
# | RM   |          | relaxivity mapping                                                | Physiological   |       |               |       |       |
# | v    | mL/cm3   | volume fraction                                                   | Physiological   | 1     | (0, 1)        |       |       |
# +-----------------------------------------------------------------------------------------------------------------------------------------------+

# +--------------------------------------------------------------------------------------------+
# |                             ConcToRelax - all outputs (n = 5)                              |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name                        | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-------+
# | R1  | Hz   | tissue R1                   | Electromagnetic | 0.65 | (0, 5) |       |       |
# | R1i | Hz   | inlet R1                    | Electromagnetic | 0.65 | (0, 5) |       |       |
# | R2  | Hz   | tissue R2                   | Electromagnetic | 2.0  | (0, 5) |       |       |
# | R2s | Hz   | tissue R2*                  | Electromagnetic | 20   | (0, 5) |       |       |
# | tR  | sec  | relaxation rate time points | Electromagnetic | 0.0  |        |       |       |
# +--------------------------------------------------------------------------------------------+

class ConcToRelax(Module): 
    configs = Relax.configs | {
        'inflow': {'none', 'pool', 'inlet'},
    }
    defaults = Relax.defaults | {'inflow': 'none'} 

    _all_inputs = {'r2se', 'tC', 'C', 'R1ib', 'r1i', 'r2', 'r1', 'RM', 'r2sq', 'R2b', 'v', 'r2s', 'ci', 'r2sv', 'R2sb', 'R1b'}
    _all_outputs = {'tR', 'R1', 'R2', 'R2s', 'R1i'}

    def __init__(self, sequence=None, imap:dict=None, omap:dict=None, iomap:dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)

        self._relax_tissue = Relax(sequence=sequence, **self.config)

        inflow = self.config['inflow']
        if sequence is not None:
            if 'R1' not in get_sequence('tissue_params', sequence):
                # Only t1_relaxation in current signal models
                inflow = 'none'

        self._relax_inlets = None
        if inflow == 'pool': 
            if self.config['t1_relaxation'] is None:
                raise InvalidConfig(f"The t1_relaxation option can't be None for T1-weighted sequences.")
            
            self._relax_inlets = R1( 
                #imap = {'C':'ci', 'vw':'vwi', 'wx':'wxi', 'R1b':'R1ib', 'r1':'r1i'},
                imap = {'C':'ci', 'RM':'RMi', 'R1b':'R1ib', 'r1':'r1i'},
                omap = {'R1':'R1i'},
                **self.config,
            )  

        self.map_io(imap, omap, iomap)  
        
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p |= self._relax_tissue(p)
        if self._relax_inlets is not None:
            p['RMi'] = np.eye(len(p['r1i'])) # No exchange between inlets
            # p['wxi'] = [[i] for i in range(len(p['r1i']))] # No exchange between inlets
            #p['vwi'] = np.ones(len(p['r1i'])) # Inlet volume fractions are 1
            p |= self._relax_inlets(p)  
        p['tR'] = p['tC']

        return self.map_results(p)

    def inputs(self):
        inputs = self._relax_tissue.mapped_inputs()
        if self._relax_inlets is not None:
            inputs |= self._relax_inlets.mapped_inputs()
            #inputs -= {'wxi', 'vwi'}
            inputs -= {'RMi'}
        inputs |= {'tC'}
        return inputs 
   
    def outputs(self):
        outputs = self._relax_tissue.mapped_outputs() 
        if self._relax_inlets is not None:
            outputs |= self._relax_inlets.mapped_outputs() 
        outputs |= {'tR'}
        return outputs

    def dummy_data(self, nc=2, nt=5): 
        data = self.init_data()
        data |= {
            'C': np.ones((nc, nt)),
            'ci': np.ones((nc, nt)),
            'tC': np.arange(nt),
            'R1b': np.ones(nc),
            'R1ib': np.ones(nc),
            'R2b': np.ones(nc),
            'r1': np.ones(nc),
            'r1i': np.ones(nc),
            'r2': np.ones(nc),
            'RM': np.eye(nc),
            'v': np.ones(nc) / nc,
            #'vw': np.ones(nc) / nc,
            # 'wx': [[0]],
        } 
        return data