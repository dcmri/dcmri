from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcTissueX
import dcmri.kinetics.functions_tissue as pk_tissue

import numpy as np

def div(a, b):
    if a != 0:
        return a / b if b != 0 else np.inf
    else:
        return a if b != 0 else np.nan

class ParsTissueX(Module):
    configs = ConcTissueX.configs
    defaults = ConcTissueX.defaults

    def _call(self, p):
        if self.config['kinetics'] == '2CX':
            p['v_bi'] = p['v_b'] + p['v_i']
            p['v_c'] = 1 - p['v_bi']
            p['v_bc'] = p['v_b'] + p['v_c']
            p['v_p'] = (1 - p['H']) * p['v_b']
            p['v_e'] = p['v_p'] + p['v_i']
            p['v_ic'] = p['v_i'] + p['v_c']

            p['F_p'] = (1 - p['H']) * p['F_b']
            p['Ktrans'] = div(p['F_p'] * p['PS'], p['F_p'] + p['PS'])
            p['E'] = div(p['PS'], p['F_p'] + p['PS'])

            p['T_p'] = div(p['v_p'], p['F_p'] + p['PS'])
            p['T_b'] = div(p['v_b'], p['F_b'])
            p['T_i'] = div(p['v_i'], p['PS'])
            p['kep'] = div(1, p['T_i'])

        if self.config['kinetics'] == 'HF':
            p['v_bi'] = p['v_b'] + p['v_i']
            p['v_c'] = 1 - p['v_bi']
            p['v_bc'] = p['v_b'] + p['v_c']
            p['v_p'] = (1 - p['H']) * p['v_b']
            p['v_e'] = p['v_p'] + p['v_i']
            p['v_ic'] = p['v_i'] + p['v_c']

            p['Ktrans'] = p['PS']
            p['T_i'] = div(p['v_i'], p['PS'])
            p['kep'] = div(1, p['T_i'])

        if self.config['kinetics'] == '2CU':
            p['v_p'] = (1 - p['H']) * p['v_b']

            p['F_p'] = (1 - p['H']) * p['F_b']
            p['Ktrans'] = div(p['F_p'] * p['PS'], p['F_p'] + p['PS'])
            p['E'] = div(p['PS'], p['F_p'] + p['PS'])

            p['T_p'] = div(p['v_p'], p['F_p'] + p['PS'])
            p['T_b'] = div(p['v_b'], p['F_b'])

        if self.config['kinetics'] == 'HFU':
            p['v_p'] = (1 - p['H']) * p['v_b']
            p['Ktrans'] = p['PS']

        if self.config['kinetics'] == 'WV':
            pass

        if self.config['kinetics'] == 'FX':
            p['F_p'] = (1 - p['H']) * p['F_b']

        if self.config['kinetics'] == 'NX':
            p['T_b'] = div(p['v_b'], p['F_b'])

        if self.config['kinetics'] == 'NXP':
            p['T_b'] = div(p['v_b'], p['F_b'])

        if self.config['kinetics'] == 'U':
            pass

        return p

    def inputs(self):
        return pk_tissue.CONC_PARAMETERS[self.config['kinetics']]

    def outputs(self):
        if self.config['kinetics'] == '2CX':
            return {'v_bi', 'v_c', 'v_bc', 'v_p', 'v_e', 'v_ic', 
                    'F_p', 'Ktrans', 'E', 
                    'T_p', 'T_b', 'T_i', 'kep'}
        if self.config['kinetics'] == 'HF':
            return {'v_bi', 'v_c', 'v_bc', 'v_p', 'v_e', 'v_ic', 
                    'Ktrans', 'T_i', 'kep'}
        if self.config['kinetics'] == '2CU':
            return {'v_p', 
                    'F_p', 'Ktrans', 'E', 
                    'T_p', 'T_b'}
        if self.config['kinetics'] == 'HFU':
            return {'v_p', 'Ktrans'}
        if self.config['kinetics'] == 'WV':
            return {}
        if self.config['kinetics'] == 'FX':
            return {'F_p'}
        if self.config['kinetics'] == 'NX':
            return {'T_b'}
        if self.config['kinetics'] == 'NXP':
            return {'T_b'}
        if self.config['kinetics'] == 'U':
            return {}
        