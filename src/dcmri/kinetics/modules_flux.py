import numpy as np
from scipy.integrate import trapezoid

from dcmri.core.module import Module
from dcmri.utils import const
from dcmri.kinetics.functions_input import ca_injection
from dcmri.kinetics.functions_blocks import flux_plug
import dcmri.kinetics.functions_tissue as pk_tissue
import dcmri.kinetics.functions_blocks as blocks


class Flux(Module):
    """Flux through an abritrary building block.
    """
    configs = {
        'block': set(blocks.FLUX_PARAMETERS.keys()),
    }
    defaults = {
        'block': 'comp',
    }
    def inputs(self):
        # TODO: Better to change the functions to use the same parametrization T1, T2 vs T=[T1, T2] etc
        inputs = set(blocks.FLUX_PARAMETERS[self.config['block']])
        if self.config['block']=='plucom':
            inputs -= {'T'}
            inputs |= {'Tc', 'Tp'}
        elif self.config['block']=='2cxm':
            inputs -= {'T'}
            inputs |= {'T1', 'T2'} 
        return inputs
    
    def outputs(self):
        outputs = {'J'}
        if self.config['block']=='plucom':
            outputs |= {'Jc', 'Jp'}
        return outputs
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        model_func = getattr(blocks, f"flux_{self.config['block']}") 

        if self.config['block']=='plucom':
            p |= {'T': [p['Tc'], p['Tp']]}
            p.pop('Tc'), p.pop('Tp')
            Jc, Jp, J = model_func(**p)
            return {'J': J, 'Jc': Jc, 'Jp': Jp}
        
        if self.config['block']=='2cxm':
            p |= {'T': [p['T1'], p['T2']]}
            p.pop('T1'), p.pop('T2')
        J = model_func(**p)
        return {'J': J}


class FluxInjection(Module):
    """Indicator flux injected"""

    configs = {
        'bolus': {'single', 'dual'}
    }
    defaults = {
        'bolus': 'single',
    }
    def inputs(self):
        inputs = {'tmax', 'dt', 'agent', 'weight'}
        if self.config['bolus'] == 'single':
            inputs |= {
                'dose', 'rate', 'BAT'
            }
        else:
            inputs |= {
                'dose_1', 'rate_1', 'BAT',
                'dose_2', 'rate_2', 'bolus_delay'
            }
        return inputs
    
    def outputs(self):
        return {'Ji'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        t = np.arange(0, p['tmax'], p['dt'])
        conc = const.ca_conc(p['agent'])

        if self.config['bolus'] == 'single':
            J = ca_injection(
                t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
            )
        else:
            J1 = ca_injection(
                t, p['weight'], conc, p['dose_1'], p['rate_1'], p['BAT']
            )
            J2 = ca_injection(
                t, p['weight'], conc, p['dose_2'], p['rate_2'], p['BAT'] + p['bolus_delay']
            )
            J = J1 + J2
        
        return {'Ji': J}


class FluxTissueX(Module):
    """Flux out of vascular-interstitial tissue.

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        params (dict, optional): override parameter defaults.
    """
    configs = {
        'kinetics': set(pk_tissue.FLUX_PARAMETERS.keys())
    }
    defaults = {
        'kinetics': '2CX',
    }  
         
    def inputs(self):
        inputs = set(pk_tissue.FLUX_PARAMETERS[self.config['kinetics']])
        inputs |= {'Ta', 'ca', 'dt'}
        return inputs
    
    def outputs(self):
        return ['J']

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        ca = flux_plug(p['ca'], dt=p['dt'], T=p['Ta'])
        p = {k: v for k, v in p.items() if k not in ['ca', 'Ta']}

        flux = 'flux_tissue_' + self.config['kinetics'].lower()   
        model_func = getattr(pk_tissue, flux)  
        return {'J':model_func(ca, **p)}
    

class FluxAorta(Module):
    """Whole-body model for indicator flux in the aorta.

    Args:
        heartlung (str, optional): Model for the heart-lung system. 
        organs (str, optional): Model for the systemic organs. 
        **params: override parameter defaults.
    """
    configs = {
        'heartlung': {'comp', 'pfcomp', 'chain'},
        'organs': {'comp', '2cxm'},
        'kidneys': {None, 'pass', 'comp', 'plug'},
        'liver': {None, 'pass', 'comp', 'plug'},
        'lagut':{None, 'pass', 'comp', 'plucom'},
        'bolus': FluxInjection.configs['bolus'],
    }
    defaults = {
        'heartlung': 'pfcomp', 
        'organs': 'comp', 
        'kidneys': None,
        'liver': None,
        'lagut': None,
        'bolus': 'single',
    }
    def __init__(self, imap:dict=None, **config):
        self.set_config(config)
        if self.config['lagut'] is not None:
            if self.config['liver'] is None:
                raise ValueError("A liver artery and gut component requires a liver component too.")
            
        self._flux_injection = FluxInjection(self.config)
        self._flux_heartlung = Flux({'T': 'Thl', 'D': 'Dhl'}, block= self.config['heartlung'])
        self._flux_organs = Flux({'T':'To', 'T1': 'To', 'T2': 'To_e', 'E':'Eo'}, block = self.config['organs'])
        if self.config['kidneys'] is not None:
            self._flux_lk = Flux({'T':'Tp_lk'}, block=self.config['kidneys'])
            self._flux_rk = Flux({'T':'Tp_rk'}, block=self.config['kidneys'])
        if self.config['liver'] is not None:
            self._flux_liver = Flux({'T':'Te_l'}, block=self.config['liver'])
        if self.config['lagut'] is not None:
            self._flux_lagut = Flux({'T':'Tg', 'Tp':'Ta', 'Tc':'Tg', 'fp':'fa'}, block=self.config['lagut'])

        self.map_inputs(imap)
        
    def inputs(self):
        inputs = self._flux_injection.mapped_inputs()
        inputs |= self._flux_heartlung.mapped_inputs()
        inputs |= {'vr_o'} | self._flux_organs.mapped_inputs()
        if self.config['kidneys'] is not None:
            inputs |= {'vr_lk'} | self._flux_lk.mapped_inputs()
            inputs |= {'vr_rk'} | self._flux_rk.mapped_inputs()
        if self.config['liver'] is not None:
            inputs |= {'vr_l'} | self._flux_liver.mapped_inputs()
        if self.config['lagut'] is not None:
            inputs |= self._flux_lagut.mapped_inputs()

        inputs |= {'dt', 'dose_tolerance'}
        inputs -= {'Ji', 'J'} # derived
        return inputs
    
    def outputs(self):
        outputs = {'Ja', 'Jv', 'Jo'}
        if self.config['kidneys'] is not None:
            outputs |= {'Jlk', 'Jrk'}
        if self.config['liver'] is not None:
            outputs |= {'Jl'}
            if self.config['lagut'] is not None:
                outputs |= {'Jlag'}
                if self._flux_lagut.config['block']=='plucom':
                    outputs |= {'Jla', 'Jpv'}

        return outputs 
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        max_it = 500

        influx = self._flux_injection(p)
        J_aorta = self._flux_heartlung(p, J=influx['Ji'])['J']
        dose = trapezoid(J_aorta, dx=p['dt'])
        min_dose = p['dose_tolerance'] * dose

        J_aorta_total = J_aorta
        it=0
        while True:
            J_aorta = self._propagate_J_aorta(p, J_aorta)['Ja']
            J_aorta_total += J_aorta

            dose = trapezoid(J_aorta, dx=p['dt'])
            if dose <= min_dose:
                break

            it += 1
            if it > max_it:
                break

        return self._propagate_J_aorta(p, J_aorta_total)
    
    # Helper function
    def _propagate_J_aorta(self, p, Ja):
        # Store all results along the way so they can be returned
        result = {}
        Jv = np.zeros_like(Ja)
        result['Jo'] = self._flux_organs(p | {'J': Ja})['J']
        Jv += p['vr_o'] * result['Jo']

        if self.config['kidneys'] is not None:
            result['Jlk'] = self._flux_lk(p | {'J': Ja})['J']
            result['Jrk'] = self._flux_rk(p | {'J': Ja})['J']
            Jv += p['vr_lk'] * result['Jlk']
            Jv += p['vr_rk'] * result['Jrk']

        if self.config['liver'] is not None:
            if self.config['lagut'] is not None:
                Jlag = self._flux_lagut(p | {'J': Ja})
                result['Jlag'] = Jlag['J']
                if self._flux_lagut.config['block']=='plucom':
                    result['Jla'] = Jlag['Jp']
                    result['Jpv'] = Jlag['Jc']
            else:
                result['Jlag'] = Ja
            result['Jl'] = self._flux_liver(p | {'J': result['Jlag']})['J']
            Jv += p['vr_l'] * result['Jl']

        Ja = self._flux_heartlung(p | {'J': Jv})['J']
        return result | {'Ja': Ja, 'Jv':Jv}
    
