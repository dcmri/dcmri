import numpy as np
from scipy.integrate import trapezoid

from dcmri.core.module import Module
from dcmri.core.module import InvalidConfig
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
        inputs = set(blocks.FLUX_PARAMETERS[self.config['block']])
        return inputs
    
    def outputs(self):
        return {'J'}
    
    def __call__(self, data: dict=None) -> dict:
        p = self.map_data(data)

        model_func = getattr(blocks, f"flux_{self.config['block']}") 
        results = {'J': model_func(**p)}

        return self.map_results(results)

    def test_data(self, nt=5, nc=2):
        data = self.init_data()

        data['J'] = np.ones(nt)
        data['h'] = [1]
        data['TT'] = [0, 1]
        if self.config['block'] in ['bicomp', 'plucom', '2cxm']:
            data['T'] = [1, 1]
        if self.config['block'] == 'ncomp':
            data['T'] = np.ones(nc)
            data['J'] = np.ones((nc, nt))
            data['E'] = np.ones((nc, nc))
        if self.config['block'] == 'nscomp':
            data['T'] = np.ones(nt)

        return self.update_data(data)
    

class FluxInjection(Module):
    """Indicator flux injected"""

    configs = {
        'bolus': {'single', 'double', 'dual'}
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
        elif self.config['bolus'] == 'double':
            inputs |= {
                'dose_1', 'rate_1', 'BAT',
                'dose_2', 'rate_2', 'bdel'
            }
        elif self.config['bolus'] == 'dual':
            inputs |= {
                'dose_1', 'rate_1', 'BAT_1',
                'dose_2', 'rate_2', 'BAT_2'
            }
        return inputs
    
    def outputs(self):
        return {'tC', 'Jinj'}

    def __call__(self, data: dict=None) -> dict:
        p = self.map_data(data)

        t = np.arange(0, p['tmax'], p['dt'])
        conc = const.ca_conc(p['agent'])

        if self.config['bolus'] == 'single':
            J = ca_injection(
                t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
            )

        elif self.config['bolus'] == 'double':
            J1 = ca_injection(
                t, p['weight'], conc, p['dose_1'], p['rate_1'], p['BAT']
            )
            J2 = ca_injection(
                t, p['weight'], conc, p['dose_2'], p['rate_2'], p['BAT'] + p['bdel']
            )
            J = J1 + J2
            
        elif self.config['bolus'] == 'dual':
            J1 = ca_injection(
                t, p['weight'], conc, p['dose_1'], p['rate_1'], p['BAT_1']
            )
            J2 = ca_injection(
                t, p['weight'], conc, p['dose_2'], p['rate_2'], p['BAT_2']
            )
            J = J1 + J2

        results = {'tC': t, 'Jinj': J}
        return self.map_results(results)


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
        inputs |= {'T_ar', 'c_ar', 'dt'}
        return inputs
    
    def outputs(self):
        return {'J'}

    def __call__(self, data: dict=None) -> dict:
        p = self.map_data(data)

        ca = flux_plug(p['c_ar'], dt=p['dt'], T=p['T_ar'])
        p = {k: v for k, v in p.items() if k not in ['c_ar', 'T_ar']}

        flux = 'flux_tissue_' + self.config['kinetics'].lower()   
        model_func = getattr(pk_tissue, flux)  
        results = {'J':model_func(ca, **p)}

        return self.map_results(results)

    def test_data(self, nt=5):
        data = self.init_data()
        data['c_ar'] *= np.ones(nt)
        return self.input_data(data)
    

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
        'renal_arteries': {None, 'pass', 'comp', 'plug'},
        'kidneys': {None, 'pass', 'comp', 'plug', 'pfcomp', 'chain'},
        'liver': {None, 'pass', 'comp', 'plug'},
        'lagut': {None, 'pass', 'comp', 'plucom'},
        'veins': {None, 'pass', 'comp', 'plug', 'pfcomp', 'chain'},
        'bolus': FluxInjection.configs['bolus'],
    }
    defaults = {
        'heartlung': 'pfcomp', 
        'organs': 'comp', 
        'renal_arteries': None,
        'kidneys': None,
        'liver': None,
        'lagut': None,
        'veins': None,
        'bolus': 'single',
    }

    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)

        # Check config
        if self.config['renal_arteries'] is not None:
            if self.config['kidneys'] is None:
                raise InvalidConfig("A renal artery component requires a renal component too.")
        if self.config['lagut'] is not None:
            if self.config['liver'] is None:
                raise InvalidConfig("A liver artery and gut component requires a liver component too.")
            
        self._flux_injection = FluxInjection(**self.config)
        self._flux_heartlung = Flux({'T': 'T_hl', 'D': 'D_hl'}, block=self.config['heartlung'])
        self._flux_organs = Flux({'T':'T_or', 'E':'E_or'}, block=self.config['organs'])
        if self.config['renal_arteries'] is not None:
            self._flux_ra_lk = Flux({'T':'T_lka'}, block=self.config['renal_arteries'])
            self._flux_ra_rk = Flux({'T':'T_rka'}, block=self.config['renal_arteries'])
        if self.config['kidneys'] is not None:
            self._flux_lk = Flux({'T':'T_p_lk', 'D':'D_lk'}, block=self.config['kidneys'])
            self._flux_rk = Flux({'T':'T_p_rk', 'D':'D_rk'}, block=self.config['kidneys'])
        if self.config['lagut'] is not None:
            self._flux_lagut = Flux({'T':'T_lag', 'ffp':'ffa'}, block=self.config['lagut'])
        if self.config['liver'] is not None:
            self._flux_liver = Flux({'T':'T_e_li'}, block=self.config['liver'])
        if self.config['veins'] is not None:
            self._flux_veins = Flux({'T':'T_vc', 'D':'D_vc'}, block=self.config['veins'])


        self.map_io(imap, omap, iomap)
        
    def inputs(self):
        inputs = set()
        if self.config['organs'] == '2cxm':
            inputs |= {'T_b_or', 'T_e_or'}
        else:
            inputs |= {'T_b_or'}
        if self.config['lagut'] == 'plucom':
            inputs |= {'T_la', 'T_gu'}
        elif self.config['lagut'] is not None:
            inputs |= {'T_gu'}
        inputs |= self._flux_injection.mapped_inputs()
        inputs |= self._flux_heartlung.mapped_inputs()
        inputs |= {'vr_or'} | self._flux_organs.mapped_inputs()
        if self.config['renal_arteries'] is not None:
            inputs |= self._flux_ra_lk.mapped_inputs()
            inputs |= self._flux_ra_rk.mapped_inputs()
        if self.config['kidneys'] is not None:
            inputs |= {'vr_lk'} | self._flux_lk.mapped_inputs()
            inputs |= {'vr_rk'} | self._flux_rk.mapped_inputs()
        if self.config['liver'] is not None:
            inputs |= {'vr_li'} | self._flux_liver.mapped_inputs()
        if self.config['lagut'] is not None:
            inputs |= self._flux_lagut.mapped_inputs()
        if self.config['veins'] is not None:
            inputs |= self._flux_veins.mapped_inputs()

        inputs |= {'dt', 'dose_tolerance'}
        inputs -= {'Jinj', 'J', 'T_or', 'T_lag'} # derived
        return inputs
    
    def outputs(self):
        outputs = {'tC', 'J_ao', 'J_vc', 'J_or'}
        if self.config['renal_arteries'] is not None:
            outputs |= {'J_lka', 'J_rka'}
        if self.config['kidneys'] is not None:
            outputs |= {'J_lk', 'J_rk'}
        if self.config['lagut'] is not None:
            outputs |= {'J_lag'}
            if self._flux_lagut.config['block']=='plucom':
                outputs |= {'J_la', 'J_pv'}
        if self.config['liver'] is not None:
            outputs |= {'J_li'}
        return outputs 
    
    def __call__(self, data: dict=None) -> dict:
        p = self.map_data(data)

        # Format MTTs
        if self.config['organs'] == '2cxm':
            p['T_or'] = [p['T_b_or'], p['T_e_or']]
        else:
            p['T_or'] = p['T_b_or'] 

        if self.config['lagut'] == 'plucom':
            p['T_lag'] = [p['T_la'], p['T_gu']]
        elif self.config['lagut'] is not None:
            p['T_lag'] = p['T_gu']

        max_it = 500

        p |= self._flux_injection(p)

        J_vc = p['Jinj']
        J_vc_total = J_vc

        dose = trapezoid(J_vc, dx=p['dt'])
        min_dose = p['dose_tolerance'] * dose

        it=0
        while True:
            J_vc = self._propagate_J_vena(p, J_vc)['J_vc']
            J_vc_total += J_vc

            dose = trapezoid(J_vc, dx=p['dt'])
            if dose <= min_dose:
                break

            it += 1
            if it > max_it:
                break

        p |= self._propagate_J_vena(p, J_vc_total)

        return self.map_results(p)

    # Helper function
    def _propagate_J_vena(self, p, Jv):
        # Store all results along the way so they can be returned
        result = {}

        Ja = self._flux_heartlung(p | {'J': Jv})['J']

        result['J_or'] = self._flux_organs(p | {'J': Ja})['J']
        Jv = p['vr_or'] * result['J_or']

        if self.config['renal_arteries'] is not None:
            result['J_lka'] = self._flux_ra_lk(p | {'J': Ja})['J']
            result['J_rka'] = self._flux_ra_rk(p | {'J': Ja})['J']
        else:
            result['J_lka'] = Ja
            result['J_rka'] = Ja

        if self.config['kidneys'] is not None:
            result['J_lk'] = self._flux_lk(p | {'J': result['J_lka']})['J']
            result['J_rk'] = self._flux_rk(p | {'J': result['J_rka']})['J']
            Jv += p['vr_lk'] * result['J_lk']
            Jv += p['vr_rk'] * result['J_rk']

        if self.config['lagut'] is not None:
            Jlag = self._flux_lagut(p | {'J': Ja})['J']
            if Jlag.ndim==2:
                result['J_lag'] = Jlag.sum(axis=0)
                result['J_la'] = Jlag[0]
                result['J_pv'] = Jlag[1]
            else:
                result['J_lag'] = Jlag
        else:
            result['J_lag'] = Ja

        if self.config['liver'] is not None:
            result['J_li'] = self._flux_liver(p | {'J': result['J_lag']})['J']
            Jv += p['vr_li'] * result['J_li']

        if self.config['veins'] is not None:
            Jv = self._flux_veins(p | {'J': Jv})['J']

        return result | {'J_ao': Ja, 'J_vc': Jv}

    def test_data(self):
        p = self.init_data()

        E_ki = 0.100
        E_or = 0.025
        fCO_ki = 0.25
        fCO_li = 0.30
        fCO_or = 0.45

        if self.config['liver'] is None:
            fCO_or += fCO_li
        else:
            p['vr_li'] = fCO_li

        if self.config['kidneys'] is None:
            fCO_or += fCO_ki
            p['vr_or'] = (1 - E_or) * fCO_or
        else:
            vr_ki = (1 - E_ki) * fCO_ki
            p['vr_lk'] = vr_ki / 2
            p['vr_rk'] = vr_ki / 2
            p['vr_or'] = fCO_or

        return self.input_data(p)