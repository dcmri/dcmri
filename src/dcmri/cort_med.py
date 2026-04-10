import copy
import numpy as np
import dcmri.pk as pk
from dcmri.ui import SuperFunc



class Conc(SuperFunc):

    _params_dict = {
        '7C': ['T_a', 'Fp', 'Eg', 'fc', 'Tglom', 'Tv', 'Tpt', 'Tlh', 'Tdt', 'Tcd'],
    }
    configs = {
        'kinetics': ['7C'],
    }
    def __init__(self, kinetics='7C', **params):
        cnfg = {'kinetics': kinetics}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        model = self._cnfg['kinetics']
        return copy.deepcopy(self._params_dict[model])
    
    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        p = self._update_pars(**params)
        kin = self._cnfg['kinetics']

        ca = pk.flux(ca, p['T_a'], dt=dt, model='plug')
        p = {k: v for k, v in p.items() if k != 'T_a'}

        if kin == '7C':
            return _conc_kidney_cm9(ca, t=t, dt=dt, **p)



def _conc_kidney_cm9(ca, t=None, dt=1.0, Fp=None, Eg=None, fc=None, Tglom=None, Tv=None, Tpt=None, Tlh=None, Tdt=None, Tcd=None):

    # Flux out of the glomeruli and arterial tree
    Jg = pk.flux(Fp*ca, Tglom, t=t, dt=dt, model='comp')

    # Flux out of the peritubular capillaries and venous system
    Jv = pk.flux((1-Eg)*Jg, Tv, t=t, dt=dt, model='comp')

    # Flux out of the proximal tubuli
    Jpt = pk.flux(Eg*Jg, Tpt, t=t, dt=dt, model='comp')

    # Flux out of the lis of Henle
    Jlh = pk.flux(Jpt, Tlh, t=t, dt=dt, model='comp')

    # Flux out of the distal tubuli
    Jdt = pk.flux(Jlh, Tdt, t=t, dt=dt, model='comp')

    # Flux out of the collecting ducts
    Jcd = pk.flux(Jdt, Tcd, t=t, dt=dt, model='comp')

    # Build cortical concentrations
    Cg = Tglom*Jg      # arteries/glomeruli
    Cv = fc*Tv*Jv   # part of the peritubular capillaries
    Cpt = Tpt*Jpt   # proximal tubuli
    Cdt = Tdt*Jdt   # distal tubuli
    Ccor = np.stack((Cg, Cv, Cpt, Cdt))

    # Build medullary concentrations
    Cv = (1-fc)*Tv*Jv   # part of the peritubular capillaries
    Clh = Tlh*Jlh       # Lis of Henle
    Ccd = Tcd*Jcd       # collecting ducts
    Cmed = np.stack((Cv, Clh, Ccd))

    return Ccor, Cmed
