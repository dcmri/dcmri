from dcmri.lexicon import QUANTITIES, init

class LayerFunction:

    # These need to be reimplemented

    configs = {}
    
    def __init__(self, **params):
        self._cnfg = {}
        self._pars = {}
        self._override_pars(**params)

    def _params(self) -> list:
        return []
    
    def __call__(self, *args, **params):
        p = self._update_pars(**params)
        return None

    # Reusable functions

    def _set_config(self, **cnfg):
        for key, value in cnfg.items():
            if value is not None: # A value can be None for an optional configuration setting
                if value not in self.configs[key]:
                    raise ValueError(f'Config {value} is not recognized. Options are {list(self.configs[key])}.')  
        self._cnfg = cnfg   
        return self._cnfg
    
    def _set_pars(self, lexicon:dict=QUANTITIES, **params):
        self._pars = init(self._params(), lexicon=lexicon, **params)
        return self._pars
    
    def params(self) -> dict:
        return self._pars
    
    def _override_pars(self, **params):
        [self._pars.update({k:v}) for k, v in params.items() if k in self._pars]

    def _update_pars(self, **params) -> dict:
        if params == {}:
            p = self._pars
        else:
            p = {k:v for k, v in self._pars.items() if k not in params}
            p = p | {k:v for k, v in params.items() if k in self._pars}
        return p