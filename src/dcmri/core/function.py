
class Function:

    # Functions that offer multiple configuration options 
    # where the parametrization depends on the configuration

    # --------------
    # User interface
    # --------------

    configs = {}
    
    def __init__(self, **defaults):
        cnfg = {}
        self._set_config(**cnfg)
        self._set_params(**defaults)

    def __call__(self, *args, **kwargs):
        p = self._update_params(**kwargs)
        return None
    
    @property
    def params(self) -> dict:
        return self._pars

    # -------
    # Backend
    # -------


    # Reimplement
    # -----------

    def _param_names(self) -> list:
        return []
    
    # Do not reimplement
    # ------------------

    def _set_config(self, **cnfg):
        for key, value in cnfg.items():
            if value is not None: # A value can be None for an optional configuration setting
                if value not in self.configs[key]:
                    raise ValueError(f'Config {value} is not recognized. Options are {list(self.configs[key])}.')  
        self._cnfg = cnfg  
    
    def _set_params(self, **kwargs):
        self._pars = {}
        for k in self._param_names():
            self._pars[k] = kwargs[k] if k in kwargs else None

    def _update_params(self, **kwargs) -> dict:
        for p in kwargs:
            if p not in self._pars:
                raise ValueError(f"{p} is not a valid keyword argument. The options are {self._pars.keys()}")
        p = {**self._pars, **kwargs}       
        return p

