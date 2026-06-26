class Function:

    # --------------
    # User interface
    # --------------

    configs = {}

    def __init__(self, defaults=None, **kwargs):
        cnfg = {}
        self._set_config(cnfg)
        self._set_params(defaults)

    def params(self) -> list:
        return []

    def __call__(self, *args, **kwargs):
        p = self._update_params(kwargs)
        return None
    
    def connect_inputs(self, input_map: dict):
        # input_map = {'from keys': 'to keys'}
        self._input_map = input_map
        return self
    
    # -------
    # Backend
    # -------

    def _set_config(self, cnfg:dict):
        for key, value in cnfg.items():
            if value not in self.configs[key]:
                raise ValueError(f'Configuration value {value} is not recognized. Options are {self.configs[key]}.')  
        self._cnfg = cnfg  
        self._input_map: dict = None
        
    def _set_params(self, defaults:dict=None):
        self._param_names = self.params()
        if defaults is None:
            self._defaults = {}
            self._pars = {}
        else:
            self._defaults = defaults
            self._pars = {k: defaults[k] for k in self._param_names if k in defaults}

    def _update_params(self, kwargs:dict):
        
        if self._input_map is not None:
            # If a input map is provided, use it to map provided arguments to internal arguments

            # Swap the keys based on the mapping
            # If a key isn't in the mapping, it keeps its original name
            kwargs = {self._input_map.get(old_key, old_key): value for old_key, value in kwargs.items()}

        for k in self._param_names:
            if k in kwargs:
                self._pars[k] = kwargs[k]
            elif k in self._defaults:
                self._pars[k] = self._defaults[k]
            else:
                raise ValueError(f"Required parameter {k} is not provided")

        return self._pars