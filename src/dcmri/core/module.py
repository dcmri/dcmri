from copy import deepcopy

class Module:

    configs = {}
    defaults = {}

    def __init__(self, config: dict=None, imap: dict=None):
        self.set_config(config)
        self.map_inputs(imap)

    def set_config(self, config: dict=None):
        # Initialise configuration
        self._config = deepcopy(self.defaults)

        # Override with user-defined values
        if config is not None:
            for key, value in config.items():
                if key in self.configs:
                    if not isinstance(self.configs[key], set):
                        raise ValueError(f"The configuration options for {key} must be a set.")
                    if value not in self.configs[key]:
                        raise ValueError(f'{value} is not a valid value for {key}. Possible values are {self.configs[key]}.')  
                    self._config[key] = value
        return self

    def map_inputs(self, imap: dict=None):
        # Set inputs 
        inputs = self.inputs()
        if not isinstance(inputs, set):
            raise ValueError('The function inputs() must return a set. Inputs are unique.')
        self._inputs = inputs
        self._imap = {i: i for i in self._inputs}
        if imap is not None:
            for key, value in imap.items():
                if key in self._inputs:
                    self._imap[key] = value

        return self

    def mapped_inputs(self):
        # Get mapped inputs
        return {self._imap[i] for i in self._inputs}


    @property
    def config(self):
        return dict(self._config) # prevent accidental overwrite
    
    def map_data(self, p):
        for i in self._inputs:
            if self._imap[i] not in p:
                raise ValueError(f'The data dictionary p is missing the input key {self._imap[i]}.')
        return {i: p[self._imap[i]] for i in self._inputs}
    
    #
    # The following functions need to be reimplemented
    #

    def inputs(self) -> set: # depends on self._cnfg
        return set()
    
    def outputs(self) -> set: # depends on self._cnfg
        return set()
    
    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)
        return {}