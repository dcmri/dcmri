from copy import deepcopy

class Module:

    configs = {}
    defaults = {}

    def __init__(self, imap: dict=None, **config):
        self.set_config(config)
        self.map_inputs(imap)

    def set_config(self, config: dict=None):
        # Initialise configuration
        self._config = deepcopy(self.defaults)

        # Override with user-defined values
        if config is not None:
            for key, value in config.items():
                if key in self.configs:
                    if value not in self.configs[key]:
                        raise ValueError(f'{value} is not a valid value for {key}. Possible values are {self.configs[key]}.')  
                    self._config[key] = value
        return self

    def map_inputs(self, imap: dict=None):
        # Set inputs 
        inputs = self.inputs()
        # if not isinstance(inputs, set):
        #     raise ValueError('The function inputs() must return a set. Inputs are unique.')
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
    
    def map_data(self, data: dict | None, override={}) -> dict:
        p = {}
        imap = self._imap
        for i in self._inputs:
            j = imap[i]
            if j in override:
                p[i] = override[j]
            elif data is None:
                raise ValueError(f'The data for input key {j} are not provided.')
            elif not isinstance(data, dict):
                raise ValueError(f"The default data argument must be a dictionary.")
            elif j in data:
                p[i] = data[j]
            else:
                raise ValueError(f'The data dictionary is missing the input key {j}.')
        return p
    
    def map_lexicon(self, qvalues):
        return self.map_data(qvalues)
        
    #
    # The following functions need to be reimplemented
    #

    def inputs(self) -> set: # depends on self._cnfg
        raise NotImplementedError('No inputs method defined')
    
    def outputs(self) -> set: # depends on self._cnfg
        raise NotImplementedError('No outputs method defined')
    
    def __call__(self, *args, data: dict=None, **kwargs) -> dict:
        # Any reimplementation needs to start with mapping the data:
        # p = self.map_data(data, kwargs)
        return_value = {}
        raise NotImplementedError('No __call__ method defined')