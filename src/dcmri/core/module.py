from copy import deepcopy
import itertools

class Module:

    configs = {}
    defaults = {}

    def __init__(self, imap: dict=None, omap: dict=None, **config):
        self.set_config(config)
        self.map_io(imap, omap)

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

    def map_io(self, imap: dict=None, omap: dict=None):
        self.map_inputs(imap)
        self.map_outputs(omap)
        return self

    def map_inputs(self, imap: dict=None):
        inputs = self.inputs()
        self._inputs = inputs
        self._imap = {i: i for i in self._inputs}
        if imap is not None:
            for key, value in imap.items():
                if key in self._inputs:
                    self._imap[key] = value
        return self

    def map_outputs(self, omap: dict=None):
        outputs = self.outputs()
        self._outputs = outputs
        self._omap = {o: o for o in self._outputs}
        if omap is not None:
            for key, value in omap.items():
                if key in self._outputs:
                    self._omap[key] = value
        return self

    @property
    def config(self):
        return dict(self._config) # prevent accidental overwrite

    def mapped_inputs(self):
        return {self._imap[i] for i in self._inputs}

    def mapped_outputs(self):
        return {self._omap[o] for o in self._outputs}
    
    def map_data(self, data: dict | None, override={}) -> dict:
        p = {}
        imap = self._imap
        for i in self._inputs:
            j = imap[i]
            if j in override:
                p[i] = override[j]
            elif data is None:
                raise ValueError(f'{self.__class__.__name__} needs a value for input {j}.')
            elif not isinstance(data, dict):
                raise ValueError(f"The default data argument must be a dictionary.")
            elif j in data:
                p[i] = data[j]
            else:
                raise ValueError(f'{self.__class__.__name__} needs a value for input {j}.')
        return p

    def map_results(self, results: dict) -> dict:
        p = {}
        omap = self._omap
        for k, v in results.items():
            if k in omap:
                p[omap[k]] = v
            else:
                p[k] = v
        return p 

    def input_data(self, data) -> dict:
        p = {}
        for i in self._inputs:
            j = self._imap[i]
            if j in data:
                p[j] = data[j]
        return p
    
    def update_data(self, data: dict, p: dict):
        return data | {self._imap[k]: v for k, v in p.items() if k in self._imap}
    
    @classmethod
    def configurations(cls):
        # Separate keys and their corresponding list of values to maintain order
        keys = list(cls.configs.keys())
        value_lists = [cls.configs[key] for key in keys]
        
        # itertools.product generates all combinations of the values
        for combination in itertools.product(*value_lists):
            # Pair each key with the current combination's value
            yield dict(zip(keys, combination))
        
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
        # return_value = {}
        raise NotImplementedError('No __call__ method defined')