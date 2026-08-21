from copy import deepcopy
import itertools

from dcmri.core.exceptions import InvalidConfiguration

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
        self._inputs = self.inputs()
        if imap is None:
            self._imap = {}
        else:
            self._imap = {k: v for k, v in imap.items() if k in self._inputs}
        return self
        # self._imap = {i: i for i in self._inputs}
        # if imap is not None:
        #     for key, value in imap.items():
        #         if key in self._inputs:
        #             self._imap[key] = value
        # return self

    def map_outputs(self, omap: dict=None):
        self._outputs = self.outputs()
        if omap is None:
            self._omap = {}
        else:
            self._omap = {k: v for k, v in omap.items() if k in self._outputs}
        return self
        # self._outputs = self.outputs()
        # self._omap = {o: o for o in self._outputs}
        # if omap is not None:
        #     for key, value in omap.items():
        #         if key in self._outputs:
        #             self._omap[key] = value
        # return self

    @property
    def config(self):
        return dict(self._config) # prevent accidental overwrite

    def mapped_inputs(self):
        result = set()
        for i in self._inputs:
            if i in self._imap:
                result.add(self._imap[i])
            else:
                result.add(i)
        return result
        # return {self._imap[i] for i in self._inputs}

    def mapped_outputs(self):
        result = set()
        for o in self._outputs:
            if o in self._omap:
                result.add(self._omap[o])
            else:
                result.add(o)
        return result
        # return {self._omap[o] for o in self._outputs}

    def map_data(self, data: dict | None, override={}, all=True) -> dict:
        p = {}
        imap = self._imap
        for i in self._inputs:
            if i in imap:
                j = imap[i]
            else:
                j = i
            if j in override:
                p[i] = override[j]
            elif data is None:
                raise ValueError(f'{self.__class__.__name__} needs a value for input {j}.')
            elif not isinstance(data, dict):
                raise ValueError(f"The default data argument must be a dictionary.")
            elif j in data:
                p[i] = data[j]
            elif all:
                raise ValueError(f'{self.__class__.__name__} needs a value for input {j}.')
        return p

    def map_results(self, data: dict) -> dict:
        results = {}
        for o in self._outputs:
            if o in self._omap:
                j = self._omap[o]
            else:
                j = o
            if o in data:
                results[j] = data[o]
        return results
        # return {self._omap[o]: results[o] for o in self._outputs}

    def input_data(self, data) -> dict:
        p = {}
        for i in self._inputs:
            if i in self._imap:
                j = self._imap[i]
            else:
                j = i
            if j in data:
                p[j] = data[j]
        return p
    
    def update_data(self, p: dict):
        results = {}
        for i in self._inputs:
            if i in self._imap:
                j = self._imap[i]
            else:
                j = i
            if i in p:
                results[j] = p[i]
        return results

    def lexicon_data(self, qvalues):
        return {}
    
    @classmethod
    def configurations(cls):
        # Separate keys and their corresponding list of values to maintain order
        keys = list(cls.configs.keys())
        value_lists = [cls.configs[key] for key in keys]
        
        # itertools.product generates all combinations of the values
        for combination in itertools.product(*value_lists):
            # Pair each key with the current combination's value
            yield dict(zip(keys, combination))

    @classmethod
    def all_inputs(cls):
        inputs = set()
        for cnfg in cls.configurations():
            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            else:
                inputs |= model.inputs()
        return inputs 

    @classmethod
    def all_outputs(cls):
        outputs = set()
        for cnfg in cls.configurations():
            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            else:
                outputs |= model.outputs()
        return outputs 

    @classmethod
    def all_io(cls):
        io = set()
        for cnfg in cls.configurations():
            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            else:
                io |= model.inputs()
                io |= model.outputs()
        return io
        
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