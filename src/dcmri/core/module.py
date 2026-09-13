from copy import deepcopy
import math
import itertools
import textwrap

from tqdm import tqdm
import numpy as np

from dcmri.core.exceptions import InvalidConfiguration
from dcmri.core.tools import get_quantity, print_quantities



class Module:

    configs = {}
    defaults = {}
    quantities = {}

    # Optional attributes
    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap: dict=None, omap: dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        self.map_io(imap, omap, iomap)

    def set_config(self, config: dict=None, cmap: dict=None):
        if cmap is not None:
            cmap = {v: k for k, v in cmap.items()}
        # Initialise configuration
        self._config = deepcopy(self.defaults)

        # Override with user-defined values
        if config is not None:
            for key, value in config.items():
                if cmap is not None:
                    if key in cmap:
                        key = cmap[key]
                if key in self.configs:
                    if value not in self.configs[key]:
                        raise ValueError(f'{value} is not a valid value for {key} in {self.__class__.__name__}. Possible values are {self.configs[key]}.')  
                    self._config[key] = value
        return self

    def map_io(self, imap: dict=None, omap: dict=None, iomap: dict=None):
        if iomap is not None:
            self.map_inputs(iomap)
            self.map_outputs(iomap)         
        else:  
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

    def new_mapped_outputs(self):
        return self.mapped_outputs() - self.mapped_inputs()

    def map_data(self, data: dict | None, override={}) -> dict:
        p = {}
        imap = self._imap
        for i in self._inputs:
            if i in imap:
                j = imap[i]
            else:
                j = i
            if j in override:
                p[i] = override[j]
            # elif data is None:
            #     raise ValueError(f'{self.__class__.__name__} needs a value for input {j}.')
            # elif not isinstance(data, dict):
            #     raise ValueError(f"The default data argument must be a dictionary.")
            elif data is not None and j in data:
                p[i] = data[j]
            else:
                try:
                    p[i] = get_quantity(i, quantities=self.quantities)['init']
                except:
                    raise ValueError(f'Unknown input {i} in call to Module {self.__class__.__name__}.')

            # elif all:
            #     raise ValueError(f'{self.__class__.__name__} needs a value for input {j}.')
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

    def init_data(self):
        data = {}
        for i in self.inputs():
            if i in self._imap:
                j = self._imap[i]
            else:
                j = i
            data[j] = get_quantity(i, quantities=self.quantities)['init']
        return data

    def dummy_data(self): # reimplement if not all inputs are scalar
        return self.init_data()

    def print_inputs(self):
        q = self.input_quantities()
        title = f"{self.__class__.__name__} instance - inputs (n = {len(q)})"
        print_quantities(title, q)

    def print_outputs(self):
        q = self.output_quantities()
        title = f"{self.__class__.__name__} instance - outputs (n = {len(q)})"
        print_quantities(title, q)

    def input_quantities(self):
        iq = {}
        for k in self.inputs():
            iq[k] = get_quantity(k, quantities=self.quantities)
            # if k in self.quantities:
            #     iq[k] = self.quantities[k] # uneccessary now
            # else:
            #     iq[k] = get_quantity(k, quantities=self.quantities)
        return iq

    def output_quantities(self):
        oq = {}
        for k in self.outputs():
            oq[k] = get_quantity(k, quantities=self.quantities)
            # if k in self.quantities:
            #     oq[k] = self.quantities[k] # uneccessary now
            # else:
            #     oq[k] = get_quantity(k, quantities=self.quantities)
        return oq

    @classmethod
    def map_configs(cls, cmap: dict):
        configs = deepcopy(cls.configs)
        return {cmap.get(k, k): v for k, v in configs.items()}

    @classmethod
    def configs_with_input(cls, input):
        configs = set()
        for cnfg in tqdm(list(cls.all_configs()), desc='Collecting configs..'):
            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            if input in model.inputs():
                configs |= {cnfg}
        return configs   

    @classmethod
    def configs_with_output(cls, output):
        configs = set()
        for cnfg in tqdm(list(cls.all_configs()), desc='Collecting configs..'):
            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            if output in model.outputs():
                configs |= {cnfg}
        return configs      

    @classmethod
    def print_configs(cls):
        title = f"{cls.__name__} - all configs (n = {len(cls.configs)})"
        print_configs(title, cls.configs, cls.defaults)

    @classmethod
    def print_all_inputs(cls, verbose=0, simple=False):
        if simple:
            print('inputs\n')
            i = cls.all_inputs(verbose)
            print(i)
        else:
            q = cls.all_input_quantities(verbose)
            title = f"{cls.__name__} - all inputs (n = {len(q)})"
            print_quantities(title, q)

    @classmethod
    def print_all_outputs(cls, verbose=0, simple=False):
        if simple:
            print('outputs\n')
            o = cls.all_outputs(verbose)
            print(o)
        else:
            q = cls.all_output_quantities(verbose)
            title = f"{cls.__name__} - all outputs (n = {len(q)})"
            print_quantities(title, q)

    @classmethod
    def print_all_io(cls, verbose=0, simple=False, sample: int = None, seed: int = None):
        if simple:
            i, o = cls.all_io(verbose, sample=sample, seed=seed)
            print('inputs\n')
            print(i)
            print('outputs\n')
            print(o)
        else:
            iq, oq = cls.all_io_quantities(verbose, sample=sample, seed=seed)
            title = f"{cls.__name__} - all inputs (n = {len(iq)})"
            print_quantities(title, iq)
            title = f"{cls.__name__} - all outputs (n = {len(oq)})"
            print_quantities(title, oq)

    @classmethod
    def all_input_quantities(cls, verbose=0):
        iq = {}
        for k in cls.all_inputs(verbose):
            iq[k] = get_quantity(k, quantities=cls.quantities)
        return iq

    @classmethod
    def all_output_quantities(cls, verbose=0):
        oq = {}
        for k in cls.all_outputs(verbose):
            oq[k] = get_quantity(k, quantities=cls.quantities)
        return oq

    @classmethod
    def all_io_quantities(cls, verbose=0, sample: int = None, seed: int = None):
        inputs, outputs = cls.all_io(verbose, sample=sample, seed=seed)
        iq = {}
        for k in inputs:
            iq[k] = get_quantity(k, quantities=cls.quantities)
        oq = {}
        for k in outputs:
            oq[k] = get_quantity(k, quantities=cls.quantities)
        return iq, oq


    @classmethod
    def all_configs(cls, sample: int = None, seed: int = None, valid=False):
        keys = list(cls.configs.keys())
        value_lists = [list(cls.configs[key]) for key in keys]  # sets -> lists, indexable

        def is_valid(cnfg):
            try:
                cls(**cnfg)
            except InvalidConfiguration:
                return False
            return True

        if sample is None:
            # Full enumeration is unavoidable here - we need every (valid) config.
            # Build lazily so we don't hold an unfiltered copy in memory before filtering.
            total = math.prod(len(v) for v in value_lists)
            combos = itertools.product(*value_lists)
            if valid:
                combos = tqdm(combos, total=total, desc='Checking configurations..')
                result = []
                for c in combos:
                    cnfg = dict(zip(keys, c))
                    if is_valid(cnfg):
                        result.append(cnfg)
                return result
            return [dict(zip(keys, c)) for c in combos]

        # Sample requested: draw random combinations directly, never touching the full product.
        rng = np.random.default_rng(seed=seed)

        seen = set()
        results = []
        pbar = tqdm(total=sample, desc='Sampling configurations..') if valid else None

        while len(results) < sample:
            combination = tuple(vals[rng.integers(len(vals))] for vals in value_lists)
            if combination in seen:
                continue  # avoid duplicate configs, same as original replace=False
            seen.add(combination)

            cnfg = dict(zip(keys, combination))
            if valid and not is_valid(cnfg):
                continue

            results.append(cnfg)
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # rng.shuffle(results)
        return results

    @classmethod
    def all_inputs(cls, verbose=0):
        if cls._all_inputs is not None:
            return cls._all_inputs
        configs = cls.all_configs()
        if verbose==0:
            iterator = configs
        else:
            iterator = tqdm(list(configs), desc='Collecting inputs..')

        inputs = set()
        for cnfg in iterator:
            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            else:
                inputs |= model.inputs()
        return inputs 

    @classmethod
    def all_outputs(cls, verbose=0):
        if cls._all_outputs is not None:
            return cls._all_outputs
        configs = cls.all_configs()
        if verbose==0:
            iterator = configs
        else:
            iterator = tqdm(list(configs), desc='Collecting outputs..')

        outputs = set()
        for cnfg in iterator:
            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            else:
                outputs |= model.outputs()
        return outputs 

    @classmethod
    def all_io(cls, verbose=0, sample: int = None, seed: int = None, split=True):
        if cls._all_inputs and cls._all_outputs:
            if not split:
                return cls._all_inputs | cls._all_outputs
            return cls._all_inputs, cls._all_outputs

        keys = list(cls.configs.keys())
        value_lists = [list(cls.configs[key]) for key in keys]

        inputs = set()
        outputs = set()

        if sample is None:
            total = math.prod(len(v) for v in value_lists)

            combos = itertools.product(*value_lists)
            if verbose:
                combos = tqdm(combos, total=total, desc='Collecting in- and outputs..')

            cnt = 0
            for combination in combos:
                cnfg = dict(zip(keys, combination))
                try:
                    model = cls(**cnfg)
                except InvalidConfiguration:
                    continue
                else:
                    cnt += 1
                    inputs |= model.inputs()
                    outputs |= model.outputs()

            if verbose:
                print(f'{cnt} configurations found!')

            if not split:
                return inputs | outputs
            return inputs, outputs

        # Sample requested: draw random combinations directly, never touching the full product.
        rng = np.random.default_rng(seed=seed)
        total = math.prod(len(v) for v in value_lists)

        seen = set()
        results = []
        pbar = tqdm(total=sample, desc='Sampling configurations..') if verbose else None

        cnt = 0
        while len(results) < sample:
            if len(seen) >= total:
                if verbose:
                    print(f'Exhausted search space after {len(results)} combinations '
                          f'({cnt} valid) - fewer than requested sample={sample}.')
                break

            combination = tuple(vals[rng.integers(len(vals))] for vals in value_lists)
            if combination in seen:
                continue  # avoid duplicate configs, same as original replace=False
            seen.add(combination)

            cnfg = dict(zip(keys, combination))
            results.append(cnfg)
            if pbar:
                pbar.update(1)

            try:
                model = cls(**cnfg)
            except InvalidConfiguration:
                continue
            else:
                cnt += 1
                inputs |= model.inputs()
                outputs |= model.outputs()

        if verbose:
            print(f'{cnt} configurations found!')

        if pbar:
            pbar.close()

        if not split:
            return inputs | outputs
        return inputs, outputs

    # @classmethod
    # def all_io(cls, verbose=0, sample: int = None, seed: int = None):
    #     if cls._all_inputs and cls._all_outputs:
    #         return cls._all_inputs, cls._all_outputs

    #     keys = list(cls.configs.keys())
    #     value_lists = [list(cls.configs[key]) for key in keys]

    #     inputs = set()
    #     outputs = set()

    #     if sample is None:
    #         total = math.prod(len(v) for v in value_lists)

    #         combos = itertools.product(*value_lists)
    #         if verbose:
    #             combos = tqdm(combos, total=total, desc='Collecting in- and outputs..')

    #         cnt = 0
    #         for combination in combos:
    #             cnfg = dict(zip(keys, combination))
    #             try:
    #                 model = cls(**cnfg)
    #             except InvalidConfiguration:
    #                 continue
    #             else:
    #                 cnt += 1
    #                 inputs |= model.inputs()
    #                 outputs |= model.outputs()

    #         if verbose:
    #             print(f'{cnt} configurations found!')

    #         return inputs, outputs

    #     # Sample requested: draw random combinations directly, never touching the full product.
    #     rng = np.random.default_rng(seed=seed)

    #     seen = set()
    #     results = []
    #     pbar = tqdm(total=sample, desc='Sampling configurations..')

    #     cnt = 0
    #     while len(results) < sample:
    #         combination = tuple(vals[rng.integers(len(vals))] for vals in value_lists)
    #         if combination in seen:
    #             continue  # avoid duplicate configs, same as original replace=False
    #         seen.add(combination)

    #         cnfg = dict(zip(keys, combination))
    #         results.append(cnfg)

    #         try:
    #             model = cls(**cnfg)
    #         except InvalidConfiguration as e:
    #             continue
    #         else:
    #             cnt += 1
    #             inputs |= model.inputs()
    #             outputs |= model.outputs()

    #         if pbar:
    #                 pbar.update(1)
    #     if verbose:
    #         print(f'{cnt} valid configurations found!')

    #     if pbar:
    #         pbar.close()

    #     return inputs, outputs
        
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


# Helper function
import textwrap

def print_configs(title, configs, defaults, width=100):

    def format_values(v):
        return ", ".join(str(x) for x in sorted(v, key=str))

    key_w = max((len(str(k)) for k in configs), default=3)
    key_w = max(key_w, len("Key"))

    default_w = max((len(str(defaults[k])) for k in configs), default=7)
    default_w = max(default_w, len("Default"))

    if width is not None:
        # fixed total width -> derive value column width from it
        table_width = width
        val_w = table_width - key_w - default_w - 10
        val_w = max(val_w, len("Values"), 10)  # keep a sane minimum
        table_width = key_w + val_w + default_w + 10  # recompute in case val_w was clamped
    else:
        # auto width, as before
        val_w = max((len(format_values(v)) for v in configs.values()), default=6)
        val_w = max(val_w, len("Values"))
        table_width = key_w + val_w + default_w + 10

    def row(key, values, default):
        return f"| {key:<{key_w}} | {values:<{val_w}} | {default:<{default_w}} |"

    def spanning_row(text):
        inner_width = table_width - 4  # account for "| " and " |"
        return f"| {text.center(inner_width)} |"

    divider = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+" + "-" * (default_w + 2) + "+"
    outer_border = "+" + "-" * (table_width - 2) + "+"

    lines = [outer_border]
    lines.append(spanning_row(title))
    lines.append(divider)
    lines.append(row("Key", "Values", "Default"))
    lines.append(divider)

    for k, v in configs.items():
        text = format_values(v)
        wrapped = textwrap.wrap(text, width=val_w, break_long_words=False, break_on_hyphens=False)
        wrapped = wrapped or [""]  # keep empty-set rows visible

        lines.append(row(str(k), wrapped[0], str(defaults[k])))
        for cont in wrapped[1:]:
            lines.append(row("", cont, ""))  # continuation rows: blank key/default columns

    lines.append(outer_border)
    lines.append("")

    print("\n".join(lines))
