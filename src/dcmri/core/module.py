from copy import deepcopy
import math
import itertools
import textwrap

from tqdm import tqdm
import numpy as np

from dcmri.core.quantity import print_quantities, Qty

class InvalidConfig(Exception):
    """Exception raised when module or model is created with an invalid configuration."""
    pass

class Module:
    """Configurable function with named inputs and outputs.

    A Module is the basic building block of the high-level functionality in dcmri. 
    It is configured once at construction, after which it behaves like a function that takes
    a dictionary of named inputs and returns a dictionary of named outputs.
    Modules can be connected by mapping the outputs of one onto the inputs
    of another, which allows more complex modules to be assembled from
    simpler ones.

    Subclasses define the class attributes ``configs``, ``defaults`` and
    ``quantities``, and implement the methods ``inputs``, ``outputs`` and
    ``_call``. The set of inputs and outputs may depend on the
    configuration.

    Names used inside the module are *internal* names. The optional
    mappings ``imap``, ``omap``, ``iomap`` and ``cmap`` translate them to
    *external* names, which are the names used in the data dictionaries
    that are passed to and returned from the module.

    Example:
        >>> model = MyModule(imap={'x': 'signal'}, mode='fast')
        >>> results = model({'signal': 1.0})
    """
    #: Valid configuration options. Maps each configuration key to a
    #: collection of allowed values.
    configs = {}

    #: Default value for each configuration key.
    defaults = {}

    #: Definitions of the quantities that appear as inputs and outputs,
    #: only needed for quantities that are not in the lexicon.
    quantities = {}

    # Optional attributes
    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap: dict=None, omap: dict=None, iomap: dict=None, cmap: dict=None, **config):
        """Create a module with a given configuration and input/output mapping.

        Args:
            imap (dict, optional): Maps internal input names to external
                names. Inputs that are not listed keep their own name.
            omap (dict, optional): Maps internal output names to external
                names. Outputs that are not listed keep their own name.
            iomap (dict, optional): Mapping applied to both inputs and
                outputs. If provided, ``imap`` and ``omap`` are ignored.
            cmap (dict, optional): Maps internal configuration keys to
                external keys, so that the module can be configured with
                different keyword names.
            **config: Configuration values. Keys must appear in
                ``configs`` and values must be among the allowed values;
                keys that are not in ``configs`` are ignored. Anything not
                specified takes its value from ``defaults``.

        Raises:
            InvalidConfig: If a configuration value is not allowed.
        """
        self.set_config(config, cmap)
        self.map_io(imap, omap, iomap)

    def set_config(self, config: dict=None, cmap: dict=None):
        """Set the configuration, starting from the defaults.

        Any previous configuration is discarded. Keys in ``config`` that
        are not in ``configs`` are silently ignored.

        Args:
            config (dict, optional): Configuration values to override
                the defaults.
            cmap (dict, optional): Maps internal configuration keys to
                external keys. It is inverted here, so that external keys
                in ``config`` are translated back to internal ones.

        Returns:
            Module: The module itself, to allow chaining.

        Raises:
            InvalidConfig: If a value is not allowed for its key.
        """
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
                        raise InvalidConfig(f'{value} is not a valid value for {key} in {self.__class__.__name__}. Possible values are {self.configs[key]}.')  
                    self._config[key] = value
        return self

    def map_io(self, imap: dict=None, omap: dict=None, iomap: dict=None):
        """Set the mapping of internal to external input and output names.

        Args:
            imap (dict, optional): Mapping for the inputs.
            omap (dict, optional): Mapping for the outputs.
            iomap (dict, optional): Mapping applied to both inputs and
                outputs. If provided, ``imap`` and ``omap`` are ignored.

        Returns:
            Module: The module itself, to allow chaining.
        """
        if iomap is not None:
            self.map_inputs(iomap)
            self.map_outputs(iomap)         
        else:  
            self.map_inputs(imap)
            self.map_outputs(omap)
        return self

    def __call__(self, data: dict=None, **kwargs) -> dict:
        """Run the module.

        The external data are translated to internal input names, missing
        inputs are filled with their default values, ``_call`` is
        executed, and the results are translated back to external names.

        Args:
            data (dict, optional): Input data, keyed by external name.
            **kwargs: Passed on to ``_call``.

        Returns:
            dict: The outputs, keyed by external name.

        Raises:
            ValueError: If an input is missing from ``data`` and has no
                default value.
        """
        p = self.map_data(data)  
        p = self._call(p, **kwargs)
        return self.map_results(p)

    def map_inputs(self, imap: dict=None):
        """Set the input mapping.

        Reads the current inputs from ``inputs()``, so this needs to be
        called again if the configuration changes. Entries in ``imap``
        that are not inputs of this module are dropped.

        Args:
            imap (dict, optional): Maps internal input names to external
                names.

        Returns:
            Module: The module itself, to allow chaining.
        """
        self._inputs = self.inputs()
        if imap is None:
            self._imap = {}
        else:
            self._imap = {k: v for k, v in imap.items() if k in self._inputs}
        return self

    def map_outputs(self, omap: dict=None):
        """Set the output mapping.

        Reads the current outputs from ``outputs()``, so this needs to be
        called again if the configuration changes. Entries in ``omap``
        that are not outputs of this module are dropped.

        Args:
            omap (dict, optional): Maps internal output names to external
                names.

        Returns:
            Module: The module itself, to allow chaining.
        """
        self._outputs = self.outputs()
        if omap is None:
            self._omap = {}
        else:
            self._omap = {k: v for k, v in omap.items() if k in self._outputs}
        return self

    def init(self):
        """Get the default value of every input.

        Returns:
            dict: Default values, keyed by internal input name.
        """
        return {
            i: Qty(i, quantities=self.quantities).value
            for i in self._inputs
        }        

    @property
    def config(self):
        """dict: A copy of the current configuration.

        Modifying the returned dictionary does not change the module.
        """
        return dict(self._config) # prevent accidental overwrite

    def mapped_inputs(self):
        """Get the input names as seen from outside the module.

        Returns:
            set: External input names.
        """
        result = set()
        for i in self._inputs:
            if i in self._imap:
                result.add(self._imap[i])
            else:
                result.add(i)
        return result

    def mapped_outputs(self):
        """Get the output names as seen from outside the module.

        Returns:
            set: External output names.
        """
        result = set()
        for o in self._outputs:
            if o in self._omap:
                result.add(self._omap[o])
            else:
                result.add(o)
        return result

    def new_mapped_outputs(self):
        """Get the external outputs that are not also external inputs.

        Useful when connecting modules, to see which quantities this
        module adds to the data.

        Returns:
            set: External names of outputs that are not inputs.
        """
        return self.mapped_outputs() - self.mapped_inputs()

    def map_data(self, data: dict | None, all=True, include:set=None) -> dict:
        """Translate external data to a dictionary with internal input names.

        Args:
            data (dict or None): Data keyed by external name.
            all (bool, optional): If True, inputs missing from ``data``
                are filled with their default values. If False, they are
                left out. Defaults to True.
            include (set, optional): Additional internal names to treat
                as inputs.

        Returns:
            dict: Input values keyed by internal name.

        Raises:
            ValueError: If an input is missing from ``data`` and has no
                default value.
        """
        p = {}
        imap = self._imap
        if include is None:
            inputs = self._inputs
        else:
            inputs = self._inputs | include
        for i in inputs:
            if i in imap:
                j = imap[i]
            else:
                j = i
            if data is not None and j in data:
                p[i] = data[j]
            elif all:
                try:
                    p[i] = Qty(i, quantities=self.quantities).value
                except:
                    raise ValueError(f'Unknown input {i} in call to Module {self.__class__.__name__}.')
        return p

    def map_results(self, data: dict) -> dict:
        """Translate results to a dictionary with external output names.

        Args:
            data (dict): Results keyed by internal output name, as
                returned by ``_call``.

        Returns:
            dict: The outputs of this module keyed by external name.
            Outputs that are missing from ``data`` are left out.
        """
        results = {}
        for o in self._outputs:
            if o in self._omap:
                j = self._omap[o]
            else:
                j = o
            if o in data:
                results[j] = data[o]
        return results

    def filter_data(self, data:dict) -> dict:
        """Keep only the entries of a dictionary that are inputs.

        Note that the keys are compared to the *internal* input names, the
        input mapping is not applied.

        Args:
            data (dict): Data to filter.

        Returns:
            dict: The entries of ``data`` whose key is an input.
        """
        return {k: v for k, v in data.items() if k in self._inputs}        
        
    def input_data(self, init, data: dict=None) -> dict:
        """Build a dictionary of input values keyed by external name.

        For each input, the value is taken from ``data`` if present, and
        from ``init`` otherwise. Inputs that are in neither are left out.

        Args:
            init (dict): Fallback values keyed by internal input name,
                for example the output of :meth:`init`.
            data (dict, optional): Data keyed by external name.

        Returns:
            dict: Input values keyed by external name.
        """
        p = {}
        for i in self._inputs:
            if i in self._imap:
                j = self._imap[i]
            else:
                j = i
            if data is not None and j in data:
                p[j] = data[j]
            elif i in init:
                p[j] = init[i]
        return p
    
    def update_data(self, p: dict):
        """Translate internal values back to external names.

        Only entries that are inputs of this module are returned.

        Args:
            p (dict): Values keyed by internal name.

        Returns:
            dict: The input entries of ``p`` keyed by external name.
        """
        results = {}
        for i in self._inputs:
            if i in self._imap:
                j = self._imap[i]
            else:
                j = i
            if i in p:
                results[j] = p[i]
        return results


    def test_data(self, data: dict=None): 
        """Get a complete set of input data for testing.

        Default values are used for every input not given in ``data``.

        Args:
            data (dict, optional): Values keyed by external name that
                override the defaults.

        Returns:
            dict: Input values keyed by external name.
        """
        p = self.init()
        return self.input_data(p, data)

    def print_inputs(self, as_dict=False):
        """Print a table of the inputs of this instance.

        Args:
            as_dict (bool, optional): Print as a dictionary instead of
                a table. Defaults to False.
        """
        q = self.input_quantities()
        title = f"{self.__class__.__name__} instance - inputs (n = {len(q)})"
        print_quantities(q, title, as_dict=as_dict)

    def print_outputs(self, as_dict=False):
        """Print a table of the outputs of this instance.

        Args:
            as_dict (bool, optional): Print as a dictionary instead of
                a table. Defaults to False.
        """
        q = self.output_quantities()
        title = f"{self.__class__.__name__} instance - outputs (n = {len(q)})"
        print_quantities(q, title, as_dict=as_dict)

    def input_quantities(self):
        """Get the quantity definitions of the inputs of this instance.

        Returns:
            dict: Maps each input name to its quantity properties.
        """
        return {k: Qty(k, quantities=self.quantities)._dct for k in self.inputs()}
        
    def output_quantities(self):
        """Get the quantity definitions of the outputs of this instance.

        Returns:
            dict: Maps each output name to its quantity properties.
        """
        return {k: Qty(k, quantities=self.quantities)._dct for k in self.outputs()}

    @classmethod
    def map_configs(cls, cmap: dict):
        """Get the valid configurations with keys renamed.

        Args:
            cmap (dict): Maps internal configuration keys to new keys.
                Keys not in ``cmap`` are unchanged.

        Returns:
            dict: A copy of ``configs`` with renamed keys.
        """
        configs = deepcopy(cls.configs)
        return {cmap.get(k, k): v for k, v in configs.items()}     

    @classmethod
    def print_configs(cls):
        """Print the configuration options with their defaults."""
        title = f"{cls.__name__} - all configs (n = {len(cls.configs)})"
        print_configs(title, cls.configs, cls.defaults)

    @classmethod
    def print_all_inputs(cls, verbose=0, simple=False):
        """Print every input that occurs in any valid configuration.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.
                Defaults to 0.
            simple (bool, optional): Print a plain set of names instead
                of a table with quantity properties. Defaults to False.
        """
        if simple:
            print('inputs\n')
            i = cls.all_inputs(verbose)
            print(i)
        else:
            q = cls._all_input_quantities(verbose)
            title = f"{cls.__name__} - all inputs (n = {len(q)})"
            print_quantities(q, title)

    @classmethod
    def print_all_outputs(cls, verbose=0, simple=False):
        """Print every output that occurs in any valid configuration.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.
                Defaults to 0.
            simple (bool, optional): Print a plain set of names instead
                of a table with quantity properties. Defaults to False.
        """
        if simple:
            print('outputs\n')
            o = cls.all_outputs(verbose)
            print(o)
        else:
            q = cls._all_output_quantities(verbose)
            title = f"{cls.__name__} - all outputs (n = {len(q)})"
            print_quantities(q, title)

    @classmethod
    def print_all_io(cls, verbose=0, simple=False, sample: int = None, seed: int = None):
        """Print every input and output that occurs in any valid configuration.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.
                Defaults to 0.
            simple (bool, optional): Print plain sets of names instead of
                tables with quantity properties. Defaults to False.
            sample (int, optional): Only inspect this many randomly drawn
                configurations instead of all of them.
            seed (int, optional): Seed for the random sampling.
        """
        if simple:
            i, o = cls.all_io(verbose, sample=sample, seed=seed)
            print('inputs\n')
            print(i)
            print('outputs\n')
            print(o)
        else:
            iq, oq = cls._all_io_quantities(verbose, sample=sample, seed=seed)
            title = f"{cls.__name__} - all inputs (n = {len(iq)})"
            print_quantities(iq, title)
            title = f"{cls.__name__} - all outputs (n = {len(oq)})"
            print_quantities(oq, title)

    @classmethod
    def _all_input_quantities(cls, verbose=0):
        """Get the quantity definitions of all possible inputs.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.

        Returns:
            dict: Maps each input name to its quantity properties.
        """
        iq = {}
        for k in cls.all_inputs(verbose):
            iq[k] = Qty(k, quantities=cls.quantities)._dct
        return iq

    @classmethod
    def _all_output_quantities(cls, verbose=0):
        """Get the quantity definitions of all possible outputs.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.

        Returns:
            dict: Maps each output name to its quantity properties.
        """
        oq = {}
        for k in cls.all_outputs(verbose):
            oq[k] = Qty(k, quantities=cls.quantities)._dct
        return oq

    @classmethod
    def _all_io_quantities(cls, verbose=0, sample: int = None, seed: int = None):
        """Get the quantity definitions of all possible inputs and outputs.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.
            sample (int, optional): Only inspect this many randomly drawn
                configurations instead of all of them.
            seed (int, optional): Seed for the random sampling.

        Returns:
            tuple: A tuple (iq, oq) of dictionaries that map input and
            output names, respectively, to their quantity properties.
        """
        inputs, outputs = cls.all_io(verbose, sample=sample, seed=seed)
        iq = {}
        for k in inputs:
            iq[k] = Qty(k, quantities=cls.quantities)._dct
        oq = {}
        for k in outputs:
            oq[k] = Qty(k, quantities=cls.quantities)._dct
        return iq, oq

    @classmethod
    def all_configs(cls, sample: int = None, seed: int = None, valid=False):
        """List the possible configurations of this class.

        Without ``sample``, this is the full product of all values in
        ``configs``, which can be very large. With ``sample``, random
        configurations are drawn without enumerating the product.

        Args:
            sample (int, optional): Number of random configurations to
                return. If None, all are returned.
            seed (int, optional): Seed for the random sampling.
            valid (bool, optional): If True, only return configurations
                that the class accepts, i.e. that do not raise
                ``InvalidConfig``. This requires instantiating each one.
                Defaults to False.

        Returns:
            list: Configurations as dictionaries of key-value pairs. With
            ``sample``, fewer are returned if the search space is
            exhausted first.
        """
        keys = list(cls.configs.keys())
        value_lists = [list(cls.configs[key]) for key in keys]  # sets -> lists, indexable

        def is_valid(cnfg):
            try:
                cls(**cnfg)
            except InvalidConfig:
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

        total = math.prod(len(vals) for vals in value_lists)

        while len(results) < sample and len(seen) < total:
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
        """Get every input that occurs in any valid configuration.

        If the class attribute ``_all_inputs`` is set, it is returned
        directly, which skips the (slow) enumeration of configurations.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.
                Defaults to 0.

        Returns:
            set: Names of all possible inputs.
        """
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
            except InvalidConfig:
                continue
            else:
                inputs |= model.inputs()
        return inputs 

    @classmethod
    def all_outputs(cls, verbose=0):
        """Get every output that occurs in any valid configuration.

        If the class attribute ``_all_outputs`` is set, it is returned
        directly, which skips the (slow) enumeration of configurations.

        Args:
            verbose (int, optional): If non-zero, show a progress bar.
                Defaults to 0.

        Returns:
            set: Names of all possible outputs.
        """

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
            except InvalidConfig:
                continue
            else:
                outputs |= model.outputs()
        return outputs 

    @classmethod
    def all_io(cls, verbose=0, sample: int = None, seed: int = None, split=True):
        """Get every input and output that occurs in any valid configuration.

        This is faster than calling :meth:`all_inputs` and
        :meth:`all_outputs` separately, as each configuration is built
        only once. If both ``_all_inputs`` and ``_all_outputs`` are set on
        the class, they are returned directly.

        Args:
            verbose (int, optional): If non-zero, show a progress bar and
                report the number of valid configurations found.
            sample (int, optional): Only inspect this many randomly drawn
                configurations (valid or not) instead of all of them.
                The result may then be incomplete.
            seed (int, optional): Seed for the random sampling.
            split (bool, optional): If True, return inputs and outputs
                separately. If False, return their union. Defaults to
                True.

        Returns:
            tuple or set: A tuple (inputs, outputs) of sets of names if
            ``split`` is True, otherwise a single set with both.
        """
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
                except InvalidConfig:
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
            except InvalidConfig:
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
        
    #
    # The following functions need to be reimplemented
    #

    def _call(self, *args) -> set: 
        """Compute the outputs from the inputs.

        Must be implemented by subclasses.

        Args:
            *args: Receives a dictionary of input values keyed by
                internal name, plus any extra arguments given to
                ``__call__``.

        Returns:
            dict: Output values keyed by internal name.
        """
        raise NotImplementedError('No _call method defined')

    def inputs(self) -> set: # depends on self._cnfg
        """Get the names of the inputs for the current configuration.

        Must be implemented by subclasses.

        Returns:
            set: Internal input names.
        """
        raise NotImplementedError('No inputs method defined')
    
    def outputs(self) -> set: # depends on self._cnfg
        """Get the names of the outputs for the current configuration.

        Must be implemented by subclasses.

        Returns:
            set: Internal output names.
        """
        raise NotImplementedError('No outputs method defined')
    



# Helper function


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
