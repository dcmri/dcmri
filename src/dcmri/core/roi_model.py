import os
from copy import deepcopy

import zarr
import numpy as np

from dcmri.core.quantities import QUANTITIES
from dcmri.core.tools import select_params, export_params, print_params

class SuperRoiModel:

    configs = {}

    def __init__(self, **kwargs):
        cnfg = {}
        self._version = '0'
        self._set_config(cnfg)
        self._set_params({})

    def params(self, select=None) -> list:
        return []
    
    def _predict(self):
        return

    # Reusable functions

    def state(self, *args, select=None) -> dict: 
        """Model parameters and their values"""
        pars = self._pars
        if select is not None:
            pars = {k: v for k, v in pars.items() if k in self.params(select)}
        if args == ():
            return pars
        for k in args:
            if k not in pars:
                raise ValueError(f"{k} is not a valid model parameter. Use print_params() to get a list of valid parameters.")
        values = [pars[k] for k in args]
        if len(args) == 1:
            return values[0]
        else:
            return values

    def _set_config(self, cnfg:dict):
        for key, value in cnfg.items():
            if value is not None: # A value can be None for an optional configuration setting
                if value not in self.configs[key]:
                    raise ValueError(f'Config {value} is not recognized. Options are {list(self.configs[key])}.')  
        self._cnfg = cnfg  
        
    def _set_params(self, defaults={}):
        self._param_names = self.params()
        self._pars = {}
        for k in self._param_names:
            try:
                self._pars[k] = defaults[k]
            except:
                raise ValueError(f"A Default value for parameter {k} is not provided")

    # def _update_params(self, kwargs:dict):
    #     for k, v in kwargs.items():
    #         if k not in self._param_names:
    #             raise ValueError(
    #                 f"'{k}' is not a valid parameter for this configuration.\n"
    #                 f"Use print_params() to print a list of all valid parameters."
    #             )
    #         self._pars.update({k:v})
    #     return self._pars


    
    def _set_free_pars(self, free: dict=None, bounds: dict=None, lexicon:dict=None):
        if lexicon is None: lexicon=QUANTITIES

        # --- 0. Set Defaults ---
        if free is None:
            free = {p: lexicon[p]['bounds'] for p in self.params('free')}
        
        # --- 1. Update Bounds ---
        if bounds is not None:
            for p, b in bounds.items():
                if b is None:
                    free.pop(p, None)
                else:
                    free[p] = b

        # --- 2. Boundary Validation ---
        pars = self._pars
        for p, bnds in free.items():
            if p not in pars:
                raise ValueError(
                    f"'{p}' is not a valid parameter for this configuration.\n"
                    f"Use print_params() to print a list of all valid parameters."
                )
            elif p in select_params(lexicon, bounds_type='add'):
                if (bnds[0] > 0) or (bnds[1] < 0):
                    raise ValueError(f"Bounds on {p} must be (negative, positive).")
            elif p in select_params(lexicon, bounds_type='mult'): 
                if not (0 <= bnds[0] < bnds[1]):
                    raise ValueError(f"Invalid bounds on {p}: Bounds are relative and must be positive.")
            elif not (bnds[0] <= np.min(pars[p]) <= np.max(pars[p]) <= bnds[1]):
                raise ValueError(f"Initial {p} is out of bounds {bnds}.")

        # --- 3. Relative to Absolute Bounds
        for par in select_params(lexicon, bounds_type='add'):
            if par in free:
                free[par] = [  
                    np.min(pars[par]) + free[par][0],
                    np.max(pars[par]) + free[par][1],
                ]
        for par in select_params(lexicon, bounds_type='mult'):
            if par in free:
                free[par] = [
                    np.min(pars[par]) * free[par][0],
                    np.max(pars[par]) * free[par][1],
                ]

        return free
    

    # I/O

    def export_params(self, lexicon:dict=None, sdev=None, num_only=False):
        if lexicon is None:
            lexicon = QUANTITIES
        return export_params(self._pars, lexicon=lexicon, sdev=sdev, num_only=num_only)

    @classmethod
    def print_configs(cls):
        # pprint(cls.configs, width=2, sort_dicts=True)
        for key, list_of_strings in cls.configs.items():
            print(f"{key}:")
            for item in list_of_strings:
                print(f"  - {item}")
            print() 

    def print_params(self, *args, lexicon:dict=None, round_to=None, group=None, fixed_only=False, free_only=False):
        """Pretty print model parameters"""
        if args == ():
            pars = self._pars
        else:
            pars = {k: v for k, v in self._pars.items() if k in args}
        if fixed_only:
            pars = {k: v for k, v in pars.items() if k not in self.params('free')}
        if free_only:
            pars = {k: v for k, v in pars.items() if k in self.params('free')}
        print_params(pars, round_to=round_to, group=group, lexicon=lexicon)
    

    def save(self, folder: str):
        root = zarr.open_group(folder, mode='w')
        
        array_keys = []
        metadata_pars = {}
        
        for k, v in self._pars.items():
            if isinstance(v, np.ndarray):
                # Store large arrays in Zarr binary chunks
                z_arr = root.create_array(
                    name=k, 
                    shape=v.shape, 
                    dtype=v.dtype, 
                    chunks=v.shape, 
                    overwrite=True
                )
                z_arr[...] = v
                array_keys.append(k)
            else:
                # Collect scalars
                metadata_pars[k] = v

        # SANITIZE and SAVE
        # This ensures scalars like np.float64 become readable 1.23 instead of binary junk
        readable_meta = {
            'model': self.__class__.__name__,
            'version': self._version,
            'config': self._cnfg,
            'pars_scalar': metadata_pars,
            'array_keys': array_keys
        }
        root.attrs.update(readable_meta)
        return self
    
    def load(self, folder: str):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory {folder} not found.")

        root = zarr.open_group(folder, mode='r')
        meta = root.attrs.asdict()

        if meta['model'] != self.__class__.__name__:
            raise ValueError(f"Model mismatch: {meta['model']} vs {self.__class__.__name__}")
        
        # Load scalars back into _pars
        self._pars = meta['pars_scalar']
        self._cnfg = meta['config']

        # Reconstruct numpy arrays from binary stores
        for key in meta['array_keys']:
            self._pars[key] = np.array(root[key])

        return self