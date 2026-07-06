import os
from pprint import pprint
from copy import deepcopy
from joblib import Parallel, delayed
from itertools import product

import zarr
import numpy as np

from dcmri.utils.fit import train, loss
from dcmri.core.quantities import QUANTITIES
from dcmri.core.tools import select_params, init, export_params, print_params
    

class SuperPixelModel:

    # Must be reimplemented

    configs = {}

    @classmethod
    def print_configs(cls):
        # pprint(cls.configs, width=2, sort_dicts=True)
        for key, list_of_strings in cls.configs.items():
            print(f"{key}:")
            for item in list_of_strings:
                print(f"  - {item}")
            print() 
    
    def __init__(self, **params):
        self._version = '0'
        self._cnfg = {}
        self._pars = {}
        self._override_pars(**params)

    def print_params(self, *args, lexicon:dict=None, round_to=None, group=None, fixed_only=False, free_only=False):
        """Pretty print model parameters"""
        if args == ():
            pars = self._pars
        else:
            pars = {k: v for k, v in self._pars.items() if k in args}
        if fixed_only:
            pars = {k: v for k, v in pars.items() if k not in self._params('free')}
        if free_only:
            pars = {k: v for k, v in pars.items() if k in self._params('free')}
        print_params(pars, round_to=round_to, group=group, lexicon=lexicon)

    def _params(self, select=None) -> list:
        return []
    
    def _predict(self):
        return
    
    @property
    def _shape(self):
        return ()
    
    # Reusable functions

    def _set_config(self, **cnfg):
        # Check if user-defined configurations are allowed
        for key, value in cnfg.items():
            if value is not None: # A value can be None for an optional configuration setting
                if key in self.configs: # Not alll configurations are set by the user - some are fixed or derived
                    if value not in self.configs[key]:
                        raise ValueError(f'Config {value} is not recognized. Options are {list(self.configs[key])}.')  
        self._cnfg = cnfg   
        return self._cnfg
    
    def _set_pars(self, lexicon:dict=QUANTITIES, **params):
        pars_list = self._params()
        for p in params:
            if p not in pars_list:
                raise ValueError(
                    f"'{p}' is not a valid parameter for this configuration.\n"
                    f"Use print_params() to print a list of all valid parameters."
                )
        self._pars = init(pars_list, lexicon=lexicon, **params)
        return self._pars
    
    def params(self, select=None) -> dict:
        return self._pars
    
    def _override_pars(self, **params):
        for k, v in params.items():
            if k not in self._params():
                raise ValueError(
                    f"'{k}' is not a valid parameter for this configuration.\n"
                    f"Use print_params() to print a list of all valid parameters."
                )
            self._pars.update({k:v})
            
    def export_params(self, lexicon:dict=None, sdev=None, num_only=False):
        if lexicon is None:
            lexicon = QUANTITIES
        return export_params(self._pars, lexicon=lexicon, sdev=sdev, num_only=num_only)
    
    def save(self, folder: str):

        # def _sanitize_for_json(obj):
        #     """Recursively convert numpy types to native python types for JSON."""
        #     if isinstance(obj, dict):
        #         return {k: _sanitize_for_json(v) for k, v in obj.items()}
        #     elif isinstance(obj, (list, tuple)):
        #         return [_sanitize_for_json(x) for x in obj]
        #     return obj
    
        # if folder.endswith('.zip') or folder.endswith('.json'):
        #     folder = os.path.splitext(folder)[0]

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
                # if hasattr(z_arr, 'update'):
                #     z_arr.update(v)
                # else:
                #     z_arr[...] = v
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

    def _set_free_pars(self, free: dict=None, bounds: dict=None, lexicon:dict=None):
        if lexicon is None: lexicon=QUANTITIES

        # --- 0. Set Defaults ---
        if free is None:
            free = {p: deepcopy(lexicon[p]['bounds']) for p in self._params('free')}
        
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
    
    def _pixel_pars(self, x):
        p = self._pars
        pixel_pars = self._params('pixel')
        pars_x = {k: v[x] for k, v in p.items() if k in pixel_pars}
        pars_x |= {k: v for k, v in p.items() if k not in pixel_pars}
        return pars_x
    
    def _run_parallel(self, pixel_func, *args, **kwargs) -> np.ndarray: # (n_pixels, ) + other dimensions
        # pixel_func must have signature pixel_func(a, b, x, c=1, d=2)
        # i.e. x must be the last of the arguments just before the keyword arguments
        nx = self._shape[0]
        if nx==1:
            # The overhead of parallellization is not worth it for 1-pixel functions
            results = [pixel_func(*(args + (0,)), **kwargs)]
        else:
            results = Parallel(n_jobs=-1)(delayed(pixel_func)(*(args + (x,)), **kwargs) for x in range(nx))
        return results

    def _train_batch_configurations(self, time, signal, free, configs, select, **kwargs):
        # Single pixel - parallellize over models
        if self._shape[0]==1:
            x = 0
            results = [self._train_batch_configurations_pixel(time, signal, free, configs, select, x, parallel=True, **kwargs)]

        # Multiple pixels - parallellize over pixels
        else:
            # results = [self._train_batch_configurations_pixel(
            #     time, signal, free, configs, select, x, parallel=False,
            #     ) for x in range(self._shape[0])
            # ]
            results = Parallel(n_jobs=-1)(
                delayed(self._train_batch_configurations_pixel)(
                    time, signal, free, configs, select, x, parallel=False,
                ) for x in range(self._shape[0])
            )
        return results
    
    
    def _train_batch_configurations_pixel(self, time, signal, free, models, metric, x, parallel=True, **kwargs):
        configs = {k: v for k, v in self.configs.items() if k in models}
        configs = configs | {k: [v] for k, v in self._cnfg.items() if k not in models}

        if parallel:
            results = Parallel(n_jobs=-1)(
                delayed(self._train_single_configuration_pixel)(
                    dict(zip(configs.keys(), args)), time, signal, free, metric, x, **kwargs
                ) for args in product(*configs.values())
            )
        else:
            results = [
                self._train_single_configuration_pixel(
                    dict(zip(configs.keys(), args)), time, signal, free, metric, x, **kwargs
                ) for args in product(*configs.values())
            ]

        # Rebuild dictionaries
        valid_results = [r for r in results if r is not None]
        key = [tuple([v for k, v in r[0].items() if k in models]) for r in valid_results]
        cost_dict = {key[i]: r[2] for i, r in enumerate(valid_results)}
        result_dict = {key[i]: r[1] for i, r in enumerate(valid_results)}

        # Find the best model
        best_model = min(cost_dict, key=cost_dict.get)
        result = result_dict[best_model] + (best_model,)

        # Update state with optimized values
        for p, v in result[0].items():
            self._pars[p][x] = v

        return result
    
    def _train_single_configuration_pixel(self, cnfg, time, signal, free, metric, x, **kwargs):
        submodel = self.__class__(**cnfg)

        # Check if the submodel is nested
        pars_topmodel = self._params()
        pars_submodel = submodel._params()
        if not set(pars_submodel).issubset(pars_topmodel):
            return None
        
        # Identify the free parameters of the submodel
        free_submodel = {k: v for k, v in free.items() if k in pars_submodel}
        if free_submodel == {}:
            return None
        
        # Initialize the submodel to match the top model
        for p in submodel._pars:
            submodel._pars[p] = deepcopy(self._pars[p])
        
        # Train single pixel to submodel
        result = train(submodel._predict, time, signal[x,...], submodel._pars, free_submodel, x, **kwargs)
        
        # Compute cost
        s_pred = submodel._predict(time, x)
        cost = loss(s_pred, signal[x,...], metric, len(free_submodel))

        # print(cost, cnfg)
        return cnfg, result, cost