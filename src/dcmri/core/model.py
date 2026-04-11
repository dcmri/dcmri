import os
from copy import deepcopy
from joblib import Parallel, delayed
from itertools import product

import zarr
import numpy as np

from dcmri.utils.fit import train, loss
from dcmri.lexicon.dicts import LEXICON
from dcmri.lexicon.tools import select_params, init

    

class Input:
    
    def __init__(
        self, 
        signal: np.ndarray=None, 
        time:np.ndarray=None,
        dt=1.0,
        R10=0.7,
        B1corr=1.0,
    ):
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        if time is None:
            time = dt * np.arange(signal.size)

        self.signal = signal
        self.time = time
        self.R10 = R10
        self.B1corr = B1corr


    

class SuperModel:

    # Must be reimplemented

    configs = {}
    
    def __init__(self, **params):
        self._version = '0'
        self._cnfg = {}
        self._pars = {}
        self._override_pars(**params)

    def _params(self, select=None) -> list:
        return []
    
    def _predict(self, time):
        return np.zeros_like(time)
    
    @property
    def _shape(self):
        return ()
    
    # Reusable functions

    def _set_config(self, **cnfg):
        for key, value in cnfg.items():
            if value is not None: # A value can be None for an optional configuration setting
                if value not in self.configs[key]:
                    raise ValueError(f'Config {value} is not recognized. Options are {list(self.configs[key])}.')  
        self._cnfg = cnfg   
        return self._cnfg
    
    def _set_pars(self, lexicon:dict=LEXICON, **params):
        self._pars = init(self._params(), lexicon=lexicon, **params)
        return self._pars
    
    def params(self, select=None) -> dict:
        return self._pars
    
    def _override_pars(self, **params):
        [self._pars.update({k:v}) for k, v in params.items() if k in self._pars]


    def save(self, folder: str):
        # Ensure directory mode
        if folder.endswith('.zip') or folder.endswith('.json'):
            folder = os.path.splitext(folder)[0]

        # mode='w' creates the directory store automatically
        root = zarr.open_group(folder, mode='w')
        
        array_keys = []
        metadata_pars = {}
        
        for k, v in self._pars.items():
            if isinstance(v, np.ndarray):
                # 1. Create the array
                # Note: 'chunks' must be a tuple, e.g., (10,) not just 10.
                z_arr = root.create_array(
                    name=k, 
                    shape=v.shape, 
                    dtype=v.dtype, 
                    chunks=v.shape, 
                    overwrite=True
                )
                
                # 2. Use the standard slice but ensure it's a full-volume write
                # If [:] fails, use .update(v) which is the V3-specific method
                if hasattr(z_arr, 'update'):
                    z_arr.update(v)
                else:
                    z_arr[...] = v  # '...' (Ellipsis) is often safer than ':' in V3
                    
                array_keys.append(k)
            else:
                metadata_pars[k] = v

        # Save the metadata into .attrs (this remains a JSON file)
        root.attrs.update({
            'model': self.__class__.__name__,
            'version': self._version,
            'config': self._cnfg,
            'pars_scalar': metadata_pars,
            'array_keys': array_keys
        })
        return self


    def load(self, folder: str):
        """Loads model state from a Zarr directory."""
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory {folder} not found.")

        root = zarr.open_group(folder, mode='r')
        meta = root.attrs.asdict()

        if meta['model'] != self.__class__.__name__:
            raise ValueError(f"Directory belongs to {meta['model']}.")
        
        self._pars = meta['pars_scalar']
        self._cnfg = meta['config']

        for key in meta['array_keys']:
            self._pars[key] = np.array(root[key])

        return self


    def _set_free_pars(self, free: dict=None, bounds: dict=None, lexicon:dict=None):
        if lexicon is None: lexicon=LEXICON

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
        pars = self.params('all')
        for p, bnds in free.items():
            if p not in pars:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
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



    def _train_batch_configurations(self, time, signal, free, configs, select, **kwargs):
        # Single pixel - parallellize over models
        if self._shape[0]==1:
            x = 0
            results = [self.__train_configurations(time, signal, free, configs, select, x, parallel=True, **kwargs)]

        # Multiple pixels - parallellize over pixels
        else:
            results = Parallel(n_jobs=-1)(
                delayed(self.__train_configurations)(
                    time, signal, free, configs, select, x, parallel=False,
                ) for x in range(self._shape[0])
            )
        return results
    
    
    def __train_configurations(self, time, signal, free, models, metric, x, parallel=True, **kwargs):
        def train_single_configuration(**cnfg):
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
            result = train(submodel._predict, time, signal[x,:,:], submodel._pars, free_submodel, x, **kwargs)
            
            # Compute cost
            s_pred = submodel._predict(time, x)
            cost = loss(s_pred, signal[x,:,:], metric, len(free_submodel))

            # print(cost, cnfg)
            return cnfg, result, cost
        
        configs = {k: v for k, v in self.configs.items() if k in models}
        configs = configs | {k: [v] for k, v in self._cnfg.items() if k not in models}

        if parallel:
            results = Parallel(n_jobs=-1)(
                delayed(train_single_configuration)(**dict(zip(configs.keys(), args))) 
                for args in product(*configs.values())
            )
        else:
            results = [
                train_single_configuration(**dict(zip(configs.keys(), args)))
                for args in product(*configs.values())
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
            self._pars[p] = v

        return result
    

def format_batch_training(results, free):
    # Format outputs
    vals = {p: [] for p in free}
    sdev = {p: [] for p in free}
    for p in free:
        for r in results:
            if p in r[0]:
                vals[p].append(r[0][p])
            else:
                vals[p].append(np.nan)
            if p in r[1]:
                sdev[p].append(r[1][p])
            else:
                sdev[p].append(np.nan)
        vals[p] = np.array(vals[p])
        sdev[p] = np.array(sdev[p])

    pcov = np.empty(len(results), dtype=object)
    pcov[:] = [r[2] for r in results]

    model = np.empty(len(results), dtype=object)
    if len(results[0]) == 4:
        model[:] = [r[3] for r in results]
    else:
        model[:] = None

    return vals, sdev, pcov, model