import os
import multiprocessing
import warnings
import pickle

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


try:
    num_workers = int(len(os.sched_getaffinity(0)))
except:
    num_workers = int(os.cpu_count())


class ArrayModel():
    # Abstract base class for end-to-end models with pixel-based analysis

    def __init__(self):
        self.shape = None
        self.free = {}

    def _par_values(self):
        return {par: getattr(self, par) for par in self.__dict__ 
                if par not in ['free', 'shape']}


    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get
            round_to (int, optional): Round to how many digits. If this is 
            not provided, the values are not rounded. Defaults to None.

        Returns:
            list or float: values of parameter values, or a scalar value if 
            only one parameter is required.
        """
        return params(self, *args, round_to=round_to)

    def set_free(self, pop=None, **kwargs):
        """Set the free model parameters.

        Args:
            pop (str or list): a single variable or a list of variables to 
              remove from the list of free parameters. 

        Raises:
            ValueError: if the pop argument contains a parameter that is not 
              in the list of free parameters.
            ValueError: If the parameter is not a model parameter, or bounds 
              are not properly formatted.
        """
        for p in kwargs:
            if not self._params(p)['pixel_par']:
                raise ValueError(
                    str(p) + ' is not a pixel-based parameter. ' +
                    'Only pixel-based parameters can be free.')
        set_free(self, pop=pop, **kwargs)

    def save(self, file=None, path=None, filename='Model'):
        """Save the current state of the model

        Args:
            file (str, optional): complete path of the file. If this is not 
              provided, a file is constructure from path amd filename 
              variables. Defaults to None.
            path (str, optional): path to store the state if file is not 
              provided. This variable is ignored if file is provided. 
              Defaults to current working directory.
            filename (str, optional): filename to store the state if file is 
              not provided. If no extension is included, the extension '.pkl' 
              is automatically added. This variable is ignored if file is 
              provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return _save(self, file, path, filename)

    def load(self, file=None, path=None, filename='Model'):
        """Load the saved state of the model

        Args:
            file (str, optional): complete path of the file. If this is not 
              provided, a file is constructure from path amd filename 
              variables. Defaults to None.
            path (str, optional): path to store the state if file is not 
              provided. This variable is ignored if file is provided. 
              Defaults to current working directory.
            filename (str, optional): filename to store the state if file is 
              not provided. If no extension is included, the extension 
              '.pkl' is automatically added. This variable is ignored if 
              file is provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return _load(self, file, path, filename)

    def _pix(self, p):
        raise NotImplementedError('No _pix() function defined')

    def _predict_curve(self, args):
        xdata, x = args
        p = np.unravel_index(x, self.shape)
        pix = self._pix(p)
        return pix.predict(xdata)

    def predict(self, xdata: np.ndarray) -> np.ndarray:
        """Predict the data for given x-values.

        Args:
            xdata (array-like): An array with x-values (time points).

        Returns:
            np.ndarray: Array of predicted y-values.
        """

        nx = np.prod(self.shape)
        nt = np.size(xdata)
        if not self.parallel:
            if self.verbose > 0:
                iterator = tqdm(
                    range(nx), 
                    desc='Running predictions for '+self.__class__.__name__)
            else:
                iterator = range(nx)
            ydata = [self._predict_curve((xdata, x)) for x in iterator]
        else:
            args = [(xdata, x) for x in range(nx)]
            pool = multiprocessing.Pool(processes=num_workers)
            ydata = pool.map(self._predict_curve, args)
            pool.close()
            pool.join()
        return np.stack(ydata).reshape(self.shape + (nt,))

    def _train_curve(self, args):
        xdata, ydata, kwargs, x = args
        p = np.unravel_index(x, self.shape)
        # if np.array_equal(p, [2,2]):
        #     print('')
        pix = self._pix(p)
        pix.train(xdata, ydata[x, :], **kwargs)
        if hasattr(pix, 'pcov'):
            sdev = np.sqrt(np.diag(pix.pcov))
        else:
            sdev = np.zeros(len(self.free.keys()))
        for i, par in enumerate(self.free.keys()):
            getattr(self, par)[p] = getattr(pix, par)
            getattr(self, 'sdev_' + par)[p] = sdev[i]
        return pix, p

    def train(self, xdata, ydata: np.ndarray, **kwargs):
        """Train the free parameters

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        nx = np.prod(self.shape)
        nt = ydata.shape[-1]
        if not self.parallel:
            if self.verbose==1:
                iterator = tqdm(range(nx), desc='Training ' +
                                self.__class__.__name__)
            else:
                iterator = range(nx)
            
            for x in iterator:
                args_x = (xdata, ydata.reshape((nx, nt)), kwargs, x)
                self._train_curve(args_x)
        else:
            args = [(xdata, ydata.reshape((nx, nt)), kwargs, x)
                    for x in range(nx)]
            pool = multiprocessing.Pool(processes=num_workers)
            pool.map(self._train_curve, args)
            pool.close()
            pool.join()
        return self

    def cost(self, xdata, ydata, metric='NRMS') -> float:
        """Goodness-of-fit value.

        Args:
            xdata (array-like): Array with x-data (time points).
            ydata (array-like): Array with y-data (signal values)
            metric (str, optional): Which metric to use - options are: 
                **RMS** (Root-mean-square);
                **NRMS** (Normalized root-mean-square); 
                **AIC** (Akaike information criterion); 
                **cAIC** (Corrected Akaike information criterion for small 
                  models);
                **BIC** (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            np.ndarray: goodness of fit in each element of the data array. 
        """
        return _cost(self, xdata, ydata, metric)

    def export_params(self) -> dict:
        """Model parameters with descriptions.

        Returns:
            dict: Dictionary with one item for each model parameter. The key 
              is the parameter symbol (short name), and the value is a 
              4-element list with [parameter name, value, unit, sdev].
        """
        pars = {}
        for p in self.free.keys():
            pars[p] = [
                p + ' name',
                getattr(self, p),
                p + ' unit',
                getattr(self, 'sdev_' + p),
            ]
        return pars

    def _add_sdev(self, pars):
        for par in pars:
            if par in self.free:
                sdev = getattr(self, 'sdev_' + par)
            else:
                sdev = None
            pars[par].append(sdev)
        return pars

    def _set_defaults(self, free=None, **params):
        for k, v in params.items():
            if hasattr(self, k):
                val = getattr(self, k)
                if isinstance(val, np.ndarray):
                    if isinstance(v, np.ndarray):
                        if v.shape != self.shape:
                            raise ValueError("""Parameter """ + str(k) + """
                                 does not have the correct shape. """)
                        else:
                            setattr(self, k, v)
                    else:
                        setattr(self, k, np.full(self.shape, v))
                elif val is None:
                    setattr(self, k, v)
                else:
                    if isinstance(v, np.ndarray):
                        msg = str(k) + " is not a pixel-based quantity. \n"
                        msg += "Please provide a scalar initial value instead."
                        raise ValueError(msg)
                    else:
                        setattr(self, k, v)
            else:
                raise ValueError(
                    str(k) + ' is not a valid parameter for this model.')
        if free is not None:
            self.set_free(**free)

# TODO: self.free - > self._free and include self.free()
class Model:
    # Abstract base class for end-to-end models.

    def __init__(self):
        self.free = {}
        self.pcov = None

    def _par_values(self, export=False):
        # Abstract method - needs to be overridden
        return {par: getattr(self, par) for par in self.__dict__ 
                if par not in ['free', 'pcov']}

    def _model_pars(self):
        # Abstract method - needs to be overridden
        return list(self._par_values().keys())
    
    def _params(self):
        # Abstract method - needs to be overridden
        return

    def _set_defaults(self, free=None, **params):

        # Model parameters
        model_pars = self._model_pars()
        pars = self._params()
        for p in model_pars:
            setattr(self, p, pars[p]['init'])
        free_pars = [p for p in model_pars if pars[p]['default_free']]
        self.free = {p: pars[p]['bounds'] for p in free_pars}

        # Override defaults
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                if k in self._params():
                    setattr(self, k, v)
                else:
                    raise ValueError(
                        str(k) + ' is not a valid parameter for this model.')
            
        # Set free
        if free is not None:
            self.free = {}
            self.set_free(**free)

    def export_params(self, type='dict') -> dict:
        """Return model parameters with their descriptions

        Args:
            type (str, optional): Type of output. If 'dict', a dictionary is 
              returned. If 'list', a list is returned. Defaults to 'dict'.

        Returns:
            dict: Dictionary with one item for each model parameter. The key 
            is the short parameter name, and the value is a 
            4-element list with [long parameter name, value, unit, sdev].

            or:

            list: List with one element for each model parameter. Each 
            element is a list with [short parameter name, 
            long parameter name, value, unit, sdev].
        """
        # Short name, full name, value, units.
        pars = self._par_values(export=True)
        params = self._params()
        pars = {p: [params[p]['name'], pars[p], params[p]['unit']]
                for p in pars}
        pars = self._add_sdev(pars)
        if type == 'dict':
            return pars
        elif type == 'list':
            return [[k] + v for k, v in pars.items()]
        else:
            raise ValueError('Type must be either "dict" or "list".')

    def save(self, file=None, path=None, filename='Model'):
        """Save the current state of the model

        Args:
            file (str, optional): complete path of the file. If this is not 
              provided, a file is constructure from path and filename 
              variables. Defaults to None.
            path (str, optional): path to store the state if file is not 
              provided. Thos variable is ignored if file is provided. 
              Defaults to current working directory.
            filename (str, optional): filename to store the state if file is 
              not provided. If no extension is included, the extension '.pkl' 
              is automatically added. This variable is ignored if file is 
              provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return _save(self, file, path, filename)

    def load(self, file=None, path=None, filename='Model'):
        """Load the saved state of the model

        Args:
            file (str, optional): complete path of the file. If this is not 
              provided, a file is constructure from path and filename 
              variables. Defaults to None.
            path (str, optional): path to store the state if file is not 
              provided. Thos variable is ignored if file is provided. 
              Defaults to current working directory.
            filename (str, optional): filename to store the state if file is 
              not provided. If no extension is included, the extension 
              '.pkl' is automatically added. This variable is ignored if file 
              is provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return _load(self, file, path, filename)

    def predict(self, xdata):
        """Predict the data at given xdata

        Args:
            xdata (array-like): Either an array with x-values (time points) 
              or a tuple with multiple such arrays

        Returns:
            tuple or array-like: Either an array of predicted y-values (if 
              xdata is an array) or a tuple of such arrays (if xdata is a 
              tuple).
        """
        raise NotImplementedError('No predict function provided')

    def train(self, xdata, ydata, **kwargs):
        """Train the free parameters

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        return train(self, xdata, ydata, **kwargs)

    def plot(self, xdata, ydata, xlim=None, ref=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form 
              (x,y), where x is an array with x-values and y is an array with 
              y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults 
              to True.
        """
        raise NotImplementedError(
            'No plot function implemented for model ' + self.__class__.__name__)

    def cost(self, xdata, ydata, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            xdata (array-like): Array with x-data (time points).
            ydata (array-like): Array with y-data (signal values)
            metric (str, optional): Which metric to use (see notes for 
              possible values). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.

        Notes:

            Available options are: 
            
            - 'RMS': Root-mean-square.
            - 'NRMS': Normalized root-mean-square. 
            - 'AIC': Akaike information criterion. 
            - 'cAIC': Corrected Akaike information criterion for small 
              models.
            - 'BIC': Baysian information criterion.

        """
        return _cost(self, xdata, ydata, metric)
    

    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is 
              not provided, the values are not rounded. Defaults to None.
        """
        pars = self.export_params()
        print('')
        print('--------------------------------')
        print('Free parameters with their stdev')
        print('--------------------------------')
        print('')
        for par in self.free:
            p = pars[par]
            if round_to is None:
                v = p[1]
                verr = p[3]
            else:
                v = round(p[1], round_to)
                verr = round(p[3], round_to)
            print(p[0] + ' ('+par+'): ' + str(v) +
                  ' (' + str(verr) + ') ' + p[2])
        print('')
        print('----------------------------')
        print('Fixed and derived parameters')
        print('----------------------------')
        print('')
        for par in pars:
            if par not in self.free:
                p = pars[par]
                if round_to is None:
                    v = p[1]
                else:
                    v = np.round(p[1], round_to)
                print(p[0] + ' ('+par+'): ' + str(v) + ' ' + p[2])

    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get
            round_to (int, optional): Round to how many digits. If this is 
            not provided, the values are not rounded. Defaults to None.

        Returns:
            list or float: values of parameter values, or a scalar value if 
            only one parameter is required.
        """
        return params(self, *args, round_to=round_to)


    def set_free(self, pop=None, **kwargs):
        """Set the free model parameters.

        Args:
            pop (str or list): a single variable or a list of variables to 
              remove from the list of free parameters. 

        Raises:
            ValueError: if the pop argument contains a parameter that is not 
              in the list of free parameters.
            ValueError: If the parameter is not a model parameter, or bounds 
              are not properly formatted.
        """
        set_free(self, pop=pop, **kwargs)

    def _getflat(self, attr: np.ndarray = None) -> np.ndarray:
        if attr is None:
            attr = self.free.keys()
        vals = []
        for a in attr:
            v = getattr(self, a)
            vals = np.append(vals, np.ravel(v))
        return vals

    def _setflat(self, vals: np.ndarray, pcov: np.ndarray = None, attr: np.ndarray = None):
        if attr is None:
            attr = self.free.keys()
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
        else:
            perr = np.zeros(np.size(vals))
        i = 0
        for p in attr:
            v = getattr(self, p)
            if np.isscalar(v):
                v = vals[i]
                e = perr[i]
                i += 1
            else: # Needs a better solution
                n = np.size(v)
                v = np.reshape(vals[i:i+n], np.shape(v))
                e = np.reshape(perr[i:i+n], np.shape(v))
                i += n
            setattr(self, p, v)
            setattr(self, p + '_sdev', e)


    def _add_sdev(self, pars):
        for par in pars:
            pars[par].append(0)
        # if not hasattr(self, 'pcov'):
        #     perr = np.zeros(len(self.free.keys()))
        # elif self.pcov is None:
        #     perr = np.zeros(len(self.free.keys()))
        # else:
        #     perr = np.sqrt(np.diag(self.pcov))
        # for i, par in enumerate(self.free.keys()):
        #     if par in pars:
        #         pars[par][-1] = perr[i]
        for par in self.free.keys():
            if par in pars:
                if hasattr(self, par + '_sdev'):
                    pars[par][-1] = getattr(self, par + '_sdev')
        return pars
    
    # def _x_scale(self):
    #     n = len(self.free)
    #     xscale = np.ones(n)
    #     for p in range(n):
    #         if np.isscalar(self.free[0]):
    #             lb = self.free[0]
    #         else:
    #             lb = self.free[0][p]
    #         if np.isscalar(self.free[1]):
    #             ub = self.free[1]
    #         else:
    #             ub = self.free[1][p]
    #         if (not np.isinf(lb)) and (not np.isinf(ub)):
    #             xscale[p] = ub-lb
    #     return xscale
    

def params(self, *args, round_to=None):
    p = self._par_values()
    if args == ():
        args = p.keys()
    pars = []
    for a in args:
        if a in p:
            v = p[a]
        else:
            if hasattr(self, a):
                v = getattr(self, a)
            else:
                raise ValueError(
                    a + ' is not a model parameter, and cannot be '
                    + 'derived from the model parameters.'
                )
        if round_to is not None:
            v = round(v, round_to)
        pars.append(v)
    if len(pars) == 1:
        return pars[0]
    else:
        return pars
    

def set_free(self, pop=None, **kwargs):
    if pop is not None:
        if np.isscalar(pop):
            if pop in self.free:
                self.free.pop(pop)
            else:
                raise ValueError(
                    pop + ' is not currently a free parameter, so cannot be '
                    'removed from the list.')
        else:
            for par in pop:
                if par in self.free:
                    self.free.pop(par)
                else:
                    raise ValueError(
                        par + ' is not currently a free parameter, so cannot '
                        'be removed from the list.')
    for k, v in kwargs.items():
        if k in self.__dict__:
            if np.size(v) == 2:
                if v[0] < v[1]:
                    if (v[0] <= getattr(self, k)) and (v[1] >= getattr(self, k)):
                        self.free[k] = v
                    else:
                        raise ValueError(
                            'Cannot set parameter bounds for ' + str(k) + '. '
                            'The value current value'
                            ' ' + str(getattr(self, k)) + ' is outside of '
                            'the bounds. ')
                else:
                    raise ValueError(
                        str(v) + ' is not a proper parameter ''bound. The '
                        'first element must be smaller than the second.')
            else:
                raise ValueError(
                    str(v) + ' is not a proper parameter bound. Bounds must be '
                    'lists or arrays with 2 elements.')
        else:
            raise ValueError(str(k) + ' is not a model parameter.')


def _save(model, file=None, path=None, filename='Model'):
    if file is None:
        if path is None:
            path = os.getcwd()
        if filename.split('.')[-1] != 'pkl':
            filename += '.pkl'
        file = os.path.join(path, filename)
    elif file.split('.')[-1] != 'pkl':
        file += '.pkl'
    with open(file, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close
    return model


def _load(model, file=None, path=None, filename='Model'):
    if file is None:
        if path is None:
            path = os.getcwd()
        if filename.split('.')[-1] != 'pkl':
            filename += '.pkl'
        file = os.path.join(path, filename)
    elif file.split('.')[-1] != 'pkl':
        file += '.pkl'
    with open(file, 'rb') as f:
        saved = pickle.load(f)
    model.__dict__.update(saved.__dict__)
    f.close
    return model


def train(model: Model, xdata, ydata, **kwargs):

    if isinstance(ydata, tuple):
        y = np.concatenate(ydata)
    else:
        y = ydata

    def fit_func(_, *p):
        model._setflat(p)
        yp = model.predict(xdata)
        if isinstance(yp, tuple):
            return np.concatenate(yp)
        else:
            return yp

    p0 = model._getflat()
    bounds = [
        [par[0] for par in model.free.values()],
        [par[1] for par in model.free.values()],
    ]
    try:
        pars, model.pcov = curve_fit(
            fit_func, None, y, p0,
            bounds=bounds,  # x_scale=model._x_scale(),
            **kwargs)
    except RuntimeError as e:
        msg = 'Runtime error in curve_fit -- \n'
        msg += str(e) + ' Returning initial values.'
        warnings.warn(msg)
        pars = p0
        model.pcov = np.zeros((np.size(p0), np.size(p0)))

    # Note pcov does not have to be an attribuite
    model._setflat(pars, pcov=model.pcov)

    return model


def _cost(model, xdata, ydata, metric='NRMS') -> float:

    # Predict data at all xvalues
    y = model.predict(xdata)
    if isinstance(ydata, tuple):
        y = np.concatenate(y)
        ydata = np.concatenate(ydata)

    # Calclulate the loss at the fitted values
    if metric == 'RMS':
        loss = np.linalg.norm(y - ydata, axis=-1)
    elif metric == 'NRMS':
        ynorm = np.linalg.norm(ydata, axis=-1)
        yerr = np.linalg.norm(y - ydata, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            loss = 100*yerr/ynorm
    elif metric == 'AIC':
        rss = np.sum((y-ydata)**2, axis=-1)
        n = ydata.shape[-1]
        k = len(model.free.keys())
        with np.errstate(divide='ignore'):
            loss = k*2 + n*np.log(rss/n)
    elif metric == 'cAIC':
        rss = np.sum((y-ydata)**2)
        n = ydata.shape[-1]
        k = len(model.free.keys())
        with np.errstate(divide='ignore'):
            loss = k*2 + n*np.log(rss/n) + 2*k*(k+1)/(n-k-1)
    elif metric == 'BIC':
        rss = np.sum((y-ydata)**2, axis=-1)
        n = ydata.shape[-1]
        k = len(model.free.keys())
        with np.errstate(divide='ignore'):
            loss = k*np.log(n) + n*np.log(rss/n)
    return loss
