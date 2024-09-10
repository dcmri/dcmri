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

    def save(self, file=None, path=None, filename='Model'):
        """Save the current state of the model

        Args:
            file (str, optional): complete path of the file. If this is not provided, a file is constructure from path amd filename variables. Defaults to None.
            path (str, optional): path to store the state if file is not provided. This variable is ignored if file is provided. Defaults to current working directory.
            filename (str, optional): filename to store the state if file is not provided. If no extension is included, the extension '.pkl' is automatically added. This variable is ignored if file is provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return _save(self, file, path, filename)

    def load(self, file=None, path=None, filename='Model'):
        """Load the saved state of the model

        Args:
            file (str, optional): complete path of the file. If this is not provided, a file is constructure from path amd filename variables. Defaults to None.
            path (str, optional): path to store the state if file is not provided. This variable is ignored if file is provided. Defaults to current working directory.
            filename (str, optional): filename to store the state if file is not provided. If no extension is included, the extension '.pkl' is automatically added. This variable is ignored if file is provided. Defaults to 'Model'.

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
    
    def predict(self, xdata:np.ndarray)->np.ndarray:
        """Predict the data for given x-values.

        Args:
            xdata (array-like): An array with x-values (time points).

        Returns:
            np.ndarray: Array of predicted y-values.
        """
       
        nx = np.prod(self.shape)
        nt = np.size(xdata)
        if not self.parallel:
            if self.verbose>0:
                iterator = tqdm(range(nx), desc='Running predictions for '+self.__class__.__name__)
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
        pix.train(xdata, ydata[x,:], **kwargs)
        if hasattr(pix, 'pcov'):
            sdev = np.sqrt(np.diag(pix.pcov))
        else:
            sdev = np.zeros(len(self.free))
        for i, par in enumerate(self.free): 
            getattr(self, par)[p] = getattr(pix, par) 
            getattr(self, 'sdev_' + par)[p] = sdev[i]
        return pix, p
    
    def train(self, xdata, ydata:np.ndarray, **kwargs):
        """Train the free parameters

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        nx = np.prod(self.shape)
        nt = ydata.shape[-1]
        if not self.parallel:
            if self.verbose>0:
                iterator = tqdm(range(nx), desc='Training '+self.__class__.__name__)
            else:
                iterator = range(nx)
            for x in iterator:
                args_x = (xdata, ydata.reshape((nx,nt)), kwargs, x)
                self._train_curve(args_x)
        else:
            args = [(xdata, ydata.reshape((nx,nt)), kwargs, x) for x in range(nx)]
            pool = multiprocessing.Pool(processes=num_workers)
            pool.map(self._train_curve, args)
            pool.close()
            pool.join()
        return self
    
    def cost(self, xdata, ydata, metric='NRMS')->float:
        """Goodness-of-fit value.

        Args:
            xdata (array-like): Array with x-data (time points).
            ydata (array-like): Array with y-data (signal values)
            metric (str, optional): Which metric to use - options are 'RMS' (Root-mean-square), 'NRMS' (Normalized root-mean-square), 'AIC' (Akaike information criterion), 'cAIC' (Corrected Akaike information criterion for small models) or 'BIC' (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            np.ndarray: goodness of fit in each element of the data array. 
        """
        return _cost(self, xdata, ydata, metric)
    
    def export_params(self)->list:
        """Model parameters with descriptions.

        Returns:
            dict: Dictionary with one item for each model parameter. The key is the parameter symbol (short name), and the value is a 4-element list with [parameter name, value, unit, sdev].
        """
        pars = {}
        for p in self.free:
            pars[p] = [
                p + ' name', 
                getattr(self, p), 
                p + ' unit', 
                getattr(self, 'sdev_' + p),
            ]
        return pars


class Model:
    # Abstract base class for end-to-end models.                  

    def __init__(self):
        self.free = []
        self.bounds = [-np.inf, np.inf]
        self.pcov = None

    def save(self, file=None, path=None, filename='Model'):
        """Save the current state of the model

        Args:
            file (str, optional): complete path of the file. If this is not provided, a file is constructure from path and filename variables. Defaults to None.
            path (str, optional): path to store the state if file is not provided. Thos variable is ignored if file is provided. Defaults to current working directory.
            filename (str, optional): filename to store the state if file is not provided. If no extension is included, the extension '.pkl' is automatically added. This variable is ignored if file is provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return _save(self, file, path, filename)

    def load(self, file=None, path=None, filename='Model'):
        """Load the saved state of the model

        Args:
            file (str, optional): complete path of the file. If this is not provided, a file is constructure from path and filename variables. Defaults to None.
            path (str, optional): path to store the state if file is not provided. Thos variable is ignored if file is provided. Defaults to current working directory.
            filename (str, optional): filename to store the state if file is not provided. If no extension is included, the extension '.pkl' is automatically added. This variable is ignored if file is provided. Defaults to 'Model'.

        Returns:
            dict: class instance
        """
        return _load(self, file, path, filename)

    def predict(self, xdata):
        """Predict the data at given xdata

        Args:
            xdata (array-like): Either an array with x-values (time points) or a tuple with multiple such arrays

        Returns:
            tuple or array-like: Either an array of predicted y-values (if xdata is an array) or a tuple of such arrays (if xdata is a tuple).
        """
        raise NotImplementedError('No predict function provided')
    
    def train(self, xdata, ydata, **kwargs):
        """Train the free parameters

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        return train(self, xdata, ydata, **kwargs)


    def plot(self, xdata, ydata, xlim=None, ref=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            xlim (array_like, optional): 2-element array with lower and upper boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form (x,y), where x is an array with x-values and y is an array with y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        raise NotImplementedError('No plot function implemented for model ' + self.__class__.__name__)
    
    
    def cost(self, xdata, ydata, metric='NRMS')->float:
        """Return the goodness-of-fit

        Args:
            xdata (array-like): Array with x-data (time points).
            ydata (array-like): Array with y-data (signal values)
            metric (str, optional): Which metric to use - options are: 
                **RMS** (Root-mean-square);
                **NRMS** (Normalized root-mean-square); 
                **AIC** (Akaike information criterion); 
                **cAIC** (Corrected Akaike information criterion for small models);
                **BIC** (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        return _cost(self, xdata, ydata, metric)


    def export_params(self)->list:
        """Return model parameters with their descriptions

        Returns:
            dict: Dictionary with one item for each model parameter. The key is the parameter symbol (short name), and the value is a 4-element list with [parameter name, value, unit, sdev].
        """
        # Short name, full name, value, units.
        pars = {}
        for p in self.free:
            pars[p] = [p+' name', getattr(self, p), p+' unit']
        return self._add_sdev(pars)
    

    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is not provided, the values are not rounded. Defaults to None.
        """
        pars = self.export_params()
        print('-----------------------------------------')
        print('Free parameters with their errors (stdev)')
        print('-----------------------------------------')
        for par in self.free:
            if par in pars:
                p = pars[par]
                if round_to is None:
                    v = p[1]
                    verr = p[3]
                else:
                    v = round(p[1], round_to)
                    verr = round(p[3], round_to)
                print(p[0] + ' ('+par+'): ' + str(v) + ' (' + str(verr) + ') ' + p[2])
        print('------------------')
        print('Derived parameters')
        print('------------------')
        for par in pars:
            if par not in self.free:
                p = pars[par]
                if round_to is None:
                    v = p[1]
                else:
                    v = round(p[1], round_to)
                print(p[0] + ' ('+par+'): ' + str(v) + ' ' + p[2])


    def get_params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get
            round_to (int, optional): Round to how many digits. If this is not provided, the values are not rounded. Defaults to None.

        Returns:
            list: values of parameter values
        """
        pars = []
        for a in args:
            v = getattr(self, a)
            if round_to is not None:
                v = round(v, round_to)
            pars.append(v)
        return pars
    
    
    def _getflat(self, attr:np.ndarray=None)->np.ndarray:
        if attr is None:
            attr = self.free
        vals = []
        for a in attr:
            v = getattr(self, a)
            vals = np.append(vals, np.ravel(v))
        return vals
    

    def _setflat(self, vals:np.ndarray, attr:np.ndarray=None):
        if attr is None:
            attr = self.free
        i=0
        for p in attr:
            v = getattr(self, p)
            if np.isscalar(v):
                v = vals[i]
                i+=1
            else:
                n = np.size(v)
                v = np.reshape(vals[i:i+n], np.shape(v))
                i+=n
            setattr(self, p, v)
                

    # def _x_scale(self):
    #     n = len(self.free)
    #     xscale = np.ones(n)
    #     for p in range(n):
    #         if np.isscalar(self.bounds[0]):
    #             lb = self.bounds[0]
    #         else:
    #             lb = self.bounds[0][p]
    #         if np.isscalar(self.bounds[1]):
    #             ub = self.bounds[1]
    #         else:
    #             ub = self.bounds[1][p]
    #         if (not np.isinf(lb)) and (not np.isinf(ub)):
    #             xscale[p] = ub-lb
    #     return xscale


    def _add_sdev(self, pars):
        for par in pars:
            pars[par].append(0)
        if not hasattr(self, 'pcov'):
            perr = np.zeros(np.size(self.free))
        elif self.pcov is None:
            perr = np.zeros(np.size(self.free))
        else:
            perr = np.sqrt(np.diag(self.pcov))
        for i, par in enumerate(self.free):
            if par in pars:
                pars[par][-1] = perr[i]
        return pars


def _save(model, file=None, path=None, filename='Model'):
    if file is None:
        if path is None:
            path=os.getcwd()
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
            path=os.getcwd()
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


def train(model:Model, xdata, ydata, **kwargs):

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
    try:
        pars, model.pcov = curve_fit(
            fit_func, None, y, p0, 
            bounds=model.bounds, #x_scale=model._x_scale(),
            **kwargs)
    except RuntimeError as e:
        msg = 'Runtime error in curve_fit -- \n'
        msg += str(e) + ' Returning initial values.'
        warnings.warn(msg)
        pars = p0
        model.pcov = np.zeros((np.size(p0), np.size(p0)))
    
    model._setflat(pars)
    
    return model


def _cost(model, xdata, ydata, metric='NRMS')->float:

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
        k = len(model.free)
        with np.errstate(divide='ignore'):
            loss = k*2 + n*np.log(rss/n)
    elif metric == 'cAIC':
        rss = np.sum((y-ydata)**2)
        n = ydata.shape[-1]
        k = len(model.free)
        with np.errstate(divide='ignore'):
            loss = k*2 + n*np.log(rss/n) + 2*k*(k+1)/(n-k-1)
    elif metric == 'BIC':
        rss = np.sum((y-ydata)**2, axis=-1)
        n = ydata.shape[-1]
        k = len(model.free)
        with np.errstate(divide='ignore'):
            loss = k*np.log(n) + n*np.log(rss/n)
    return loss