import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class CurveFit():

    # defaults
    x = np.arange(0, 1, 0.05)
    xname = 'x'
    xunit = 'unit'
    yname = 'y'
    yunit = 'unit'

    # Define constants (if any)
    power = 3.0

    def function(self, x, p):

        return p.a*x**self.power + p.b      

    def parameters(self):
        
        return [
            ['a', "slope", 1, "", -np.inf, np.inf, True, 3],
            ['b', "intercept", 0, '', -np.inf, np.inf, True, 3],
        ]

    def __init__(self):

        self.reset_p()

    def reset_p(self):
        """Reset parameters to factory settings"""

        p = self.parameters()
        if type(p[0]) is str:
            p = [[variable, variable, 0, "", -np.inf, np.inf, True, 3] for variable in p]
        cols = ['symbol', "name", "initial value", "unit", "lower bound", "upper bound", "fit", "digits"]
        self.p = pd.DataFrame(p, columns=cols)
        self.p.set_index('symbol', inplace=True)
        self.p['value'] = self.p['initial value']
        cols = cols[1:]
        cols.insert(2, 'value')
        self.p = self.p[cols]       

    def predict_y(self, x=None):

        if x is not None:
            self.set_x(x)
        self.yp = self.function(self.x, self.p.value)
        return self.yp

    def estimate_p(self):
        pass

    def plabel(self):
        label = ''
        for _, p in self.p.iterrows():
            v = str(p.value).split('.')
            digits = p.digits-len(v[0])
            if digits >= 0:
                v = round(p.value, digits)
            else:
                v = p.value
            label += '\n'
            label += p.name + " = " + str(v) + " " + p.unit
        return label

    def _fit_function(self, _, *params):

        self.p.loc[self.p.fit,'value'] = params
        return self.predict_y()

    def fit_p(self, x=None, y=None):

        if x is not None:
            self.set_x(x)
        if y is not None:
            self.set_y(y)
        try:
            self.p.loc[self.p.fit, 'value'], _ = curve_fit(
                self._fit_function, self.x, self.y, 
                p0 = self.p.loc[self.p.fit, 'value'].values, 
                bounds = (
                    self.p.loc[self.p.fit, 'lower bound'].values,
                    self.p.loc[self.p.fit, 'upper bound'].values,
                ),
            )
        except ValueError as e:
            print(e)
        except RuntimeError as e:
            print(e)
        self.predict_y()

    def plot_prediction(self, show=True, save=False, path=None):

        name = self.__class__.__name__
        plt.title(name + ' - model prediction')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(self.x, self.yp, 'g-', label='prediction ' + self.plabel())
        plt.legend()
        if save:
            if path is None:
                path = self.path()
            plt.savefig(fname=os.path.join(path, name + '_prediction' + '.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_data(self, show=True, save=False, path=None): 

        name = self.__class__.__name__
        plt.title(name + " - data")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(self.x, self.y, 'ro', label='data')
        plt.legend()
        if save:
            if path is None:
                path = self.path()            
            plt.savefig(fname=os.path.join(path, name + '_data' + '.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_fit(self, xrange=None, show=True, save=False, path=None): 

        if xrange is None:
            x0 = self.x[0]
            x1 = self.x[-1]
            win_str = ''
        else:
            x0 = xrange[0]
            x1 = xrange[1]
            win_str = ' [' + str(round(x0)) + ', ' + str(round(x1)) + ']'
        name = self.__class__.__name__
        i = np.nonzero((self.x>=x0) & (self.x<=x1))[0]
        plt.title(name + " - model fit"+ win_str)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(self.x[i], self.y[i], 'ro', label='data')
        plt.plot(self.x[i], self.yp[i], 'b--', label='fit ' + self.plabel())
        plt.legend()
        if save:
            if path is None:
                path = self.path()            
            plt.savefig(fname=os.path.join(path, name + '_fit' + win_str + '.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def export_p(self, path=None):

        if path is None: 
            path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
        save_file = os.path.join(path, self.__class__.__name__ + '_fitted_parameters.csv')
        try:
            self.p.to_csv(save_file)
        except:
            print("Can't write to file ", save_file)
            print("Please close the file before saving data")

    # Export curves
    # -------------
    # df_results = pd.DataFrame({"Time fit (s)": time})
    # df_results["Liver fit (a.u.)"] = subject.liver_signal
    # df_output = pd.concat([df_data, df_results], axis=1)
    # save_file = data.results_path() + 'fit_' + filename + ".csv"
    # try:
    #     df_output.to_csv(save_file)
    # except:
    #     print("Can't write to file ", save_file)
    #     print("Please close the file before saving data")


    def path(self):

        path = os.path.dirname(__file__)
        path = os.path.join(path, 'results')
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def set_p(self, p):
        self.p = p

    def set_x(self, value=None, name=None, unit=None):
        if value is not None:
            self.x = value
        if name is not None:
            self.xname = name
        if unit is not None:
            self.xunit = unit

    def set_y(self, value=None, name=None, unit=None):
        if value is not None:
            self.y = value
        if name is not None:
            self.yname = name
        if unit is not None:
            self.yunit = unit

    def set_xy(self, x, y):
        self.set_x(x)
        self.set_y(y)

    @property
    def xlabel(self):
        return self.xname + ' (' + self.xunit + ')'

    @property
    def ylabel(self):
        return self.yname + ' (' + self.yunit + ')'

    @property
    def parameter_values(self):
        return self.p['value'].values


class BiExponential(CurveFit):

    def function(self, x, p):
        return p.A*np.exp(-p.a*x) + p.B*np.exp(-p.b*x)

    def parameters(self):
        return ['A', 'a', 'B', 'b']


def test_biexp_fit():

    x = np.arange(0, 1, 0.05)
    y = 3*x**2 + 200

    c = BiExponential()
    c.p['upper bound'] = [np.inf,1,100,1]
    c.fit_p(x,y)
    c.plot_fit(save=True)

def test_curve_fit():

    x = np.arange(0, 1, 0.05)
    y = 3*x**2 - 200

    c = CurveFit()
    c.set_x(x)
    c.predict_y()
    c.plot_prediction(save=True)
    c.set_y(y)
    c.plot_data(save=True)
    c.fit_p()
    c.plot_fit(save=True)
    c.export_p()

if __name__ == "__main__":
    test_biexp_fit()
    test_curve_fit()
