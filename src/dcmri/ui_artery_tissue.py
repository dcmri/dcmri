import os
import sys
import pickle
import json
import uuid
from pathlib import Path

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.analysis import plot_summary

from dcmri.pk_aorta import flux_aorta
from dcmri.lib import ca_conc, ca_injection, relaxivity
from dcmri.tissue import conc_tissue
from dcmri.sig import signal_ss
from dcmri.utils import sample, add_noise


PARAMS = {
    'R10': {
        'name': 'Tissue precontrast R1',
        'unit': 'Hz',
    },
    'S0': {
        'name': 'Signal scaling factor',
        'unit': '',
    },
    'R10a': {
        'name': 'Arterial precontrast R1',
        'unit': 'Hz',
    },
    'S0a': {
        'name': 'Arterial signal scaling factor',
        'unit': '',
    },
    'BAT': {
        'name': 'Bolus arrival time',
        'unit': 'sec',
    },
    'CO': {
        'name': 'Cardiac output',
        'unit': 'mL/sec',
    },
    'Thl': {
        'name': 'Heart-lung transit time',
        'unit': 'sec',
    },
    'Dhl': {
        'name': 'Heart-lung dispersion',
        'unit': 'sec',
    },
    'To': {
        'name': 'Organ transit time',
        'unit': 'sec',
    },
    'vb': {
        'name': 'Parenchymal blood volume fraction',
        'unit': 'mL/cm3',
    },
    'Fb': {
        'name': 'Parenchymal blood flow',
        'unit': 'mL/sec/cm3',
    },
    'weight': {
        'name': 'Weight',
        'unit': 'kg',
    },
    'B1corr': {
        'name': 'B1 correction factor',
        'unit': '',
    },
    'B1corr_a': {
        'name': 'Arterial B1 correction factor',
        'unit': '',
    },
    'TS': {
        'name': 'Time between acquisitions',
        'unit': 'sec',
    },
    'nt': {
        'name': 'Number of time points',
        'unit': '',
    },
    'agent': {
        'name': 'Contrast agent',
        'unit': '',
    },
    'dose': {
        'name': 'Dose',
        'unit': 'mL/kg',
    },
    'rate': {
        'name': 'Injection rate',
        'unit': 'mL/sec',
    },
    'field_strength': {
        'name': 'Field strength',
        'unit': 'T',
    },
    'FA': {
        'name': 'Flip angle',
        'unit': 'deg',
    },
    'TR': {
        'name': 'Repetition time',
        'unit': 'sec',
    },
    'VFA': {
        'name': 'Variable flip angles',
        'unit': 'deg',
    },
    'dt': {
        'name': 'Simulation time step',
        'unit': 'sec',
    },
}

class ArteryTissue():
    """A model for aorta and 2-compartment exchange tissue.
    """

    def __init__(self, **kwargs):

        self.vars = {
            'R10': [0, 5], 
            'S0': [0, 10], 
            'R10a': [0, 5], 
            'S0a': [0, 10], 
            'BAT': [0, 10], 
            'CO': [1, 15], 
            'Thl': [0, 30], 
            'Dhl': [0, 1], 
            'To': [0, 30], 
            'vb': [0, 1], 
            'Fb': [0, 1],
        }
        self.const = {
            'weight': 54, # kg
            'B1corr': 1,
            'B1corr_a': 1, 
            'TS': 0.558, # sec
            'nt': 48,
            'agent': 'gadobutrol',
            'dose': 0.05, # mL/kg
            'rate': 4, # mL / sec
            'field_strength': 3, # T
            'FA': 20, # deg
            'TR': 0.002, # sec
            'VFA': [2,4,10,30], #deg
            'dt': 0.25, # sec
        }

        for p in kwargs:
            if p in self.vars:
                self.vars[p] = kwargs[p]
            elif p in self.const:
                self.const[p] = kwargs[p]

        self.pcov = None
        self.posterior = None


    def to_json(self, fname):
        """Save the model parameters to a JSON file

        Args:
            fname (str): Filepath to save the JSON file.
        """
        model_dict = {
            'vars': self.vars,
            'const': self.const,
        }
        with open(fname, 'w') as f:
            json.dump(model_dict, f, indent=4)


    def from_json(self, fname):
        """Load the model parameters from a JSON file

        Args:
            fname (str): Filepath to load the JSON file.
        """
        with open(fname, 'r') as f:
            model_dict = json.load(f)
        self.vars = model_dict['vars']
        self.const = model_dict['const']
        return self


    def predict_arterial_conc(self, vars):

        # Combine constants and variables in the same dictionary
        p = self.const | {p: vars[i] for i, p in enumerate(self.vars)}

        # Injection
        conc = ca_conc(p['agent'])
        tsim = np.arange(0, p['TS']*(p['nt']+1), p['dt'])
        Ji = ca_injection(tsim, p['weight'], conc, p['dose'], p['rate'], p['BAT'])

        # Arterial concentration
        Jb = flux_aorta(
            Ji, tsim, E=0,
            heartlung = ['chain', (p['Thl'], p['Dhl'])],
            organs = ['comp', (p['To'],)]
        )
        return Jb/p['CO']
 

    def predict_tissue_conc(self, vars, ca=None):
        """Return the tissue concentration

        Returns:
            np.ndarray: Concentration in M
        """
        # Combine constants and variables in the same dictionary
        p = self.const | {p: vars[i] for i, p in enumerate(self.vars)}

        if ca is None:
            ca = self.predict_arterial_conc(vars)
        return conc_tissue(
            ca, 
            dt=p['dt'],
            kinetics='NXP',
            vb=p['vb'], 
            Fb=p['Fb'],
        )

    def predict(self, vars, SNR=None):
        """Predict the signal at specific time points

        Returns:
            np.ndarray: Array of predicted signals for each time point.
        """ 
        # Combine constants and variables in the same dictionary
        p = self.const | {p: vars[i] for i, p in enumerate(self.vars)}

        # vFA signals
        vFAa = np.array([signal_ss(p['S0a'], p['R10a'], p['TR'], p['B1corr_a'] * fa) for fa in p['VFA']])
        vFAt = np.array([signal_ss(p['S0'], p['R10'], p['TR'], p['B1corr'] * fa) for fa in p['VFA']])

        # Injection
        ca = self.predict_arterial_conc(vars)
        Ct = self.predict_tissue_conc(vars, ca)

        r1 = relaxivity(p['field_strength'], 'blood', p['agent'])
        R1a = p['R10a'] + r1 * ca
        R1t = p['R10'] + r1 * Ct

        siga = signal_ss(p['S0a'], R1a, p['TR'], p['B1corr_a'] * p['FA'])
        sigt = signal_ss(p['S0'], R1t, p['TR'], p['B1corr'] * p['FA'])

        # Sample signals
        tsim = np.arange(0, p['TS']*(p['nt']+1), p['dt'])
        tacq = p['TS'] * np.arange(p['nt'])
        
        siga = sample(tacq, tsim, siga, p['TS'])
        sigt = sample(tacq, tsim, sigt, p['TS'])

        if SNR is not None:
            S = np.max(np.concatenate((vFAa, vFAt, [siga[0]], [sigt[0]])))
            noise_sdev = S / SNR
            vFAa = add_noise(vFAa, noise_sdev)
            vFAt = add_noise(vFAt, noise_sdev)
            siga = add_noise(siga, noise_sdev)
            sigt = add_noise(sigt, noise_sdev)

        return np.concatenate((vFAa, vFAt, siga, sigt))

    
    def fit_ls(self, signal, vars0=None, verbose=True, **kwargs):
        """Train the free parameters using a least-squares method.

        Args:
            signal (array-like): Array with measured signals, normalised to the range [0,1].
            tol: cut-off value for the singular values in the 
                computation of the matrix pseudo-inverse.

        Returns:
            self
        """ 
        def fit_func(_, *vars):
            return self.predict(vars)
        
        if vars0 is None:
            vars0 = [(p[0] + p[1]) / 2 for p in self.vars.values()]
        b0 = [p[0] for p in self.vars.values()]
        b1 = [p[1] for p in self.vars.values()]

        try:
            vars_opt, pcov = curve_fit(
                fit_func, None, signal, vars0, bounds=(b0, b1), **kwargs,
            )
        except RuntimeError as e:
            self.pcov = None
            if verbose:
                print(f"\n Fitting failed - returning initial values: {e}")
        except ValueError as e:
            self.pcov = None
            raise ValueError(f"Fitting failed: {e}")
        else:
            self.pcov = pcov
        
        return vars_opt


    def fit_dl(self, signal, n_samples=10**3):
        if self.posterior is None:
            try:
                self.posterior = self.load_posterior()
            except Exception:
                raise ValueError("No matching posterior found - please train a new posterior first.")
        signal = torch.tensor(signal, dtype=torch.float32) 
        pars_posterior = self.posterior.sample((n_samples,), x=signal, show_progress_bars=False)
        pars_mean = torch.mean(pars_posterior, axis=0)
        return pars_mean.numpy()
    
    
    def load_posterior(self):
        # TODO: look on zenodo
        dir = importlib_resources.files('dcmri.datafiles')
        for json_file in list(dir.glob(f"{self.__class__.__name__}_Posterior_*.json")):
            with open(json_file, 'r') as f:
                json_dict = json.load(f) 
                if json_dict['vars'] != self.vars:
                    continue
                if json_dict['const'] != self.const:
                    continue
                pkl_file = str(json_file).replace('.json', '.pkl')
                with open(pkl_file, "rb") as handle:
                    posterior = pickle.load(handle) 
                return posterior  
        raise ValueError("No matching posterior found.")     
    

    def sample(self, n_samples):
        """Return a prior distribution for the free parameters

        Returns:
            BoxUniform: Uniform prior distribution.
        """
        low = torch.tensor([p[0] for p in self.vars.values()])
        high = torch.tensor([p[1] for p in self.vars.values()])
        prior = BoxUniform(low=low, high=high)  
        samples = prior.sample((n_samples,))
        return samples.numpy() # no need to use BoxUniform here - use numpy or scipy method


    def simulate(self, vars:np.ndarray, SNR=100) -> np.ndarray:
        n_sim = vars.shape[0]
        n_sig = 2 * self.const['nt'] + 2 * len(self.const['VFA'])
        output = np.zeros((n_sim, n_sig))
        for i_sim in tqdm(range(n_sim), desc='Simulating..'):
            output[i_sim, :] = self.predict(vars[i_sim, :], SNR=SNR) 
        return output


    def learn_inverse(self, num_training_data=10**6):

        # Generate training data
        vars = self.sample(num_training_data)
        signal = self.simulate(vars)

        # Build prior
        low = torch.tensor([p[0] for p in self.vars.values()])
        high = torch.tensor([p[1] for p in self.vars.values()])
        prior = BoxUniform(low=low, high=high) 

        # Training new posterior
        vars = torch.tensor(vars, dtype=torch.float32) 
        signal = torch.tensor(signal, dtype=torch.float32)
        inference = NPE(prior=prior)
        inference = inference.append_simulations(vars, signal)
        inference.train()
        posterior = inference.build_posterior()

        # Validate posterior
        if self.is_best_posterior(posterior):
            print("\nThe new posterior is better than the previous one! Saving as new default.")
            self.posterior = posterior
            dir = importlib_resources.files('dcmri.datafiles')
            fname = f"{self.__class__.__name__}_Posterior_{uuid.uuid4()}"
            file_json = str(dir.joinpath(fname + '.json'))
            file_pkl = str(dir.joinpath(fname + '.pkl'))
            self.to_json(file_json)
            with open(file_pkl, "wb") as handle:
                pickle.dump(self.posterior, handle)
        else:
            print("\nThe new posterior is not better than the current one - not saving.")

        # # Show convergence
        # _ = plot_summary(
        #     inference,
        #     tags=["training_loss", "validation_loss"],
        #     figsize=(10, 2),
        # )


    def is_best_posterior(self, new_posterior, num_test_data=100):

        # If this is the first posterior for the current configuration
        # it is automatically the best.
        try:
            current_posterior = self.load_posterior()
        except Exception:
            return True
        
        # Generate training data
        test_vars = self.sample(num_test_data)
        test_signal = self.simulate(test_vars)

        init_posterior = self.posterior

        print('Testing current posterior..')
        self.posterior = current_posterior
        pars_err = self.benchmark_posterior(test_vars, test_signal)
        pars_err_upper = np.percentile(np.abs(pars_err), 97.5, axis=0)
        current_benchmark = np.max(pars_err_upper)

        print('Testing new posterior..')
        self.posterior = new_posterior
        pars_err = self.benchmark_posterior(test_vars, test_signal)
        pars_err_upper = np.percentile(np.abs(pars_err), 97.5, axis=0)
        new_benchmark = np.max(pars_err_upper)

        self.posterior = init_posterior

        return new_benchmark < current_benchmark


    def benchmark_posterior(self, test_vars, test_signal):
        nsim = test_vars.shape[0]
        vars_err = np.zeros((nsim, test_vars.shape[1]))
        for i in tqdm(range(nsim)):
            vars_i = self.fit_dl(test_signal[i,:], n_samples = 10**3)
            vars_err[i,:] = 100 * (vars_i - test_vars[i,:])/test_vars[i,:]
        return vars_err


    def print(self, vars, round_to=None):
        """Print the model parameters

        Args:
            round_to (int, optional): Round to how many digits. If this is
              not provided, the values are not rounded. Defaults to None.
        """

        print('')
        print('---------------------')
        print('Free parameter values')
        print('---------------------')
        print('')  

        p = {p: vars[i] for i, p in enumerate(self.vars)}

        for par in p:
            val = p[par]
            val = round(val, round_to) if round_to is not None else val
            name = PARAMS[par]['name']
            unit = PARAMS[par]['unit']
            print(f"{name} ({par}): {val} {unit}")



    def plot(self, vars, signal=None, round_to=None, title=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            signal (array-like, optional): Array with measured signals.
            round_to (int, optional): Rounding for the model parameters.
            fname (path, optional): Filepath to save the image. If no value is
              provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to
              True.
        """
        p = self.const | {p: vars[i] for i, p in enumerate(self.vars)}
        nFA = len(p['VFA'])
        if signal is not None:
            vFAa, vFAt, siga, sigt = np.split(signal, [nFA, 2*nFA, p['nt']+2*nFA])
        signal_pred = self.predict(vars)
        vFAa_pred, vFAt_pred, siga_pred, sigt_pred = np.split(signal_pred, [nFA, 2*nFA, p['nt']+2*nFA]) 
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        if title is not None:
            fig.suptitle(title)

        # Plot VFA
        i=0
        ax[i].set_title('Variable flip angle')
        ax[i].set(ylabel='MRI signal (a.u.)', xlabel='FA (deg)')
        ax[i].plot(
            p['VFA'], vFAt_pred, linestyle='-', 
            linewidth=3.0, color='cornflowerblue', 
            label='Tissue (predicted)',
        )
        if vFAt is not None:
            ax[i].plot(
                p['VFA'], vFAt, marker='x', linestyle='None',
                color='darkblue', label='Tissue (measured)',
            )
        ax[i].plot(
            p['VFA'], vFAa_pred, linestyle='-', 
            linewidth=3.0, color='lightcoral', 
            label='Artery (predicted)',
        )
        if vFAa is not None:
            ax[i].plot(
                p['VFA'], vFAa, marker='x', linestyle='None',
                color='darkred', label='Artery (measured)',
            )
        ax[i].legend()

        # Plot predicted signals and measured signals
        i=1
        tacq = p['TS'] * np.arange(p['nt'])
        ax[i].set_title('MRI signals')
        ax[i].set(ylabel='MRI signal (a.u.)', xlabel='Time (min)')
        ax[i].plot(
            tacq / 60, sigt_pred, linestyle='-', 
            linewidth=3.0, color='cornflowerblue', 
            label='Tissue (predicted)',
        )
        if sigt is not None:
            ax[i].plot(
                tacq / 60, sigt, marker='x', linestyle='None',
                color='darkblue', label='Tissue (measured)',
            )
        ax[i].plot(
            tacq / 60, siga_pred, linestyle='-', 
            linewidth=3.0, color='lightcoral', 
            label='Artery (predicted)',
        )
        if sigt is not None:
            ax[i].plot(
                tacq / 60, siga, marker='x', linestyle='None',
                color='darkred', label='Artery (measured)',
            )
        ax[i].legend()

        # Plot predicted concentrations and measured concentrations
        i=2
        tsim = np.arange(0, p['TS']*(p['nt']+1), p['dt'])
        ca = self.predict_arterial_conc(vars)
        Ct = self.predict_tissue_conc(vars, ca)
        ax[i].set_title('Tissue concentrations')
        ax[i].set(ylabel='Concentration (mM)', xlabel='Time (min)')
        ax[i].plot(
            tsim / 60, 1000 * ca, linestyle='-', linewidth=3.0,
            color='lightcoral', label='Arterial blood',
        )
        ax[i].plot(
            tsim / 60, 1000 * Ct, linestyle='-', linewidth=3.0, 
            color='cornflowerblue', label='Tissue (predicted)', 
        )
        ax[i].legend()

        # Plot text
        i = 3
        msg = []
        for par in self.vars:
            val = float(p[par])
            val = round(val, round_to) if round_to is not None else val
            name = PARAMS[par]['name']
            unit = PARAMS[par]['unit']
            msg.append(f"{name} ({par}): {val} {unit} \n")
        msg = "\n".join(msg)
        ax[i].set_title('Free parameters')
        ax[i].axis("off")  # hide axes
        ax[i].text(0, 0.9, msg, fontsize=10, transform=ax[3].transAxes, ha="left", va="top")

        # Show and/or save plot
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()
