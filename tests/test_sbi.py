import os
import time

from tqdm import tqdm
import numpy as np

from dcmri.ui_artery_tissue import ArteryTissue


VARS = {
    'R10': [0.3, 3], 
    'S0': [0.01, 10], 
    'R10a': [0.3, 3], 
    'S0a': [0.1, 10], 
    'BAT': [0.5, 5], 
    'CO': [2, 15], 
    'Thl': [1, 20], 
    'Dhl': [0.1, 0.9], 
    'To': [1, 30], 
    'vb': [0.01, 0.9], 
    'Fb': [0.01, 0.5],
}
CONST = { 
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
PARS = VARS | CONST


def test_lung_dce_model():

    model = ArteryTissue(**PARS)
    nsim = 5
    test_vars = model.sample(nsim)
    test_signal = model.simulate(test_vars)

    # Test DL
    # TODO: avoid loop by fit_dl(test_signal)
    start = time.time()
    vars_err = np.zeros((nsim, test_vars.shape[1]))
    for i in tqdm(range(nsim)):
        vars_i = model.fit_dl(test_signal[i,:])
        # vars_i = model.fit_ls(test_signal[i,:], vars0=vars_i, xtol=1e-6) # Hybrid method
        model.plot(vars_i, test_signal[i,:], round_to=3, title='Deep learning')
        vars_err[i,:] = 100 * (vars_i - test_vars[i,:])/test_vars[i,:]
    vars_err_mean = np.mean(vars_err, axis=0)
    vars_err_std = np.std(vars_err, axis=0)

    print(f"\nDL computation time: {time.time() - start:.2f} sec")
    for j, p in enumerate(model.vars):
        print(f"DL-Error {p}: {np.round(vars_err_mean[j])} +/- {np.round(1.96*vars_err_std[j]/np.sqrt(nsim))} %")

    # Test LS
    start = time.time()
    vars_err = np.zeros((nsim, test_vars.shape[1]))
    for i in tqdm(range(nsim)):
        vars_i = model.fit_ls(test_signal[i,:], xtol=1e-6)
        model.plot(vars_i, test_signal[i,:], round_to=3, title='Least squares')
        vars_err[i,:] = 100 * (vars_i - test_vars[i,:])/test_vars[i,:]
    vars_err_mean = np.mean(vars_err, axis=0)
    vars_err_std = np.std(vars_err, axis=0)

    print(f"\nLS computation time: {time.time() - start:.2f} sec")
    for j, p in enumerate(model.vars):
        print(f"LS-Error {p}: {np.round(vars_err_mean[j])} +/- {np.round(1.96*vars_err_std[j]/np.sqrt(nsim))} %")
        

if __name__ == "__main__":

    # ArteryTissue(**PARS).learn_inverse(num_training_data=10**6)
    test_lung_dce_model()

    print('All sbi tests passed!!')



