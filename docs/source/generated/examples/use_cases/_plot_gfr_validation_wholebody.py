"""
=========================================================
GFR validation study
=========================================================

SK-GFR validation

Reference
--------- 

TBC
"""

# %%
# Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc
from tqdm import tqdm


# Create output folder in current working directory
path = os.path.join(os.getcwd(), 'KRUK')
if not os.path.exists(path):
    os.makedirs(path)

# data = dc.fetch('KRUK')


# # %%
# # Fit all datasets, export a plot for each subject and save all parameters in a dataframe:

# results = []
# const = ['TR','FA','field_strength','agent','dose','rate','weight','n0']
# for scan in tqdm(data):
#     if ('LK' in scan) and ('RK' in scan): # consider single kidney case
#         B0 = scan['field_strength']
#         xdata = (
#             scan['time'],
#             scan['time'],
#             scan['time'],
#         )
#         ydata = (
#             scan['aorta'],
#             scan['LK'],
#             scan['RK'],
#         )
#         T1_lk = scan['LK T1']
#         T1_rk = scan['RK T1']
#         T1_lk = dc.T1(B0, 'kidney') if T1_lk is None else T1_lk
#         T1_rk = dc.T1(B0, 'kidney') if T1_rk is None else T1_rk
#         model = dc.AortaKidneys(
#             dt = 0.5,
#             organs = 'comp',
#             heartlung = 'chain',
#             kidneys = '2CF',
#             vol_lk = scan['LK vol'], # include in params
#             vol_rk = scan['RK vol'], # include in params
#             R10_lk = 1/T1_lk, # include in params
#             R10_rk = 1/T1_rk, # include in params
#             R10b = 1/dc.T1(B0, 'blood'), # include in params
#             **{p:scan[p] for p in const},
#         )
#         model.train(xdata, ydata, xtol=1e-6)
#         model.plot(xdata, ydata, show = False,
#             fname = os.path.join(path, scan['subject']+'_'+scan['visit']+'.png')
#         )
#         pars = model.export_params()

#         # Add the isotope values and goodness of fit to the export
#         pars['NRMS'] = [
#             'Normalized root-mean-square',
#             model.cost(xdata, ydata), 
#             '%', 0,
#         ]
#         pars['LK iso-SK-GFR'] = [
#             'LK - Isotope single-kidney GFR',
#             scan['LK iso-SK-GFR'], 
#             'mL/sec', 0
#         ]
#         pars['RK iso-SK-GFR'] = [
#             'RK - Isotope single-kidney GFR',
#             scan['RK iso-SK-GFR'],
#             'mL/sec', 0
#         ]
#         pars['iso-DRF'] = [
#             'Isotope differential renal function',
#             scan['LK iso-SK-GFR']/(scan['LK iso-SK-GFR']+scan['RK iso-SK-GFR']), 
#             '', 0
#         ]

#         # convert the parameter dictionary to a dataframe and append to results
#         pars = pd.DataFrame.from_dict(pars, orient='index', 
#                 columns = ["name", "value", "unit", 'stdev'])
#         val = pars.value.values
#         cov = np.zeros(len(val))
#         cov[val!=0] = 100*pars.stdev.values[val!=0]/val[val!=0]
#         pars['subject'] = scan['subject']
#         pars['visit'] = scan['visit']
#         pars['parameter'] = pars.index
#         pars['CoV (%)'] = cov
#         pars['B0'] = B0
#         results.append(pars)

# # Combine all results into a single dataframe and save to working directory.
# results = pd.concat(results).reset_index(drop=True)
# results.to_csv(os.path.join(path, 'results.csv'))


# # %%
# # Plot MRI values and reference values

results = pd.read_csv(os.path.join(path, 'results.csv'))

v1T = pd.pivot_table(results[results.B0==1], values='value', columns='parameter', index=['subject','visit'])
v3T = pd.pivot_table(results[results.B0==3], values='value', columns='parameter', index=['subject','visit'])

plt.plot(60*v1T['LK iso-SK-GFR'].values, 60*v1T['LK-GFR'].values, 'bo', linestyle='None', markersize=4, label='1T')
plt.plot(60*v1T['RK iso-SK-GFR'].values, 60*v1T['RK-GFR'].values, 'bo', linestyle='None', markersize=4)
plt.plot(60*v3T['LK iso-SK-GFR'].values, 60*v3T['LK-GFR'].values, 'ro', linestyle='None', markersize=4, label='3T')
plt.plot(60*v3T['RK iso-SK-GFR'].values, 60*v3T['RK-GFR'].values, 'ro', linestyle='None', markersize=4)
plt.plot(60*v3T['RK iso-SK-GFR'].values, 60*v3T['RK iso-SK-GFR'].values, linestyle='-', color='black')
plt.xlabel("Isotope single-kidney GFR (mL/min)")
plt.ylabel("MRI single-kidney GFR (mL/min)")
plt.legend()
plt.show()

plt.plot(100*v1T['iso-DRF'].values, 100*v1T['DRF'].values, 'bo', linestyle='None', markersize=4, label='1T')
plt.plot(100*v3T['iso-DRF'].values, 100*v3T['DRF'].values, 'ro', linestyle='None', markersize=4, label='3T')
plt.plot([0,100], [0,100], linestyle='-', color='black')
plt.xlabel("Isotope split renal function (%)")
plt.ylabel("MRI split renal function (%)")
plt.legend()
plt.show()

# %%
# Compute bias and accuracy

v = pd.pivot_table(results, values='value', columns='parameter', index=['subject','visit'])

iso = np.concat((60*v['LK iso-SK-GFR'].values, 60*v['RK iso-SK-GFR'].values))
mri = np.concat((60*v['LK-GFR'].values, 60*v['RK-GFR'].values))

diff = mri-iso
bias = np.mean(diff)
err =  1.96*np.std(diff)
bias_err = 1.96*np.std(diff)/np.sqrt(np.size(diff))

print('-----------------')
print('Single-kidney GFR')
print('-----------------')
print('95% CI on the bias (ml/min): ', bias-bias_err, bias+bias_err)
print('95% CI on an individual measurement (ml/min): ', bias-err, bias+err)

iso = np.concat((100*v['iso-DRF'].values, 100*v['iso-DRF'].values))
mri = np.concat((100*v['DRF'].values, 100*v['DRF'].values))

diff = mri-iso
bias = np.mean(diff)
err =  1.96*np.std(diff)
bias_err = 1.96*np.std(diff)/np.sqrt(np.size(diff))

print('--------------------')
print('Split renal function')
print('--------------------')
print('95% CI on the bias (%): ', bias-bias_err, bias+bias_err)
print('95% CI on an individual measurement (%): ', bias-err, bias+err)

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
