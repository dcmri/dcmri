"""
========================================
Single-kidney glomerular filtration rate
========================================

This examples illustrates the use of `~dcmri.Kidney` for measurement of 
single kidney glomerular filtration rate (SK-GFR). The script aims to 
replicate a validation study comparing MRI-derived measurement of SK-GFR 
against reference measurement performed with radio-isotopes Basak et al 2018). 
The study used 124 historical datasets collected in between the years 2000 and 
2010 at 1 Tesla and 3 Tesla MRI. The study concluded that while the 
MRI-derived values were unbiased, the precision was low and significant 
improvements in data quality would be needed before this technique can be 
applied in clinical practice. The study was funded by 
`Kidney Research UK <https://www.kidneyresearchuk.org/>`_.

**Reference**

Basak S, Buckley DL, Chrysochou C, Banerji A, Vassallo D, Odudu A, Kalra PA, 
Sourbron SP. Analytical validation of single-kidney glomerular filtration 
rate and split renal function as measured with magnetic resonance renography. 
Magn Reson Imaging. 2019 Jun;59:53-60. doi: 10.1016/j.mri.2019.03.005. 
`[URL] <https://pubmed.ncbi.nlm.nih.gov/30849485/>`_.
"""

# %%
# Setup
# -----

# Import packages and fetch data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dcmri as dc

# Fetch the data
data = dc.fetch('KRUK')

# %%
# Model definition
# ----------------
# In order to avoid some repetition in this script, we define a function that 
# returns a trained model for a single dataset:

def kidney_model(scan, kidney):

    # Get B0 and precontrast T1
    B0 = scan['field_strength']
    T1 = scan[kidney+' T1']
    T1 = dc.T1(B0, 'kidney') if T1 is None else T1

    # Set kidney model parameters
    model = dc.Kidney(
        aif = scan['aorta'], 
        t = scan['time'],
        vol = scan[kidney+' vol'],
        R10 = 1/T1,
        R10a = 1/dc.T1(B0, 'blood'),
        sequence = 'SS',
        TR = scan['TR'],
        FA = scan['FA'],
        field_strength = B0,
        agent = scan['agent'],
        n0 = scan['n0'],
    )

    # Customize free parameter ranges
    model.set_free(
        pop = 'Ta', 
        Fp = [0, 0.05], 
        FF = [0, 0.3], 
        Tt = [30, np.inf],
    )

    # Train the kidney model on the data
    xdata = scan['time']
    ydata = scan[kidney]
    model.train(xdata, ydata)

    return xdata, ydata, model

# %%
# Check model fit
# ---------------
# Before running the full analysis on all cases, lets illustrate the results 
# by fitting the left kidney of the first subject:

time, signal, model = kidney_model(data[0], 'LK')

# %%
# Plot the results to check that the model has fitted the data:

model.plot(time, signal)

# %%
# Print the measured model parameters and any derived parameters and check 
# that standard deviations of measured parameters are small relative to the 
# value, indicating that the parameters are measured reliably:

model.print_params(round_to=3)

#%%
# The measured SK-GFR for this kidney (0.43) is somewhat higher than the 
# radio-isotope reference value (0.28):

print('-----------------------------')
print('Comparison to reference value')
print('-----------------------------')
print('Radio-isotope SK-GFR: ', data[0]['LK iso-SK-GFR'])


# %%
# Fit all data
# ------------
# Now that we have illustrated an individual result in some detail, we proceed 
# to determine SK-GFR for all datasets:

results = []

for scan in data:
    for kidney in ['LK', 'RK']:
        if kidney not in scan:
            continue
        xdata, ydata, model = kidney_model(scan, kidney)

        # Export parameters and add reference value
        pars = model.export_params()
        pars['iso-SK-GFR'] = [
            'Isotope single-kidney GFR', 
            scan[kidney + ' iso-SK-GFR'], 
            'mL/sec', 
            0,
        ]

        # Convert to a dataframe
        pars = pd.DataFrame.from_dict(
            pars, 
            orient = 'index', 
            columns = ["name", "value", "unit", "stdev"])
        pars['subject'] = scan['subject']
        pars['kidney'] = kidney
        pars['visit'] = scan['visit']
        pars['parameter'] = pars.index
        pars['B0'] = scan['field_strength']

        # Append to results
        results.append(pars)

# Combine all results into a single dataframe
results = pd.concat(results).reset_index(drop=True)


# %%
# Plot MRI values and reference values

# Validation
v1T = pd.pivot_table(results[results.B0==1], values='value', columns='parameter', index=['subject','kidney','visit'])
v3T = pd.pivot_table(results[results.B0==3], values='value', columns='parameter', index=['subject','kidney','visit'])

iso1T, iso3T = 60*v1T['iso-SK-GFR'].values, 60*v3T['iso-SK-GFR'].values
mri1T, mri3T = 60*v1T['SK-GFR'].values, 60*v3T['SK-GFR'].values

plt.title('Single-kidney GFR (SK-GFR)')
plt.plot(iso1T, mri1T, 'bo', linestyle='None', markersize=4, label='1T')
plt.plot(iso3T, mri3T, 'ro', linestyle='None', markersize=4, label='3T')
plt.plot(iso3T, iso3T, linestyle='-', color='black')
plt.ylabel("MRI SK-GFR (mL/min)")
plt.xlabel("Isotope SK-GFR (mL/min)")
#plt.xlim(0,100)
#plt.ylim(0,200)
plt.legend()
plt.show()


# %%
# Compute bias and accuracy

v = pd.pivot_table(results, values='value', columns='parameter', index=['subject','kidney','visit'])

iso = 60*v['iso-SK-GFR'].values
mri = 60*v['SK-GFR'].values

diff = mri-iso
bias = np.mean(diff)
err =  1.96*np.std(diff)
bias_err = 1.96*np.std(diff)/np.sqrt(np.size(diff))

print('-----------------')
print('Single-kidney GFR')
print('-----------------')
print('95% CI on the bias (ml/min): ', bias-bias_err, bias+bias_err) # paper 0.56
print('95% CI on individual error (ml/min): ', bias-err, bias+err) # paper [-28, 29]

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

