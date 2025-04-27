"""
=====================================================================
Clinical - rifampicin effect in subjects with impaired liver function
=====================================================================

The data show in this example aimed to demonstrates the effect of rifampicin 
on liver function of patients with impaired function. The use 
case is provided by the liver work package of the 
`TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging 
biomarkers for drug safety assessment. 

The data were acquired in the aorta and liver in 3 patients with 
dynamic gadoxetate-enhanced MRI. The study participants take rifampicin 
as part of their routine clinical workup, with an aim to promote their liver 
function. For this study, they were taken off rifampicin 3 days before the 
first scan, and placed back on rifampicin 3 days before the second scan. The 
aim was to determine the effect if rifampicin in uptake and 
excretion function of the liver.

The data confirmed that patients had significantly reduced uptake and excretion 
function in the absence of rifampicin. Rifampicin adminstration promoted their 
excretory function but had no effect on their uptake function. 

Reference
--------- 

Manuscript in preparation..
"""

# %%
# Setup
# -----

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import pydmr
import dcmri as dc

# Fetch the data from the TRISTAN rifampicin study:
dmrfile = dc.fetch('tristan_humans_patients_rifampicin')
data = pydmr.read(dmrfile, 'nest')
rois, pars = data['rois'], data['pars']

# %%
# Model definition
# ----------------
# In order to avoid some repetition in this script, we define a function that 
# returns a trained model for a single dataset with 2 scans:

def tristan_human_2scan(roi, par, **kwargs):

    model = dc.AortaLiver2scan(

        # Injection parameters
        weight = par['weight'],
        agent = 'gadoxetate',
        dose = par['dose_1'],
        dose2 = par['dose_2'],
        rate = 1,

        # Acquisition parameters
        field_strength = 3,
        t0 = par['t0'],
        TR = par['TR'],
        FA = par['FA_1'],
        FA2 = par['FA_2'],
        TS = roi['time_1'][1]-roi['time_1'][0],

        # Signal parameters
        R10a = 1/par['T1_aorta_1'],
        R10l = 1/par['T1_liver_1'],
        R102a = 1/par['T1_aorta_3'],
        R102l = 1/par['T1_liver_3'],

        # Tissue parameters
        vol = par['liver_volume'],
    )

    xdata = (
        roi['time_1'][roi['aorta_1_accept']] - roi['time_1'][0], 
        roi['time_2'][roi['aorta_2_accept']] - roi['time_1'][0], 
        roi['time_1'][roi['liver_1_accept']] - roi['time_1'][0],
        roi['time_2'][roi['liver_2_accept']] - roi['time_1'][0],
    )
    ydata = (
        roi['aorta_1'][roi['aorta_1_accept']], 
        roi['aorta_2'][roi['aorta_2_accept']], 
        roi['liver_1'][roi['liver_1_accept']],
        roi['liver_2'][roi['liver_2_accept']],
    )
    
    model.train(xdata, ydata, **kwargs)

    return xdata, ydata, model

# %%
# Before running the full analysis on all cases, lets illustrate the results 
# by fitting the baseline visit for the first subject. We use maximum 
# verbosity to get some feedback about the iterations: 

xdata, ydata, model = tristan_human_2scan(
    rois['001']['control'], 
    pars['001']['control'],
    xtol=1e-3, 
    verbose=2,
)

# %%
# Plot the results to check that the model has fitted the data. The plot also 
# shows the concentration in the two liver compartments separately:

model.plot(xdata, ydata)

# %%
# Print the measured model parameters and any derived parameters. Standard 
# deviations are included as a measure of parameter uncertainty, indicate that 
# all parameters are identified robustly:

model.print_params(round_to=3)

# %%
# Fit all data
# ------------
# Now that we have illustrated an individual result in some detail, we 
# proceed with fitting the data for all 3 patients, at baseline and 
# rifampicin visit. We do not print output for these individual computations 
# and instead store results in one single dataframe:

results = []

# Loop over all datasets
for subj in rois.keys():
    for visit in rois[subj].keys():

        roi = rois[subj][visit]
        par = pars[subj][visit]

        # Generate a trained model for the scan:
        _, _, model = tristan_human_2scan(roi, par, xtol=1e-3)

        # Export fitted parameters as lists
        rows = model.export_params(type='list')

        # Add visit and subject info
        rows = [row + [visit, subj] for row in rows]

        # Add to the list of all results
        results += rows

# Combine all results into a single dataframe.
cols = ['parameter', 'name', 'value', 'unit', 'stdev',
        'visit', 'subject']
results = pd.DataFrame(results, columns=cols)

# Print all results
print(results.to_string())


# %%
# Plot individual results
# -----------------------
# Now lets visualise the main results from the study by plotting the drug 
# effect for all volunteers, and for both biomarkers: uptake rate ``khe`` 
# and excretion rate ``kbh``:

# Set up the figure
clr = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
fs = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
fig.subplots_adjust(wspace=0.5)
ax1.set_title('Hepatocellular uptake rate', fontsize=fs, pad=10)
ax1.set_ylabel('khe (mL/min/100mL)', fontsize=fs)
ax1.set_ylim(0, 60)
ax1.tick_params(axis='x', labelsize=fs)
ax1.tick_params(axis='y', labelsize=fs)
ax2.set_title('Biliary excretion rate', fontsize=fs, pad=10)
ax2.set_ylabel('kbh (mL/min/100mL)', fontsize=fs)
ax2.set_ylim(0, 10)
ax2.tick_params(axis='x', labelsize=fs)
ax2.tick_params(axis='y', labelsize=fs)

# Pivot data for both visits to wide format for easy access:
v1 = pd.pivot_table(results[results.visit=='control'], values='value', 
                    columns='parameter', index='subject')
v2 = pd.pivot_table(results[results.visit=='drug'], values='value', 
                    columns='parameter', index='subject')

# Plot the rate constants in units of mL/min/100mL
for s in v1.index:
    x = ['control']
    khe = [6000*v1.at[s,'khe']]
    kbh = [6000*v1.at[s,'kbh']] 
    if s in v2.index:
        x += ['drug']
        khe += [6000*v2.at[s,'khe']]
        kbh += [6000*v2.at[s,'kbh']] 
    color = clr[int(s)-1]
    ax1.plot(x, khe, '-', label=s, marker='o', markersize=6, color=color)
    ax2.plot(x, kbh, '-', label=s, marker='o', markersize=6, color=color)
plt.show()

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
