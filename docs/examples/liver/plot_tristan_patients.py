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
import dcmri as dc

# Fetch the data from the TRISTAN rifampicin study:
data = dc.fetch('tristan_gothenburg')

# %%
# Model definition
# ----------------
# In order to avoid some repetition in this script, we define a function that 
# returns a trained model for a single dataset with 2 scans:

def tristan_human_2scan(data, **kwargs):

    model = dc.AortaLiver2scan(

        # Injection parameters
        weight = data['weight'],
        agent = data['agent'],
        dose = data['dose'][0],
        dose2 = data['dose'][1],
        rate = data['rate'],

        # Acquisition parameters
        field_strength = data['field_strength'],
        t0 = data['t0'],
        TR = data['TR'],
        FA = data['FA'],

        # Signal parameters
        R10a = data['R10b'],
        R102a = data['R102b'],
        R10l = data['R10l'],
        R102l = data['R102l'],

        # Tissue parameters
        H = data['Hct'],
        vol = data['vol'],
    )

    xdata = (
        data['time1aorta'], 
        data['time2aorta'], 
        data['time1liver'], 
        data['time2liver'],
    )
    ydata = (
        data['signal1aorta'], 
        data['signal2aorta'], 
        data['signal1liver'], 
        data['signal2liver'],
    )
    
    model.train(xdata, ydata, **kwargs)

    return xdata, ydata, model

# %%
# Before running the full analysis on all cases, lets illustrate the results 
# by fitting the baseline visit for the first subject. We use maximum 
# verbosity to get some feedback about the iterations: 

xdata, ydata, model = tristan_human_2scan(data[0], xtol=1e-3, verbose=2)

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
for scan in data:

    # Generate a trained model for the scan:
    _, _, model = tristan_human_2scan(scan, xtol=1e-3)

    # Convert the parameter dictionary to a dataframe
    pars = model.export_params()
    pars = pd.DataFrame.from_dict(pars, 
        orient = 'index', 
        columns = ["name", "value", "unit", 'stdev'])
    pars['parameter'] = pars.index
    pars['visit'] = scan['visit']
    pars['subject'] = scan['subject']

    # Add the dataframe to the list of results
    results.append(pars)

# Combine all results into a single dataframe.
results = pd.concat(results).reset_index(drop=True)

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
