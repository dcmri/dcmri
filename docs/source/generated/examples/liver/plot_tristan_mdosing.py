"""
=====================================================
Preclinical - repeat dosing effects on liver function
=====================================================

This example illustrates the use of `~dcmri.Liver` for fitting of signals 
measured in liver. The use case is provided by the liver work package of the 
`TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging 
biomarkers for drug safety assessment. The manuscript relating to this data
and analysis is currently in preparation. 

The specific objective of the study was to investigate the potential of
gadoxetate-enhanced DCE-MRI to study acute inhibition of hepatocyte
transporters of drug-induced liver injury (DILI) causing drugs, and to study
potential changes in transporter function after chronic dosing.

The study presented here measured gadoxetate uptake and excretion in healthy 
rats scanned after administration of vehicle and repetitive dosing regimes
of either Rifampicin, Cyclosporine, or Bosentan. Studies were performed in
preclinical MRI scanners at 3 different centers and 2 different field strengths.

**Reference**

Mikael Montelius, Steven Sourbron, Nicola Melillo, Daniel Scotcher, 
Aleksandra Galetin, Gunnar Schuetz, Claudia Green, Edvin Johansson, 
John C. Waterton, and Paul Hockings. Acute and chronic rifampicin effect on 
gadoxetate uptake in rats using gadoxetate DCE-MRI. Int Soc Mag Reson Med 
2021; 2674.
"""

# %%
# Setup
# -----

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dcmri as dc

# Fetch the data
data = dc.fetch('tristan_mdosing')


# %%
# Model definition
# ----------------
# In order to avoid some repetition in this script, we define a function that 
# returns a trained model for a single dataset. 
# 
# The model uses a standardized, population-average input function and fits 
# for only 2 parameters, fixing all other free parameters to typical values 
# for this rat model:

def tristan_rat(data, **kwargs):

    # High-resolution time points for prediction
    t = np.arange(0, np.amax(data['time'])+0.5, 0.5)

    # Standard input function
    ca = dc.aif_tristan_rat(t, BAT=data['BAT'], duration=data['duration'])

    # Liver model with population input function
    model = dc.Liver(

        # Input parameters
        t = t,
        ca = ca,

        # Acquisition parameters
        field_strength = data['field_strength'],
        TR = data['TR'],
        FA = data['FA'],
        n0 = data['n0'],

        # Kinetic paramaters
        kinetics = '1I-IC-HF',
        H = 0.418,
        ve = 0.23,
        free = {
            'khe': [0, np.inf], 
            'Th': [0, np.inf],
        },

        # Tissue paramaters
        R10 = 1/dc.T1(data['field_strength'], 'liver'),
    )

    return model.train(data['time'], data['liver'], **kwargs)


# %%
# Check model fit
# ---------------
# Before running the full analysis on all cases, lets illustrate the results 
# by fitting the baseline visit for the first subject. We use maximum 
# verbosity to get some feedback about the iterations: 

#model = tristan_rat(data[0], xtol=1e-3, verbose=2)
model = tristan_rat(data[0], xtol=1e-3, verbose=2)

# %%
# Plot the results to check that the model has fitted the data:

model.plot(data[0]['time'], data[0]['liver'])

# %%
# Print the measured model parameters and any derived parameters and check 
# that standard deviations of measured parameters are small relative to the 
# value, indicating that the parameters are measured reliably:

model.print_params(round_to=3)

# %%
# Fit all data
# ------------
# Now that we have illustrated an individual result in some detail, we proceed 
# with fitting all the data. Results are stored in a dataframe in long format:

results = []

# Loop over all datasets
for scan in data:

    # Generate a trained model for scan i:
    model = tristan_rat(scan, xtol=1e-3)
    
    # Save fitted parameters as a dataframe.
    pars = model.export_params()
    pars = pd.DataFrame.from_dict(pars, 
        orient = 'index', 
        columns = ["name", "value", "unit", 'stdev'])
    pars['parameter'] = pars.index
    pars['study'] = scan['study']
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
# Now let's plot the biomarker values across visits for each study group.
# For this exercise, let's specify khe and kbh as the biomarker parameters that
# we are interested in. For each subject, we can visualise the change in
# biomarker values between visits. For reference, in the below plots, the
# studies are numbered as follows:
# Study 1: Rifampicin repetitive dosing regime
# Study 2: Cyclosporine repetitive dosing regime
# Study 3: Bosentan repetitive dosing regime

# Customise plot settings
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.labelweight"] = 'bold'
plt.rcParams["axes.titleweight"] = 'bold'
plt.rcParams["font.weight"] = 'bold'
plt.rc('axes', linewidth=1.5)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams['lines.markersize'] = 2

# Extract results of interest, i.e., for parameters khe and kbh
filtered_data = results.query("parameter == 'khe' | parameter == 'kbh'")

# Plot distributions across visits per study groups and per biomarker
g = sns.catplot(data=filtered_data,
                x='visit',
                y='value',
                palette='rocket',
                hue='subject',
                row='parameter',
                col='study',
                kind='point',
                sharey=False)

g.set_titles(pad=15) # increase white space between subplots and titles

# Set limits for y-axes
for i in range(0, 3):
    g.axes[0, i].set(ylim=([0, 0.05]))

for i in range(0, 3):
    g.axes[1, i].set(ylim=([0, 0.005]))

g.set_ylabels("Value [mL/sec/cm3]") # set labels for y-axis

# reposition legend
sns.move_legend(g, "lower right", bbox_to_anchor=(0.95, 0.7))

plt.tight_layout()
plt.show()

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
