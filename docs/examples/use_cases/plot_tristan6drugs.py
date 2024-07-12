"""
============================
The TRISTAN 6 compound study
============================

This example illustrates the use of `~dcmri.Liver` for fitting of signals measured in liver. The use case is provided by the liver work package of the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging biomarkers for drug safety assessment. The data and analysis were first published in Melillo et al (2023). 

The specific objective of the study was to determine the effect of selected drugs on hepatocellular uptake and excretion of the liver-specific contrast agent gadoxetate. If a drug inhibits uptake into liver cells, then it might cause other drugs to circulate in the blood stream for longer than expected, potentially causing harm to other organs. Alternatively, if a drug inhibits excretion from the liver, then it might cause other drugs to pool in liver cells for much longer than expected, potentially causing liver injury. These so-called drug-drug interactions (DDI's) pose a significant risk to patients and trial participants. A direct in-vivo measurement of drug effects on liver uptake and excretion can potentially help improve predictions of DDI's and inform dose setting strategies to reduce the risk.

The study presented here measured gadoxetate uptake and excretion in healthy rats before and after injection of 6 test drugs. Studies were performed in preclinical MRI scanners at 3 different centers and 2 different field strengths. Results demonstrated that two of the tested drugs (rifampicin and cyclosporine) showed strong inhibition of both uptake and excretion. One drug (ketoconazole) inhibited uptake but not excretion. Three drugs (pioglitazone, bosentan and asunaprevir) inhibited excretion but not uptake. 

**Reference**

Melillo N, Scotcher D, Kenna JG, Green C, Hines CDG, Laitinen I, Hockings PD, Ogungbenro K, Gunwhy ER, Sourbron S, et al. Use of In Vivo Imaging and Physiologically-Based Kinetic Modelling to Predict Hepatic Transporter Mediated Drug–Drug Interactions in Rats. Pharmaceutics. 2023; 15(3):896. `[DOI] <https://doi.org/10.3390/pharmaceutics15030896>`_ 
"""

# %%
# Setup
# -----

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc

# Fetch the data
data = dc.fetch('tristan6drugs')


# %%
# Model definition
# ----------------
# In order to avoid some repetition in this script, we define a function that returns a trained model for a single dataset. 
# 
# The model uses a standardized, population-average input function and fits for only 2 parameters, fixing all other free parameters to typical values for this rat model:

def tristan_rat(data, **kwargs):

    # High-resolution time points for prediction
    t = np.arange(0, np.amax(data['time']), 0.5)

    # Liver model with population input function
    model = dc.Liver(

        # Input parameters
        t = t,
        ca = dc.aif_tristan_rat(t, BAT=data['BAT'], duration=data['duration']),

        # Acquisition parameters
        field_strength = data['field_strength'],
        TR = data['TR'],
        FA = data['FA'],
        n0 = data['n0'],

        # Kinetic paramaters
        kinetics = 'IC-HF',
        Hct = 0.418,
        ve = 0.23,
        free = ['khe', 'Th'],
        bounds = [0, np.inf],

        # Tissue paramaters
        R10 = 1/dc.T1(data['field_strength'], 'liver'),
    )
    return model.train(data['time'], data['liver'], **kwargs)


# %%
# Check model fit
# ---------------
# Before running the full analysis on all cases, lets illustrate the results by fitting the baseline visit for the first subject. We use maximum verbosity to get some feedback about the iterations: 

model = tristan_rat(data[0], xtol=1e-3, verbose=2)

# %%
# Plot the results to check that the model has fitted the data:

model.plot(data[0]['time'], data[0]['liver'])

# %%
# Print the measured model parameters and any derived parameters and check that standard deviations of measured parameters are small relative to the value, indicating that the parameters are measured reliably:

model.print_params(round_to=3)


# %%
# Fit all data
# ------------
# Now that we have illustrated an individual result in some detail, we proceed with fitting all the data. Results are stored in a dataframe in long format:

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
# Now lets visualise the main results from the study by plotting the drug effect for all rats, and for both biomarkers: uptake rate ``khe`` and excretion rate ``kbh``:

# Set up the figure
clr = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
fs = 10
fig, ax = plt.subplots(2, 6, figsize=(6*1.5, 8))
fig.subplots_adjust(wspace=0.2, hspace=0.1)

# Loop over all studies
studies = [5,10,8,7,6,12]
drugs = ['Asunaprevir','Bosentan','Cyclosporine','Ketoconazole','Pioglitazone','Rifampicin']
for i, s in enumerate(studies):

    # Set up subfigures for the study
    ax[0,i].set_title(drugs[i], fontsize=fs, pad=10)
    ax[0,i].set_ylim(0, 300)
    ax[0,i].set_xticklabels([])
    ax[1,i].set_ylim(0, 30)
    ax[1,i].set_xticklabels([])
    if i==0:
        ax[0,i].set_ylabel('khe (mL/min/100mL)', fontsize=fs)
        ax[0,i].tick_params(axis='y', labelsize=fs)
        ax[1,i].set_ylabel('kbh (mL/min/100mL)', fontsize=fs)
        ax[1,i].tick_params(axis='y', labelsize=fs)
    else:
        ax[0,i].set_yticklabels([])
        ax[1,i].set_yticklabels([])

    # Pivot data for both visits of the study for easy access:
    study = results[results.study==s]
    v1 = pd.pivot_table(study[study.visit==1], values='value', columns='parameter', index='subject')
    v2 = pd.pivot_table(study[study.visit==2], values='value', columns='parameter', index='subject')

    # Plot the rate constants in units of mL/min/100mL
    for s in v1.index:
        x = [1]
        khe = [6000*v1.at[s,'khe']]
        kbh = [6000*v1.at[s,'kbh']] 
        if s in v2.index:
            x += [2]
            khe += [6000*v2.at[s,'khe']]
            kbh += [6000*v2.at[s,'kbh']] 
        color = clr[int(s)-1]
        ax[0,i].plot(x, khe, '-', label=s, marker='o', markersize=6, color=color)
        ax[1,i].plot(x, kbh, '-', label=s, marker='o', markersize=6, color=color)

plt.show()

# %%
# Plot effect sizes
# -----------------
# Now lets calculate the effect sizes (relative change) for each drug, along with 95% confidence interval, and show these in a plot. Results are presented in **red** if inhibition is more than 20% (i.e. upper value of the 95% CI is less than -20%), in **orange** if inhbition is less than 20% (i.e. upper value of the 95% CI is less than 0%), and in **green** if no inhibition was detected with 95% confidence (0% in the 95% CI):

# Set up figure
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 5))
fig.subplots_adjust(left=0.3, right=0.7, wspace=0.25)

ax0.set_title('khe effect (%)', fontsize=fs, pad=10)
ax1.set_title('kbh effect (%)', fontsize=fs, pad=10)
ax0.set_xlim(-100, 50)
ax1.set_xlim(-100, 50)
ax0.grid(which='major', axis='x', linestyle='-')
ax1.grid(which='major', axis='x', linestyle='-')
ax1.set_yticklabels([])

# Loop over all studies
for i, s in enumerate(studies):

    # Pivot data for both visits of the study for easy access:
    study = results[results.study==s]
    v1 = pd.pivot_table(study[study.visit==1], values='value', columns='parameter', index='subject')
    v2 = pd.pivot_table(study[study.visit==2], values='value', columns='parameter', index='subject')
    
    # Calculate effect size for the drug in %
    effect = 100*(v2-v1)/v1

    # Get descriptive statistics
    stats = effect.describe()

    # Calculate mean effect sizes and 59% CI on the mean.
    khe_eff = stats.at['mean','khe']
    kbh_eff = stats.at['mean','kbh']
    khe_eff_err = 1.96*stats.at['std','khe']/np.sqrt(stats.at['count','khe'])
    kbh_eff_err = 1.96*stats.at['std','kbh']/np.sqrt(stats.at['count','kbh'])

    # Plot mean effect size for khe along with 95% CI
    # Choose color based on magnitude of effect
    if khe_eff + khe_eff_err < -20:
        clr = 'tab:red'
    elif khe_eff + khe_eff_err < 0:
        clr = 'tab:orange'
    else:
        clr = 'tab:green'
    ax0.errorbar(khe_eff, drugs[i], xerr=khe_eff_err, fmt='o', color=clr)

    # Plot mean effect size for kbh along with 95% CI
    # Choose color based on magnitude of effect
    if kbh_eff + kbh_eff_err < -20:
        clr = 'tab:red'
    elif kbh_eff + kbh_eff_err < 0:
        clr = 'tab:orange'
    else:
        clr = 'tab:green'
    ax1.errorbar(kbh_eff, drugs[i], xerr=kbh_eff_err, fmt='o', color=clr)

# Plot dummy values out of range to show a legend
ax1.errorbar(-200, drugs[0], marker='o', color='tab:red', label='inhibition > 20%')
ax1.errorbar(-200, drugs[0], marker='o', color='tab:orange', label='inhibition')
ax1.errorbar(-200, drugs[0], marker='o', color='tab:green', label='no inhibition')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
