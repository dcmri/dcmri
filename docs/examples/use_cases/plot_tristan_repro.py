"""
=================================
The TRISTAN reproducibility study
=================================

This example illustrates the use of `~dcmri.Liver` for fitting of signals 
measured in liver. The use case is provided by the liver work package of the 
`TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging 
biomarkers for drug safety assessment. The data and analysis were first 
published in Gunwhy et al. (2024). 

The specific objective of the study was to identify the main sources of
variability in DCE-MRI biomarkers of hepatocellular function in rats. This was
done by comparing data measured at different centres and field strengths, at
different days in the same subjects, and over the course of several months
in the same centre.

The study presented here measured gadoxetate uptake and excretion in healthy 
rats either scanned once with vehicle or twice with either vehicle or 10 mg/kg
of rifampicin at follow-up. Studies were performed in preclinical MRI scanners
at 3 different centers and 2 different field strengths. Results demonstrated
significant differences between substudies for uptake and excretion.
Within-subject differences were substantially smaller for excretion but less so
for uptake. Rifampicin-induced inhibition was safely above the detection limits
for both uptake and excretion. Most of the variability in individual data was
accounted for by between-subject and between-centre variability, substantially
more than the between-day variation. Significant differences in excretion were
observed between field strengths at the same centre, between centres at the same
field strength, and between repeat experiments over 2 months apart in the same
centre.

**Reference**

Gunwhy ER, Hines CDG, Green C, Laitinen I, Tadimalla S, Hockings PD, Schütz G,
Kenna JG, Sourbron S, Waterton JC. Assessment of hepatic transporter function
in rats using dynamic gadoxetate-enhanced MRI: a reproducibility study. MAGMA.
2024 Aug;37(4):697-708. `[DOI] <https://doi.org/10.1007/s10334-024-01192-5>`_
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
data = dc.fetch('tristan_repro')


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
        ca = ca*(1-0.418),

        # Acquisition parameters
        field_strength = data['field_strength'],
        agent = 'gadoxetate',
        TR = data['TR'],
        FA = data['FA'],
        n0 = data['n0'],

        # Kinetic paramaters
        kinetics = '1I-IC-HF',
        H = 0.418,
        ve = 0.23,
        Fp = 0.022019, # mL/sec/cm3
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
# Now let's calculate the average biomarker values per substudy for saline
# data only. For this exercise, let's specify khe and kbh as the biomarker
# parameters that we are interested in. For each biomarker, we can plot the
# avergae biomarker values along with their 95% confidence intervals for each
# study group. We can also calculate an average 'benchmark' value across all
# study groups for each biomarker, and overlay these on the graphs to see
# whether the observed values lie within these ranges.
# **red** lines indicate the average benchmark value, while **blue** represents
# the upper and lower limits of the 95% confidence intervals (CIs) associated
# with these benchmarks.

# Customise plot settings
plt.rcParams["axes.labelsize"] = 50
plt.rcParams["axes.titlesize"] = 50
plt.rcParams["axes.labelweight"] = 'bold'
plt.rcParams["axes.titleweight"] = 'bold'
plt.rcParams["font.weight"] = 'bold'
plt.rc('axes', linewidth=2)
plt.rc('xtick', labelsize=40)
plt.rc('ytick', labelsize=40)
plt.rcParams["lines.linewidth"] = 4
plt.rcParams['lines.markersize'] = 12

# Create list of biomarkers (parameters) of interest
params = ['khe', 'kbh']

# Extract data of interest, i.e., visit 1 data for parameters of interest
visitOneData = results.query('parameter in @params and visit==1')

# Get statistical summaries per parameter and study group
stat_summary = (visitOneData
                .groupby(['parameter', 'study'])['value']
                .agg(['mean']))

# Calculate benchmark values per parameter by averaging all study group averages
benchmarks = (stat_summary
              .groupby(['parameter'])['mean']
              .agg(['mean', 'sem']))

# Calculate the 95% confidence intervals for each parameter benchmark
benchmarks['CI95'] = (benchmarks['sem'].mul(1.96))

# Sort dataframes
visitOneData_sorted = visitOneData.sort_values(['parameter'], ascending=[False])
benchmarks_sorted = benchmarks.sort_values(['parameter'], ascending=[False])

# Plot distributions across all study groups per biomarker of interest
g = sns.catplot(data=visitOneData_sorted,
                x='study',
                y='value',
                col='parameter',
                kind='point',
                capsize=0.2,
                sharey=False,
                linestyle='none',
                height=14,
                aspect=1.2,
                color='k',
                errorbar=('ci', 95))

g.set_titles("") # set custom subplot titles

# Set limits for y-axes
g.axes[0, 0].set(ylim=([0, 0.05]))
g.axes[0, 1].set(ylim=([0, 0.006]))

ylabels = ['$k_{he}$', '$k_{bh}$'] # define labels for y-axis

# Assign values from benchmarks dataframe to be
# used as horizontal lines to overlay on plots
means = benchmarks_sorted['mean']
lower_cis = means - benchmarks_sorted['CI95']
upper_cis = means + benchmarks_sorted['CI95']

# iterate through subplots to overlay y-labels and axis lines
for i in range(len(ylabels)):
    g.axes[0, i].set_ylabel(f"{ylabels[i]} [mL/sec/cm3]")
    g.axes[0, i].axhline(means.iloc[i], color='blue', ls=':')
    g.axes[0, i].axhline(lower_cis.iloc[i], color='red', ls='--')
    g.axes[0, i].axhline(upper_cis.iloc[i], color='red', ls='--')

plt.tight_layout()
plt.show()
