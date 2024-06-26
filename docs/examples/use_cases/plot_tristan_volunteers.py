"""
=========================================================
The TRISTAN experimental medicine study (1 scan protocol)
=========================================================

This example illustrates the use of `~dcmri.AortaLiver` for joint fitting of aorta and liver signals to a whole-body model. The use case is provided by the liver work package of the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging biomarkers for drug safety assessment. The data and analysis was first presented at the ISMRM in 2024 (Min et al 2024, manuscript in press). 

The data were acquired in the aorta and liver of 10 healthy volunteers with dynamic gadoxetate-enhanced MRI, before and after administration of a drug (rifampicin) which is known to inhibit liver function. The assessments were done on two separate visits at least 2 weeks apart. 

The research question was to what extent rifampicin inhibits gadoxetate uptake rate from the extracellular space into the liver hepatocytes (khe, mL/min/100mL) and excretion rate from hepatocytes to bile (kbh, mL/100mL/min). 

2 of the volunteers only had the baseline assessment, the other 8 volunteers completed the full study. The results showed consistent and strong inhibition of khe (95%) and kbh (40%) by rifampicin. This implies that rifampicin poses a risk of drug-drug interactions (DDI), meaning it can cause another drug to circulate in the body for far longer than expected, potentially causing harm or raising a need for dose adjustment.

**Note**: this example is different to the 2 scan example of the same study in that this uses only the first scan to fit the model. 

Reference
--------- 

Thazin Min, Marta Tibiletti, Paul Hockings, Aleksandra Galetin, Ebony Gunwhy, Gerry Kenna, Nicola Melillo, Geoff JM Parker, Gunnar Schuetz, Daniel Scotcher, John Waterton, Ian Rowe, and Steven Sourbron. *Measurement of liver function with dynamic gadoxetate-enhanced MRI: a validation study in healthy volunteers*. Proc Intl Soc Mag Reson Med, Singapore 2024.
"""

# %%
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc

# %%
# Fetch the 1-scan data from the TRISTAN experimental medicine study:

data = dc.fetch('tristan1scan')

# %%
# Before running the full analysis on all cases, lets illustrate the results by fitting the baseline visit for the first subject. We use maximum verbosity to get some feedback about the iterations: 

data_subj = data['baseline']['001']
model = dc.AortaLiver(**data_subj['params'])
model.train(data_subj['xdata'], data_subj['ydata'], xtol=1e-3, verbose=2)

# %%
# Plot the results to check that the model has fitted the data. The plot also shows the concentration in the two liver compartments separately:

model.plot(data_subj['xdata'], data_subj['ydata'])

# %%
# Print the measured model parameters and any derived parameters. Standard deviations are included as a measure of parameter uncertainty, indicate that all parameters are identified robustly:

model.print_params(round_to=3)


# %%
# Now that we have illustrated an individual result in some detail, we proceed with fitting the data for all 10 volunteers, at baseline and rifampicin visit. We do not print output for these individual computations and instead store results in one single dataframe:

results = []
for visit in data:
    for subj in data[visit]:

        # Get the data for the subject and visit
        data_subj = data[visit][subj]

        # Use ``dcmri`` to fit the model and export the parameters:
        model = dc.AortaLiver(**data_subj['params'])
        model.train(data_subj['xdata'], data_subj['ydata'], xtol=1e-3)
        pars_subj = model.export_params()

        # Convert the parameter dictionary to a dataframe
        pars_subj = pd.DataFrame.from_dict(pars_subj, 
            orient = 'index', columns = ["name", "value", "unit", 'stdev'])
        pars_subj['subject'] = subj
        pars_subj['visit'] = visit
        pars_subj['parameter'] = pars_subj.index

        # Add the dataframe to the list of results
        results.append(pars_subj)

# Combine all results into a single dataframe.
results = pd.concat(results).reset_index(drop=True)

# Print all results
print(results.to_string())



# %%
# Now lets visualise the main results from the study by plotting the drug effect for all volunteers, and for both biomarkers: uptake rate ``khe`` and excretion rate ``kbh``:

# First pivot data for both visits to wide format for easy access:
v1 = pd.pivot_table(results[results.visit=='baseline'], values='value', columns='parameter', index='subject')
v2 = pd.pivot_table(results[results.visit=='rifampicin'], values='value', columns='parameter', index='subject')

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
ax2.set_ylim(0, 6)
ax2.tick_params(axis='x', labelsize=fs)
ax2.tick_params(axis='y', labelsize=fs)

# Plot the rate constants in units of mL/min/100mL
for s in v1.index:
    x = ['baseline']
    khe = [6000*v1.at[s,'khe']]
    kbh = [6000*v1.at[s,'kbh']] 
    if s in v2.index:
        x += ['rifampicin']
        khe += [6000*v2.at[s,'khe']]
        kbh += [6000*v2.at[s,'kbh']] 
    color = clr[int(s)-1]
    ax1.plot(x, khe, '-', label=s, marker='o', markersize=6, color=color)
    ax2.plot(x, kbh, '-', label=s, marker='o', markersize=6, color=color)
plt.show()

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
