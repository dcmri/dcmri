import numpy as np
import pydmr
import dcmri as dc
#
# Read the dataset:
#
datafile = dc.fetch('minipig_renal_fibrosis')
data = pydmr.read(datafile, 'nest')
rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']
#
# Initialize the tissue:
#
aorta = dc.Aorta(
    sequence='SSI',
    heartlung='chain',
    organs='comp',
    field_strength=pars['B0'],
    t0=15,
    agent="gadoterate",
    weight=pars['weight'],
    dose=pars['dose'],
    rate=pars['rate'],
    TR=pars['TR'],
    FA=pars['FA'],
    TS=pars['TS'],
    CO=60,
    R10=1/dc.T1(pars['B0'], 'blood'),
)
#
# Create an array of time points:
#
time = pars['TS'] * np.arange(len(rois['Aorta']))
#
# Train the system to the data:
#
aorta.train(time, rois['Aorta'])
#
# Plot the reconstructed signals and concentrations:
#
aorta.plot(time, rois['Aorta'])
#
# Print the model parameters:
#
aorta.print_params(round_to=4)
