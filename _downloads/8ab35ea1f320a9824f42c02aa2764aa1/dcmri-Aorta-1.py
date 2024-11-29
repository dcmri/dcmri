import matplotlib.pyplot as plt
import dcmri as dc
#
# Use `fake_tissue` to generate synthetic test data from experimentally-derived concentrations:
#
time, aif, _, gt = dc.fake_tissue()
#
# Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_):
#
aorta = dc.Aorta(
    dt = 1.5,
    weight = 70,
    agent = 'gadodiamide',
    dose = 0.2,
    rate = 3,
    field_strength = 3.0,
    TR = 0.005,
    FA = 15,
    R10 = 1/dc.T1(3.0,'blood'),
)
#
# Train the model on the data:
#
aorta.train(time, aif)
#
# Plot the reconstructed signals and concentrations and compare against the experimentally derived data:
#
aorta.plot(time, aif)
#
# We can also have a look at the model parameters after training:
#
aorta.print_params(round_to=3)
# Expected:
## -----------------------------------------
## Free parameters with their errors (stdev)
## -----------------------------------------
## Bolus arrival time (BAT): 18.485 (5.656) sec
## Cardiac output (CO): 228.237 (29.321) mL/sec
## Heart-lung mean transit time (Thl): 9.295 (6.779) sec
## Heart-lung transit time dispersion (Dhl): 0.459 (0.177)
## Organs mean transit time (To): 29.225 (11.646) sec
## Extraction fraction (Eb): 0.013 (0.972)
## Organs extraction fraction (Eo): 0.229 (0.582)
## Extracellular mean transit time (Te): 97.626 (640.454) sec
## ------------------
## Derived parameters
## ------------------
## Mean circulation time (Tc): 38.521 sec
