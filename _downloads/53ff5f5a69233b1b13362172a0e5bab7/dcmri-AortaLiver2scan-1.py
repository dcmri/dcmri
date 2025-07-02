import matplotlib.pyplot as plt
import dcmri as dc
#
# Use `fake_tissue` to generate synthetic test data from
# experimentally-derived concentrations:
#
time, aif, roi, gt = dc.fake_tissue2scan(R10=1/dc.T1(3.0,'liver'))
#
# Since this model generates four time curves, the x- and y-data are
# tuples:
#
xdata = (time[0], time[1], time[0], time[1])
ydata = (aif[0], aif[1], roi[0], roi[1])
#
# Build an aorta-liver model and parameters to match the conditions of
# the fake tissue data:
#
model = dc.AortaLiver2scan(
    dt = 0.5,
    weight = 70,
    agent = 'gadodiamide',
    dose = 0.2,
    dose2 = 0.2,
    rate = 3,
    field_strength = 3.0,
    t0 = 10,
    TR = 0.005,
    FA = 15,
    FA2 = 15,
)
#
# Train the model on the data:
#
model.train(xdata, ydata, xtol=1e-3)
#
# Plot the reconstructed signals and concentrations and compare against
# the experimentally derived data:
#
model.plot(xdata, ydata)
#
# We can also have a look at the model parameters after training:
#
model.print_params(round_to=3)
# Expected:
## -----------------------------------------
## Free parameters with their errors (stdev)
## -----------------------------------------
## Bolus arrival time (BAT): 17.13 (1.771) sec
## Cardiac output (CO): 208.547 (9.409) mL/sec
## Heart-lung mean transit time (Thl): 12.406 (2.137) sec
## Heart-lung transit time dispersion (Dhl): 0.459 (0.04)
## Organs mean transit time (To): 30.912 (3.999) sec
## Extraction fraction (Eb): 0.064 (0.032)
## Liver extracellular mean transit time (Te): 2.957 (0.452) sec
## Liver extracellular dispersion (De): 1.0 (0.146)
## Liver extracellular volume fraction (ve): 0.077 (0.007) mL/cm3
## Hepatocellular uptake rate (khe): 0.002 (0.001) mL/sec/cm3
## Hepatocellular transit time (Th): 600.0 (1173.571) sec
## Organs extraction fraction (Eo): 0.2 (0.057)
## Organs extracellular mean transit time (Toe): 87.077 (56.882) sec
## Hepatocellular uptake rate (final) (khe_f): 0.001 (0.001) mL/sec/cm3
## Hepatocellular transit time (final) (Th_f): 600.0 (623.364) sec
## ------------------
## Derived parameters
## ------------------
## Blood precontrast T1 (T10a): 1.629 sec
## Mean circulation time (Tc): 43.318 sec
## Liver precontrast T1 (T10l): 0.752 sec
## Biliary excretion rate (kbh): 0.002 mL/sec/cm3
## Hepatocellular tissue uptake rate (Khe): 0.023 mL/sec/cm3
## Biliary tissue excretion rate (Kbh): 0.002 mL/sec/cm3
## Hepatocellular uptake rate (initial) (khe_i): 0.003 mL/sec/cm3
## Hepatocellular transit time (initial) (Th_i): 600.0 sec
## Hepatocellular uptake rate variance (khe_var): 0.934
## Biliary tissue excretion rate variance (Kbh_var): 0.0
## Biliary excretion rate (initial) (kbh_i): 0.002 mL/sec/cm3
## Biliary excretion rate (final) (kbh_f): 0.002 mL/sec/cm3
