import matplotlib.pyplot as plt
import dcmri as dc
#
# Use `fake_tissue` to generate synthetic test data from
# experimentally-derived concentrations:
#
# Use `fake_liver` to generate synthetic test data:
#
time, aif, vif, roi, _ = dc.fake_liver(sequence='SSI')
#
# Since this model generates 3 time curves, the x- and y-data are
# tuples:
#
xdata, ydata = (time, time, time), (aif, vif, roi)
#
# Build an aorta-portal-liver model and parameters to match the
# conditions of the fake liver data:
#
model = dc.AortaPortalLiver(
    kinetics = '2I-IC',
    sequence = 'SSI',
    dt = 0.5,
    tmax = 180,
    weight = 70,
    agent = 'gadoxetate',
    dose = 0.2,
    rate = 3,
    field_strength = 3.0,
    t0 = 10,
    TR = 0.005,
    FA = 15,
    TS = 0.5,
)
#
# Train the model on the data:
#
model.train(xdata, ydata, xtol=1e-3)
#
# Plot the reconstructed signals and concentrations and compare
# against the experimentally derived data:
#
model.plot(xdata, ydata)
#
# We can also have a look at the model parameters after training:
#
model.print_params(round_to=3)
# Expected:
## --------------------------------
## Free parameters with their stdev
## --------------------------------
## First bolus arrival time (BAT): 14.616 (1.1) sec
## Cardiac output (CO): 100.09 (2.736) mL/sec
## Heart-lung mean transit time (Thl): 14.402 (1.375) sec
## Heart-lung dispersion (Dhl): 0.391 (0.013)
## Organs blood mean transit time (To): 27.811 (5.291) sec
## Organs extraction fraction (Eo): 0.29 (0.105)
## Organs extravascular mean transit time (Toe): 70.621 (102.614) sec
## Body extraction fraction (Eb): 0.013 (0.23)
## Aorta inflow time (TF): 0.409 (0.014) sec
## Liver extracellular volume fraction (ve): 0.479 (0.112) mL/cm3
## Liver plasma flow (Fp): 0.018 (0.001) mL/sec/cm3
## Arterial flow fraction (fa): 0.087 (0.074)
## Arterial transit time (Ta): 2.398 (1.356) sec
## Hepatocellular uptake rate (khe): 0.006 (0.003) mL/sec/cm3
## Hepatocellular mean transit time (Th): 683.604 (2554.75) sec
## Gut mean transit time (Tg): 10.782 (0.614) sec
## Gut dispersion (Dg): 0.893 (0.07)
## ----------------------------
## Fixed and derived parameters
## ----------------------------
## Hematocrit (H): 0.45
## Arterial venous blood flow (Fa): 0.002 mL/sec/cm3
## Portal venous blood flow (Fv): 0.016 mL/sec/cm3
## Extracellular mean transit time (Te): 27.216 sec
## Biliary tissue excretion rate (Kbh): 0.001 mL/sec/cm3
## Hepatocellular tissue uptake rate (Khe): 0.012 mL/sec/cm3
## Biliary excretion rate (kbh): 0.001 mL/sec/cm3
