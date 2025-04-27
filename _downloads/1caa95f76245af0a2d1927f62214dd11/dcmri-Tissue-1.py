import dcmri as dc
#
# Use `fake_tissue` to generate synthetic test data:
#
time, aif, roi, gt = dc.fake_tissue(CNR=50)
#
# Build a tissue and set the parameters to match the experimental
# conditions of the synthetic data:
#
tissue = dc.Tissue(
    aif = aif,
    dt = time[1],
    r1 = dc.relaxivity(3, 'blood','gadodiamide'),
    TR = 0.005,
    FA = 15,
    n0 = 15,
)
#
# Train the tissue on the data:
#
tissue.train(time, roi)
#
# Print the optimized tissue parameters, their standard deviations and
# any derived parameters:
#
tissue.print_params(round_to=2)
# Expected:
## <BLANKLINE>
## --------------------------------
## Free parameters with their stdev
## --------------------------------
## <BLANKLINE>
## Blood volume (vb): 0.03 (0.0) mL/cm3
## Interstitial volume (vi): 0.2 (0.0) mL/cm3
## Permeability-surface area product (PS): 0.0 (0.0) mL/sec/cm3
## <BLANKLINE>
## ----------------------------
## Fixed and derived parameters
## ----------------------------
## <BLANKLINE>
## Tissue Hematocrit (H): 0.45
## Plasma volume (vp): 0.02 mL/cm3
## Interstitial mean transit time (Ti): 58.92 sec
## B1-corrected Flip Angle (FAcorr): 15 deg
#
# Plot the fit to the data and the reconstructed concentrations, using
# the noise-free ground truth as reference:
#
tissue.plot(time, roi, ref=gt)
