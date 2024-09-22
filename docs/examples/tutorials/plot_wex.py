"""
========================================
An exploration of water exchange effects
========================================

This tutorial will explore the effect of water exchange in a standard two-compartmental tissue with extended Tofts kinetics. Data acquisition is modelled using the steady-state model of a spoiled gradient echo sequence.

The three tissue compartments involved are the blood, interstitium and tissue cells. The effect of water exchange between blood and interstitium (**transendothelial** water exchange) and between interstitium and tissue cells (**transcytolemmal** water exchange) is separately investigated. The water exchange in the blood compartment, particularly between plasma and red blood cells, is assumed to be in the fast exhange limit throughout. 
"""

# %%
# Simulation setup
# ----------------
# First we set up the simulation by importing the necessary packages and defining the constants that will be fixed throughout. 

# %%
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc

# The constants defining the signal model and simulation settings
const = {
    'TR': 0.005,                # Repetition time (sec)
    'FA': 15.0,                 # Flip angle (deg)
    'agent': 'gadoxetate',      # Contrast agent
    'field_strength': 3.0,      # Magnetic field strength (T)
    'R10': 1/dc.T1(3.0, 'muscle'),
    'R10b': 1/dc.T1(3.0, 'blood'), 
}

tacq, aif, _, _ = dc.fake_tissue(tacq=300, **const)

# The ground-truth kinetic parameters of the extended Tofts model
ptruth = {
    'S0':1,             # arbitrary units
    'vp':0.05,          # mL/cm3
    'Ktrans':0.1/60,    # mL/sec/cm3
    'vi':0.30,          # mL/cm3
} 
truth = np.array(list(ptruth.values()))

# %% 
# Visualising water exchange effects
# ----------------------------------
# We'll start by exploring how the level of water exchange affects the measured signal. We will predict signals using explicit models in the limiting scenarios of fast water exchange (F) and no water exchange (N):

# %% 

# The fast water exchange limit (all barriers fully transparent to water) - this is the default:
ffx = dc.Tissue(aif=aif, t=tacq, **(const | ptruth)).predict(tacq)

# The no water exchange limit (NN, both barriers impermeable to water). 
nnx = dc.Tissue(aif=aif, t=tacq, water_exchange='NN', **(const | ptruth)).predict(tacq)

# A mixed case, with no transendothelial water exchange (N), but fast transcytolemmal water exchange (F):
nfx = dc.Tissue(aif=aif, t=tacq, water_exchange='NF', **(const | ptruth)).predict(tacq) 

# The other mixed case (FN) with fast transendothelial water exchange (F), but no transcytolemmal water exchange (N):
fnx = dc.Tissue(aif=aif, t=tacq, water_exchange='FN', **(const | ptruth)).predict(tacq)

# %%
# Now we turn to the intermediate case of restricted water exchange (R), where the water exchange is neither infinite or zero. A choice of ``PSe = 1`` mL/sec/cm3 and ``PSc = 2`` mL/sec/cm3 produces a curve that lies in between the extremes:

# %%

# Signal with restricted (R) transendothelial and transcytolemmal water exchange (RR):
iix = dc.Tissue(aif=aif, t=tacq, water_exchange='RR', PSe=1, PSc=2, **(const | ptruth)).predict(tacq)

# %%
# We now plot the different results, using fast- and no-exchange limits for visual reference:

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(15,5))

ax0.set_title('No transendothelial exchange \n Fast transcytolemmal exchange')
ax0.plot(tacq, ffx, 'g-', label='Fast exchange')
ax0.plot(tacq, nnx, 'b-', label='No exchange')
ax0.plot(tacq, nfx, 'r--', label='Intermediate exchange')
ax0.set_xlabel('Time (sec)')
ax0.set_ylabel('Signal (a.u.)')
ax0.legend()

ax1.set_title('Fast transendothelial exchange \n No transcytolemmal exchange')
ax1.plot(tacq, ffx, 'g-', label='Fast exchange')
ax1.plot(tacq, nnx, 'b-', label='No exchange')
ax1.plot(tacq, fnx, 'r--', label='Intermediate exchange')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Signal (a.u.)')
ax1.legend()

ax2.set_title('Intermediate transendothelial exchange \n Intermediate transcytolemmal exchange')
ax2.plot(tacq, ffx, 'g-', label='Fast exchange')
ax2.plot(tacq, nnx, 'b-', label='No exchange')
ax2.plot(tacq, iix, 'r--', label='Intermediate exchange')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Signal (a.u.)')
ax2.legend()

plt.show()

# %%
# These figures show the expected observations: 
#
# 1. Water exchange levels have a measureable effect on signals, as shown by the clear difference between fast and slow exchange scenarios (blue vs. green curves). 
#
# 2. With fast transcytolemmal exchange but impermeable endothelium (left panel), the slowy changing extravascular part of the signal aligns with the fast exchange curve, and the first pass aligns with the no-exchange curve  
#
# 3. Without transcytolemmal exchange the extravascular curve lines up with the no-exchange model (middle panel). The first pass is closer to the fast-exchange signal but does not align with it completely as it is partly obscured by already extravasated indicator.
#
# 4. When both exchange levels are intermediate (right panel), then the signal is also intermediate between the extremes of fast and no exchange.
#
# **Note** while the effect of water exchange is detectable, it is comparatively small considering the difference between the blue and green curves represent the extremes of zero to maximal levels of water exchange. It is easily verified that changing kinetic parameters such as Ktrans over their entire range (zero to infinity) has a much larger impact on the signal. Water exchange is in that sense a second order effect.


# %% 
# Understanding water exchange bias
# ---------------------------------
# Since the level of water exchange affects the signal, making inaccurate assumptions on the level of water exchange will create a bias in any measurement of the kinetic parameters. 
#
# One way to explore the scale of the water exchange bias is by generating data for a tissue in the fast exchange limit and analysing them making the opposite assumption that water exchange is negligible:

# Launch a no-exchange model
model = dc.Tissue(aif=aif, t=tacq, water_exchange='NN', **const)

# To check the effect of training, first predict the signal using the untrained model
nnx0 = model.predict(tacq)

# Fit the model to the fast-exchange tissue
model.train(tacq, ffx)

# Predict the signal using the trained model
nnx1 = model.predict(tacq)

# Calculate the bias in the fitted parameters in %
pars = model.get_params('S0','vp','Ktrans','vi')
bias = 100*(np.array(pars)-truth)/truth

# Plot the model fits
fig, ax0 = plt.subplots(1,1,figsize=(6,5))
ax0.set_title('Water exchange bias')
ax0.plot(tacq, ffx, 'g-', linewidth=3, label='Signal data (fast exchange tissue)')
ax0.plot(tacq, nnx0, 'b-', label='Prediction (before training)')
ax0.plot(tacq, nnx1, 'b--', label='Prediction (after training)')
ax0.set_xlabel('Time (sec)')
ax0.set_ylabel('Signal (a.u.)')
ax0.legend()
plt.show()

# Print the parameter bias
print('')
print('Bias in kinetic model parameters')
print('--------------------------------')
print('vp error:', round(bias[1],1), '%')
print('vi error:', round(bias[3],1), '%')
print('Ktrans error:', round(bias[2],1), '%')

# %%
# The plot shows that the trained model predicts the data with high accuracy, despite the inaccurate assumption of no water exchange. However the false assumption does lead to fitted parameters that are severely biased.

# %% 
# Removing water exchange bias
# ----------------------------
# The model bias can be removed by generalizing the model to allow for any level of water exchange, avoiding the risk of making a false assumption on this point:

# Launch a general water exchange model with default settings for all free parameters
model = dc.Tissue(aif=aif, t=tacq, water_exchange='RR', **const)

# Predict the signal using the untrained model as a reference
iix0 = model.predict(tacq)

# Fit the model to the fast-exchange data and predict the signal again 
# (we reduce here the x-tolerance to speed up convergence). 
iix1 = model.train(tacq, ffx, xtol=1e-4).predict(tacq)

# Calculate the bias in the fitted parameters
pars = model.get_params('S0','vp','Ktrans','vi')
bias = 100*(np.array(pars)-truth)/truth

# Plot the model fits
fig, ax0 = plt.subplots(1,1,figsize=(6,5))
ax0.set_title('Water exchange bias')
ax0.plot(tacq, ffx, 'g-', linewidth=3, label='Signal data (fast exchange tissue)')
ax0.plot(tacq, iix0, 'r-', label='Prediction (before training)')
ax0.plot(tacq, iix1, 'r--', label='Prediction (after training)')
ax0.set_xlabel('Time (sec)')
ax0.set_ylabel('Signal (a.u.)')
ax0.legend()
plt.show()

# Print the parameter bias
print('')
print('Bias in kinetic model parameters')
print('--------------------------------')
print('vp error:', round(bias[1],2), '%')
print('vi error:', round(bias[3],2), '%')
print('Ktrans error:', round(bias[2],2), '%')

# Print the water permeability estimates
print('')
print('Water permeability estimates')
print('----------------------------')
print('PSe:', model.get_params('PSe', round_to=0), 'mL/sec/cm3')
print('PSc:', model.get_params('PSc', round_to=0), 'mL/sec/cm3')


# %%
# Plotting the results now shows a practically perfect fit to the data, and the measurements of the kinetic parameters are more accurate.  
# 
# As a bonus the water-exchange sensitive model also estimates the water permeability, which as expected produces values in the fast-exchange range. As the actual PS-values are infinite the estimates can never approximate the ground truth, but at this level the predicted data are effectively indistinguishable from fast-exchange signals. 
#
# Note this does not automatically imply that water exchanges rates should always be included in the modelling. The data in this tutorial are noise free, and noise in the data may well mask more subtle structure such as that imposed by restricted water exchange.

# %% 
# Additional sources of bias
# --------------------------
# The results show that small residual errors remain in the kinetic parameters, even after removing the model bias. While the error may be negligible for practical purposes, it is useful and illustrative to explore its origin further.
# 
# Any remaining bias must be due to one or more of the three remaining sources of error: (1) *sampling bias* - temporal undersampling in the data used for training, which at 1.5s creates a small mismatch with the exact (pseudo)continuous signals; (2) *convergence bias* - imperfect convergence of the model training; (3) *numerical bias* - numerical errors in the computation of the model solutions. 
#
# We can get some insight by fitting the data with an unbiased model, i.e. fitting the data with the same model that was used to generate it. This is a simple model that is likely to be much less susceptible to convergence or numerical bias, so this analysis exposes the sampling bias (alternatively we can generate data with much smaller temporal sampling intervals):

# Train a fast-exchange model on the fast exchange data
model = dc.Tissue(aif=aif, t=tacq, **const).train(tacq, ffx)

# Calculate the bias relative to the ground truth
pars = model.get_params('S0','vp','Ktrans','vi')
bias = 100*(np.array(pars)-truth)/truth

# Print the bias for each kinetic parameter
print('')
print('Bias in kinetic model parameters')
print('--------------------------------')
print('vp error:', round(bias[1],2), '%')
print('vi error:', round(bias[3],2), '%')
print('Ktrans error:', round(bias[2],2), '%')

# %%
# Any remaining bias is smaller than 0.01%, which shows that temporal undersampling in this case only causes a minor error, and the residual errors observed with the more general model are due to imperfect convergence or numerical error. 
# 
# Imperfect convergence is likely to play a role, as actual water PS values are infinite in this case, convergence is likely to be slow. Indeed - reducing the xtolerance and increasing the maximum number of iterations leads to increasingly accurate results.
# 
# Numerical error may also play a role, since the general water exchange model is implemented using linear algebra involving operations such as matrix exponentials and numerical matrix inversion, which are likely to come with some numerical error. However the effect is likely small compared to other sources of error at play.
