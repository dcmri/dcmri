"""
===========================
Dealing with inflow effects
===========================

`Nichlas Vous Christensen <https://www.au.dk/en/nvc@clin.au.dk>`_, 
`Mohsen Redda <https://www.au.dk/en/au569527@biomed.au.dk>`_, 
`Steven Sourbron <https://www.sheffield.ac.uk/smph/people/academic/clinical-medicine/steven-sourbron>`_.

What are inflow effects?
------------------------
Inflow effects in the arterial input function (AIF) are a common 
type of artefact in DCE-MRI which severely biases the results if it 
is not properly acccounted for. The effect is caused by unsaturated 
blood flowing into the imaging slab causing an artefactual increase 
of the baseline signal. A naive analysis of inflow-corrupted data 
underestimates the arterial concentration and consequently 
overestimates tissue perfusion parameters. 

Inflow effects can be detected visually by 
inspecting a precontrast image. Since blood has a long T1, arteries 
should be dark on a T1-weighted image. If they are instead 
brighter than surrounding tissue, this is evidence of inflow effects. 
The brightness typically shows a gradient with the vessel being 
brightest at the edge of the slab where flow-induced enhancement is 
most severe, then gradually darking as the blood travels down the slab 
and is pushed into steady-state by the excitation pulses. 

Minimizing inflow effects
-------------------------
The most robust approach to dealing with inflow effects is by 
eliminating them at the source, by optimizing the acquisition. In 
abdominal imaging this can often be achieved by positioning the slab 
coronally and extending the field of view to include the heart. 

This may not always be possible though, for instance for prostate imaging 
where the heart is too far from the region of interest. Even when it 
is technically possible, such as for liver or renal imaging, it may 
require compromises in other areas: the approach forbids axial 
imaging, which is sometimes preferred, and may also come at a cost of 
spatial resolution. 

Other approaches may be possible, such as 
increasing the flip angle to accelerate transition to steady-state; 
but this comes with other compromises such as increased tissue 
heating and reduced signal enhancement.

Correcting inflow effects
-------------------------
If inflow effects cannot be fully avoided by optimizing the 
acquisition, or data with inflow effects are analysed retrospectively, 
the problem must be addressed at image analysis stage. 

One approach 
that has sometimes been applied is to measure the input function 
deeper down in the slab where inflow effects have largely decayed. 
This is not always feasible and may cause variability in result for 
instance due to differences in blood velocity.

This tutorial illustrates two alternative approaches: using a 
standardised input function rather than attempting to measure it; 
and correcting for inflow by adding flow-related enhancement in the 
signal model.

The solutions are illustrated for the use case of MR renography in a 
minipig with unilateral kidney fibrosis. The data are taken from 
Bøgh et al (2024).

Reference
---------
Nikolaj Bøgh, Lotte B Bertelsen, 
Camilla W Rasmussen, Sabrina K Bech, Anna K Keller, Mia G Madsen, 
Frederik Harving, Thomas H Thorsen, Ida K Mieritz, Esben Ss Hansen, 
Alkwin Wanders, Christoffer Laustsen. Metabolic MRI With 
Hyperpolarized 13C-Pyruvate for Early Detection 
of Fibrogenic Kidney Metabolism. 
[`DOI <https://doi.org/10.1097/rli.0000000000001094>`_].
"""

# %%
# Setup
# -----

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dcmri as dc

# Read the dataset
dmrfile = dc.fetch('minipig_renal_fibrosis')
data = dc.read_dmr(dmrfile, 'nest')
rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']

# %%
# Plot data
# ---------
# Let's start by plotting the data:

time = pars['TS'] * np.arange(len(rois['Aorta']))
plt.plot(time, rois['Aorta'], 'r-', label='Aorta')
plt.plot(time, rois['LeftKidney'], 'g-', label='LeftKidney')
plt.plot(time, rois['RightKidney'], 'b-', label='RightKidney')
plt.xlabel('Time (sec)')
plt.ylabel('Signal (a.u.)')
plt.legend()
plt.show()

# %%
# We see a clear difference between left kidney (healthy) and right 
# kidney (fibrotic). We also 
# see that the peak signal change in the aorta is similar to that in 
# the kidney. As peak aorta concentrations in pure blood are always 
# substantially higher than those in tissue, this shows that 
# signal changes underestimate concentrations - consistent with the 
# effect of inflow artefacts.
# 
# Verification on the data confirms this suspicion: looking at a 
# precontrast slice through the aorta (below) we see that the aorta 
# is bright compared to the surrounding tissue, and gradually becomes 
# darker as we move further down into the slab - this is the signature 
# sign of inflow artefacts. 

img = mpimg.imread('../../source/_static/tutorial-inflow.png')
plt.figure(figsize=(6, 4))
plt.imshow(img)
plt.axis("off")
plt.title("Baseline image")
plt.show()

# %%
# Standard analysis
# -----------------
# Ignoring the inflow effects leads to significant bias in the 
# results. For the kidney 
# the most common modelling approach is implemented in the function 
# `dcmri.Kidney`. Let's run it on the left kidney and see what we get:

kidney = dc.Kidney(

    # Configuration
    aif=rois['Aorta'],
    dt=pars['TS'],

    # General parameters
    field_strength=pars['B0'],
    agent="gadoterate",
    t0=pars['TS'] * pars['n0'],

    # Sequence parameters
    TS=pars['TS'], 
    TR=pars['TR'],
    FA=pars['FA'],

    # Tissue parameters
    R10=1/dc.T1(pars['B0'], 'kidney'),
    R10a=1/dc.T1(pars['B0'], 'blood'),
)

kidney.train(time, rois['LeftKidney'])
kidney.plot(time, rois['LeftKidney'])
kidney.print_params(round_to=4)

# %%
# The model is not fitting the data because by default the model 
# parameters are not allowed to enter into unphysical regimes. So 
# the model stops converging when they hit their bounds, as 
# can be seen from the plasma flow which converged to its upper 
# bound of 0.05 mL/sec/cm3. 

# %%
# If we free up the parameters then the model will fit, albeit
# with unphysical values for the parameters:

kidney.set_free(Fp=[0,np.inf], vp=[0,1], FF=[0,1])
kidney.train(time, rois['LeftKidney'])
kidney.plot(time, rois['LeftKidney'])
kidney.print_params(round_to=4)

# %%
# The plasma flow now has a value of 1.2 mL/sec/cm3 or 
# 7200 mL/min/100cm3. This is around 40 times higher than what 
# is realistic for a kidney - confirming the massive bias caused by 
# inflow artefacts. This is separately evidenced by the peak 
# concentration around 0.8mM (right of plot) - substantially lower 
# than the values of 4-5mM that are typically seen after injection 
# of a standard dose.

# %%
# Using a standard input function
# -------------------------------
# One approach that can always be considered if arterial 
# concentrations are not reliable is to use a modelled rather than a 
# measured input function. 
#
# There are no input function models available for minipigs so 
# our best option is to use one derived for humans and adjust the 
# parameters. We will use the function `dcmri.aif_tristan` which is 
# built on a model of the circulation and thefore defined in terms 
# of physiological parameters. 
# 
# We set the cardiac output (CO) to a typical value for the 
# minipig (3.6 L/min or 60 mL/sec). The bolus arrival time (BAT) can 
# be estimated from the 
# peak of the aorta concentration - it does not have to be exact as the 
# value is optimized in the fit. All other parameters are left at 
# default values as no data exist for the minipig:

dt = 0.25
t = np.arange(0, np.amax(time) + dt, dt)  
ca = dc.aif_tristan(
    t, 
    agent="gadoterate",
    dose=pars['dose'],
    rate=pars['rate'],
    weight=pars['weight'],
    CO=60,
    BAT=time[np.argmax(rois['Aorta'])] - 20,
)

# %%
# Now we can use this fixed concentration as input in the kidney 
# model instead of the measured aorta signal. Since the 
# artery signal is not measured in the aorta we will allow the 
# arterial transit time to vary over a larger range than the default 
# of [0, 3] sec:

kidney = dc.Kidney(

    # Configuration
    ca=ca,
    dt=dt,

    # General parameters
    field_strength=pars['B0'],
    agent="gadoterate",
    t0=pars['TS'] * pars['n0'],

    # Sequence parameters
    TS=pars['TS'], 
    TR=pars['TR'],
    FA=pars['FA'],

    # Tissue parameters
    R10=1/dc.T1(pars['B0'], 'kidney'),
    R10a=1/dc.T1(pars['B0'], 'blood'),
)

kidney.set_free(Ta=[0,30])
kidney.train(time, rois['LeftKidney'])
kidney.plot(time, rois['LeftKidney'])
kidney.print_params(round_to=4)

# %%
# This now fits a lot better without unphysical 
# parameter values, but the plasma flow still hits the upper limit,
# and the fit remains poor - indicating the input function does not 
# represent reality very well even after adapting the parameters.

# %%
# Model-based inflow correction
# -----------------------------
# An alternative solution is to fit an 
# aorta model to the data and use a signal model (SSI) that 
# accounts for inflow effects. 
# 
# The CO is set to the same values as for the modelled AIF above, 
# but here this serves as initial guess rather than a fixed 
# parameter:

aorta = dc.Aorta(

    # Configuration
    sequence='SSI',
    heartlung='chain',
    organs='comp',

    # General parameters
    dt=dt,
    field_strength=pars['B0'],
    t0=pars['TS']*pars['n0'],

    # Injection protocol
    agent="gadoterate",
    weight=pars['weight'],
    dose=pars['dose'],
    rate=pars['rate'],

    # Sequence parameters
    TR=pars['TR'],
    FA=pars['FA'],
    TS=pars['TS'],
    
    # Aorta parameters
    CO=60, 
    R10=1/dc.T1(pars['B0'], 'blood'), 
)

aorta.train(time, rois['Aorta'])
aorta.plot(time, rois['Aorta'])
aorta.print_params(round_to=4)

# %%
# This produces a good fit to the data and also reasonable 
# values for the parameters. A cardiac output of 73 mL/sec 
# equates to 4.4 L/min, which is in the right range for a minipig. 
# Also a travel time of 280 msec (Inflow time) from heart to kidneys 
# seems reasonable. The peak concentrations (right) are also in the 
# expected range for a standard injection (5mM). 

# %%
# Kidney model
# ------------
# We can now use the trained aorta model to generate 
# concentrations and use those as input for the kidney model, 
# instead of the modelled concentration:

t, ca = aorta.conc() # get arterial concentrations

kidney = dc.Kidney(

    # Configuration
    ca=ca,
    t=t,

    # General parameters
    field_strength=pars['B0'],
    agent="gadoterate",
    t0=pars['TS']*pars['n0'],

    # Sequence parameters
    TS=pars['TS'], 
    TR=pars['TR'],
    FA=pars['FA'],

    # Tissue parameters
    R10=1/dc.T1(pars['B0'], 'kidney'),
    R10a=1/dc.T1(pars['B0'], 'blood'),
)

kidney.train(time, rois['LeftKidney'])
kidney.plot(time, rois['LeftKidney'])
kidney.print_params(round_to=4)

# %%
# This now gives a good fit with reasonable values for all parameters. 
# Let's run it on the right kidney as well so we can compare 
# parameters:

kidney.train(time, rois['RightKidney'])
kidney.plot(time, rois['RightKidney'])
kidney.print_params(round_to=4)

# %%
# The right (fibrotic) kidney now shows a substantially lower 
# perfusion and function than the left. The perfusion of the left 
# kidney has not hit the maximum value. It is still relatively high 
# (Fp = 0.035 mL/sec/cm3 or 210 mL/min/100mL) but it is possible 
# that it is overcompensating to some extent for the damage on the 
# right kidney.

# %%
# Joint aorta-kidneys fit
# -----------------------
# The method above produces a good solution for these data, but if the 
# volumes of the kidneys are known this can be further refined 
# by performing a joint fit of aorta and both kidneys. 
# 
# This is not only more compact but should also be more robust 
# as shared parameters can be eliminated and all data are accounted 
# for equally. In this example the volumes are not actually known, 
# so for the purpose of illustration we use a typical value of 
# 85mL: 

aorta_kidneys = dc.AortaKidneys(

    # Configuration
    sequence='SSI',
    heartlung='chain',
    organs='comp',
    agent="gadoterate",

    # General parameters
    field_strength=pars['B0'],
    t0=pars['TS']*pars['n0'], 

    # Injection protocol
    weight=pars['weight'],
    dose=pars['dose'],
    rate=pars['rate'],

    # Sequence parameters
    TR=pars['TR'],
    FA=pars['FA'],
    TS=pars['TS'],

    # Aorta parameters
    CO=60,  
    R10a=1/dc.T1(pars['B0'], 'blood'),

    # Kidney parameters
    vol_lk=85,
    vol_rk=85,
    R10_lk=1/dc.T1(pars['B0'], 'kidney'),
    R10_rk=1/dc.T1(pars['B0'], 'kidney'),
)

# Define time and signal data
t = (time, time, time)
signal = (rois['Aorta'], rois['LeftKidney'], rois['RightKidney'])

# Train model and show result
aorta_kidneys.train(t, signal)
aorta_kidneys.plot(t, signal)
aorta_kidneys.print_params(round_to=4)

# %%
# This produces all results for aorta and kidneys in one go, which 
# also allows to derive some secondary parameters such as 
# differential function which would otherwise have to be computed 
# from separate results. Since the model uses the volumes this can 
# also automatically output important whole kidney parameters such 
# as GFR and RPF. Numaerically the values are similar - though not 
# identical - to those produced from a separate Aorta and Kidneys fit.

# sphinx_gallery_start_ignore
# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
# sphinx_gallery_end_ignore