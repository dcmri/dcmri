.. _dcmri-basics:

**************
Basic concepts
**************

The basic question in DC-MRI is this: given a signal measured as a function 
of time on a certain tissue, what are the values of critical tissue parameters 
such as perfusion, vascularity or capillary permeability? 

By default, `dcmri` uses a **forward-type approach** to solve the problem. 
The approach involves building a **forward model** that predicts the measured 
signal for given values of the unknown parameters. The parameters are then 
adjusted until the forward model correctly predicts the measured signals. 

The forward model must be well-defined, i.e. one and only 
one set of tissue parameters can correctly predict any given signal. This means 
the model should have enough freedom to predict the variety of signals that 
can be measured in any given context, but not so much freedom that there are 
multiple ways of predicting any given signal.

Building a suitable forward model in DC-MRI involves three different different 
types of physics. They are described in more detail in the following sections:

- :ref:`Pharmacokinetics <basics-pharmacokinetics>` models the concentration 
  of indicator (usually an MRI contrast agent) in terms of parameters such as 
  volume fractions of tissue compartments and the exchange rates between them. 

- :ref:`Relaxation theory <relaxation-theory>` models the effect of indicator 
  concentration on electromagnetic tissue proprerties, in particular the 
  longitudinal and transverse relaxation rates.

- :ref:`MRI signal theory <imaging-sequences>` models the effect of MRI 
  pulse sequences and gradients on the magnetisation and the measured signal. 

The process of adjusting the model parameters until the signals are correctly 
predicted is called **optimization** and is described in more detail in the 
section on :ref:`optimization methods <model-fitting>`. 


.. toctree::
   :maxdepth: 2
   :hidden:

   pharmacokinetics
   relaxation
   sequences
   optimization
   







