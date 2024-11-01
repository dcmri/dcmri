.. _imaging-sequences:

Signals
-------

The measured MRI signal is determined by the free relaxation of tissue, 
but also by the impact of magnetic pulses and gradients applied during an 
MRI acquisition to generate transverse magnetization and localize the signals. 

This section discusses the effect of these pulses on the tissue magnetization 
and the measured signal for common pulse sequences. See the 
:ref:`table with definitions <sequence-params>` for a summary of relevant 
terms and definitions.


Spoiled gradient-echo steady state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common pulse sequence for T1-weighted DC-MRI involves applying 
slective readiofrequency pulses with a low flip angle (FA) on rapid succession, 
typically with a repetition time (TR) in the range of a few msec. After each 
pulse the signal is collected and any transverse residual magnetization is 
actively destroyed before applying the next pulse. 

If the longitudinal magnetization is :math:`M_z` before a pulse with flip 
angle :math:`\alpha`, it has a value :math:`M_z\cos\alpha` afterwards. Until 
the next pulse is applied, it relaxes freely to equilibrium (see section 
:ref:`basics-relaxation-T1`). The pulses are initially applied to the 
equilibrium magnetization :math:`M_0`, but after a short number of 
:math:`\alpha`-pulses a new equilibrium arises where the regrowth after each 
pulse exactly equals the reduction in :math:`M_z` by the pulse itself: 

.. math::

  M_z = e^{-T_RR_1}M_z\cos\alpha + \left(1-e^{-T_RR_1}\right)M_0

Solving for :math:`M_z` produces the steady-state under a series of repeated 
pulses:

.. math::

  M_z = M_0\frac{1-e^{-T_RR_1}}{1-\cos\alpha\; e^{-T_RR_1}}

The signal itself is proportional to the transverse magnetization 
:math:`M_z\sin\alpha` after the pulse, with a proportionality constant 
:math:`\Omega` that depends in unknown factors such as coil sensitivity. In 
practice these factors are combined with the equilibrium magnitization into 
a single scaling factor :math:`S_\infty=\Omega M_0` to produce the final signal 
model for the spoiled gradient echo in steady state:

.. math::

  S = S_\infty\frac{1-e^{-T_RR_1}}{1-\cos\alpha\; e^{-T_RR_1}}



[... coming soon...] Other common pulse sequences.


Terms and definitions
^^^^^^^^^^^^^^^^^^^^^

.. _sequence-params:
.. list-table:: **List of sequence parameters**
    :widths: 15 20 40 10 10
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
      - Sequences
    * - TS
      - Sampling time
      - Duration of the signal readout for a single time point. If TS=0, the 
        signals are sampled by interpolation. If TS is a finite value, the 
        signals are averaged over a time TS around the acquisition time. 
        Defaults to None.
      - sec
      - SS, SR
    * - S0
      - Signal scale factor
      - Amplitude of the signal model
      - a.u.
      - SS, SR
    * - FA
      - Flip angle
      - Angle of magnetization after excitation
      - degrees
      - SS, SR
    * - TR
      - Repetition time
      - Time between excitation pulses
      - sec
      - SS, SR
    * - TC
      - Time to center
      - Time between the preparation pulse and the k-space center
      - sec
      - SR
    * - TP
      - Preparation delay
      - Time between the preparation pulse and the first k-space line
      - sec
      - SR
    * - B1corr
      - B1 correction factor
      - Factor to correct flip angles for B1-variations
      - dimensionless
      - SR, SS


.. _params-per-sequence:
.. list-table:: **Sequence parameters**
    :widths: 20 40
    :header-rows: 1 

    * - sequence
      - parameters
    * - SS
      - S0, FA, TR
    * - SR
      - S0, FA, TR, TC, TP








