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

Definitions and notations
^^^^^^^^^^^^^^^^^^^^^^^^^

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


Spoiled gradient-echo steady state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common pulse sequence for T1-weighted DC-MRI involves applying 
selective radiofrequency pulses with a low flip angle (FA) in rapid succession, 
typically with a repetition time (TR) in the range of a few msec. After each 
pulse the signal is collected and any residual transverse magnetization is 
actively destroyed before applying the next pulse. 

The pulses are initially applied to the 
equilibrium magnetization :math:`M_{ze}`, but after a short number of 
pulses a new equilibrium arises where the regrowth after each 
pulse exactly equals the reduction in :math:`M_z` by the pulse itself. This 
is called the steady-state, and in a steady-state sequence, it is in this 
regime that data are collected.

Fast water exchange
+++++++++++++++++++

We first consider the steady-state longitudinal magnetization in a 
tissue with fast water exchange (section :ref:`T1-FX`). If the longitudinal 
magnetization is :math:`M_z` before a pulse with flip 
angle :math:`\alpha`, it has a value :math:`M_z\cos\alpha` afterwards. Until 
the next pulse is applied, it relaxes freely to equilibrium. In the 
steady-state, the end result of that free relaxation is again the initial 
state :math:`M_z` before the pulse. Using the explicit solution in 
Eq. :eq:`Mz-FX solution const J` with :math:`M_z(0)=M_z\cos\alpha`, where 
:math:`\alpha` is the flip angle:

.. math::

  M_z = e^{-T_RK}M_z\cos\alpha + \left(1-e^{-T_RK}\right)K^{-1} J

The parameters :math:`K, J` are a function of :math:`R_1` and are therefore 
changing during contrast agent injection. However, since the period of free 
relaxation TR is much shorter than typical T1-values, 
we have assumed here that :math:`K, J` are constant during this time. 

Solving the equation for :math:`M_z` produces the steady-state magnetization 
under a series of repeated pulses:

.. math::

  M_z = \frac{1-e^{-T_R\,K}}{1-\cos\alpha\; e^{-T_R\,K}} K^{-1} J 

In this context it is safe to ignore flow effects so that 
:math:`J/K\approx M_{ze}`, which is constant, and :math:`K\approx R_1` (see 
section :ref:`T1-FX`). The signal itself is proportional to the transverse 
magnetization 
:math:`M_z\sin\alpha` after the pulse, with a proportionality constant 
that depends on factors such as coil sensitivity. In 
practice all constants are assembled into 
a single scaling factor :math:`S_\infty` to produce the final signal 
model for the spoiled gradient echo in steady state:

.. math::

  S = S_\infty\sin\alpha\frac{1-e^{-T_R\,R_1}}{1-\cos\alpha\; e^{-T_R\,R_1}}

In `dcmri` this model is implemented in the function `dcmri.signal_ss`:

.. code-block:: python

  S = signal_ss(R1, Sinf, TR, FA)

Here *R1* is either a scalar, or a 1D array if it is variable. The other 
variables are scalar.

Restricted water exchange
+++++++++++++++++++++++++

If water exchange is restricted, the signal derivation is similar except that 
now we must use the vector form of the magnetization (see section 
:ref:`T1-RX`): 

.. math::

  \mathbf{M} = e^{-T_R\,\mathbf{K}}\mathbf{M}\cos\alpha 
  + \left(1-e^{-T_R\,\mathbf{K}}\right) \mathbf{K}^{-1}\mathbf{J}

The solution is also similar, though we must take care not to commute the 
matrices:

.. math::

  \mathbf{M} = \left(1 - \cos\alpha\, e^{-T_R\,\mathbf{K}}\right)^{-1} 
  \left(1-e^{-T_R\,\mathbf{K}}\right) \mathbf{K}^{-1}\mathbf{J} 

As before the signal is proportional to the total magnetization with now 
takes the form :math:`\mathbf{e}^T\mathbf{M}` with :math:`\mathbf{e}^T=[1,1]`. 
Assuming the equilibrium magnetization is the same :math:`m_e` in both 
compartments we can extract it by 
defining :math:`\mathbf{j}=\mathbf{J}/m_e` and absorbing the constant 
:math:`m_e` in the global scaling factor :math:`S_\infty`:

.. math::

  S = S_\infty\, \sin\alpha\, \mathbf{e}^T 
  \left(1 - \cos\alpha\, e^{-T_R\,\mathbf{K}}\right)^{-1} 
  \left(1-e^{-T_R\,\mathbf{K}}\right) \mathbf{K}^{-1}\mathbf{j} 

If we ignore the inflow effects then :math:`\mathbf{j}` is determined by 
relaxation rates :math:`R_{1,k}` and volume fractions :math:`v_{k}` of 
both compartments, and 
:math:`\mathbf{K}` additionally depends on the water permeabilities 
:math:`PS_{kl}` between the compartments. 

This signal model is available 
in `dcmri` through the same function `dcmri.signal_ss`. The calling sequence 
is the same as in fast water exchange, except that now the volume fractions of 
the compartments need to be provided, and the water PS values across the 
barriers between them:

.. code-block:: python

  S = signal_ss(R1, Sinf, TR, FA, v, PS)

In this case *R1* is a 2-element array, or a 2xn array with *n* time points if 
*R1* is variable, *v* is a 2-element array, and *PS* is a 
2x2 array with zeros on the diagonal and water PS values on 
the off-diagonal. The same function also applies when the 
number of compartments is larger than 2.

As mentioned flow effects can usually be ignored in steady-state sequences, 
but it is possible to include them by adding the water outflow from 
each compartment on the diagonal of *PS*, and providing the influx of 
normalized magnetization *j* with the same dimensions as *R1*:

.. code-block:: python

  S = signal_ss(R1, Sinf, TR, FA, v, PS, j)


Saturation-recovery spoiled gradient-echo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[... coming soon...] 











