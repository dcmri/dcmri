.. _basics-pharmacokinetics:

Pharmacokinetics
----------------

An **indicator** is a foreign substance added to a body fluid 
which is separately detectable. A **tracer** is a special kind of indicator 
which behaves in exactly the same way as a component of the body fluid.  
Labelled water such as that used in arterial spin labelling is a tracer. MRI 
contrast agents are an example of an indicator that is *not* a tracer.

Pharmacokinetics creates the critical link between tissue parameters and 
concentration-time profiles of an indicator. In 
DC-MRI the indicator is usually an MR contrast agent. It is initially 
added to venous blood, and then carried through the circulation and 
distributed through the rest of the body. 

``dcmri`` provides a library of simple pharmacokinetic models, which can be 
assembled to build more complex pharmacokinetic models (see section 
:ref:`PK blocks`). For each system, functions are included which return the 
tissue concentration *C(t)* and outflux *J(t)* of the system. 

The sections 
below define the models and their solutions in more detail. 
See the :ref:`table with definitions <PK-terms>` for a summary of 
relevant terms and notations. This introduction aims to provide sufficient 
detail to unambiguously define all models included in ``dcmri``, but is not 
intended to provide a full introduction to pharmacokinetics. For that we 
refer to classic textbooks or review papers.

Definitions and notations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _PK-terms:
.. list-table:: **Pharmacokinetic models: definitions and notations**
    :widths: 15 20 40 10
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
    * - J
      - Indicator flux
      - The amount of indicator molecules entering or leaving 
        a system in a unit of time.
      - mmol/sec
    * - c
      - Indicator concentration
      - The amount of indicator molecules relative to the volume of 
        distribution.
      - mmol/mL
    * - C
      - Indicator tissue concentration
      - The amount of indicator molecules relative to the volume of tissue.
      - mmol/cm3
    * - v
      - Indicator volume of distribution
      - The fraction of the tissue accessible to the indicator.
      - mL/cm3
    * - T
      - Indicator mean transit time
      - The average time an indicator molecule needs to pass through the tissue.
      - sec
    * - R
      - Residue function
      - The fraction of indicator left at time t of an injection at time t=0.
      - dimensionless
    * - h
      - Propagator
      - The transit time distribution
      - 1/sec
    * - K
      - Compartment rate constant
      - The ratio between outflux and tissue concentration.
      - 1/sec


Indicator transit times
^^^^^^^^^^^^^^^^^^^^^^^

Explicit definitions of pharmacokinetic models are built on the conservation 
of indicator mass:

.. math::

    \frac{dC}{dt}(t) = J_\mathrm{in}(t) - J_\mathrm{out}(t)

For linear and stationary systems, solutions can always be written in terms 
of the residue function *R(t)* and the propagator *h(t)*:

.. math::

    J_\mathrm{out}(t) = h(t)*J_\mathrm{in}(t)
    \qquad\textrm{and}\qquad
    C(t) = R(t)*J_\mathrm{in}(t)

Here the symbol :math:`*` denotes the **convolution product** of two 
functions, which is defined in this context as:

.. math::

    (f*g)(t) = \int_0^t d\tau f(\tau)g(t-\tau)

The propagator *h(t)* can be interpreted as the probability distribution of 
transit times through the tissue. The expectation value of *h(t)* is 
therefore the **mean transit time** *T* of the 
system. Substituting the definitions of *R(t)* and *h(t)* in the equation 
of mass conservation we find a relationship between them:

.. math::

    R(t) = 1 - \int_0^t h(\tau) d\tau

This implies that the mean transit time is also the area under *R(t)*. 

``dcmri`` includes discrete approximations of *R(t)* and *h(t)* for most 
systems. In some cases these are trivial, but implementations are nevertheless 
provided for completeness. We are assuming throughout this section and all 
implementations that *t=0* at the start of the acquisition and the system 
contains no indicator concentrations at that time.

.. _define-trap:

Trap
^^^^

The simplest pharmacokinetic system is the **trap**, which captures all the 
indicator that enters it (see section :ref:`reference-trap` 
for implementations):

.. math::

    C(t) = \int_0^t J_\mathrm{in}(\tau)d\tau

In practice a system behaves like a trap when the shortest transit times are 
longer than the acquisition time. In that case any loss of indicator falls 
beyond the measurement window and the concentrations are accurately predicted 
by modelling the system as a trap.


.. _define-pass:

Pass
^^^^

A **pass** is a space where the concentration is proportional to the input 
(see section :ref:`reference-pass` for implementations). For a 
pass with mean transit time *T* and volume fraction *v*, the tissue 
concentration is proportional to the influx or inlet concentration:

.. math::

    C(t) = T J_\mathrm{in}(t) = vc_\mathrm{in}(t)

In practice it is used to model tissues where the transit times are shorter 
than the temporal sampling interval. Under these conditions any bolus 
broadening is not detectable. 


.. _define-compartment:

Compartment
^^^^^^^^^^^

A **compartment** is a space where the outflux is proportional to the 
concentration (see section :ref:`reference-compartment` for implementations). 
This is particularly true in systems that are *well-mixed*, i.e. have a 
uniform concentration throughout. 

Expressing conservation of indicator mass provides the mathematical 
definition of a compartment:

.. math::

    \frac{dC}{dt}(t) = J_\mathrm{in}(t) - KC(t)

Here *K* is the **rate constant** of the compartment. The solution is:

.. math::

    C(t) = e^{-Kt}*J_\mathrm{in}(t)

This shows that the residue function of a compartment is a mono-exponential, 
and its mean transit time is the therefore reciprocal *1/K*.


.. _define-plug-flow:

Plug flow
^^^^^^^^^

A **plug-flow system** is a space where all indicator particles have a 
constant velocity *u* (see section :ref:`reference-plug-flow` for 
implementations). Indicator motion through a plug-flow system can be 
modelled as a one-dimensional system with mass conservation at each point:

.. math::

    \frac{\partial C}{\partial t}(x,t) = 
    -u\frac{\partial C}{\partial x}(x,t)

A plug flow system is in many ways the opposite of a compartment as it 
does not allow for any mixing at all. Indicator concentrations at the outlet 
are shifted in time but are not otherwise distorted:

.. math::

    J_\mathrm{out} (t) = J_\mathrm{in}(t-T)

The mean transit time *T* equals *u/L*, where *L* is the distance between in- 
and outlet. The concentration inside a plug flow system is found by 
integrating the mass conservation:

.. math::

    C(t) = \int_0^t d\tau \left(J_\mathrm{in}(\tau) - J_\mathrm{in}(\tau-T)\right)


.. _define-chain:

Chain
^^^^^

A **chain** is a serial arrangement of *n* identical compartments, each with a 
transit time *T/n* (see section :ref:`reference-chain` for 
implementations). The mean transit time of a chain is *T* and the 
propagator is a convolution of *n* exponentials (see also `dcmri.nexpconv`). 
This takes the form of a normalized gamma-variate function which is known to 
provide a good model for concentration-time curves after rapid indicator 
injection:

.. math::
    h(t) = \frac{1}{\Gamma(n)}\left(\frac{t}{T/n}\right)^{n-1} \frac{e^{-t/T/n}}{T/n} 

With :math:`n\to\infty` a chain becomes a plug flow system, and with 
:math:`n=1` a chain is a single compartment. If we introduce a **dispersion 
parameter** *D = 1/n* with values in the range of [0,1], then a chain is 
fully characterized by two numbers *(T,D)* which has a compartment (D=1) and a 
plug flow system (D=0) as special cases. Moreover while the physical definition 
involves a discrete system of n compartments, the solution allows for D to take 
any value in the range [0,1]. 

In practice a chain can therefore be used to model tissues that 
cause an unknown level of indicator dispersion in between the extremes of no 
dispersion (D=0) and maximal dispersion (D=1).

.. _define-pfcomp:

Plug-flow compartment
^^^^^^^^^^^^^^^^^^^^^

A **plug-flow compartment** is a serial arrangement of a compartment with 
mean transit time *DT* and and plug-flow system with mean transit time 
*(1-D)T* (see section :ref:`reference-pfcomp` for implementations). The total 
mean transit time of a plug-flow compartment is *T*. The dimensionless 
parameter *D* can take any values in the interal [0,1] and the system has 
a plug-flow system (D=0) and a compartment (D=1) as special cases. 

The propagator :math:`h_\mathrm{PC}` of a plug-flow compartment is a delayed 
exponential function:

.. math::

    h_\mathrm{PC}(t<(1-D)T) &= 0 
    \\
    h_\mathrm{PC}(t>(1-D)T) &= \frac{e^{-(t-(1-D)T)/DT}}{DT}

A plug-flow compartment is similar to a chain in that it 
can be used to model tissues with an unknown level of dispersion by varying 
the dispersion parameter *D*. Its internal structure is coarser but it 
is computationally more efficient than a chain.

.. _define-step:

Step
^^^^

A **step** is a system where the transit time distribution is a step function 
with a constant non-zero value between the time points *(1-D)T* and *(1+D)T* 
(see section :ref:`reference-step` for implementations):

.. math::

    h( (1-D)T < t < (1+D)T ) = 1/(2DT)

And h(t)=0 otherwise. The mean transit time of a step is T and just like the 
chain and the plug-flow compartment, the dispersion parameter *D* takes 
values in the range [0,1] where *D=1* represents maximum dispersion, 
and *D=0* is a plug-flow system with minimal dispersion.  

.. _define-free:

Free
^^^^

A **free** system is one where the transit time distribution can take any 
required form. The transit time distribution is parametrized as a histogram 
with any number of bins. The model parameters are the n+1 boundaries of the 
n bins, and the n frequencies of each bin (see section :ref:`reference-free` 
for the available functions). For model fitting the boundaries are 
usually treated as fixed parameters, and the frequencies are treated as 
unknowns. 

.. _define-ncomp:

N-compartment system
^^^^^^^^^^^^^^^^^^^^

An **n-compartment system** is a collection of *n* interacting compartments (see 
section :ref:`reference-ncomp` for the available functions). Each compartment 
in the system can exchange with any other compartment, and with the external 
environment. The system is therefore characterized by *n* equations of the 
following form:

.. math::

    \frac{dC_i}{dt}(t) = J_i(t) - \sum_{j=1}^n K_{ji}C_i(t) + \sum_{j\neq i}K_{ij}C_j(t)

Here :math:`j_i(t)` is the influx into compartment *i* fom the environment. 
The system is fully determined by the :math:`n^2` rate constants :math:`K_{ji}`
which represent the rate constants for the outflux from 
*i* to *j* if :math:`i\neq j`, and the rate constant for the outflux from 
*i* to the environment if :math:`i= j`. Arranging the *n* concentrations and 
influxes in arrays :math:`\mathbf{C}` and :math:`\mathbf{J}` we can write this 
in a form very similar to the one-compartment case:

.. math::

    \frac{d\mathbf{C}}{dt} = \mathbf{J} - \mathbf{\Lambda} \mathbf{C}

Here :math:`\Lambda` is a square matrix which has off-diagonal elements 
:math:`-K_{ij}` and diagonal elements :math:`K_i = \sum_j K_{ji}`. The general 
solution also has the same form as the one-compartment case, except that 
it now involves a matrix exponential:

.. math::

    \mathbf{C}(t) = e^{-\mathbf{\Lambda}t} * \mathbf{J}(t)

The mean transit time of each compartment is :math:`T_i=1/K_i` and the 
extraction fraction :math:`E_{ji}` from *i* to *j* is the ratio 
:math:`K_{ji}/K_i`. An alternative way of characterizing the system is 
therefore in terms of the *n* mean transit times :math:`T_i` and the *n(n-1)* 
extraction fractions :math:`E_{ji}`.

.. _define-2comp:

2-compartment exchange
^^^^^^^^^^^^^^^^^^^^^^

A two-compartment exchange model is a 2-compartment model with a central 
compartment that exchanges with the environment, and a second compartment that 
only exchanges with the central compartment (see section 
:ref:`reference-2comp`). It is characterised by 3 
parameters: the mean transit times of both compartments, and the extraction 
fraction from the central compartment into the second compartment. Since this 
is an example of an n-compartment model, the solution can be obtained from 
these more general functions, but a dedicated solution for the 2-compartment 
exchange model is more convenient to use. 


.. _define-nscomp:

Non-stationary compartment
^^^^^^^^^^^^^^^^^^^^^^^^^^

A non-stationary compartment is a compartment with a rate constant that is a 
function of time (see section :ref:`reference-nscomp` for implementations):

.. math::

    \frac{dC}{dt}(t) = J_\mathrm{in}(t) - K(t) C(t)

In this case the solution can no longer be expressed as a convolution. Instead 
the equation must be solved numerically, for instance by forward propagation 
over small time steps *dt*:

.. math::

    C(t+dt) = dt J_\mathrm{in}(t) + (1 - dt K(t)) C(t)

The solution is stable if the time steps are small enough, i.e. 
:math:`dt K(t) < 1` for any time *t*. This also states that the time step *dt* 
must be smaller than the shortest mean transit time *T(t)* of the compartment. 
When fitting data with unknown transit time, suitable lower-bounds must be 
placed on the values of *T(t)* to avoid very small values of *dt* and 
correspondingly large computation times.

A non-stationary compartment would be used in situations where the tissue 
properties themselves change in the course of the measurement - for instance 
because the acquisition is very long, or because rapid physiological changes
are taking place. In practice the number of free parameters can be reduced 
by interpolating between values at particular times. For instance, parameters 
:math:`K(t)` can be determined for each :math:`t` by interpolating between 
two values :math:`(K_i, K_f)` at the initial and final time points, 
respectively. When the model is used to explain measured data, those two 
values would then be treated as free parameters.

.. _define-mmcomp:

Michaelis-Menten compartment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Michaelis-Menten compartment is a compartment where the rate constant 
depends on the concentration (see section :ref:`reference-mmcomp` for 
implementations):

.. math::
    \frac{dC}{dt} = -K(C) C
    \qquad\textrm{with}\qquad
    K(C) = \frac{V_\max}{K_m+C}

For small enough concentrations :math:`C << K_m` this reduces to a standard 
linear compartment with :math:`K=V_\max/K_m`. The Michaelis-Menten compartment 
would therefore mainly be used in situations where higher doses of contrast 
agent are injected. It is a classic example of a non-linear system and 
an analytical solution is available through the work of 
`Schnell and Mendoza <https://www.sciencedirect.com/science/article/pii/S0022519397904252>`_.










