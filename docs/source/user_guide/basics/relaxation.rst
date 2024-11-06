.. _relaxation-theory:

Relaxation
----------

Contrast agents and their concentration are visible in MRI because the agents
are designed to modify the relaxation rates of tissues. This section deals 
with the relationship between contrast agent concentration and magnetic 
relaxation rates. 

The detailed interaction between contrast agent 
molecules and magnetic tissue properties can be complex, but fortunately 
the relationship between concentrations and relaxation rates can be modelled 
relatively easily with simple approximations. See the 
:ref:`table with definitions <relaxation-params>` for a summary of relevant 
terms and notations.

Definitions and notations
^^^^^^^^^^^^^^^^^^^^^^^^^

Models of magnetic relaxation are determined by the following parameters:

.. _relaxation-params:
.. list-table:: **Relaxation model parameters**
    :widths: 15 20 40 10
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
    * - :math:`R_1`
      - Longitudinal relaxation rate
      - Reciprocal of longitudinal relaxation time
      - Hz
    * - :math:`M_z`
      - Longitudinal magnetization
      - Component of the tissue magnetization parallel to the magnetic field
      - A/cm
    * - :math:`M_{ze}`
      - Equilibrium longitudinal magnetization
      - Longitudinal magnetization at rest
      - A/cm
    * - :math:`m_z`
      - Longitudinal water magnetization
      - Longitudinal magnetization per unit water volume
      - A/cm/mL
    * - :math:`m_{ze}`
      - Equilibrium longitudinal water magnetization
      - Water magnetization at rest
      - A/cm/mL
    * - :math:`m_{zi}`
      - Inlet water magnetization
      - Magnetization of the water flowing into the tissue
      - A/cm/mL
    * - :math:`R_{10}`
      - Precontrast longitudinal relaxation rate in tissue
      - Native longitudinal relaxation rate in the absence of contrast agent
      - Hz
    * - :math:`r_1`
      - Longitudinal relaxivity
      - Increase in longitudinal relaxation rate :math:`R_1` per unit 
        concentration
      - Hz/M
    * - :math:`r^*_2`
      - Transverse relaxivity
      - Increase in transverse relaxation rate :math:`R^*_2` per unit 
        concentration
      - Hz/M
    * - :math:`v`
      - Water volume fraction
      - Volume fraction of the space occupied by water
      - mL/cm3
    * - :math:`f_i`
      - Water inflow
      - Volume of water flowing in per unit of time and per unit of tissue
      - mL/sec/cm3
    * - :math:`f_o`
      - Water outflow
      - Volume of water flowing out per unit of time and per unit of tissue
      - mL/sec/cm3
    * - :math:`PS_{kl}`
      - Magnetization permeability-surface area from l to k
      - Magnetization transfer rate from compartment l to compartment k
      - mL/sec/cm3


.. _basics-relaxation-T1:

Longitudinal relaxation
^^^^^^^^^^^^^^^^^^^^^^^

.. _T1-FX:

Fast water exchange
+++++++++++++++++++

We consider a tissue with uniform magnetization. Magnetization is carried in by 
inflow of magnetized water and carried out by water flow and relaxation. The 
longitudinal magnetization is governed by the Bloch equation:

.. math::

    v\frac{dm_z}{dt} = f_i m_{zi} - f_o m_z + R_1 v (m_{ze} - m_z)

After regrouping terms and writing this in terms of the total magnetization 
:math:`M_z=vm_z`:

.. math::
  :label: Mz-FX

    \frac{dM_z}{dt} = J - KM_z

where we define influx and rate constants:

.. math::

    J &= R_1 v\, m_{ze} + f_i m_{zi}
    \\
    K &= R_1 + \frac{f_o}{v}

Note this is in fact just another one-compartment model (see 
section :ref:`define-compartment`), with the water magnetization :math:`M_z` 
playing the role of tracer. If :math:`K` is a constant, or we are 
considering sufficiently short time scales so that it can be assumed to be 
constant, the solution is:

.. math::
  :label: Mz-FX solution

    M_z = e^{-tK}M_z(0) + e^{-tK}*J(t)

If additionally the influx :math:`J` can be assumed constant, we can compute 
the convolution:

.. math::
  :label: Mz-FX solution const J

    M_z = e^{-tK}M_z(0) + \left(1-e^{-tK}\right)K^{-1} J

The flow component in :math:`K, J` is often negligible, in which case 
:math:`J=R_1M_{ze}`, :math:`K=R_1` and :math:`J/K=M_{ze}`. This produces the 
familiar solution for free longitudinal relaxation in a closed system:

.. math::
  :label: Mz-FX solution noflow

    M_z = e^{-tR_1}M_z(0) + \left(1-e^{-tR_1}\right)M_{ze}


.. _T1-RX:

Restricted water exchange
+++++++++++++++++++++++++

The above solution assumes the tissue magnetization is uniform, i.e. the water 
moves so quickly between tissue compartments that any differences in 
magnetization are immediately levelled out. If that is not the case, the 
exchange of magnetization between the tissue compartments must be explicitly 
incorporated. 

We consider this for the example of two interacting water compartments 
:math:`1,2`. The generalization to N compartments is then straightforward. We 
can write a Bloch equation for each and now explicitly include the exchange 
of magnetization between them. As there is no confusion possible we drop 
the z-indices for this section to avoid overloading the notations:

.. math::

    v_1\frac{dm_1}{dt} &= f_{i,1}m_{i,1} - f_{o,1}m_1 
    + R_{1,1}v_1(m_{e,1}-m_1) + PS_{12}m_2 - PS_{21}m_1 
    \\
    v_2\frac{dm_2}{dt} &= f_{i,2}m_{i,2} - f_{o,2}m_2 
    + R_{1,2}v_2(m_{e,2}-m_2) + PS_{21}m_1 - PS_{12}m_2 

The magnetization transfer :math:`PS_{lk}m_k` will be mediated by 
physical water flow, but other mechanisms of magnetization transfer between 
compartments may also be at play. The basic assumption is that the 
transfer is proportional to the water magnetization - as long as this is true 
the equation is valid and the precise mechanism of transfer only affects the
physical interpretion of :math:`PS`.

Gathering terms and expressing the result in terms of the total magnetization 
:math:`M=vm`, this takes the familiar form of a two-compartment model 
(see section :ref:`define-ncomp`):

.. math::

    \frac{dM_1}{dt} &= J_1 - K_1M_1 + K_{12}M_2 
    \\
    \frac{dM_2}{dt} &= J_2 - K_2M_2 + K_{21}M_1

Here we define rate constants:

.. math::

    K_1 &= R_{1,1} + \frac{f_{o,1} + PS_{21}}{v_1} \qquad 
    K_{12}=\frac{PS_{12}}{v_2}
    \\
    K_2 &= R_{1,2} + \frac{f_{o,2} + PS_{12}}{v_2} \qquad 
    K_{21}=\frac{PS_{21}}{v_1}

and an \`\`influx\'\' of magnetization:

.. math::

    J_1 &=  R_{1,1}v_1 m_{e,1} + f_{i,1}m_{i,1}
    \\
    J_2 &=  R_{1,2}v_2 m_{e,2} + f_{i,2}m_{i,2}

In matrix form the Bloch equations are exactly the same as the n-compartment 
kinetic equations:

.. math::
  :label: Mz-RX

    \frac{d\mathbf{M}}{dt} = \mathbf{J} - \mathbf{\Lambda} \mathbf{M}

Here :math:`\Lambda` is a square matrix which has off-diagonal elements 
:math:`-K_{ij}` and diagonal elements :math:`K_i`. 

The equations, and therefore their solutions, are formally identical to the 
fast-exchange situation (Eq. :eq:`Mz-FX`). If the relaxation rates :math:`R_1` 
are constant in time, or changing slowly on the time scale we are interested 
in, the solution is a direct generalization of the fast exchange case (see 
Eq. :eq:`Mz-FX solution`):

.. math::

  \mathbf{M}(t) = e^{-t\mathbf{\Lambda}}\mathbf{M}(0) 
  + e^{-t\mathbf{\Lambda}}*\mathbf{J}

If additionally the influx :math:`\mathbf{J}` is constant, the result is 
formall the same as Eq. :eq:`Mz-FX solution const J`:

.. math::

  \mathbf{M}(t) = e^{-t\mathbf{\Lambda}}\mathbf{M}(0) 
  + \left(1-e^{-t\mathbf{\Lambda}}\right) \mathbf{\Lambda}^{-1}\mathbf{J}


The effect of contrast agents
+++++++++++++++++++++++++++++

With standard doses of contrast agents used in in-vivo MRI acquisitions, the 
contrast agent increases the longitudinal relaxation rate of tissue in 
proportion to its concentration:

.. math::
  :label: R1 lin

  R_1(c) = R_{10} + r_1 c

The **relaxivity** :math:`r_1` is a constant which depends on the contrast agent. 
It generally has at most a weak dependence on tissue type, except for contrast 
agents which exihibit stronger levels of protein binding. This linear 
relationship is a very good approximation under most conditions. 

In the absence of contrast agent, tissues with different :math:`R_1` values 
nevertheless show mono-exponential longitudinal relaxation because of 
the fast water exchange between them. The magnetization in this fast 
water-exchange limit relaxes with a single :math:`R_1` which is a weighted 
average of the :math:`R_1` values of the different compartments:

.. math::

  R_1 = \sum_i v_i R_{1,i}

The result can be proven by considering the limit :math:`PS>>R_1` in a 
multi-compartment model. 

If each tissue component has a different concentration :math:`c_i`, but each 
compartment has the same relaxivity :math:`r_1`, the relaxation rate shows a 
linear dependence on the total tissue concentration :math:`C`:

.. math::

  R_1 = R_{10} + r_1 C 
  \quad\textrm{with}\quad
  R_{10} = \sum_i v_i R_{10,i}
  \quad\textrm{and}\quad
  C = \sum_i v_i c_i

In this regime the longitudinal relaxation is not affected by how the 
indicator is distributed over the compartments exactly. 
This is no longer the case if the tissue compartments have 
different relaxivities. In that case the result must be generalized:

.. math::

  R_1 = R_{10} + \sum_i  r_{1,i} v_ic_i

In this case, the change in :math:`R_1` is explicitly dependent on the exact 
distribution of the indicator over the tissue compartments. In other words, 
two states with 
the same total tissue concentration :math:`C` can nevertheless 
have different :math:`R_1` values. In such a scenario, the concentrations 
cannot be derived directly from the relaxation rates. A relevant example is 
the use of the hepatobiliary agent gadoxetate, which at 
most field strengths shows a 2-fold increase in relaxivity as soon as it 
enters the hepatocytes.

If the tissue is not in the fast water exchange limit, it is no longer 
characterised by a single :math:`R_1` value, and the effect of concentration 
must be determined by applying Eq. :eq:`R1 lin` to the relaxation rates of 
each compartment individually. 


Transverse relaxation
^^^^^^^^^^^^^^^^^^^^^

Like longitudinal relaxation, transverse magnetization is often approximated 
by a linear relationship:

.. math::

  R^*_2(C) = R^*_{10} + r^*_2 C

However, unlike the longitudinal relaxivity :math:`r_1`, the transverse 
relaxivity :math:`r^*_2` is strongly dependend on tissue type. Hence using 
literature values is not usually realistic.  

[... coming soon ...] The effect of contrast agent leakage.















