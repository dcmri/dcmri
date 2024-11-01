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

.. _basics-relaxation-T1:

Longitudinal relaxation
^^^^^^^^^^^^^^^^^^^^^^^

In the absence of external drivers, the longitudinal magnetization :math:`M_z` 
of tissue relaxes mono-exponentially from an inital state :math:`M_z(0)` to 
an equilibrium state :math:`M_0`:

.. math::

  M_z(t) = e^{-tR_1}M_z(0) + \left(1-e^{-tR_1}\right)M_0

With standard doses of contrast agents used in in-vivo MRI acquisitions, the 
contrast agent increases the longitudinal relaxation rate of tissue in 
proportion to its concentration:

.. math::

  R_1(c) = R_{10} + r_1 c

The **relaxivity** :math:`r_1` is a constant which depends on the contrast agent. 
It generally has at most a weak dependence on tissue type, except for contrast 
agents which exihibit stronger levels of protein binding. This linear 
relationship is a very good approximation under most conditions. 

In the absence of contrast agent, tissues that are built on components with 
different :math:`R_1` nevertheless show mono-exponential relaxation of their 
longitudinal magnetization. This is because the magnetization is carried by 
water which exchanges rapidly between the tissue compartments. It acts as a 
mixer for the magnetization which effectively levels out any differences 
between compartments. The magnetization in this **fast water-exchange limit** 
relaxes with a single :math:`R_1` which is a weighted average of the 
:math:`R_1` values of the different compartments:

.. math::

  R_1 = \sum_i v_i R_{1,i}

If the relaxivity is a constant but each tissue recieves a different 
concentration :math:`c_i`, the end-result is a linear dependence on the tissue 
concentration:

.. math::

  R_1 = R_{10} + r_1 C 
  \qquad\textrm{with:}\qquad
  R_{10} = \sum_i v_i R_{10,i}
  \quad\textrm{and}\quad
  C = \sum_i v_i c_i

Hence in the limit of fast water exchange, the magnetic properties of the 
tissue are fully determined by the total tissue concentrations - and is not 
affected by how the indicator is distributed over the compartments exactly. 
This is no longer the case if the different tissue compartments have 
different relaxivities. In that case the result must be generalized:

.. math::

  R_1 = R_{10} + \sum_i  r_{1,i} v_ic_i
  \qquad\textrm{with:}\qquad
  R_{10} = \sum_i v_i R_{10,i}

In this case, the change in :math:`R_1` is explicitly dependent on the exact 
distribution of the indicator over the tissue compartments. A relevant example 
of this scenario is the use of the hepatobiliary agent gadoxetate, which at 
most field strengths shows a 2-fold increase in relaxivity in the hepatocytes.

[... coming soon ...] The effect of restricted water exchange.


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


Terms and definitions
^^^^^^^^^^^^^^^^^^^^^

Models of magnetic relaxation are determined by the following parameters:

.. _relaxation-params:
.. list-table:: **Relaxation model parameters**
    :widths: 15 20 40 10
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
    * - R10
      - Precontrast longitudinal relaxation rate in tissue
      - Native longitudinal relaxation rate in the absence of contrast agent
      - Hz
    * - R10a
      - Precontrast longitudinal relaxation rate in arterial blood
      - Native longitudinal relaxation rate in blood in the absence of 
        contrast agent
      - Hz
    * - r1
      - Longitudinal relaxivity
      - Increase in longitudinal relaxation rate R1 per unit concentration
      - Hz/M
    * - r2*
      - Longitudinal relaxivity
      - Increase in transverse relaxation rate R2* per unit concentration
      - Hz/M













