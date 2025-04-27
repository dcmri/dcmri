.. _signal mods:

*****************
MRI signal models
*****************

MRI signal models available in `dcmri`, as well as their inverses and some 
utilities such as sampling and adding noise. For more background on these 
models, see the section on :ref:`imaging sequences <imaging-sequences>`.

.. currentmodule:: dcmri


Signal models
=============

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   signal_dsc
   signal_t2w
   signal_ss
   signal_spgr
   signal_free
   signal_lin


.. _inverse-signal-models:

Inverse signal models
=====================

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_t2w
   conc_ss
   conc_src
   conc_lin
