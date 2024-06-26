*********
Utilities
*********

.. currentmodule:: dcmri


Example datasets
================

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   fetch
   fake_tissue
   fake_tissue2scan
   fake_kidney_cortex_medulla


Special functions
=================


.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   influx_step
   aif_parker
   aif_tristan_rat


Useful constants
================


.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   ca_conc
   ca_std_dose
   relaxivity
   T1


Convolution
===========

Convolution is an essential mathematical tool for solving linear and stationary compartment models. 
Explicit numerical convolution is slow, and `dcmri`` includes apart from a generic convolution method 
also some faster functions for use in special cases where one or both of the factors have a known form.

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   conv
   expconv
   biexpconv
   nexpconv
   stepconv


Helper functions
================


.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   sample
   add_noise
   interp
