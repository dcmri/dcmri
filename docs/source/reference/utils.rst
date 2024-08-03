*********
Utilities
*********

.. currentmodule:: dcmri

A collection of helper functions that may be useful for testing code, building examples, or to construct new models.


Real data
=========

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   fetch


Synthetic data
==============

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   fake_brain
   fake_tissue
   fake_tissue2scan
   fake_kidney_cortex_medulla


Synthetic images
================

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   shepp_logan


Special functions
=================


.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   influx_step


Population AIFs
===============


.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   aif_parker
   aif_tristan
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
   T2 
   PD
   perfusion


Convolution
===========

Convolution is an essential mathematical tool for solving linear and stationary compartment models. 
Explicit numerical convolution is slow, and `dcmri` includes apart from a generic convolution method 
also some faster and more accurate functions for use in special cases where one or both of the factors have a known form.

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
