.. _utilities:

*********
Utilities
*********

.. currentmodule:: dcmri

A collection of helper functions that may be useful for testing code, 
building examples, or to construct new models.

.. _real-data:


Reading and writing
===================

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   read_dmr
   write_dmr


Real data
=========

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   fetch

.. _synthetic-data:

Synthetic data
==============

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   fake_aif
   fake_brain
   fake_tissue
   fake_liver
   fake_kidney
   fake_tissue2scan

.. _synthetic-images:

Synthetic images
================

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   shepp_logan


.. _input-functions:

Input functions
===============

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   aif_parker
   aif_tristan
   aif_tristan_rat
   ca_injection

.. _useful-constants:

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


.. _convolution-functions:

Convolution
===========

Convolution is an essential mathematical tool for solving linear and 
stationary compartment models. Explicit numerical convolution is slow, and 
`dcmri` includes apart from a generic convolution method also some faster and 
more accurate functions for use in special cases where one or both of the 
factors have a known form.

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   conv
   expconv
   biexpconv
   nexpconv
   stepconv


.. _sampling-functions:

Helper functions
================


.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   sample
   add_noise
   interp
