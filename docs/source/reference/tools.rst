****************
Helper functions
****************

A collection of helper functions available in `dcmri`.

.. currentmodule:: dcmri


Convolution
===========

Convolution is an essential mathematical tool for solving linear and stationary compartment models. 
Explicit numerical convolution is slow, and `dcmri`` therefore includes apart from a generic convolution method 
also some faster functions for use in special cases where one or both of the factors have a known form.

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   conv
   expconv
   biexpconv
   nexpconv
   stepconv


Helpers
=======

`dcmri` includes a number of convenient wrapper functions useful for building novel models.

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   interp
