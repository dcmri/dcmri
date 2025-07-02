.. _PK blocks:

*********
PK blocks
*********

A collection of basic pharmacokinetic models that can be assembled to build 
more complex models. For more background on these models, see the section on 
:ref:`basic pharmacokinetics <basics-pharmacokinetics>`.

.. currentmodule:: dcmri


Any model
=========

These are high-level wrappers for all many of the other functions in this 
section.

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc
   flux


No-parameter models
===================

.. _reference-trap:

*Trap*
------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_trap
   flux_trap
   res_trap
   prop_trap


One-parameter models
====================

.. _reference-pass:

*Pass*
------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_pass
   flux_pass
   res_pass
   prop_pass


.. _reference-compartment:

*Compartment*
-------------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_comp
   flux_comp
   res_comp
   prop_comp

.. _reference-plug-flow:

*Plug flow*
-----------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_plug
   flux_plug
   res_plug
   prop_plug


Two-parameter models
====================

.. _reference-chain:

*Chain*
-------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_chain
   flux_chain
   res_chain
   prop_chain

.. _reference-pfcomp:

*Plug-flow compartment*
-----------------------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   flux_pfcomp

.. _reference-step:

*Step*
------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_step
   flux_step
   res_step
   prop_step

.. _reference-mmcomp:

*Michaelis-Menten compartment*
------------------------------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_mmcomp
   flux_mmcomp


Three-parameter models
======================

.. _reference-2comp:

*2-compartment exchange*
------------------------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_2cxm
   flux_2cxm


N-parameter models
==================

.. _reference-free:

*Free*
------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_free
   flux_free
   res_free
   prop_free

.. _reference-ncomp:

*N-compartment models*
----------------------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_ncomp
   flux_ncomp
   res_ncomp
   prop_ncomp


.. _reference-nscomp:

*Non-stationary compartment*
----------------------------

.. autosummary::
   :toctree: ../api/
   :template: autosummary.rst

   conc_nscomp
   flux_nscomp