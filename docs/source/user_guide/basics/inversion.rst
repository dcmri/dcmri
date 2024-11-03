.. _model-inversion:


Inversion
---------

In some cases, some of the analysis steps can be performed by direct inversion 
rather than optimization of a non-linear cost function. Direct inversion 
methods typically first determine relaxation rates from signals, then derive 
concentrations and finally fit for the parameters. Direct inversion methods 
are common in the literature, but they don't always apply.

An example is the conversion from signal to concentration using steady-state 
sequences, which 
can be performed analytically for tissues with fast water exchange. Some 
of these methods are available as functions in `dcmri` 
(see :ref:`inverse-signal-models`) and as optional fitting methods in 
end-to-end applications.

[ ... coming soon ...] Direct inversion methods.

  