.. _model-fitting:


Optimization
------------

Fitting a forward model to the data always involves the definition of a **cost 
function** which measures the distance between the measured signal, and the 
prediction by the forward model. In most cases the cost function is the 
mean-square-difference between data and prediction. The problem of 
optimization then involves finding the parameters that minimize this cost 
function. 

The most common approach is **iterative optimization**, where starting values 
are defined for the free parameters, and these are then adjusted iteratively to 
bring the prediction closer to the data. The adjustment is done by a 
gradient-descent method, where each parameter is first tested
to measure its effect on the cost function. Then parameters are 
modified in proportion to that effect. 

``dcmri`` performs iterative optimization by default in end-to-end 
applications and uses `scipy.optimize.curve_fit`. Iterative optimization is 
flexible and convenient but also has some downsides - notably the risk of 
returning local optima where the solution depends on the choice of initial 
values. 

The simplest alternative to iterative optimization is a **brute-force** 
approach where the forward model is used to 
predict signals for all possible combinations of tissue parameters. For each 
prediction the fit to the data is measured, and the prediction with the best 
fit is identified. The brute-force approach is reliable and robust, but it 
is often computationally prohibitive. Brute force optimization is available 
in scipy through `scipy.optimize.brute`.

In some cases a **dictionary-based** approach can be used where signals are 
computed up front for all possible combinations of parameters, and these are 
retained in a dictionary and saved. For any given signal it then suffices to 
load the dictionary and find the prediction closest to the measured signal. 
This is fast but requires different dictionaries to be generated if, 
for instance, experimental parameters are modified.

An alternative data-driven method uses **deep learning** to approximate the 
inverse relationship that maps the signals to the tissue parameters. Just like 
in a dictionary-based method, the forward model is used to generate signals 
for many combinations of tissue parameters. Rather than storing them in a 
dictionary these are then used to train a neural network to perform the 
inverse transformation. 


  