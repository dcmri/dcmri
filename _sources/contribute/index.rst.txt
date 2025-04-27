.. _contributor-guide:

##########
Contribute
##########


The ``dcmri`` source code is freely accessible on 
`github <https://github.com/dcmri/dcmri>`_. We welcome contributions of any 
kind, large or small, including new features, bug fixes or documentation. 

If you have a suggestion for something that could be improved, but are 
not able to contribute yourself, please post an
`issue <https://github.com/dcmri/dcmri/issues>`_ describing your idea, problem 
or request, and perhaps someone else will pick it up.


**************************
How to make a contribution
**************************

The methodology for contributing to open source projects on github 
is well explained in 
`other resources <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_. 

You can make a contribution by submitting a 
`pull request <https://github.com/dcmri/dcmri/pulls>`_. Before 
doing this though, it is best to open an 
`issue <https://github.com/dcmri/dcmri/issues>`_ 
first with some detail on what you would like to 
contribute, and why, so we can make a plan before diving in. This will 
avoid code conflicts and duplicate work. 


*********************
Contributing examples
*********************

By far the easiest way to get started with a new software package 
is to look for an example that does something similar to what you have in mind, 
and then build on that. 

As `dcmri` is a new package, we are at this stage particularly interested in 
contributions of new :ref:`examples <examples>` to help others get started 
and act as independent tests of the package. 

If you are using ``dcmri`` to 
analyse your data, or develop methods - please consider packaging this up as an example. 
This is not only of great value to the wider community but will also 
increase the visibility and impact of your work. You can add your name 
as author, including links to your profile and relevant papers.

Types of examples
-----------------

Examples can be 
different things but we generally consider the following types:

- **Tutorial**: an educational notebook on an aspect of DCMRI that is 
  explained using methods in the package.
- **Methodological study**: this can be an aspect of method development or 
  evaluation, such as sensitivity analysis, model comparisons, or 
  replications of published results.
- **Use case**: this is an application of `dcmri` in a real world problem, 
  typically an analysis of real data with sufficient power to show an effect 
  (or absence thereof) on meaningful endpoints - such as effect of treatment, 
  differences between groups, changes over time or predictions of outcomes.  

How to write an example
-----------------------

The source code for all existing examples can be found in the folder 
dcmri/docs/examples. Each example is a single python script, saved in a file 
that must start with *plot_*. 

You must follow proper formatting of text and 
titles to make sure your example shows up correctly on the website in notebook 
format. Easiest is to look at some other examples in the source 
code, and copy the formatting.

You can develop and test your example outside of dcmri as a 
standalone script. To include it in the documentation, 
drop it into dcmri/docs/examples. The next 
time the documentation is built, your example will 
appear under the `examples tab <https://dcmri.org/examples>`_ on 
the website, in a notebook format. 

In order to test what your use case will look like before committing it, 
you can build the documentation locally:

1. In your terminal, cd to the folder *dcmri/docs*
2. Then type the command *./make html*
3. When the execution finishes, double-click on the file *index.html* in the 
   folder *dcmri/docs/build/html/*


How to include data?
--------------------

If your example uses data, these must be made accessible to `dcmri` 
users via the `dcmri.fetch` function. For this you need to: 

1. Store your data on `zenodo <https://zenodo.org/>`_ and make them 
   publicly available under a `CC-BY license <https://creativecommons.org/licenses/by/4.0/>`_
2. Add an entry in the `DATASETS` dictionary in 'dcmri/src/data.py'

For developing and testing your example the data do not have to 
be uploaded to zenodo. You can save them in src/datafiles locally 
and this will also make thenm accessible via `dcmri.fetch`. 

When you are happy with your example and want to contribute it to `dcmri`, 
you can upload to zenodo, add to `DATASETS` and delete it from src/datafiles 
locally.

The data can be in any format. For ROI-type data we prefer the 
`dmr format <https://openmiblab.github.io/pydmr/>`_ for consistency 
with other examples. 

Requirements in examples
------------------------

Your example can import any other python packages, but they must be 
listed in the requirements file of the dcmri documentation (dcmri/docs/requirements.txt).
Add it if not.

Review of examples
------------------

We will review examples for clarity in code and text, and also scientific 
content and general fit with the design concepts of dcmri. At some point 
in the future we may formalize this process and define transparent 
criteria, but this will need to be informed by more experience. At this stage 
we recommend posting an issue presenting your idea first, before spending too 
much time working it out in detail. 