import sys
import os
import pickle

# filepaths need to be identified with importlib_resources
# rather than __file__ as the latter does not work at runtime 
# when the package is installed via pip install

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources


def fetch(dataset:str)->dict:
    """Fetch a dataset included in dcmri

    Args:
        dataset (str): name of the dataset. See below for options.

    Returns:
        dict: Data as a dictionary. 

    Notes:

        The following datasets are currently available:

        **tristan2scan**

            *Data format*: a dictionary with two fields, one for each visit, labelled as 'baseline' and 'rifampicin'. Each of those is a dictionary with one field per subject, labelled as '001', '002' until '010'. Then for each visit and each subject the value is a dictionary containing xdata (tuple), ydata (tuple) and params (dictionary with experimental parameters such as sequence parameters and injection protocol). 

            *Background*: data are provided by the liver work package of the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging biomarkers for drug safety assessment. The data and analysis was first presented at the ISMRM in 2024 (Min et al 2024, manuscript in press). 

            The data were acquired in the aorta and liver of 10 healthy volunteers with dynamic gadoxetate-enhanced MRI, before and after administration of a drug (rifampicin) which is known to inhibit liver function. The assessments were done on two separate visits at least 2 weeks apart. On each visit, the volunteer had two scans each with a separate contrast agent injection of a quarter dose each. the scans were separated by a gap of about 1 hour to enable gadoxetate to clear from the liver. This design was deemed necessary for reliable measurement of excretion rate when liver function was inhibited.

            The research question was to what extent rifampicin inhibits gadoxetate uptake rate from the extracellular space into the liver hepatocytes (khe, mL/min/100mL) and excretion rate from hepatocytes to bile (kbh, mL/100mL/min). 2 of the volunteers only had the baseline assessment, the other 8 volunteers completed the full study. The results showed consistent and strong inhibition of khe (95%) and kbh (40%) by rifampicin. This implies that rifampicin poses a risk of drug-drug interactions (DDI), meaning it can cause another drug to circulate in the body for far longer than expected, potentially causing harm or raising a need for dose adjustment.

            Please reference the following abstract when using these data:

            Thazin Min, Marta Tibiletti, Paul Hockings, Aleksandra Galetin, Ebony Gunwhy, Gerry Kenna, Nicola Melillo, Geoff JM Parker, Gunnar Schuetz, Daniel Scotcher, John Waterton, Ian Rowe, and Steven Sourbron. *Measurement of liver function with dynamic gadoxetate-enhanced MRI: a validation study in healthy volunteers*. Proc Intl Soc Mag Reson Med, Singapore 2024.

        **tristan1scan**

            Data from the same study as those that produced **tristan2scan**, but this time only included the data from the first scan. These were used for a secondary objective to test if results are significantly improved by including the second scan.

            Please reference the above abstract when using these data in publications.

    Example:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc

        Use the AortaLiver model to fit the **tristan1scan** data:     

        >>> data = dc.fetch('tristan1scan')

        Fit the baseline visit for the first subject:

        >>> data_subj = data['baseline']['001']
        >>> model = dc.AortaLiver(**data_subj['params'])
        >>> model.train(data_subj['xdata'], data_subj['ydata'], xtol=1e-3)

        Plot the results to check that the model has fitted the data:

        >>> model.plot(data_subj['xdata'], data_subj['ydata'])
    """


    f = importlib_resources.files('dcmri.datafiles')
    datafile = str(f.joinpath(dataset + '.pkl'))

    with open(datafile, 'rb') as fp:
        data_dict = pickle.load(fp)

    return data_dict