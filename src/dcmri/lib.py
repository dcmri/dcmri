import math
import sys
import pickle

import numpy as np

import dcmri.utils as utils


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

        **tristan_rifampicin**

            **Background**: data are provided by the liver work package of the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging biomarkers for drug safety assessment. The data and analysis was first presented at the ISMRM in 2024 (Min et al 2024, manuscript in press). 

            The data were acquired in the aorta and liver of 10 healthy volunteers with dynamic gadoxetate-enhanced MRI, before and after administration of a drug (rifampicin) which is known to inhibit liver function. The assessments were done on two separate visits at least 2 weeks apart. On each visit, the volunteer had two scans each with a separate contrast agent injection of a quarter dose each. the scans were separated by a gap of about 1 hour to enable gadoxetate to clear from the liver. This design was deemed necessary for reliable measurement of excretion rate when liver function was inhibited.

            The research question was to what extent rifampicin inhibits gadoxetate uptake rate from the extracellular space into the liver hepatocytes (khe, mL/min/100mL) and excretion rate from hepatocytes to bile (kbh, mL/100mL/min). 2 of the volunteers only had the baseline assessment, the other 8 volunteers completed the full study. The results showed consistent and strong inhibition of khe (95%) and kbh (40%) by rifampicin. This implies that rifampicin poses a risk of drug-drug interactions (DDI), meaning it can cause another drug to circulate in the body for far longer than expected, potentially causing harm or raising a need for dose adjustment.

            **Data format**: The fetch function returns a list of dictionaries, one per subject visit. Each dictionary contains the following items: 

            - **time1aorta**: array of signals in arbitrary units, for the aorta in the first scan.
            - **time2aorta**: array of signals in arbitrary units, for the aorta in the second scan.
            - **time1liver**: array of signals in arbitrary units, for the liver in the first scan.
            - **time2liver**: array of signals in arbitrary units, for the liver in the second scan.
            - **signal1aorta**: array of signals in arbitrary units, for the aorta in the first scan.
            - **signal2aorta**: array of signals in arbitrary units, for the aorta in the second scan.
            - **signal1liver**: array of signals in arbitrary units, for the liver in the first scan.
            - **signal2liver**: array of signals in arbitrary units, for the liver in the second scan.  
            - **weight**: subject weight in kg.
            - **agent**: contrast agent generic name (str).
            - **dose**: 2-element list with contrast agent doses of first scan and second scan in mL/kg.
            - **rate**: contrast agent injection rate in mL/sec.
            - **FA**: Flip angle in degrees
            - **TR**: repretition time in sec
            - **t0**: baseline length in subject       
            - **subject**: Volunteer number. 
            - **visit**: either 'baseline' or 'rifampicin'.
            - **field_strength**: B0-field of scanner.
            - **R10b**: precontrast R1 of blood (1st scan).
            - **R10l**: precontrast R1 of liver (1st scan).
            - **R102b**:  precontrast R1 of blood (2nd scan).
            - **R102l**: precontrast R1 of liver (2nd scan).
            - **Hct**: hematocrit.
            - **vol**: liver volume in mL.
        
            Please reference the following abstract when using these data:

            Thazin Min, Marta Tibiletti, Paul Hockings, Aleksandra Galetin, Ebony Gunwhy, Gerry Kenna, Nicola Melillo, Geoff JM Parker, Gunnar Schuetz, Daniel Scotcher, John Waterton, Ian Rowe, and Steven Sourbron. *Measurement of liver function with dynamic gadoxetate-enhanced MRI: a validation study in healthy volunteers*. Proc Intl Soc Mag Reson Med, Singapore 2024.


        **tristan6drugs**

            **Background**: data are provided by the liver work package of the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging biomarkers for drug safety assessment. The data and analysis were first published in Melillo et al (2023). 

            The study presented here measured gadoxetate uptake and excretion in healthy rats before and after injection of 6 test drugs (up to 6 rats per drug). Studies were performed in preclinical MRI scanners at 3 different centers and 2 different field strengths. 
            
            Results demonstrated that two of the tested drugs (rifampicin and cyclosporine) showed strong inhibition of both uptake and excretion. One drug (ketoconazole) inhibited uptake but not excretion. Three drugs (pioglitazone, bosentan and asunaprevir) inhibited excretion but not uptake. 

            **Data format**: The fetch function returns a list of dictionaries, one per scan. Each dictionary contains the following items: 

            - **time**: array of time points in sec
            - **spleen**: array of spleen signals in arbitrary units
            - **liver**: array of liver signals in arbitrary units.    
            - **FA**: Flip angle in degrees
            - **TR**: repretition time in sec
            - **n0**: number of precontrast acquisitions        
            - **study**: an integer identifying the substudy the scan was taken in
            - **subject**: a study-specific identifier of the subject in the range 1-6.  
            - **visit**: either 1 (baseline) or 2 (drug or vehicle/saline).
            - **center**: center wehere the study was performed, either E, G or D.
            - **field_strength**: B0-field of scanner on whuch the study was performed
            - **substance**: what was injected, eg. saline, vehicle or drug name.
            - **BAT**: Bolus arrival time
            - **duration**: duration on the injection in sec.

            Please reference the following paper when using these data:

            Melillo N, Scotcher D, Kenna JG, Green C, Hines CDG, Laitinen I, Hockings PD, Ogungbenro K, Gunwhy ER, Sourbron S, et al. Use of In Vivo Imaging and Physiologically-Based Kinetic Modelling to Predict Hepatic Transporter Mediated Drug–Drug Interactions in Rats. Pharmaceutics. 2023; 15(3):896. `[DOI] <https://doi.org/10.3390/pharmaceutics15030896>`_ 

            The data were first released as supplementary material in csv format with this paper on Zenodo. Use this DOI to reference the data themselves:

            Gunwhy, E. R., & Sourbron, S. (2023). TRISTAN-RAT (v3.0.0). `Zenodo <https://doi.org/10.5281/zenodo.8372595>`_

        **tristan_repro**

            **Background**: data are provided by the liver work package of the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging biomarkers for drug safety assessment. The data and analysis were first published in Gunwhy et al (2024). 

            The study presented here aimed to determine the repreducibility and rpeatability of gadoxetate uptake and excretion measurements in healthy rats. Data were acquired in different centers and field strengths to identify contributing factors. Some of the studies involved repeat scans in the same subject. In some studies data on the second day were taken after adminstration of a drug (rifampicin) to test if effect sizes were reproducible.

            **Data format**: The fetch function returns a list of dictionaries, one per scan. The dictionaries in the list contain the following items: 

            - **time**: array of time points in sec
            - **spleen**: array of spleen signals in arbitrary units
            - **liver**: array of liver signals in arbitrary units.    
            - **FA**: Flip angle in degrees
            - **TR**: repretition time in sec
            - **n0**: number of precontrast acquisitions        
            - **study**: an integer identifying the substudy the scan was taken in
            - **subject**: a study-specific identifier of the subject in the range 1-6.  
            - **visit**: either 1 (baseline) or 2 (drug or vehicle/saline).
            - **center**: center wehere the study was performed, either E, G or D.
            - **field_strength**: B0-field of scanner on whuch the study was performed
            - **substance**: what was injected, eg. saline, vehicle or drug name.
            - **BAT**: Bolus arrival time
            - **duration**: duration on the injection in sec.

            Please reference the following paper when using these data:

            Ebony R. Gunwhy, Catherine D. G. Hines, Claudia Green, Iina Laitinen, Sirisha Tadimalla, Paul D. Hockings, Gunnar Schütz, J. Gerry Kenna, Steven Sourbron, and John C. Waterton. Assessment of hepatic transporter function in rats using dynamic gadoxetate-enhanced MRI: A reproducibility study. In review.

            The data were first released as supplementary material in csv format with this paper on Zenodo. Use this to reference the data themselves:

            Gunwhy, E. R., Hines, C. D. G., Green, C., Laitinen, I., Tadimalla, S., Hockings, P. D., Schütz, G., Kenna, J. G., Sourbron, S., & Waterton, J. C. (2023). Rat gadoxetate MRI signal dataset for the IMI-WP2-TRISTAN Reproducibility study [Data set]. `Zenodo. <https://doi.org/10.5281/zenodo.7838397>`_


    Example:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc

        Use the AortaLiver model to fit the **tristan_rifampicin** data:     

        >>> data = dc.fetch('tristan_rifampicin')

        Fit the first scan of the baseline visit for the first subject:

        >>> data = data[0]

        >>> model = dc.AortaLiver(
        >>>     # Injection parameters
        >>>     weight = data['weight'],
        >>>     agent = data['agent'],
        >>>     dose = data['dose'][0],
        >>>     rate = data['rate'],
        >>>     # Acquisition parameters
        >>>     field_strength = data['field_strength'],
        >>>     t0 = data['t0'],
        >>>     TR = data['TR'],
        >>>     FA = data['FA'],
        >>>     # Signal parameters
        >>>     R10b = data['R10b'],
        >>>     R10l = data['R10l'],
        >>>     # Tissue parameters
        >>>     Hct = data['Hct'],
        >>>     vol = data['vol'],
        >>> )

        >>> xdata = (data['time1aorta'], data['time1liver'])
        >>> ydata = (data['signal1aorta'], data['signal1liver'])
        
        >>> model.train(xdata, ydata, xtol=1e-3)

        Plot the results to check that the model has fitted the data:

        >>> model.plot(xdata, ydata)
    """


    f = importlib_resources.files('dcmri.datafiles')
    datafile = str(f.joinpath(dataset + '.pkl'))

    with open(datafile, 'rb') as fp:
        data_dict = pickle.load(fp)

    return data_dict




def influx_step(t:np.ndarray, weight:float, conc:float, dose:float, rate:float, t0:float)->np.ndarray:
    """Contrast agent flux (mmol/sec) generated by step injection.

    Args:
        t (numpy.ndarray): time points in sec where the flux is to be calculated.
        weight (float): weight of the subject in kg.
        conc (float): concentration of the contrast agent in mmol/mL.
        dose (float): injected dose in mL/kg body weight.
        rate (float): rate of contrast agent injection in mL/sec.
        t0 (float): start of the injection in sec.
        
    Raises:
        ValueError: if the injection duration is zero, or smaller than the time step of the time array.

    Returns:
        numpy.ndarray: contrast agent flux for each time point in units of mmol/sec.

    Example:

        >>> import numpy as np
        >>> import dcmri as dc

        Create an array of time points covering 20sec in steps of 1.5sec. 

        >>> t = np.arange(0, 20, 1.5)

        Inject a dose of 0.2 mL/kg bodyweight at a rate of 3mL/sec starting at t=5sec. 
        For a subject weighing 70 kg and a contrast agent with concentration 0.5M, this produces the flux:

        >>> dc.influx_step(t, 70, 0.5, 5, 0.2, 3)
        array([0. , 0. , 0. , 0. , 1.5, 1.5, 1.5, 0. , 0. , 0. , 0. , 0. , 0. ,0. ])
    """

    # Get timings
    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    dt = np.amin(t[1:]-t[:-1])

    # Check consistency of timings
    if dose > 0:
        if duration==0:
            msg = 'Invalid input variables. \n' 
            msg = 'The injection duration is zero.'
            raise ValueError(msg)
        if dt >= duration:
            msg = 'Invalid input variables. \n' 
            msg = 'The smallest time step dt ('+dt+' sec) is larger than the injection duration 1 (' + duration + 'sec). \n'
            msg = 'We would recommend dt to be at least 5 times smaller.'
            raise ValueError(msg)

    # Build flux 
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    J = np.zeros(t.size)
    J[(0 < t) & (t < duration)] = Jmax
    return np.interp(t-t0, t, J, left=0)


def ca_conc(agent:str)->float:
    """Contrast agent concentration

    Args:
        agent (str): Generic contrast agent name, all lower case. Examples are 'gadobutrol', 'gadobenate', etc.

    Raises:
        ValueError: If no data are available for the agent.

    Returns:
        float: concentration in mmol/mL

    Notes:
        Sources: 
            `<https://mriquestions.com/so-many-gd-agents.html>`_
            `<https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf>`_
            `<https://www.medicines.org.uk/emc/product/2876/smpc#gref>`_

    Example:

        Print the concentration of the agents gadobutrol and gadoterate:

    .. exec_code::

        import dcmri as dc

        print('gadobutrol is available in a solution of', dc.ca_conc('gadobutrol'), 'M')
        print('gadoterate is available in a solution of', dc.ca_conc('gadoterate'), 'M')
    """    
    if agent == 'gadoxetate':
        return 0.25     # mmol/mL
    if agent == 'gadobutrol':
        return 1.0      # mmol/mL
    if agent in [
            'gadopentetate',
            'gadobenate',
            'gadodiamide',
            'gadoterate',
            'gadoteridol',
            'gadopiclenol',
        ]:
        return 0.5  # mmol/mL 
    raise ValueError('No concentration data for contrast agent ' + agent)


def ca_std_dose(agent:str)->float:
    """Standard injection volume (dose) in mL per kg body weight.

    Args:
        agent (str): Generic contrast agent name, all lower case. Examples are 'gadobutrol', 'gadobenate', etc.

    Raises:
        ValueError: If no data are available for the agent.

    Returns:
        float: Standard injection volume in mL/kg.

    Notes:
        Sources: 
            `<https://mriquestions.com/so-many-gd-agents.html>`_
            `<https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf>`_
            `<https://www.medicines.org.uk/emc/product/2876/smpc#gref>`_

    Example:

        >>> import dcmri as dc
        >>> print('The standard clinical dose of gadobutrol is', dc.ca_std_dose('gadobutrol'), 'mL/kg')
        The standard clinical dose of gadobutrol is 0.1 mL/kg
    """    
    #"""Standard dose in mL/kg""" # better in mmol/kg, or offer it as an option
    if agent == 'gadoxetate':
        # https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf
        return 0.1  # mL/kg
    if agent == 'gadobutrol':
        return 0.1      # mL/kg
    if agent == 'gadopiclenol':
        return 0.1      # mL/kg
    if agent in [
            'gadopentetate',
            'gadobenate',
            'gadodiamide',
            'gadoterate',
            'gadoteridol',
            ]:
        return 0.2      # mL/kg  # 0.5 mmol/mL = 0.1 mmol/kg
    raise ValueError('No dosage data for contrast agent ' + agent)


def relaxivity(field_strength=3.0, tissue='plasma', agent='gadoxetate', type='T1')->float: 
    """Contrast agent relaxivity values in units of Hz/M

    Args:
        field_strength (float, optional): Field strength in Tesla. Defaults to 3.0.
        tissue (str, optional): Tissue type - options are 'plasma', 'hepatocytes'. Defaults to 'plasma'.
        agent (str, optional): Generic contrast agent name, all lower case. Examples are 'gadobutrol', 'gadobenate', etc.. Defaults to 'gadoxetate'.
        type (str, optional): transverse (T2) or longitudinal (T1) relaxivity. Defaults to 'T1'.

    Returns:
        float: relaxivity in Hz/M or 1/(sec*M)

    Notes:
        Sources: 
            `<https://journals.lww.com/investigativeradiology/FullText/2005/11000/Comparison_of_Magnetic_Properties_of_MRI_Contrast.5.aspx>`_
            Szomolanyi P, et al. Comparison of the Relaxivities of Macrocyclic Gadolinium-Based Contrast Agents in Human Plasma at 1.5, 3, and 7 T, and Blood at 3 T. Invest Radiol. 2019 Sep;54(9):559-564. doi: 10.1097/RLI.0000000000000577.

    Example:

        >>> import dcmri as dc
        >>> print('The plasma relaxivity of gadobutrol at 3T is', 1e-3*dc.relaxivity(3.0, 'plasma', 'gadobutrol'), 'Hz/mM')
        The plasma relaxivity of gadobutrol at 3T is 5.0 Hz/mM    
    """    
    # Blood and plasma have (theoretically) the same relaxivity
    if tissue == 'blood':
        tissue = 'plasma'
    rel = {}
    rel['T1'] = {
        'plasma': {
            'gadopentetate':{ #Magnevist
                0.47: 3.8, 
                1.5: 4.1,
                3.0: 3.7,
                4.7: 3.8,
            },
            'gadobutrol': { # Gadovist
                0.47: 6.1, 
                1.5: 5.2,
                3.0: 5.0,
                4.7: 4.7,
            },
            'gadoteridol': { #Prohance
                0.47: 4.8, 
                1.5: 4.1,
                3.0: 3.7,
                4.7: 3.7,
            },
            'gadobenade': { #Multihance
                0.47: 9.2, 
                1.5: 6.3,
                3.0: 5.5,
                4.7: 5.2,
            },
            'gadoterate': { # Dotarem
                0.47: 4.3, 
                1.5: 3.6,
                3.0: 3.5,
                4.7: 3.3,
            },
            'gadodiamide': { #Omniscan
                0.47: 4.4, 
                1.0: 4.35, # Interpolated
                1.5: 4.3,
                3.0: 4.0,
                4.7: 3.9,
            },
            'mangafodipir': { #Teslascan
                0.47: 3.6, 
                1.5: 3.6,
                3.0: 2.7,
                4.7: 2.2,
            },
            'gadoversetamide': {#Optimark
                0.47: 5.7, 
                1.5: 4.7,
                3.0: 4.5,
                4.7: 4.4,
            },
            'ferucarbotran': { #Resovist
                0.47: 15, 
                1.5: 7.4,
                3.0: 3.3,
                4.7: 1.7,
            },
            'ferumoxide': { #Feridex
                1.5: 4.5,
                3.0: 2.7,
                4.7: 1.2,
            },
            'gadoxetate': { # Primovist
                0.47: 8.7, 
                1.5: 8.1,
                3.0: 6.4,
                4.7: 6.4,
                7.0: 6.2,
                9.0: 6.1
            },
        },
    }
    rel['T1']['hepatocytes'] = rel['T1']['plasma']
    rel['T1']['hepatocytes']['gadoxetate'] = {
        1.5: 14.6,
        3.0: 9.8,
        4.7: 7.6,
        7.0: 6.0,
        9.0: 6.1,
    }
    try:
        return 1000*rel[type][tissue][agent][field_strength]
    except:
        msg = 'No relaxivity data for ' + agent + ' at ' + str(field_strength) + ' T.'
        raise ValueError(msg)


def T1(field_strength=3.0, tissue='blood', Hct=0.45)->float:
    """T1 value of selected tissue types.

    Values are taken from literature, mostly from `Stanisz et al 2005 <https://doi.org/10.1002/mrm.20605>`_

    Args:
        field_strength (float, optional): Field strength in Tesla. Defaults to 3.0.
        tissue (str, optional): Tissue type. Defaults to 'blood'.
        Hct (float, optional): Hematocrit value - ignored when tissue is not blood. Defaults to 0.45.

    Raises:
        ValueError: If the requested T1 values are not available.

    Returns:
        float: T1 values in sec

    Example:

        >>> import dcmri as dc
        >>> print('The T1 of liver at 1.5T is', 1e3*dc.T1(1.5, 'liver'), 'msec')
        The T1 of liver at 1.5T is 602.0 msec
    """    
    T1val = {
        'muscle':{
            1.5: 1.008,
            3.0: 1.412,  
        },
        'heart':{
            1.5: 1.030,
            3.0: 1.471,  
        },
        'cartilage':{
            1.5: 1.024,
            3.0: 1.168,  
        },
        'white matter':{
            1.5: 0.884,
            3.0: 1.084,  
        },
        'gray matter':{
            1.5: 1.124,
            3.0: 1.820,  
        },
        'optic nerve':{
            1.5: 0.815,
            3.0: 1.083,  
        },
        'spinal cord':{
            1.5: 0.745,
            3.0: 0.993,  
        },
        'blood':{
            1.0: 1.378, # Extrapolated
            1.5: 1.441,
            3.0: 1/(0.52 * Hct + 0.38),  # Lu MRM 2004
        },
        'spleen':{
            4.7: 1/0.631,
            7.0: 1/0.611,
            9.0: 1/0.600,
        },
        'liver':{
            1.5: 0.602, # liver R1 in 1/sec (Waterton 2021)
            3.0: 0.752, # liver R1 in 1/sec (Waterton 2021)
            4.7: 1/1.281, # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
            7.0: 1/1.109,  # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
            9.0: 1/0.920, # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)
        },
        'kidney':{
            # Reference values average over cortex and medulla from Cox et al
            # https://academic.oup.com/ndt/article/33/suppl_2/ii41/5078406
            1.0: 1.017, # Extrapolated
            1.5: (1.024+1.272)/2,
            3.0: (1.399+1.685)/2,
        },
    }  
    try:
        return T1val[tissue][field_strength]
    except:
        msg = 'No T1 values for ' + tissue + ' at ' + str(field_strength) + ' T.'
        raise ValueError(msg)


def aif_parker(t, BAT:float=0.0)->np.ndarray:
    """Population AIF model as defined by `Parker et al (2006) <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21066>`_

    Args:
        t (array_like): time points in units of sec. 
        BAT (float, optional): Time in seconds before the bolus arrives. Defaults to 0 sec (no delay). 

    Returns:
        np.ndarray: Concentrations in M for each time point in t. If t is a scalar, the return value is a scalar too.

    References:
        Adapted from a contribution by the QBI lab of the University of Manchester to the `OSIPI code repository <https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection>`_. 
        
    Example:

        >>> import numpy as np
        >>> import dcmri as dc

        Create an array of time points covering 20sec in steps of 1.5sec, which rougly corresponds to the first pass of the Paeker AIF:

        >>> t = np.arange(0, 20, 1.5)

        Calculate the Parker AIF at these time points, and output the result in units of mM:

        >>> 1000*dc.aif_parker(t)
        array([0.08038467, 0.23977987, 0.63896354, 1.45093969, 
        2.75255937, 4.32881325, 5.6309778 , 6.06793854, 5.45203828,
        4.1540079 , 2.79568217, 1.81335784, 1.29063036, 1.08751679])
    """

    # Check input types
    if not np.isscalar(BAT):
        raise ValueError('BAT must be a scalar')

    # Convert from secs to units used internally (mins)
    t_offset = np.array(t)/60 - BAT/60

    #A1/(SD1*sqrt(2*PI)) * exp(-(t_offset-m1)^2/(2*var1))
    #A1 = 0.833, SD1 = 0.055, m1 = 0.171
    gaussian1 = 5.73258 * np.exp(
        -1.0 *
        (t_offset - 0.17046) * (t_offset - 0.17046) /
        (2.0 * 0.0563 * 0.0563) )
    
    #A2/(SD2*sqrt(2*PI)) * exp(-(t_offset-m2)^2/(2*var2))
    #A2 = 0.336, SD2 = 0.134, m2 = 0.364
    gaussian2 = 0.997356 * np.exp(
        -1.0 *
        (t_offset - 0.365) * (t_offset - 0.365) /
        (2.0 * 0.132 * 0.132))
    # alpha*exp(-beta*t_offset) / (1+exp(-s(t_offset-tau)))
    # alpha = 1.064, beta = 0.166, s = 37.772, tau = 0.482
    sigmoid = 1.050 * np.exp(-0.1685 * t_offset) / (1.0 + np.exp(-38.078 * (t_offset - 0.483)))

    pop_aif = gaussian1 + gaussian2 + sigmoid
    
    return pop_aif/1000 # convert to M



def aif_tristan_rat(t, 
        BAT = 4.6*60,
        duration = 30,   # sec
    ) -> np.ndarray:
    """Population AIF model for rats measured with a standard dose of gadoxetate. 

    Args:
        t (array_like): time points in units of sec. 
        BAT (float, optional): Time in seconds before the bolus arrives. Defaults to 4.6 min. 
        duration (float, optional): Duration of the injection. Defaults to 30s.

    Returns:
        np.ndarray: Concentrations in M for each time point in t. If t is a scalar, the return value is a scalar too.

    References: 

        - Melillo N, Scotcher D, Kenna JG, Green C, Hines CDG, Laitinen I, et al. Use of In Vivo Imaging and Physiologically-Based Kinetic Modelling to Predict Hepatic Transporter Mediated Drug-Drug Interactions in Rats. `Pharmaceutics 2023;15(3):896 <https://doi.org/10.3390/pharmaceutics15030896>`_.

        - Gunwhy, E. R., & Sourbron, S. (2023). TRISTAN-RAT (v3.0.0). `Zenodo <https://doi.org/10.5281/zenodo.8372595>`_
        
    Example:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Create an array of time points over 30 minutes

        >>> t = np.arange(0, 30*60, 0.1)

        Generate the rat input function for these time points:

        >>> cp = dc.aif_tristan_rat(t)

        Plot the result:

        >>> plt.plot(t/60, 1000*cp, 'r-')
        >>> plt.title('TRISTAN rat AIF')
        >>> plt.xlabel('Time (min)')
        >>> plt.ylabel('Plasma concentration (mM)')
        >>> plt.show()
    """
    # Constants
    
    dose = 0.0075   # mmol

    Fb = 2.27/60  # https://doi.org/10.1021/acs.molpharmaceut.1c00206
                  # (Changed from 3.61/60 on 07/03/2022)
                  # From Brown the cardiac output of rats is
                  # 110.4 mL/min (table 3-1) ~ 6.62L/h
                  # From table 3-4, sum of hepatic artery and portal vein
                  # blood flow is 17.4% of total cardiac output ~ 1.152 L/h
                  # Mass of liver is 9.15g, with density of 1.08 kg/L,
                  # therefore ~8.47mL
                  #  9.18g refers to the whole liver, i.e. intracellular tissue
                  # + extracellular space + blood
                  # Dividing 1.152L/h for 8.47mL we obtain ~2.27 mL/h/mL liver
                  # Calculation done with values in Table S2 of our article
                  # lead to the same results
    Hct = 0.418  # Cremer et al, J Cereb Blood Flow Metab 3, 254-256 (1983)
    VL = 8.47    # Scotcher et al 2021, DOI: 10.1021/acs.molpharmaceut.1c00206
                 # Supplementary material, Table S2
    GFR = 0.023  # https://doi.org/10.1152/ajprenal.1985.248.5.F734
    P = 0.172    # Estimated from rat repro study data using PBPK model
                 # Table 3 in Scotcher et al 2021
                 # DOI: 10.1021/acs.molpharmaceut.1c00206
    VB = 15.8    # 0.06 X BW + 0.77, Assuming body weight (BW) = 250 g
                 # Lee and Blaufox. Blood volume in the rat.
                 # J Nucl Med. 1985 Jan;26(1):72-6.
    VE = 30      # All tissues, including liver.
                 # Derived from Supplementary material, Table S2
                 # Scotcher et al 2021
                 # DOI: 10.1021/acs.molpharmaceut.1c00206
    E = 0.4      # Liver extraction fraction, estimated from TRISTAN data.

    # Derived constants
    VP = (1-Hct)*VB
    Fp = (1-Hct) * Fb
    K = GFR + E*Fp*VL
    
    # Model parameters
    KP = (K + P)/VP
    KE = P/VE
    KB = K/VP

    # Influx in mmol/sec
    J = np.zeros(np.size(t))
    Jmax = dose/duration
    J[(t > BAT) & (t < BAT + duration)] = Jmax

    cp, ce = _propagate_2cxm(t, J/K, KP, KE, KB)

    return cp


# TODO: replace by conc_tissue()
def _propagate_2cxm(t: np.ndarray,
                   ca: np.ndarray,
                   KP: float,
                   KE: float,
                   KB: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Calculates propagators for individual compartments in the 2CXM.

    For details and notations see appendix of Sourbron et al. Magn Reson Med 62:672–681 (2009).

    Args:
        t: time points (sec) where the input function is defined
        ca: input function (mmol/mL)
        KP: inverse plasma MTT (sec) = VP/(FP+PS)
        KE: inverse extracellular MTT (sec) = VE/PS
        KB: inverse blood MTT (sec) = VP/FP

    Returns:
        A tuple (cp, ce), where cp is the concentration in the plasma
        compartment, and ce is the concentration in the extracellular
        compartment. Both are in mmol/mL.
    """
    KT = KP + KE
    sqrt = math.sqrt(KT**2-4*KE*KB)

    Kpos = 0.5*(KT + sqrt)
    Kneg = 0.5*(KT - sqrt)

    cpos = utils.expconv(ca, 1/Kpos, t) # normalized
    cneg = utils.expconv(ca, 1/Kneg, t)

    Eneg = (Kpos - KB)/(Kpos - Kneg)

    cp = (1-Eneg)*cpos + Eneg*cneg
    ce = (cneg*Kpos - cpos*Kneg) / (Kpos - Kneg)

    return cp, ce





