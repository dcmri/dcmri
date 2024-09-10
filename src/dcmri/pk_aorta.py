import numpy as np
import dcmri.pk as pk
import dcmri.lib as lib


def aif_tristan(
        t:np.ndarray, 
        agent = 'gadoterate',
        dose = 0.2, 
        rate = 3, 
        BAT = 0, 
        weight = 73, 
        CO = 97, 
        E = 0.07, 
        Thl = 13,  
        Dhl = 0.5,  
        Tp = 25, 
        Te = 350, 
        Ee = 0.18, 
        dtol = 0.01, 
    )->np.ndarray:
    """Arterial input function with default parameters for young healthy volunteers.

    This AIF was measured in the TRISTAN project (Min et al 2024). The default values are for young healthy volunteers but since the AIF is built on a whole-body model of the circulation, they can be modified to generate virtual populations. 

    Reference:
    
    Thazin Min, Marta Tibiletti, Paul Hockings, Aleksandra Galetin, Ebony Gunwhy, Gerry Kenna, Nicola Melillo, Geoff JM Parker, Gunnar Schuetz, Daniel Scotcher, John Waterton, Ian Rowe, and Steven Sourbron. *Measurement of liver function with dynamic gadoxetate-enhanced MRI: a validation study in healthy volunteers*. Proc Intl Soc Mag Reson Med, Singapore 2024.

    Args:
        t (np.ndarray): Array of time points
        agent (str, optional): Contrast agent generic name. Defaults to 'gadoterate'.
        dose (float, optional): Contrast agent dose in mL/kg. Defaults to 0.2.
        rate (float, optional): Contrast agent injection rate in mL/sec. Defaults to 3.
        BAT (float, optional): Bolus arrival time in sec. Defaults to 0.
        weight (float, optional): Subject weight in kg. Defaults to 73.
        CO (float, optional): Cardiac output in mL/sec. Defaults to 97.
        E (float, optional): Body extraction fraction. Defaults to 0.07.
        Thl (float, optional): Mean transit time of the heart-lung system in sec. Defaults to 13.
        Dhl (float, optional): Transit time dispersion of the heart-lung systen. Defaults to 0.5.
        Tp (float, optional): Plasma mean transit time in sec of the other organs. Defaults to 25.
        Te (float, optional): Extravascular mean transit time in sec. Defaults to 350.
        Ee (float, optional): Extraction fraction into the extravascular space. Defaults to 0.18.
        dtol (float, optional): Dose tolerance. Defaults to 0.01.

    Returns:
        np.ndarray: Aorta concentrations in mmol/mL.

    Example:

        Generate AIFs with different levels of cardiac output:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Time points in sec
        t = np.arange(0, 180, 2.0)

        # Plot aifs with different levels of cardiac output, including the default of 100mL/sec.
        plt.plot(t/60, 1000*dc.aif_tristan(t, BAT=30), 'r-', label='AIF (default)')
        plt.plot(t/60, 1000*dc.aif_tristan(t, BAT=30, CO=150), 'g-', label='AIF (increased cardiac output)')
        plt.plot(t/60, 1000*dc.aif_tristan(t, BAT=30, CO=75), 'b-', label='AIF (reduced cardiac output)')
        plt.xlabel('Time (min)')
        plt.ylabel('Concentration (mmol/mL)')
        plt.legend()
        plt.show()    
    """
    conc = lib.ca_conc(agent)
    Ji = lib.influx_step(t, weight, 
            conc, dose, rate, BAT,
    )
    Jb = flux_aorta(Ji, t, E=E,
            heartlung = ['chain', (Thl, Dhl)], 
            organs=['2cxm', ([Tp, Te], Ee)],
            tol=dtol,   
    )
    return Jb/CO



def flux_aorta(J_vena:np.ndarray, 
        t=None, dt=1.0, E=0.1, FFkl=0.0, FFk=0.5, 
        heartlung = ['pfcomp', (10, 0.2)],
        organs = ['2cxm', ([20, 120], 0.15)],
        kidneys = ['comp', (10,)],
        liver = ['pfcomp', (10, 0.2)],
        tol = 0.001):
    
    """Indicator flux through the body modelled with a two-system whole body model (heart-lung system and organ system).

    The heart-lung system and the other organs are organised in a loop, with the heart and lungs modelled as a chain and the other organs as a 2-compartment exchange model (`flux_ncomp`).

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        heartlung (list): 3-element list specifying the model to use for the heart-lung system (see notes for detail).
        organs (dict): 3-element list specifying the model to use for the organs (see notes for detail). 
        kidneys (dict): 3-element list specifying the model to use for the kidneys (see notes for detail). 
        liver (dict): 3-element list specifying the model to use for the liver (see notes for detail). 
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava and aorta.

    Notes:

        1. The lists specifying each organ system consist of 3 elements: the model (str), it parameters (tuple) and any keyword parameters (dict). Any of the basic pk blocks can be used, for instance *chain*, *plug-flow compartment*, and *compartment* would be specified as follows:
        - **chain**: ['chain', (Thl, Dhl), {'solver':'step'}]
        - **plugflow-compartment**: ['pfcomp', (Thl, Dhl), {'solver':'interp'}}
        - **compartment**: ['comp', Thl]
        - **2cxm**: ['2cxm', ([To, Te], Eo)]
        - and so on..

        2. Different architectures can be modelled by setting the flow fractions:
        - Set FFkl=0 to model organs only
        - Set FFk=1 to model organs+kidney
        - Set FFk=0 to model organs+liver.


    Example:

        Generate flux through aorta:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Generate a stepwise injection:
        t = np.arange(0, 120, 2.0)
        Ji = dc.influx_step(t, 70, 0.5, 0.2, 3, 30)

        # Calculate the fluxes in mmol/sec:
        Ja = dc.flux_aorta(Ji, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.legend()
        plt.show()
    """

    dose = np.trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Rk = FFk*FFkl*(1-E)
    Rl = (1-FFk)*FFkl*(1-E)
    Ro = (1-FFkl)*(1-E)

    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)

    while dose > min_dose:

        # Aorta flux of the current pass
        J_aorta = pk.flux(J_vena, *heartlung[1], t=t, dt=dt, kinetics=heartlung[0])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = Ro*pk.flux(J_aorta, *organs[1], t=t, dt=dt, kinetics=organs[0])
        if Rl>0:
            J_vena += Rl*pk.flux(J_aorta, *liver[1], t=t, dt=dt, kinetics=liver[0])
        if Rk>0:
            J_vena += Rk*pk.flux(J_aorta, *kidneys[1], t=t, dt=dt, kinetics=kidneys[0])

        # Get residual dose in current pass
        dose = np.trapezoid(J_vena, x=t, dx=dt)

    return J_aorta_total

