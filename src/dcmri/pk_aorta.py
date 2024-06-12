import numpy as np
import dcmri.pk as pk



def flux_aorta(J_vena:np.ndarray, 
        # set FFkl=0 to have organs only, set FFk=1 to have organs+kidney, set FFk=0 to have organs+liver
        t=None, dt=1.0, E=0.1, FFkl=0.0, FFk=0.5, 
        heartlung = ['pfcomp', (10, 0.2)],
        organs = ['2cxm', (20, 120, 0.15)],
        kidneys = ['comp', 10],
        liver = ['pfcomp', (10, 0.2)],
        tol = 0.001):
    
    """Indicator flux through the body modelled with a two-system whole body model (heart-lung system and organ system).

    The heart-lung system and the other organs are organised in a loop, with the heart and lungs modelled as a chain (`flux_chain`) and the other organs as a 2-compartment exchange model (`flux_2comp`).

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        heartlung (dict): Dictionary specifying the model to use for the heart-lung system, and its parameters. In order of reducing complexity and computation time, the options are chain (`flux_chain`, default), plug-flow compartment (`flux_pfcomp`), and compartment (`flux_comp`). Each is specified as follows:
        - **chain**: heartlung = {'model':'chain', 'Thl':10, 'Dhl':0.2, 'solver':'step'}
        - **plugflow-compartment**: heartlung = {'model':'pfcomp', 'Thl':10, 'Dhl':0.2, 'solver':'interp'}
        - **compartment**: heartlung = {'model':'comp', 'Thl':10}
        organs (dict): Dictionary specifying the model to use for the organ system, and its parameters. In order of reducing complexity and computation time, the options are two-compartment exchange model (2cxm, default), and compartment. Each is specified as follows:
        - **2cxm**: organs = {'model':'2cxm', 'Eo':0.15, 'To':20, 'Te':120, 'Eb':0.05}
        - **compartment**: organs = {'model':'comp', 'To':20}
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava and aorta.

    See Also:
        `body_flux_2c`

    Example:

        Generate fluxes through aorta and vena cava:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Generate a stepwise injection:
        t = np.arange(0, 120, 2.0)
        Ji = dc.influx_step(t, 70, 0.5, 0.2, 3, 30)

        # Calculate the fluxes in mmol/sec:
        Jv, Ja = dc.body_flux_hlo(Ji, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.plot(t/60, Jv, 'b-', label='Vena cava')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.legend()
        plt.show()
    """

    dose = np.trapz(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Rk = FFk*FFkl*(1-E)
    Rl = (1-FFk)*FFkl*(1-E)
    Ro = (1-FFkl)*(1-E)

    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)

    while dose > min_dose:

        # Aorta flux of the current pass
        J_aorta = pk.flux(J_vena, t=t, dt=dt, system=heartlung)

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = Ro*pk.flux(J_aorta, t=t, dt=dt, system=organs)
        if Rl>0:
            J_vena += Rl*pk.flux(J_aorta, t=t, dt=dt, system=liver)
        if Rk>0:
            J_vena += Rk*pk.flux(J_aorta, t=t, dt=dt, system=kidneys)

        # Get residual dose in current pass
        dose = np.trapz(J_vena, x=t, dx=dt)

    return J_aorta_total

