import numpy as np
import dcmri

# 3 parameters
def body_flux_2c(J_vena, Tlh, To, Eb, t=None, dt=1.0, tol = 0.001):
    """Indicator flux through the body modelled with a 2-compartment whole body model.

    The two compartments are arranged in a loop and model the heart-lung system and the other organs, respectively.

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        Tlh (float): Mean transit time of the heart-lung system (sec).
        To (float): Mean transit time of the organs (sec).
        Eb (float): Extraction fraction from the body.
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava and aorta.

    See Also:
        `body_flux_chc`

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
        Jv, Ja = dc.body_flux_2c(Ji, 15, 20, 0.1, t)

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
    J_aorta_total = np.zeros(J_vena.size)
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_comp(J_vena, Tlh, t=t, dt=dt)
        # Propagate through the other organs
        J_vena = (1-Eb)*dcmri.flux_comp(J_aorta, To, t=t, dt=dt)
        # Add to the total flux
        J_aorta_total += J_aorta
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, x=t, dx=dt)
    return J_vena_total, J_aorta_total

# 4 parameters
def body_flux_chc(J_vena, Tlh, Dlh, To, Eb,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    """Indicator flux through the body modelled with a chain-compartment whole body model.

    The chain and compartments are arranged in a loop and model the heart-lung system and the other organs, respectively.

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        Tlh (float): Mean transit time of the heart-lung system (sec).
        Dlh (float): Dispersion of the heart-lung system. Dlh takes values in the range [0,1].
        To (float): Mean transit time of the organs (sec).
        Eb (float): Extraction fraction from the body.
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.
        solver (str, optional): Solver to use for the chain model. Options are 'step' or 'trap', with the latter slower but more accurate at low temporal resolution. Defaults to 'step'.

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
        Jv, Ja = dc.body_flux_chc(Ji, 15, 0.5, 20, 0.1, t)

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
    J_aorta_total = np.zeros(J_vena.size)
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_vena, Tlh, Dlh, t=t, dt=dt, solver=solver)
        # Propagate through the other organs
        J_vena = (1-Eb)*dcmri.flux_comp(J_aorta, To, t=t, dt=dt)
        # Add to the total flux
        J_aorta_total += J_aorta
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, x=t, dx=dt)
    return J_vena_total, J_aorta_total

# 5 parameters
def body_flux_3c(J_vena, Tlh, Eo, Tob, Toe, Eb,
        t=None, dt=1.0, tol = 0.001):
    """Indicator flux through the body modelled with a 3-compartment whole body model.

    The heart-lung system and the other organs are organised in a loop, with the heart and lungs modelled as a single compartment and the other organs as a 2-compartment exchange model.

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        Tlh (float): Mean transit time of the heart-lung system (sec).
        Eo (float): Extraction fraction into the extravascular leakage space.
        Tob (float): Mean transit time of the organ blood space (sec).
        Toe (float): Mean transit time of the organ extravascular space (sec).
        Eb (float): Extraction fraction from the body.
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
        Jv, Ja = dc.body_flux_3c(Ji, 15, 0.2, 20, 180, 0.1, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.plot(t/60, Jv, 'b-', label='Vena cava')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.legend()
        plt.show()
    """
    E_o = [[1-Eo,1],[Eo,0]]
    dose = np.trapz(J_vena, x=t, dx=dt)
    min_dose = tol*dose
    J_aorta_total = np.zeros(J_vena.size)
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_comp(J_vena, Tlh, t=t, dt=dt)
        # Propagate through the other organs
        Ja = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_vena = (1-Eb)*dcmri.flux_2comp(Ja, [Tob, Toe], E_o, t=t, dt=dt)[0,0,:]
        # Add to the total flux
        J_aorta_total += J_aorta
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, x=t, dx=dt)
    return J_vena_total, J_aorta_total

# 6 parameters
def body_flux_ch2c(J_vena, Tlh, Dlh, Eo, Tob, Toe, Eb,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    """Indicator flux through the body modelled with a chain/2-compartment whole body model.

    The heart-lung system and the other organs are organised in a loop, with the heart and lungs modelled as a chain (`flux_chain`) and the other organs as a 2-compartment exchange model (`flux_2comp`).

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        Tlh (float): Mean transit time of the heart-lung system (sec).
        Dlh (float): Dispersion of the heart-lung system. Dlh takes values in the range [0,1].
        Eo (float): Extraction fraction into the extravascular leakage space.
        Tob (float): Mean transit time of the organ blood space (sec).
        Toe (float): Mean transit time of the organ extravascular space (sec).
        Eb (float): Extraction fraction from the body.
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.
        solver (str, optional): Solver to use for the chain model. Options are 'step' or 'trap', with the latter slower but more accurate at low temporal resolution. Defaults to 'step'.

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
        Jv, Ja = dc.body_flux_ch2c(Ji, 15, 0.5, 0.2, 20, 180, 0.1, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.plot(t/60, Jv, 'b-', label='Vena cava')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.legend()
        plt.show()
    """
    E_o = [[1-Eo,1],[Eo,0]] 
    dose = np.trapz(J_vena, x=t, dx=dt)
    min_dose = tol*dose
    J_aorta_total = np.zeros(J_vena.size)
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_vena, Tlh, Dlh, t=t, dt=dt, solver=solver)
        # Propagate through the other organs
        Ja = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_vena = (1-Eb)*dcmri.flux_2comp(Ja, [Tob, Toe], E_o, t=t, dt=dt)[0,0,:]
        # Add to the total flux
        J_aorta_total += J_aorta
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, x=t, dx=dt)
    return J_vena_total, J_aorta_total

# 9 parameters
def body_flux_hlol(J_vena, Tlh, Dlh, Ro, Eo, Tob, Toe, Rl, Tl, Dl,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    """Indicator flux through the body modelling the heart-lung system (chain), the organs (2CXM) and the liver (plug flow compartment).

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        Tlh (float): Mean transit time of the heart-lung system (sec).
        Dlh (float): Dispersion of the heart-lung system. Dlh takes values in the range [0,1].
        Ro (float): fraction of the aorta flux passing through the organs.
        Eo (float): Extraction fraction into the extravascular leakage space.
        Tob (float): Mean transit time of the organ blood space (sec).
        Toe (float): Mean transit time of the organ extravascular space (sec).
        Rl (float): fraction of the aorta flux passing through the liver.
        Tl (float): Mean transit time of the liver (sec).
        Dl (float): Dispersion of the liver. Dl takes values in the range [0,1].
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.
        solver (str, optional): Solver to use for the chain model. Options are 'step' or 'trap', with the latter slower but more accurate at low temporal resolution. Defaults to 'step'.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava, aorta, liver and other organs.

    See Also:
        `body_flux_2c`

    Example:

        Generate fluxes through all compartments:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Generate a stepwise injection:
        t = np.arange(0, 120, 2.0)
        Ji = dc.influx_step(t, 70, 0.5, 0.2, 3, 30)

        # Calculate the fluxes in mmol/sec:
        Jv, Ja, Jl, Jo = dc.body_flux_hlol(Ji, 
            15, 0.5, 0.3, 0.2, 30, 180, 0.5, 30, 0.8, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.plot(t/60, Jv, 'b-', label='Vena cava')
        plt.plot(t/60, Jl, 'g-', label='Liver')
        plt.plot(t/60, Jo, 'k-', label='Organs')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.ylim(0,0.4)
        plt.legend()
        plt.show()
    """
    E_o = [[1-Eo,1],[Eo,0]]
    dose = np.trapz(J_vena, x=t, dx=dt)
    min_dose = tol*dose
    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)
    J_liver_total = np.zeros(J_vena.size)
    J_organs_total = np.zeros(J_vena.size)
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_vena, Tlh, Dlh, t=t, dt=dt, solver=solver)
        # Propagate through liver and other organs
        # Rl = (1-E_liver)*FF_liver
        # Ro = (1-E_kidneys)*(1-FF_liver)
        J_liver = Rl*dcmri.flux_pfcomp(J_aorta, Tl, Dl, t=t, dt=dt)
        Ja = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_organs = Ro*dcmri.flux_2comp(Ja, [Tob, Toe], E_o, t=t, dt=dt)[0,0,:]
        J_vena = J_liver + J_organs
        # Add to the total flux
        J_aorta_total += J_aorta
        J_liver_total += J_liver
        J_organs_total += J_organs
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, x=t, dx=dt)
    return J_vena_total, J_aorta_total, J_liver_total, J_organs_total

# 8 parameters
def body_flux_hlok(J_vena, Tlh, Dlh, Ro, Eo, Tob, Toe, Rk, Tk, 
        t=None, dt=1.0, tol = 0.001, solver='step'):
    """Indicator flux through the body modelling the heart-lung system (chain), the organs (2CXM) and the kidney (compartment).

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        Tlh (float): Mean transit time of the heart-lung system (sec).
        Dlh (float): Dispersion of the heart-lung system. Dlh takes values in the range [0,1].
        Ro (float): fraction of the aorta flux passing through the organs.
        Eo (float): Extraction fraction into the extravascular leakage space.
        Tob (float): Mean transit time of the organ blood space (sec).
        Toe (float): Mean transit time of the organ extravascular space (sec).
        Rk (float): fraction of the aorta flux passing through the kidneys.
        Tk (float): Mean transit time of the kidney (sec).
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.
        solver (str, optional): Solver to use for the chain model. Options are 'step' or 'trap', with the latter slower but more accurate at low temporal resolution. Defaults to 'step'.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava, aorta, kidneys and other organs.

    See Also:
        `body_flux_2c`

    Example:

        Generate fluxes through all compartments:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Generate a stepwise injection:
        t = np.arange(0, 120, 2.0)
        Ji = dc.influx_step(t, 70, 0.5, 0.2, 3, 30)

        # Calculate the fluxes in mmol/sec:
        Jv, Ja, Jl, Jo = dc.body_flux_hlok(Ji, 
            15, 0.5, 0.3, 0.2, 30, 180, 0.5, 15, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.plot(t/60, Jv, 'b-', label='Vena cava')
        plt.plot(t/60, Jl, 'g-', label='Kidneys')
        plt.plot(t/60, Jo, 'k-', label='Organs')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.ylim(0,0.4)
        plt.legend()
        plt.show()
    """
    E_o = [[1-Eo,1],[Eo,0]]
    dose = np.trapz(J_vena, x=t, dx=dt)
    min_dose = tol*dose
    J_aorta_total = np.zeros(J_vena.size)
    J_kidneys_total = np.zeros(J_vena.size)
    J_organs_total = np.zeros(J_vena.size)
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_vena, Tlh, Dlh, t=t, dt=dt, solver=solver)
        # Propagate through kidney and other organs
        # Rk = (1-E_kidneys)*FF_kidneys
        # Ro = (1-E_liver)*(1-FF_kidneys)
        J_kidneys = Rk*dcmri.flux_comp(J_aorta, Tk, t)
        Ja = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_organs = Ro*dcmri.flux_2comp(Ja, [Tob, Toe], E_o, t)[0,0,:]
        J_vena = J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_aorta_total += J_aorta
        J_kidneys_total += J_kidneys
        J_organs_total += J_organs
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, t)
    return J_vena_total, J_aorta_total, J_kidneys_total, J_organs_total


def body_flux_hlolk(J_vena, Tlh, Dlh, Eo, Tob, Toe,
        FFl, El, Tl, Dl, FFk, Ek, Tk,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    """Indicator flux through the body modelling the heart-lung system (chain), the organs (2CXM), the liver (plug flow compartment) and the kidneys (compartment).

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        Tlh (float): Mean transit time of the heart-lung system (sec).
        Dlh (float): Dispersion of the heart-lung system. Dlh takes values in the range [0,1].
        Eo (float): Extraction fraction into the extravascular leakage space.
        Tob (float): Mean transit time of the organ blood space (sec).
        Toe (float): Mean transit time of the organ extravascular space (sec).
        FFl (float): fraction of the aorta flux passing through the liver.
        El (float): Extraction fraction of the liver.
        Tl (float): Mean transit time of the liver (sec).
        Dl (float): Dispersion of the liver. Dl takes values in the range [0,1].
        FFk (float): fraction of the aorta flux passing through the kidneys.
        Ek (float): Extraction fraction of the kidneys.
        Tk (float): Mean transit time of the kidneys (sec).
        t (np.ndarray, optional): Array of time points (sec), must be of equal size as J_vena. If not provided, the time points are uniformly sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        tol (float, optional): Dose tolerance in the solution. The solution propagates the input through the system, until the dose that is left in the system is given by tol*dose0, where dose0 is the initial dose. Defaults to 0.001.
        solver (str, optional): Solver to use for the chain model. Options are 'step' or 'trap', with the latter slower but more accurate at low temporal resolution. Defaults to 'step'.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava, aorta, liver and other organs.

    See Also:
        `body_flux_2c`

    Example:

        Generate fluxes through all compartments:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Generate a stepwise injection:
        t = np.arange(0, 120, 2.0)
        Ji = dc.influx_step(t, 70, 0.5, 0.2, 3, 30)

        # Calculate the fluxes in mmol/sec:
        Jv, Ja, Jl, Jk, Jo = dc.body_flux_hlolk(Ji, 
            15, 0.5, 0.3, 30, 180, 
            0.2, 0.2, 30, 0.8, 
            0.5, 0.4, 20, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.plot(t/60, Jv, 'b-', label='Vena cava')
        plt.plot(t/60, Jl, 'g-', label='Liver')
        plt.plot(t/60, Jk, 'g--', label='Kidneys')
        plt.plot(t/60, Jo, 'k-', label='Organs')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.ylim(0,0.4)
        plt.legend()
        plt.show()
    """
    E_o = [[1-Eo,1],[Eo,0]]
    dose = np.trapz(J_vena, x=t, dx=dt)
    min_dose = tol*dose
    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)
    J_liver_total = np.zeros(J_vena.size)
    J_kidneys_total = np.zeros(J_vena.size)
    J_organs_total = np.zeros(J_vena.size)
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_vena, Tlh, Dlh, t=t, dt=dt, solver=solver)
        # Split into liver, kidneys and other organs
        J_liver = FFl*J_aorta
        J_kidneys = FFk*J_aorta
        J_organs = (1-FFl-FFk)*J_aorta
        # Propagate through liver, kidneys and other organs
        J_liver = (1-El)*dcmri.flux_pfcomp(J_liver, Tl, Dl, t=t, dt=dt)
        J_kidneys = (1-Ek)*dcmri.flux_comp(J_kidneys, Tk, t)
        Jo = np.stack((J_organs, np.zeros(J_organs.size)))
        J_organs = dcmri.flux_2comp(Jo, [Tob, Toe], E_o, t)[0,0,:]
        J_vena = J_liver + J_kidneys + J_organs
        # Add to the total flux
        J_aorta_total += J_aorta
        J_liver_total += J_liver
        J_kidneys_total += J_liver
        J_organs_total += J_organs
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, t)
    return J_vena_total, J_aorta_total, J_liver_total, J_kidneys_total, J_organs_total

# 10 params kidneys
def body_flux_hlok_ns(J_vena,
        Tlh, Dlh,
        Eo, To, Toe,
        E_liver,
        FF_kidneys, E_kidneys, Ke_kidneys, Ta_kidneys,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    E_o = [[1-Eo,1],[Eo,0]]
    dose0 = np.trapz(J_vena, t)
    dose = dose0
    min_dose = tol*dose0
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_vena, Tlh, Dlh, t=t, dt=dt, solver=solver)
        # Split into liver and other organs
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys)*J_aorta
        # Propagate through liver and other organs
        J_kidneys = dcmri.flux_plug(J_kidneys, Ta_kidneys, t)
        J_kidneys = (1-E_kidneys)*dcmri.flux_nscomp(J_kidneys, 1/Ke_kidneys, t)
        J_organs = (1-E_liver)*dcmri.flux_2comp(J_organs, [To, Toe], E_o, t)[0,0,:]
        # Add up outfluxes from liver and other organs
        J_vena = J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, t)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_chain(J_vena_total, Tlh, Dlh, t=t, dt=dt)
    # Return total flux into aorta
    return J_aorta_total

# 12 params liver kidneys
def body_flux_hlolk_ns(J_vena,
        Tlh, Dlh,
        Eo, To, Toe,
        T_gut, FF_liver, E_liver, Ke_liver, 
        FF_kidneys, E_kidneys, Kp_kidneys,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    E_o = [[1-Eo,1],[Eo,0]]
    dose0 = np.trapz(J_vena, t)
    dose = dose0
    min_dose = tol*dose0
    J_vena_total = J_vena
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_vena, Tlh, Dlh, t=t, dt=dt, solver=solver)
        # Split into liver, kidneys and other organs
        J_liver = FF_liver*J_aorta
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_liver-FF_kidneys)*J_aorta
        # Propagate through liver, kidneys and other organs
        J_liver = dcmri.flux_comp(J_liver, T_gut, t)
        J_liver = (1-E_liver)*dcmri.flux_nscomp(J_liver, 1/Ke_liver, t)
        J_kidneys = (1-E_kidneys)*dcmri.flux_nscomp(J_kidneys, Kp_kidneys, t)
        J_organs = dcmri.flux_2comp(J_organs, [To, Toe], E_o, t)[0,0,:]
        # Add up outfluxes from liver, kidneys and other organs
        J_vena = J_liver + J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_vena_total += J_vena
        # Get residual dose
        dose = np.trapz(J_vena, t)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_chain(J_vena_total, Tlh, Dlh, t=t, dt=dt)
    # Return total flux into aorta
    return J_aorta_total