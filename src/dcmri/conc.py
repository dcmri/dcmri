import numpy as np

from dcmri import pk_aorta
from dcmri.func import SuperFunc
from dcmri.utils import lib


class ConcAorta(SuperFunc):
    """Whole-body model for indicator flux through the aorta.

    See section :ref:`whole-body-tissues` for a more detailed description of 
    this model.

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        t (np.ndarray, optional): Array of time points (sec), must be of 
          equal size as J_vena. If not provided, the time points are uniformly 
          sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        E (float, optional): Body extraction fraction. Defaults to 0.1.
        FFkl (float, optional): Fraction of the cardiac output that passes 
          through kidney and liver. Set FFkl=0 for a whole body model without 
          explicit kidney and liver spaces. Defaults to 0.0.
        FFk (float, optional): Kidney fraction of the flow to kidney and liver. 
          With FFk=0, only the liver is modelled. With FFk=1, only the kidneys 
          are modelled. Defaults to 0.5.
        heartlung (list): 3-element list specifying the model to use for the 
          heart-lung system (see notes for detail).
        organs (list): 3-element list specifying the model to use for the 
          organs (see notes for detail). 
        kidneys (list): 3-element list specifying the model to use for the 
          kidneys (see notes for detail). This keyword is ignored if FFkl=0 
          or FFk=0.
        liver (list): 3-element list specifying the model to use for the 
          liver (see notes for detail). This keyword is ignored if FFkl=0 
          or FFk=1.
        tol (float, optional): Dose tolerance in the solution. The solution 
          propagates the input through the system, until the dose that is 
          left in the system is given by tol*dose0, where dose0 is the 
          initial dose. Defaults to 0.001.
        max_it (int, optional): Maximum number of iterations.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava and aorta.

    Notes:

        The lists specifying each organ system consist of 3 elements: the 
        model (str), its parameters (tuple) and any keyword parameters (dict). 
        Any of the basic pharmacokinetic blocks can be used. For instance,
        *chain*, *plug-flow compartment*, and *compartment* would be specified 
        as follows:

          - **chain**: ['chain', (Thl, Dhl), {'solver':'step'}]
          - **plug-flow compartment**: ['pfcomp', (Thl, Dhl), {'solver':'interp'}}
          - **compartment**: ['comp', Thl]
          - **2cxm**: ['2cxm', ([To, Te], Eo)]


    Example:

        Generate flux through aorta:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Generate a stepwise injection:
        t = np.arange(0, 120, 2.0)
        Ji = dc.ca_injection(t, 70, 0.5, 0.2, 3, 30)

        # Calculate the fluxes in mmol/sec:
        Ja = dc.flux_aorta(Ji, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.legend()
        plt.show()
    """
    configs = {
        'heartlung': ['comp', 'pfcomp', 'chain'],
        'organs': ['comp', '2cxm'],
        
    }
    def __init__(self, heartlung='pfcomp', organs='comp', **params):
        cnfg = {
            'heartlung': heartlung, 
            'organs': organs, 
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self, select=None):
        organs = {
            'comp': ['To'],
            '2cxm': ['To', 'To_e', 'Eo']
        }[self._cnfg['organs']]

        heartlung = {
            'comp': ['Thl'],
            'pfcomp': ['Thl', 'Dhl'],
            'chain': ['Thl', 'Dhl'],
        }[self._cnfg['heartlung']]

        body = ['BAT', 'CO', 'Eb']
        const = [
            'dt', 'tmax', 'dose_tolerance', 'field_strength',
            'agent', 'weight', 'dose', 'rate', 
        ]
        if select is None:
            return heartlung + organs + body + const 
        if select=='body':
            return heartlung + organs + body 
    
    def __call__(self, **params) -> np.ndarray:
        p = self._update_pars(**params)
        t = np.arange(0, p['tmax'], p['dt'])

        hl, orgs = self._cnfg['heartlung'], self._cnfg['organs']

        if hl=='comp':
            heartlung = ['comp', (p['Thl'],)]
        elif hl=='pfcomp':
            heartlung = ['pfcomp', (p['Thl'], p['Dhl'])]
        elif hl=='chain':
            heartlung = ['chain', (p['Thl'], p['Dhl'])]

        if orgs=='comp':
            organs = ['comp', (p['To'],)]
        elif orgs=='2cxm':
            organs = ['2cxm', ([p['To'], p['To_e']], p['Eo'])]

        conc = lib.ca_conc(p['agent'])
        Ji = lib.ca_injection(
            t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
        )
        Jb = pk_aorta.flux(
            Ji, E=p['Eb'], dt=p['dt'], tol=p['dose_tolerance'],
            heartlung=heartlung, organs=organs,
        )
        return Jb / p['CO']