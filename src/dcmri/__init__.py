# dcmri/__init__.py

__version__ = "0.3.1"

# Helper functions imported for testing but not exposed to package users
from . import tools
from . import pk

# Functions exposed to package users
from .tools import (
    stepconv,
    nexpconv,
    biexpconv,
    expconv,
    conv,
)

from .lib import (
    aif_parker,
)

from .pk import (

    # Trap

    res_trap,
    flux_trap,
    conc_trap,
    prop_trap,

    # Pass

    res_pass,
    flux_pass,
    conc_pass,
    prop_pass,

    # Compartment

    res_comp,
    flux_comp,
    conc_comp,
    prop_comp,

    # Plug flow
    
    res_plug,
    flux_plug,
    conc_plug,
    prop_plug,

    # Chain
    
    res_chain,
    flux_chain,
    conc_chain,
    prop_chain,

    # Step
    
    res_step,
    flux_step,
    conc_step,
    prop_step,

    # Free
    
    res_free,
    flux_free,
    conc_free,
    prop_free,

    # N-comp

    conc_ncomp,
    flux_ncomp,
    res_ncomp,
    prop_ncomp,

    # 2-comp

    conc_2comp,
    flux_2comp,
    res_2comp,
    prop_2comp,

    # 2cxm

    conc_2cxm,
    flux_2cxm,
    res_2cxm,
    prop_2cxm,

    # 2cfm

    conc_2cfm,
    flux_2cfm,
    res_2cfm,
    prop_2cfm,

    # nscomp

    conc_nscomp,
    flux_nscomp,

    # mmcomp

    conc_mmcomp,
    flux_mmcomp,
)