from dcmri.utils.const import (
    ca_conc,
    ca_std_dose,
    relaxivity,
    r2s,
    r1,
    r2,
    T1,
    T2,
    PD,
    perfusion,
)
from dcmri.utils.convolution import (
   conv,
   stepconv,
   expconv,
   biexpconv,
   nexpconv,
   deconv,
   convmat,
   invconvmat,
)
from dcmri.utils.data import (
    fetch
)
from dcmri.utils.misc import (
    sample,
    add_noise,
    mle_rice,
)
from dcmri.core.quantities import (
    QUANTITIES,
    QVALUES,
    QBOUNDS,
)
from dcmri.core.tools import (
    init,
    bounds,
    select_params,
)

from dcmri.kinetics.modules_conc import (
   ConcAorta,
   ConcAortaKidneys,
   ConcAortaLiver,
   ConcLiver,
   ConcKidney,
   ConcCortMed,
   ConcTissueX,   
)
from dcmri.kinetics.modules_flux import (
    Flux,
    FluxInjection,
    FluxAorta,
    FluxTissueX
)
from dcmri.kinetics.functions_blocks import (
    res_trap,
    res_pass,
    res_comp,
    res_plug,
    res_chain,
    res_step,
    res_free,
    res_ncomp, 

    prop_trap,
    prop_pass,
    prop_comp,
    prop_plug,
    prop_chain,
    prop_step,
    prop_free,
    prop_ncomp,

    conc, # TODO: remove, replace by new module Conc()
    conc_trap,
    conc_pass,
    conc_comp,
    conc_bicomp,
    conc_plug,
    conc_chain,
    conc_step,
    conc_free,
    conc_ncomp,
    conc_nscomp,
    conc_mmcomp,
    conc_2cxm,

    flux, # TODO: remove, replace by new module Flux()
    flux_trap,
    flux_pass,
    flux_comp,
    flux_bicomp,
    flux_plucom,
    flux_plug,
    flux_chain,
    flux_step,
    flux_pfcomp,
    flux_free,
    flux_ncomp,
    flux_nscomp,
    flux_mmcomp,
    flux_2cxm,
)
from dcmri.kinetics.functions_tissue import (
    conc_tissue_u,
    conc_tissue_fx,
    conc_tissue_nx,
    conc_tissue_nxp,
    conc_tissue_wv,
    conc_tissue_hfu,
    conc_tissue_hf,
    conc_tissue_2cu,
    conc_tissue_2cx,

    flux_tissue_u,
    flux_tissue_nx,
    flux_tissue_nxp,
    flux_tissue_fx,
    flux_tissue_wv,
    flux_tissue_hfu,
    flux_tissue_hf,
    flux_tissue_2cu,
    flux_tissue_2cx,
)
from dcmri.kinetics.functions_liver import (
    # conc_liver_1i_ec_d,
    conc_liver_1i_ec,
    conc_liver_1i_ec_hf,
    conc_liver_1i_ic,
    conc_liver_1i_ic_nsu,
    conc_liver_1i_ic_nse,
    conc_liver_1i_ic_nsue,
    conc_liver_1i_ic_hf,
    conc_liver_1i_ic_hf_nsu,
    conc_liver_1i_ic_hf_nse,
    conc_liver_1i_ic_hf_nsue,
    # conc_liver_1i_ic_hfd,
    # conc_liver_1i_ic_hfd_nsu,
    # conc_liver_1i_ic_hfd_nse,
    # conc_liver_1i_ic_hfd_nsue,
    # conc_liver_1i_ic_hfdu,
    # conc_liver_1i_ic_hfdu_nsu,
    conc_liver_2i_ec_hf,
    conc_liver_2i_ec,
    conc_liver_2i_ic_hf,
    conc_liver_2i_ic_hf_nse,
    conc_liver_2i_ic_hf_nsu,
    conc_liver_2i_ic_hf_nsue,
    conc_liver_2i_ic,
    conc_liver_2i_ic_nse,
    conc_liver_2i_ic_nsu,
    conc_liver_2i_ic_nsue,
    conc_liver_2i_ic_u,
    conc_liver_2i_ic_u_nsu,
)
from dcmri.kinetics.functions_kidney import (
    conc_kidney_2cf,
    conc_kidney_2pf,
    conc_kidney_cpf,
    conc_kidney_hf,
    conc_kidney_fn,
    conc_kidney_2cfu,
    conc_kidney_2pfu,
    conc_kidney_hfu,
    conc_kidney_cm9,
)
from dcmri.kinetics.functions_aorta import ( 
    flux_aorta,
)
from dcmri.kinetics.functions_inv import (
    linfit_2cfm
)
from dcmri.kinetics.functions_input import (
    ca_injection
)


# Function functions built on standalone tools

from dcmri.relaxivity.modules_tissue import (
    R1,
    R2,
    R2s,
    Relax,
)
from dcmri.relaxivity.modules_tissue_x import (
   R1TissueX,
   R2TissueX,
   R2sTissueX,
   RelaxTissueX,
)
from dcmri.relaxivity.functions_relaxivity import (
   relax_t2s,
   relax_t2,
   relax_t1,
   conc_t1,
)

from dcmri.bloch.modules_tissue import (
    Magnetization,
)
from dcmri.bloch.modules_tissue_x import (
    MagnetizationTissueX,
    WaterVolumesTissueX,
    WaterFlowsTissueX,
)
from dcmri.bloch.functions_sequences import (
    Mz_ss,
    Mz_ssi,
    Mz_ss_spgr,
    Mz_ss_pr_spgr,
    Mz_prop,
    Mz_prop_pr_spgr,
    mz_readout,
)
from dcmri.bloch.functions_dynamic_sequences import (
   Mz_spgr_in_ss,
   Mz_spgr_in_ssi,
   Mz_pr_spgr,
   Mz_pr_spgr_in_ss,
   Mz_se,
)
from dcmri.signal.modules_tissue import (
    Signal,
    CalibrateSignal,
)
from dcmri.inverse.sig2conc import (
    SignalToConc
)
from dcmri.inverse.lib import (
    estimate_bat,
    conc_dce,
    conc_dsc,
    conc_ss,
    conc_dce_lin,
    vfa_nonlinear,
    vfa_linear,
)

# End-to-end forward Models
from dcmri.models.aorta import AortaModel

# End user tools
from dcmri.e2e.aorta import Aorta
from dcmri.e2e.aorta_liver import AortaLiver
from dcmri.e2e.aorta_kidneys import AortaKidneys
from dcmri.e2e.aorta_portal_liver import AortaPortalLiver
from dcmri.e2e.aorta_liver_dynamic import AortaLiverDynamic
from dcmri.e2e.aorta_liver_drug import AortaLiverDrug
from dcmri.e2e.aorta_liver_dynamic_drug import AortaLiverDynamicDrug
from dcmri.e2e.kidney import Kidney
from dcmri.e2e.liver import Liver
from dcmri.e2e.cort_med import CortMed
# from dcmri.e2e.tissue_x import TissueX
from dcmri.e2e.tissue_ls import TissueLS

from dcmri.dro.aif import (
    parker,
    tristan,
    tristan_rat,
)
from dcmri.dro.phantoms import (
    shepp_logan,
)

# Utilities with internal dependencies

from dcmri.dro.fake import (
   aif,
   brain,
   tissue,
   liver,
   kidney,
   tissue2scan,    
)