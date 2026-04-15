# conc => tconc

from dcmri.pk.input import (
    ca_injection,
)
from dcmri.pk.inv import (
    linfit_2cfm,
)

from dcmri.pk.blocks import (
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

    conc, # eliminate - replaced by kinetics
    conc_trap,
    conc_pass,
    conc_comp,
    conc_plug,
    conc_chain,
    conc_step,
    conc_free,
    conc_ncomp,
    conc_nscomp,
    conc_mmcomp,
    conc_2cxm,

    flux, # eliminate - replaced by kinetics
    flux_trap,
    flux_pass,
    flux_comp,
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
from dcmri.pk.aorta import (
    flux_aorta
)
from dcmri.pk.kidney import (
    conc_kidney_2cf,
    conc_kidney_hf,
    conc_kidney_fn,
    conc_kidney_cm9,
)
from dcmri.pk.liver import ( # change double __ notation
    conc_liver_1i_ec_d,
    conc_liver_1i_ec,
    conc_liver_1i_ic,
    conc_liver_1i_ic__u,
    conc_liver_1i_ic__e,
    conc_liver_1i_ic__ue,
    conc_liver_1i_ic_hf,
    conc_liver_1i_ic_hf__u,
    conc_liver_1i_ic_hf__e,
    conc_liver_1i_ic_hf__ue,
    conc_liver_1i_ic_hfd,
    conc_liver_1i_ic_hfd__u,
    conc_liver_1i_ic_hfd__e,
    conc_liver_1i_ic_hfd__ue,
    conc_liver_1i_ic_hfdu,
    conc_liver_1i_ic_hfdu__u,
    conc_liver_2i_ec_hf,
    conc_liver_2i_ec,
    conc_liver_2i_ic_hf,
    conc_liver_2i_ic_hf__e,
    conc_liver_2i_ic_hf__u,
    conc_liver_2i_ic_hf__ue,
    conc_liver_2i_ic,
    conc_liver_2i_ic__e,
    conc_liver_2i_ic__u,
    conc_liver_2i_ic__ue,
    conc_liver_2i_ic_u,
    conc_liver_2i_ic_u__u,
)
from dcmri.pk.tissue import (
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