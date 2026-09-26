import numpy as np
from scipy.linalg import expm


def Mz_ss_vectorized(R1_pulses, v, Fw, j_pulses, me, seq):
    """
    Vectorized steady-state Mz over n_periods.
    R1_pulses, j_pulses: shape (n_comps, n_periods)
    Returns: Mz, shape (n_comps, n_periods)
    """
    if np.isscalar(v) or (hasattr(v, "size") and v.size == 1):
        return _Mz_ss_1c_vec(R1_pulses, v, Fw, j_pulses, me, seq)
    else:
        return _Mz_ss_nc_vec(R1_pulses, v, Fw, j_pulses, me, seq)


def _Mz_ss_1c_vec(R1_pulses, v, Fw, j_pulses, me, seq):
    R1 = R1_pulses.ravel() if R1_pulses.ndim > 1 else R1_pulses  # (n_periods,)
    j = j_pulses.ravel() if j_pulses.ndim > 1 else j_pulses

    # K, KinvJ must now be shape (n_periods,) each
    K, KinvJ = _Mz_KinvJ_vec_1c(R1, v, Fw, j, me)  # <-- needs batching support

    n = K.shape[0]
    prod = np.ones(n)
    acc = np.zeros(n)
    for FA, TR in seq:
        cFA = np.cos(np.radians(FA))
        E = np.exp(-TR * K)          # (n_periods,)
        C = cFA * E
        F = 1 - E
        acc = C * acc + F
        prod = C * prod

    A = 1 - prod
    M = np.where(A != 0, acc / A * KinvJ, 0.0)
    return M.reshape(R1_pulses.shape)


def _Mz_ss_nc_vec(R1_pulses, v, Fw, j_pulses, me, seq):
    nc, n_periods = R1_pulses.shape
    Id = np.eye(nc)[None, :, :].repeat(n_periods, axis=0)   # (n_periods, nc, nc)

    K, KinvJ = _Mz_KinvJ_vec_nc(R1_pulses, v, Fw, j_pulses, me)

    prod = Id.copy()
    acc = np.zeros((n_periods, nc, nc))
    for FA, TR in seq:
        cFA = np.cos(np.radians(FA))
        E = expm(-TR * K)
        C = cFA * E
        F = Id - E
        acc = C @ acc + F
        prod = C @ prod

    A = Id - prod
    rhs = np.einsum('pij,pj->pi', acc, KinvJ)   # (n_periods, nc)

    M = np.linalg.solve(A, rhs[..., None])[..., 0]   # force unambiguous batch shape
    return M.T


def _Mz_K_vec(R1_periods, v, Fw):
    """
    nc==1: R1_periods shape (n_periods,); v, Fw scalars -> returns K shape (n_periods,)
    nc>1:  R1_periods shape (n_periods, nc); v shape (nc,), Fw shape (nc, nc) -> returns K shape (n_periods, nc, nc)
    """
    nc = np.size(v)

    if nc == 1:
        return R1_periods + Fw / v          # R1_periods already (n_periods,); Fw, v scalars

    n_periods = R1_periods.shape[0]
    K_offdiag = -Fw / v
    total_outflow = np.sum(Fw, axis=0)

    K = np.broadcast_to(K_offdiag, (n_periods, nc, nc)).copy()
    diag = R1_periods + total_outflow / v

    idx = np.arange(nc)
    K[:, idx, idx] = diag

    return K


def _Mz_KJ_vec(R1_periods, v, Fw, j_periods, me):
    """
    R1_periods, j_periods: (n_periods, nc)
    v, me: (nc,) or scalar (nc==1)
    Returns K, J with leading n_periods axis
    """
    K = _Mz_K_vec(R1_periods, v, Fw)
    J = R1_periods * v * me + j_periods              # broadcasts fine for both nc==1 and nc>1
    return K, J


def _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me):
    nc = np.size(v)
    K, J = _Mz_KJ_vec(R1_periods, v, Fw, j_periods, me)

    if nc == 1:
        KinvJ = np.where(K != 0, J / K, 0.0)          # (n_periods,)
    else:
        # J: (n_periods, nc) -> (n_periods, nc, 1) to force correct batch broadcasting
        KinvJ = np.linalg.solve(K, J[..., None])       # (n_periods, nc, 1)
        KinvJ = KinvJ[..., 0]                           # back to (n_periods, nc)

    return K, KinvJ


def _Mz_KinvJ_vec_1c(R1_pulses, v, Fw, j_pulses, me):
    R1_periods = R1_pulses.reshape(-1)   # (n_periods,)
    j_periods = j_pulses.reshape(-1)     # (n_periods,)
    return _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)

def _Mz_KinvJ_vec_nc(R1_pulses, v, Fw, j_pulses, me):
    R1_periods = R1_pulses.T   # (n_periods, nc)
    j_periods = j_pulses.T     # (n_periods, nc)
    return _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)





def Mz_ss_spgr_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    """
    R1_pulses, j_pulses: shape (n_comps, n_periods)
    Returns Mz: shape (n_comps, n_periods)
    """
    if np.isscalar(v):
        return _Mz_ss_spgr_1c_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)
    elif v.size == 1:
        Mz = _Mz_ss_spgr_1c_vec(R1_pulses[0], v[0], Fw[0, 0], j_pulses[0], me, FA, TR)
        return Mz.reshape(R1_pulses.shape)
    else:
        return _Mz_ss_spgr_nc_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)


def _Mz_ss_spgr_nc_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    off_diag = ~np.eye(Fw.shape[0], dtype=bool)
    PSw = Fw[off_diag]

    if np.all(PSw == 0):
        return _Mz_ss_nex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)
    
    elif np.all(np.isinf(PSw)):
        return _Mz_ss_fex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)
    
    elif 0 < np.count_nonzero(np.isinf(PSw)):
        raise NotImplementedError(
            'Water exchange with some (but not all) infinite PS '
            'values is currently not implemented.')
    else:
        return _Mz_ss_aex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR)


def _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # reuse existing helper
    E = np.exp(-TR * K)
    cFA = np.cos(np.radians(FA))
    n = (1 - E) / (1 - cFA * E)
    return n * KinvJ


def _Mz_ss_fex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    R1fex_periods = np.sum(v[:, None] * R1_pulses, axis=0) / np.sum(v)   # (n_periods,)
    fo = np.diag(Fw)
    j_sum_periods = np.sum(j_pulses, axis=0)                              # (n_periods,)

    M_periods = _Mz_ss_spgr_1c_vec(
        R1fex_periods, np.sum(v), np.sum(fo), j_sum_periods, me, FA, TR
    )                                                                      # (n_periods,)

    Mc = M_periods[None, :] * (v / np.sum(v))[:, None]                    # (nc, n_periods)
    return Mc


def _Mz_ss_nex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    nc = v.size
    fo = np.diag(Fw)
    Mc = [
        _Mz_ss_spgr_1c_vec(R1_pulses[c], v[c], fo[c], j_pulses[c], me, FA, TR)
        for c in range(nc)
    ]
    return np.stack(Mc)   # (nc, n_periods) -- loop is only over nc, not n_periods


def _Mz_ss_aex_vec(R1_pulses, v, Fw, j_pulses, me, FA, TR):
    nc, n_periods = R1_pulses.shape
    R1_periods = R1_pulses.T   # (n_periods, nc)
    j_periods = j_pulses.T     # (n_periods, nc)

    K = _Mz_K_vec(R1_periods, v, Fw)             # (n_periods, nc, nc), reuse existing helper
    J = R1_periods * v * me + j_periods           # (n_periods, nc)

    E = expm(-TR * K)                              # batched
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))
    cFA = np.cos(np.radians(FA))

    A = K @ (Id - cFA * E)                          # (n_periods, nc, nc)
    rhs = (Id - E) @ J[..., None]                    # (n_periods, nc, 1)

    Mz = np.zeros((n_periods, nc))
    try:
        Mz = np.linalg.solve(A, rhs)[..., 0]
    except np.linalg.LinAlgError:
        # fall back per-period only where the batched solve fails,
        # mirroring the original's try/except -> zeros behaviour
        for p in range(n_periods):
            try:
                Mz[p] = np.linalg.solve(A[p], rhs[p, :, 0])
            except np.linalg.LinAlgError:
                Mz[p] = 0.0

    return Mz.T   # (nc, n_periods)


# PR-SPGR SS before the prepulse
def Mz_ss_pr_spgr_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, Nph, TP, TD, PA):
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_ss_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        M = _Mz_ss_pr_spgr_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, Nph, TP, TD, PA)
        return M.reshape(R1_pulses.shape)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        M = _Mz_ss_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA)
        return M.T   # (nc, n_periods)


def _Mz_ss_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)          # (n_periods,)
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)  # (n_periods,)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n

    En = np.exp(-T_read * K)
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    W = nFA * ED * En
    A = 1 - cPA * W * EP
    B = ED - W
    C = W * (1 - EP) + (1 - ED)

    RH = B * Mss + C * KinvJ
    return np.where(A != 0, RH / A, 0.0)


def _Mz_ss_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # (n_periods,nc,nc), (n_periods,nc)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T  # (n_periods,nc)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    nFA = cFA**n

    En = expm(-T_read * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    W = nFA * (ED @ En)
    A = Id - cPA * (W @ EP)
    B = ED - W
    C = W @ (Id - EP) + (Id - ED)

    RH = np.einsum('pij,pj->pi', B, Mss) + np.einsum('pij,pj->pi', C, KinvJ)
    return np.linalg.solve(A, RH[..., None])[..., 0]   # (n_periods, nc)



# PR-SPGR SS at the k-space center
def Mz_ss_k0_pr_spgr_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, Nph, TP, TD, PA, Nk0):
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_ss_k0_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        M = _Mz_ss_k0_pr_spgr_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0)
        return M.reshape(R1_pulses.shape)
    else:
        R1_periods = R1_pulses.T
        j_periods = j_pulses.T
        M = _Mz_ss_k0_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0)
        return M.T


def _Mz_ss_k0_pr_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    n0, n1 = Nk0, n - Nk0

    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = np.exp(-TR * n0 * K)
    E1 = np.exp(-TR * n1 * K)
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    steps = [
        (cFA1 * E1, (1 - cFA1 * E1) * Mss),
        (ED,        (1 - ED) * KinvJ),
        (cPA,       0.0),
        (EP,        (1 - EP) * KinvJ),
        (cFA0 * E0, (1 - cFA0 * E0) * Mss),
    ]

    prod, seq = steps[0]
    for C, b in steps[1:]:
        seq = C * seq + b
        prod = C * prod

    A = 1 - prod
    return np.where(A != 0, seq / A, 0.0)


def _Mz_ss_k0_pr_spgr_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, Nph, TP, TD, PA, Nk0):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T  # (n_periods,nc)

    T_read = TR * Nph
    n = np.floor(T_read / TR)
    n0, n1 = Nk0, n - Nk0

    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = expm(-TR * n0 * K)
    E1 = expm(-TR * n1 * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    def mv(C, x):  # batched matrix @ vector
        return np.einsum('pij,pj->pi', C, x)

    steps = [
        (cFA1 * E1, mv(Id - cFA1 * E1, Mss)),
        (ED,        mv(Id - ED, KinvJ)),
        (cPA * Id,  np.zeros((n_periods, nc))),
        (EP,        mv(Id - EP, KinvJ)),
        (cFA0 * E0, mv(Id - cFA0 * E0, Mss)),
    ]

    prod, seq = steps[0]
    for C, b in steps[1:]:
        seq = mv(C, seq) + b
        prod = C @ prod

    A = Id - prod
    return np.linalg.solve(A, seq[..., None])[..., 0]






def Mz_ss_spgri_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, TF, SA):
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_ss_spgri_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        M = _Mz_ss_spgri_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, TF, SA)
        return M.reshape(R1_pulses.shape)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        M = _Mz_ss_spgri_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA)
        return M.T                # (nc, n_periods)


def _Mz_ss_spgri_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR)
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me   # scalar, constant across periods

    K_t = _Mz_K_vec(R1_periods, v, Fw)          # (n_periods,) -- reuse existing helper
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)  # (n_periods,)

    En_t = np.exp(-TF * K_t)                     # (n_periods,)
    M_sig = Mss + nFA * En_t * (M0 - Mss)

    return M_sig


def _Mz_ss_spgri_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, TF, SA):
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR)
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me   # (nc,), constant across periods

    K_t = _Mz_K_vec(R1_periods, v, Fw)                                    # (n_periods, nc, nc)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T  # (n_periods, nc)

    En_t = expm(-TF * K_t)                                                 # batched

    diff = M0[None, :] - Mss                                               # (n_periods, nc)
    M_sig = Mss + nFA * np.einsum('pij,pj->pi', En_t, diff)

    return M_sig




############################
# VECTORIZED IMPLEMENTATIONS - PROPAGATOR
############################



def _Mz_prop_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, seq):
    """
    R1_periods, j_periods: (n_periods,)
    Returns A, B: (n_periods,) such that M_end = A * M_start + B
    """
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # (n_periods,)
    n_periods = K.shape[0]

    A = np.ones(n_periods)
    B = np.zeros(n_periods)

    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = np.exp(-TR * K)                # (n_periods,)
        C = cFA * E
        b = (1 - E) * KinvJ

        B = C * B + b
        A = C * A

    return A, B


def _Mz_prop_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, seq):
    """
    R1_periods, j_periods: (n_periods, nc)
    Returns A: (n_periods, nc, nc), B: (n_periods, nc)
    """
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    # (n_periods, nc, nc), (n_periods, nc)

    A = Id.copy()
    B = np.zeros((n_periods, nc))

    for pulse in seq:
        FA, TR = pulse[0], pulse[1]
        cFA = np.cos(np.radians(FA))
        E = expm(-TR * K)                                  # batched, (n_periods, nc, nc)
        C = cFA * E
        b = np.einsum('pij,pj->pi', Id - E, KinvJ)          # (n_periods, nc)

        B = np.einsum('pij,pj->pi', C, B) + b
        A = C @ A

    return A, B.T



def _Mz_prop_coeffs_vectorized(R1_pulses, v, Fw, j_pulses, me, seq):
    """
    R1_pulses, j_pulses: (n_comps, n_periods)
    Returns A, B with the batch (n_periods) as the leading axis, ready to
    drive a cheap sequential recurrence over periods.
    """
    if len(seq) == 0:
        # identity map: A=1/Id, B=0
        if np.isscalar(v):
            n_periods = R1_pulses.shape[-1]
            return np.ones(n_periods), np.zeros(n_periods)
        else:
            nc, n_periods = R1_pulses.shape
            return np.broadcast_to(np.eye(nc), (n_periods, nc, nc)).copy(), np.zeros((n_periods, nc))

    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, seq)
    elif v.size == 1:
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_coeffs_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, seq)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        return _Mz_prop_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, seq)



def Mz_prop_vectorized(M0, R1_periods, v, Kw, j_periods, me, seq):
    nc = R1_periods.shape[0]

    R1_periods = R1_periods.reshape((nc, -1))
    j_periods = j_periods.reshape((nc, -1))
    
    # Precompute once, fully vectorized across all periods
    A, B = _Mz_prop_coeffs_vectorized(R1_periods, v, Kw, j_periods, me, seq)

    n_periods = R1_periods.shape[1]
    Mz = np.zeros((M0.size, n_periods))

    for period in range(n_periods):
        M_init = M0 if period==0 else Mz[:, period - 1]

        if np.isscalar(v) or v.size == 1:
            Mz[:, period] = A[period] * M_init  + B[period]
        else:
            Mz[:, period] = A[period] @ M_init + B[:, period]

    return Mz


# SPGR

def _Mz_prop_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)   # (n_periods,)

    cFA = np.cos(np.radians(FA))
    E = np.exp(-TR * K)          # (n_periods,) -- one exp call instead of n_pulses
    C = cFA * E
    b = (1 - E) * KinvJ

    cFA_n = cFA ** n_pulses
    A = cFA_n * np.exp(-n_pulses * TR * K)
    # geometric series, guarding C == 1
    B = np.where(C != 1, (1 - A) / (1 - C) * b, n_pulses * b)

    return A, B


def _Mz_prop_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)

    cFA = np.cos(np.radians(FA))
    E = expm(-TR * K)                                    # single-pulse map, still needed for C, b
    C = cFA * E
    b = np.einsum('pij,pj->pi', Id - E, KinvJ)

    cFA_n = cFA ** n_pulses
    A = cFA_n * expm(-n_pulses * TR * K)                   # replaces matrix_power(C, n_pulses)

    rhs = np.einsum('pij,pj->pi', Id - A, b)
    B = np.linalg.solve(Id - C, rhs[..., None])[..., 0]

    return A, B.T


def _Mz_prop_spgr_coeffs_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, n_pulses):
    """
    R1_pulses, j_pulses: (n_comps, n_periods)
    Returns A, B with the batch (n_periods) as the leading axis, ready to
    drive a cheap sequential recurrence over periods.
    """
    if n_pulses == 0:
        # identity map: A=1/Id, B=0
        if np.isscalar(v):
            n_periods = R1_pulses.shape[-1]
            return np.ones(n_periods), np.zeros(n_periods)
        else:
            nc, n_periods = R1_pulses.shape
            return np.broadcast_to(np.eye(nc), (n_periods, nc, nc)).copy(), np.zeros((n_periods, nc))

    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses)
    elif v.size == 1:
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_spgr_coeffs_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, n_pulses)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        return _Mz_prop_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, n_pulses)



def Mz_prop_spgr_vectorized(M0, R1_periods, v, Kw, j_periods, me, FA, TR, n_pulses):
    nc = R1_periods.shape[0]

    R1_periods = R1_periods.reshape((nc, -1))
    j_periods = j_periods.reshape((nc, -1))
    
    # Precompute once, fully vectorized across all periods
    A, B = _Mz_prop_spgr_coeffs_vectorized(R1_periods, v, Kw, j_periods, me, FA, TR, n_pulses)

    n_periods = R1_periods.shape[1]
    Mz = np.zeros((M0.size, n_periods))

    for period in range(n_periods):
        M_init = M0 if period==0 else Mz[:, period - 1]

        if np.isscalar(v) or v.size == 1:
            Mz[:, period] = A[period] * M_init  + B[period]
        else:
            Mz[:, period] = A[period] @ M_init + B[:, period]

    return Mz




# PR-SPGR


def Mz_prop_pr_spgr_coeffs_vectorized(R1_pulses, v, Fw, j_pulses, me, FA, TR, N, TP, TD, PA, Nk0):
    """
    Precompute per-period affine coefficients A, B such that
    M_end = A @ M_start + B for one prepared-SPGR cycle, starting and
    ending at the k-space center.

    R1_pulses, j_pulses: (n_comps, n_periods)
    Returns A, B with n_periods as the leading batch axis.
    """
    if np.isscalar(v):
        R1_periods = R1_pulses.reshape(-1)
        j_periods = j_pulses.reshape(-1)
        return _Mz_prop_pr_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0)
    elif v.size == 1:
        R1_periods = R1_pulses[0].reshape(-1)
        j_periods = j_pulses[0].reshape(-1)
        return _Mz_prop_pr_spgr_coeffs_1c_vec(R1_periods, v[0], Fw[0, 0], j_periods, me, FA, TR, N, TP, TD, PA, Nk0)
    else:
        R1_periods = R1_pulses.T   # (n_periods, nc)
        j_periods = j_pulses.T
        return _Mz_prop_pr_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0)


def _Mz_prop_pr_spgr_coeffs_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0):
    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)             # (n_periods,)
    Mss = _Mz_ss_spgr_1c_vec(R1_periods, v, Fw, j_periods, me, FA, TR)     # (n_periods,)

    n0, n1 = Nk0, N - Nk0
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = np.exp(-TR * n0 * K)
    E1 = np.exp(-TR * n1 * K)
    EP = np.exp(-TP * K)
    ED = np.exp(-TD * K)

    C1 = cFA1 * E1
    C0 = cFA0 * E0

    steps = [
        (C1,             (1 - C1) * Mss),                # remaining (N-Nk0) readout pulses
        (ED,             (1 - ED) * KinvJ),               # TD recovery
        (cPA * np.ones_like(K), np.zeros_like(K)),        # preparation pulse
        (EP,             (1 - EP) * KinvJ),                # TP recovery
        (C0,             (1 - C0) * Mss),                  # Nk0 readout pulses to k-space center
    ]

    A, B = steps[0]
    for C, b in steps[1:]:
        B = C * B + b
        A = C * A

    return A, B


def _Mz_prop_pr_spgr_coeffs_nc_vec(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0):
    nc = v.size
    n_periods = R1_periods.shape[0]
    Id = np.broadcast_to(np.eye(nc), (n_periods, nc, nc))

    K, KinvJ = _Mz_KinvJ_vec(R1_periods, v, Fw, j_periods, me)
    Mss = Mz_ss_spgr_vectorized(R1_periods.T, v, Fw, j_periods.T, me, FA, TR).T   # (n_periods, nc)

    n0, n1 = Nk0, N - Nk0
    cPA = np.cos(np.radians(PA))
    cFA = np.cos(np.radians(FA))
    cFA0, cFA1 = cFA**n0, cFA**n1

    E0 = expm(-TR * n0 * K)
    E1 = expm(-TR * n1 * K)
    EP = expm(-TP * K)
    ED = expm(-TD * K)

    C1 = cFA1 * E1
    C0 = cFA0 * E0

    def mv(C, x):
        return np.einsum('pij,pj->pi', C, x)

    steps = [
        (C1,        mv(Id - C1, Mss)),                    # remaining (N-Nk0) readout pulses
        (ED,        mv(Id - ED, KinvJ)),                   # TD recovery
        (cPA * Id,  np.zeros((n_periods, nc))),            # preparation pulse
        (EP,        mv(Id - EP, KinvJ)),                    # TP recovery
        (C0,        mv(Id - C0, Mss)),                       # Nk0 readout pulses to k-space center
    ]

    A, B = steps[0]
    for C, b in steps[1:]:
        B = mv(C, B) + b
        A = C @ A

    return A, B.T   # B as (nc, n_periods)


def Mz_prop_pr_spgr_vectorized(M0, R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0):
    nc = R1_periods.shape[0]
    R1_periods = R1_periods.reshape((nc, -1))
    j_periods = j_periods.reshape((nc, -1))

    A, B = Mz_prop_pr_spgr_coeffs_vectorized(R1_periods, v, Fw, j_periods, me, FA, TR, N, TP, TD, PA, Nk0)

    n_periods = R1_periods.shape[1]
    Mz = np.zeros((M0.size, n_periods))

    for period in range(n_periods):
        M_init = M0 if period == 0 else Mz[:, period - 1]
        if np.isscalar(v) or v.size == 1:
            Mz[:, period] = A[period] * M_init + B[period]
        else:
            Mz[:, period] = A[period] @ M_init + B[:, period]

    return Mz
