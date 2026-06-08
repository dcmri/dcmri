
import numpy as np
from dcmri.utils import const


def _ellipse(array, center, axes, angle, value):
    n = array.shape[0]
    major = np.amax(axes)
    minor = np.amin(axes)
    c = np.sqrt(major**2 - minor**2)
    dx = np.cos(angle * np.pi / 180)
    dy = np.sin(angle * np.pi / 180)
    f1 = [center[0] + dx * c, center[1] + dy * c]
    f2 = [center[0] - dx * c, center[1] - dy * c]
    d = 2.0 / n
    for i in range(n):
        for j in range(n):
            x = d * (i - (n - 1) / 2)
            y = d * (j - (n - 1) / 2)
            d1 = np.sqrt((x - f1[0])**2 + (y - f1[1])**2)
            d2 = np.sqrt((x - f2[0])**2 + (y - f2[1])**2)
            if d1 + d2 <= 2 * major:
                array[i, j] = value
    return array


def _shepp_logan_mask(n=256) -> dict:

    mask = {}

    # Background
    array = np.ones((n, n)).astype(bool)
    array = _ellipse(array, [0, 0], [0.72, 0.95], 0, 0)
    mask['background'] = np.flip(array, axis=0)

    # 1 Scalp
    array = _ellipse(np.zeros((n, n)).astype(
        bool), [0, 0], [0.72, 0.95], 0, 1)
    array = _ellipse(array, [0, 0], [0.69, 0.92], 0, 0)
    mask['scalp'] = np.flip(array, axis=0)

    # 2 Bone and marrow
    array = _ellipse(np.zeros((n, n)).astype(
        bool), [0, 0], [0.69, 0.92], 0, 1)
    array = _ellipse(array, [-0.0184, 0], [0.6624, 0.874], 0, 0)
    mask['bone'] = np.flip(array, axis=0)

    # 3 CSF
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [-0.0184, 0], [0.6624, 0.874], 0, 1)
    array = _ellipse(array, [-0.0184, 0], [0.6524, 0.864], 0, 0)
    mask['CSF skull'] = np.flip(array, axis=0)

    # 4 Gray matter
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [-0.0184, 0], [0.6524, 0.864], 0, 1)
    array = _ellipse(array, [0, -0.22], [0.41, 0.16], 180 - 18, 0)
    array = _ellipse(array, [0, 0.22], [0.31, 0.11], 180 + 18, 0)
    array = _ellipse(array, [0.35, 0], [0.21, 0.25], 0, 0)
    array = _ellipse(array, [0.1, 0], [0.046, 0.046], 0, 0)
    array = _ellipse(array, [-0.605, -0.08], [0.046, 0.023], 90, 0)
    array = _ellipse(array, [-0.605, 0.06], [0.023, 0.046], 0, 0)
    array = _ellipse(array, [-0.1, 0.0], [0.046, 0.046], 0, 0)
    array = _ellipse(array, [-0.605, 0.0], [0.023, 0.023], 0, 0)
    array = _ellipse(array, [-0.83, 0], [0.05, 0.05], 0, 0)
    array = _ellipse(array, [0.66, 0], [0.02, 0.02], 0, 0)
    mask['gray matter'] = np.flip(array, axis=0)

    # 5 CSF
    array = _ellipse(np.zeros((n, n)).astype(bool), [
                     0, -0.22], [0.41, 0.16], 180 - 18, 1)
    array = _ellipse(array, [0.35, 0], [0.21, 0.25], 0, 0)
    array = _ellipse(array, [-0.1, 0.0], [0.046, 0.046], 0, 0)
    mask['CSF right'] = np.flip(array, axis=0)

    # 6 CSF
    array = _ellipse(np.zeros((n, n)).astype(bool), [
                     0, 0.22], [0.31, 0.11], 180 + 18, 1)
    mask['CSF left'] = np.flip(array, axis=0)

    # 7 Tumor
    array = _ellipse(np.zeros((n, n)).astype(
        bool), [0.35, 0], [0.21, 0.25], 0, 1)
    array = _ellipse(array, [0.1, 0], [0.046, 0.046], 0, 0)
    mask['tumor 1'] = np.flip(array, axis=0)

    # 8 Tumor
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [0.1, 0], [0.046, 0.046], 0, 1)
    mask['tumor 2'] = np.flip(array, axis=0)

    # 9 Tumor
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [-0.605, -0.08], [0.046, 0.023], 90, 1)
    mask['tumor 4'] = np.flip(array, axis=0)

    # 10 Tumor
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [-0.605, 0.06], [0.023, 0.046], 0, 1)
    mask['tumor 6'] = np.flip(array, axis=0)

    # 11 Tumor
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [-0.1, 0.0], [0.046, 0.046], 0, 1)
    mask['tumor 3'] = np.flip(array, axis=0)

    # 12 Tumor
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [-0.605, 0.0], [0.023, 0.023], 0, 1)
    mask['tumor 5'] = np.flip(array, axis=0)

    # 13 Sagittal sinus
    array = _ellipse(np.zeros((n, n)).astype(bool),
                     [-0.83, 0], [0.05, 0.05], 0, 1)
    mask['sagittal sinus'] = np.flip(array, axis=0)

    # 14 Anterior cerebral artery
    array = _ellipse(np.zeros((n, n)).astype(
        bool), [0.66, 0], [0.02, 0.02], 0, 1)
    mask['anterior artery'] = np.flip(array, axis=0)

    return mask


def _shepp_logan(param, n=256, B0=3) -> np.ndarray:
    if param == 'PD':
        scalp = const.PD('skin')
        bone = const.PD('bone marrow')
        csf = const.PD('csf')
        gm = const.PD('gray matter')
        tumor = 0.95
        blood = 0.9
    elif param == 'T2':
        scalp = const.T2(B0, 'skin')
        bone = const.T2(B0, 'bone marrow')
        csf = const.T2(B0, 'csf')
        gm = const.T2(B0, 'gray matter')
        tumor = 0.25
        blood = 0.1
    elif param == 'T1':
        scalp = const.T1(B0, 'skin')
        bone = const.T1(B0, 'bone marrow')
        csf = const.T1(B0, 'csf')
        gm = const.T1(B0, 'gray matter')
        tumor = 0.926 * (B0**0.217)
        blood = const.T1(B0, 'blood')
    elif param == 'Fb':
        scalp = const.perfusion('Fb', 'skin')
        bone = const.perfusion('Fb', 'bone marrow')
        csf = const.perfusion('Fb', 'csf')
        gm = const.perfusion('Fb', 'gray matter')
        tumor = 0.02
        blood = 0
    elif param == 'vb':
        scalp = const.perfusion('vb', 'skin')
        bone = const.perfusion('vb', 'bone marrow')
        csf = const.perfusion('vb', 'csf')
        gm = const.perfusion('vb', 'gray matter')
        tumor = 0.1
        blood = 1
    elif param == 'PS':
        scalp = const.perfusion('PS', 'skin')
        bone = const.perfusion('PS', 'bone marrow')
        csf = const.perfusion('PS', 'csf')
        gm = const.perfusion('PS', 'gray matter')
        tumor = 0.001
        blood = 0
    elif param == 'vi':
        scalp = const.perfusion('vi', 'skin')
        bone = const.perfusion('vi', 'bone marrow')
        csf = const.perfusion('vi', 'csf')
        gm = const.perfusion('vi', 'gray matter')
        tumor = 0.3
        blood = 0

    array = np.zeros((n, n)).astype(np.float64)
    # 1 Scalp
    array = _ellipse(array, [0, 0], [0.72, 0.95], 0, scalp)
    # 2 Bone and marrow
    array = _ellipse(array, [0, 0], [0.69, 0.92], 0, bone)
    # 3 CSF
    array = _ellipse(array, [-0.0184, 0], [0.6624, 0.874], 0, csf)
    # 4 Gray matter
    array = _ellipse(array, [-0.0184, 0], [0.6524, 0.864], 0, gm)
    # 5 CSF
    array = _ellipse(array, [0, -0.22], [0.41, 0.16], 180 - 18, csf)
    # 6 CSF
    array = _ellipse(array, [0, 0.22], [0.31, 0.11], 180 + 18, csf)
    # 7 Tumor
    array = _ellipse(array, [0.35, 0], [0.21, 0.25], 0, tumor)
    # 8 Tumor
    array = _ellipse(array, [0.1, 0], [0.046, 0.046], 0, tumor)
    # 9 Tumor
    array = _ellipse(array, [-0.605, -0.08], [0.046, 0.023], 90, tumor)
    # 10 Tumor
    array = _ellipse(array, [-0.605, 0.06], [0.023, 0.046], 0, tumor)
    # 11 Tumor
    array = _ellipse(array, [-0.1, 0.0], [0.046, 0.046], 0, tumor)
    # 12 Tumor
    array = _ellipse(array, [-0.605, 0.0], [0.023, 0.023], 0, tumor)
    # 13 Sagittal sinus
    array = _ellipse(array, [-0.83, 0], [0.05, 0.05], 0, blood)
    # 14 Anterior cerebral artery
    array = _ellipse(array, [0.66, 0], [0.02, 0.02], 0, blood)
    # Flip
    array = np.flip(array, axis=0)
    return array


def shepp_logan(*params, n=256, B0=3):
    """Modified Shepp-Logan phantom mimicking an axial slice through the brain.

    The phantom is based on an MRI adaptation of the Shepp-Logan phantom 
    (Gach et al 2008), but with added features for use in a DC-MRI setting: 
    (1) additional regions for anterior cerebral artery and sinus sagittalis; 
    (2) additional optional contrasts blood flow (BF), blood volume (BV), 
    permeability-surface area product (PS) and interstitial volume (IV).

    Args:
        params (str or tuple): parameter or parameters shown in the image. 
          The options are 'PD' (proton density), 'T1', 'T2', 'Fb' (blood flow), 
          'vb' (Blood volume), 'PS' (permeability-surface area product) and 
          'vi' (interstitial volume). If no parameters are provided, the 
          function returns a dictionary with 14 masks, one for each region.
        n (int, optional): matrix size. Defaults to 256.
        B0 (int, optional): field strength in T. Defaults to 3.

    Reference:
        H. M. Gach, C. Tanase and F. Boada, "2D & 3D Shepp-Logan Phantom 
        Standards for MRI," 2008 19th International Conference on Systems 
        Engineering, Las Vegas, NV, USA, 2008, pp. 521-526, 
        `doi 10.1109/ICSEng.2008.15 <https://ieeexplore.ieee.org/document/4616690>`_.

    Returns:
        numpy.array or dict: if only one parameter is provided, this returns 
        an array. In all other conditions this returns a dictionary where keys 
        are the parameter- or region names, and values are square arrays with 
        image values.

    Note:
        Mask names:
            - background
            - scalp
            - bone
            - CSF skull
            - CSF left
            - CSF right
            - gray matter
            - tumor 1 to tumor 6
            - sagittal sinus
            - anterior artery

    Example:

    Generate a single contrast:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Simulate a synthetic blood flow image:

        >>> im = dc.shepp_logan('Fb')

        Plot the result in units of mL/min/100mL:

        >>> fig, ax = plt.subplots(figsize=(5, 5), ncols=1)
        >>> pos = ax.imshow(6000*im, cmap='gray', vmin=0.0, vmax=80)
        >>> fig.colorbar(pos, ax=ax, label='blood flow (mL/min/100mL)')
        >>> plt.show()

    Generate multiple contrasts in one function call:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Generate the MR Shepp-Logan phantom in low resolution:

        >>> im = dc.shepp_logan('PD', 'T1', 'T2', n=64)

        Plot the result:

        >>> fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 5), ncols=3)
        >>> ax1.imshow(im['PD'], cmap='gray')
        >>> ax2.imshow(im['T1'], cmap='gray')
        >>> ax3.imshow(im['T2'], cmap='gray')
        >>> plt.show()

    Generate masks for the different regions-of-interest:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Generate the MR Shepp-Logan phantom masks:

        >>> im = dc.shepp_logan(n=128)

        Plot all masks:

        >>> fig, ax = plt.subplots(figsize=(8, 8), ncols=4, nrows=4)
        >>> ax[0,0].imshow(im['background'], cmap='gray')
        >>> ax[0,1].imshow(im['scalp'], cmap='gray')
        >>> ax[0,2].imshow(im['bone'], cmap='gray')
        >>> ax[0,3].imshow(im['CSF skull'], cmap='gray')
        >>> ax[1,0].imshow(im['CSF left'], cmap='gray')
        >>> ax[1,1].imshow(im['CSF right'], cmap='gray')
        >>> ax[1,2].imshow(im['gray matter'], cmap='gray')
        >>> ax[1,3].imshow(im['tumor 1'], cmap='gray')
        >>> ax[2,0].imshow(im['tumor 2'], cmap='gray')
        >>> ax[2,1].imshow(im['tumor 3'], cmap='gray')
        >>> ax[2,2].imshow(im['tumor 4'], cmap='gray')
        >>> ax[2,3].imshow(im['tumor 5'], cmap='gray')
        >>> ax[3,0].imshow(im['tumor 6'], cmap='gray')
        >>> ax[3,1].imshow(im['sagittal sinus'], cmap='gray')
        >>> ax[3,2].imshow(im['anterior artery'], cmap='gray')
        >>> for i in range(4):
        >>>     for j in range(4):
        >>>         ax[i,j].set_yticklabels([])
        >>>         ax[i,j].set_xticklabels([])
        >>> plt.show()
    """
    if len(params) == 0:
        return _shepp_logan_mask(n=n)
    elif len(params) == 1:
        return _shepp_logan(params[0], n=n, B0=B0)
    else:
        phantom = {}
        for p in params:
            phantom[p] = _shepp_logan(p, n=n, B0=B0)
        return phantom
