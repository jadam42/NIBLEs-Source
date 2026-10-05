"""
Library of instances of samplegen.Material() classes pre-filled with
relevant material parameters.
"""

import numpy as np

from NIBLEs import samplegen

class InfiniteRelax(samplegen.Material):
    """
    Instance of Material Class, pre-filled with all relaxation 
    values set to 1e9 seconds except t2star = 1e8 seconds
    """
    def __init__(self, index = 1,
        pd = 1,
        chem_shift = 0,
        t1 = 1e9,
        t2 = 1e9,
        t2_star = 1e8,
        dynamic_t1 = [0],
        dynamic_t2 = [0],
        t1_alt = 1e9,
        t2_alt = 1e9,
        dynamic_t1_alt = [0],
        dynamic_t2_alt = [0]):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

#~~~~~ Brain Materials ~~~~~#
class GMSynth(samplegen.Material):
    """
    Instance of Material class, pre-filled with static parametres for
    human Grey Matter at 3T. Contains parameters for dynamic relaxation
    through interpolation from 0.05T to 3T.

    References
    ----------
    .. [1] Bottomley, P.A., Foster, T.H., Argersinger, R.E. and Pfeifer,
       L.M. (1984), A review of normal tissue hydrogen NMR relaxation
       times and relaxation mechanisms from 1-100 MHz: Dependence on
       tissue type, NMR frequency, temperature, species, excision, and
       age. Med. Phys., 11: 425-448. https://doi.org/10.1118/1.595535.

    .. [2] O'Reilly T, Webb AG. In vivo T1 and T2 relaxation time maps
       of brain tissue, skeletal muscle, and lipid measured in healthy
       volunteers at 50 mT. Magn Reson Med. 2021; 87: 884-895.
       https://doi.org/10.1002/mrm.29009.

    .. [3] Rooney, W.D., Johnson, G., Li, X., Cohen, E.R., Kim, S.-G.,
       Ugurbil, K. and Springer, C.S., Jr. (2007), Magnetic field and
       tissue dependencies of human brain longitudinal 1H2O relaxation
       in vivo. Magn. Reson. Med., 57: 308-318.
       https://doi.org/10.1002/mrm.21122.

    .. [4] Zhou, J., Golay, X., van Zijl, P.C.M., Silvennoinen, M.J.,
       Kauppinen, R., Pekar, J. and Kraut, M. (2001), Inverse T2
       contrast at 1.5 Tesla between gray matter and white matter in the
       occipital lobe of normal adult human brain. Magn. Reson. Med.,
       46: 401-406. https://doi.org/10.1002/mrm.1204.

    .. [5] Wansapura, J.P., Holland, S.K., Dunn, R.S. and Ball, W.S., Jr
       (1999), NMR relaxation times in the human brain at 3.0 tesla.
       J. Magn. Reson. Imaging, 9: 531-538.
       https://doi.org/10.1002/
       (SICI)1522-2586(199904)9:4<531::AID-JMRI4>3.0.CO;2-L
    """

    def __init__(self, index = 1,
        pd = 0.825,
        chem_shift = 0,
        t1 = 1.331,
        t2 = 110e-3,
        t2_star = 52e-3,
        dynamic_t1 = np.array([[0.05, 0.15, 0.5, 1.5, 3],
                  [0.329, 0.525, 0.815, 1.188, 1.331]]),
        dynamic_t2 = np.array([[0.05, 0.15, 0.5, 1.5, 3],
                  [0.097, 0.11, 0.11, 0.088, 0.11]]),
        t1_alt = 8e-3,
        t2_alt = 1e-3,
        dynamic_t1_alt = [0],
        dynamic_t2_alt = [0]):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

class WMSynth(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for
    human White Matter at 3T. Contains parameters for dynamic relaxation
    through interpolation from 0.05T to 3T.

    References
    ----------
    .. [1] Bottomley, P.A., Foster, T.H., Argersinger, R.E. and Pfeifer,
       L.M. (1984), A review of normal tissue hydrogen NMR relaxation
       times and relaxation mechanisms from 1-100 MHz: Dependence on
       tissue type, NMR frequency, temperature, species, excision, and
       age. Med. Phys., 11: 425-448. https://doi.org/10.1118/1.595535.

    .. [2] O'Reilly T, Webb AG. In vivo T1 and T2 relaxation time maps
       of brain tissue, skeletal muscle, and lipid measured in healthy
       volunteers at 50 mT. Magn Reson Med. 2021; 87: 884-895.
       https://doi.org/10.1002/mrm.29009.

    .. [3] Rooney, W.D., Johnson, G., Li, X., Cohen, E.R., Kim, S.-G.,
       Ugurbil, K. and Springer, C.S., Jr. (2007), Magnetic field and
       tissue dependencies of human brain longitudinal 1H2O relaxation
       in vivo. Magn. Reson. Med., 57: 308-318.
       https://doi.org/10.1002/mrm.21122.

    .. [4] Zhou, J., Golay, X., van Zijl, P.C.M., Silvennoinen, M.J.,
       Kauppinen, R., Pekar, J. and Kraut, M. (2001), Inverse T2
       contrast at 1.5 Tesla between gray matter and white matter in the
       occipital lobe of normal adult human brain. Magn. Reson. Med.,
       46: 401-406. https://doi.org/10.1002/mrm.1204.

    .. [5] Wansapura, J.P., Holland, S.K., Dunn, R.S. and Ball, W.S., Jr
       (1999), NMR relaxation times in the human brain at 3.0 tesla.
       J. Magn. Reson. Imaging, 9: 531-538.
       https://doi.org/10.1002/
       (SICI)1522-2586(199904)9:4<531::AID-JMRI4>3.0.CO;2-L
    """

    def __init__(self, index = 2,
        pd = 0.677,
        chem_shift = 0,
        t1 = 0.832,
        t2 = 79.6e-3,
        t2_star = 45e-3,
        dynamic_t1 = np.array([[0.05, 0.15, 0.5, 1.5, 3],
                  [0.273, 0.352, 0.687, 0.656, 0.832]]),
        dynamic_t2 = np.array([[0.05, 0.15, 0.5, 1.5, 3],
                  [0.092, 0.105, 0.107, 0.084, 0.0796]]),
        t1_alt = 10e-3, t2_alt = 1e-3,
        dynamic_t1_alt = [0],
        dynamic_t2_alt = [0]):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

class CSFSynth(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for
    human CSF at 3T. Contains parameters for dynamic relaxation through
    interpolation from 0.05T to 3T.

    References
    ----------
    .. [1] O'Reilly T, Webb AG. In vivo T1 and T2 relaxation time maps
       of brain tissue, skeletal muscle, and lipid measured in healthy
       volunteers at 50 mT. Magn Reson Med. 2021; 87: 884-895.
       https://doi.org/10.1002/mrm.29009.

    .. [2] Rooney, W.D., Johnson, G., Li, X., Cohen, E.R., Kim, S.-G.,
       Ugurbil, K. and Springer, C.S., Jr. (2007), Magnetic field and
       tissue dependencies of human brain longitudinal 1H2O relaxation
       in vivo. Magn. Reson. Med., 57: 308-318.
       https://doi.org/10.1002/mrm.21122.

    .. [3] A. Daoust, S. Dodd, G. Nair, N. Bouraoud, S. Jacobson,
       S. Walbridge, D.S. Reich, A. Koretsky, Transverse relaxation of
       cerebrospinal fluid depends on glucose concentration, Magnetic
       Resonance Imaging, Volume 44, 2017, Pages 72-81,
       https://doi.org/10.1016/j.mri.2017.08.001.

    .. [4] Lin C., Bernstein M., Huston J., Fain S.. Measurements of T1
       Relaxation times at 3.0T: Implications for clinical MRA. Proc.
       Intl. Soc. Mag. Reson. Med 9 (2001)
    """

    def __init__(self, index = 3,
        pd = 1,
        chem_shift = 0,
        t1 = 4,
        t2 = 2.02,
        t2_star = 0.44,
        dynamic_t1 = np.array([[0.05, 0.15, 0.5, 1.5, 3],
                  [3.528, 4.36, 4.22, 4.07, 4.163]]),
        dynamic_t2 = np.array([[0.05, 0.15, 0.5, 1.5, 3],
                  [1.627, 1.76, 2.19, 2.1, 2.02]]),
        t1_alt = 10e-3,
        t2_alt = 1e-3,
        dynamic_t1_alt = [0],
        dynamic_t2_alt = [0]):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

#~~~~~ Agarose Phantoms ~~~~~#
class Agar2pc(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for
    2% Agarose gel at 3T. Does not contain information for dynamic
    relaxation. Static t1rho relaxation value acquired at 500 Hz. t1 and
    t2 information calculated from the formula published in [1]. t1rho
    values from [2]. t2star and t2rho arbitrarily set to 20 ms.

    References
    ----------
    .. [1] Woletz, M., Roat, S., Hummer, A., Tik, M. and Windischberger,
       C. (2021), Technical Note: Human tissue-equivalent MRI phantom
       preparation for 3 and 7 Tesla. Med. Phys., 48: 4387-4394.
       https://doi.org/10.1002/mp.14986

    .. [2] X. Li, V. Pedoia, D. Kumar, J. Rivoire, C. Wyatt,
       D. Lansdown, K. Amano, N. Okazaki, D. Savic, M.F. Koff,
       J. Felmlee, S.L. Williams, S. Majumdar, Cartilage T1rho and T2
       relaxation times: longitudinal reproducibility and variations
       using different coils, MR systems and sites, Osteoarthritis and
       Cartilage, Volume 23, Issue 12, 2015, 2214-2223,
       https://doi.org/10.1016/j.joca.2015.07.006.
    """

    def __init__(self, index = 1,
        pd = 1,
        chem_shift = 0,
        t1 = 2.710,
        t2 = 79.7e-3,
        t2_star = 20e-3,
        dynamic_t1 = [0],
        dynamic_t2 = [0],
        t1_alt = 58.8e-3,
        t2_alt = 20e-3,
        dynamic_t1_alt = [0],
        dynamic_t2_alt = [0]):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

class Agar4pc(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for
    4% Agarose gel at 3T. Does not contain information for dynamic
    relaxation. Static t1rho relaxation value acquired at 500 Hz. t1 and
    t2 information calculated from the formula published in [1]. t1rho
    values from [2]. t2star and t2rho arbitrarily set to 20 ms.


    References
    ----------
    .. [1] Woletz, M., Roat, S., Hummer, A., Tik, M. and Windischberger,
       C. (2021), Technical Note: Human tissue-equivalent MRI phantom
       preparation for 3 and 7 Tesla. Med. Phys., 48: 4387-4394.
       https://doi.org/10.1002/mp.14986

    .. [2] Buck FM, Bae WC, Diaz E, Du J, Statum S, Han ET, Chung CB.
       Comparison of T1rho measurements in agarose phantoms and human
       patellar cartilage using 2D multislice spiral and 3D
       magnetization prepared partitioned k-space spoiled gradient-echo
       snapshot techniques at 3 T. AJR Am J Roentgenol.
       2011 Feb;196(2):W174-9. doi: 10.2214/AJR.10.4570.
    """
        
    def __init__(self, index = 2,
        pd = 1,
        chem_shift = 0,
        t1 = 2.270,
        t2 = 44.9e-3,
        t2_star = 20e-3,
        dynamic_t1 = [0],
        dynamic_t2 = [0],
        t1_alt = 29e-3,
        t2_alt = 20e-3,
        dynamic_t1_alt = [0],
        dynamic_t2_alt = [0]):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

class CFMMAgar2pc(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for 2%
    Agarose gel at 3T. Contains parameters for dynamic t1rho relaxation
    using interpolation. Static parameters denote t1rho relaxation
    at 500Hz. All relaxation values taken from scans performed at
    Western University [1], except t2rho which has been set to 1/2
    t1rho.

    References
    ----------
    .. [1] Unpublished, Under review
    """

    def __init__(self, index = 1,
        pd = 1,
        chem_shift = 0,
        t1 = 2.887,
        t2 = 86.2e-3,
        t2_star = 56.3e-3,
        dynamic_t1 = [0],
        dynamic_t2 = [0],
        t1_alt = 64.1e-3,
        t2_alt = 32e-3,
        dynamic_t1_alt = np.array([[0, 3.131e-6, 3.914e-6],
                       [56.3e-3, 64.4e-3, 64.1e-3]]),
        dynamic_t2_alt = np.array([[0, 3.131e-6, 3.914e-6],
                       [28e-3, 32e-3, 32e-3]])):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

class CFMMAgar4pc(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for 4%
    Agarose gel at 3T. Contains parameters for dynamic t1rho relaxation
    using interpolation. Prefilled with static parameters for t1rho
    relaxation at 500Hz. All relaxation values taken from scans
    performed at Western University [1], except t2rho which has been set
    to 1/2 t1rho.

    References
    ----------
    .. [1] Unpublished, under review
    """

    def __init__(self, index = 2,
        pd = 1,
        chem_shift = 0,
        t1 = 2.625,
        t2 = 52.4e-3,
        t2_star = 31.9e-3,
        dynamic_t1 = [0],
        dynamic_t2 = [0],
        t1_alt = 36.7e-3,
        t2_alt = 18e-3,
        dynamic_t1_alt = np.array([[0, 3.131e-6, 3.914e-6],
                       [31.9e-3, 36.2e-3, 36.7e-3]]),
        dynamic_t2_alt = np.array([[0, 3.131e-6, 3.914e-6],
                       [16e-3, 18e-3, 18e-3]])):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

#~~~~~ Cartilage Materials ~~~~~#

class CartilageHealthy(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for
    healthy cartilage. Contains parameters for dynamic relaxation
    calculations at field strengths <= 3T using
    bottomley_curve_offset_spinlock(). Also contains parameters
    for dynamic t1rho for locking fields from 0 to 550 Hz. Constant
    relaxation values are set for B0 = 3T. Relaxation times have been
    aggregated from the literature [1].

    References
    ----------
    .. [1] Unpublished, manuscript under preparation
    """

    def __init__(self, index = 1,
        pd = 1,
        chem_shift = 0,
        t1 = 1.24,
        t2 = 32.37e-3,
        t2_star = 21.7e-3,
        dynamic_t1 = np.array([0.7724867, 0.44195728, 0]),
        dynamic_t2 = np.array([0, 0, 21.7e-3]),
        t1_alt = 47.6e-3,
        t2_alt = 21.7e-3,
        dynamic_t1_alt = np.array([1, 0.36920115, 0.03083421]),
        dynamic_t2_alt = np.array([0, 0, 21.7e-3])):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)

class CartilageMildOA(samplegen.Material):
    """
    Instance of Material Class, pre-filled with static parametres for
    cartilage affected by mild osteoarthritis. Contains parameters
    for dynamic relaxation calculations at field strengths <=3T
    using bottomley_curve_offset_spinlock(). Also contains parameters
    for dynamic t1rho for locking fields from 0 to 550 Hz. Constant
    relaxation values are set for B0 = 3T. Relaxation times are
    equivalent to healthy cartilage [1] +14%, per the diagnostic
    criteria established by [2].

    References
    ----------
    .. [1] Unpublished, manuscript under preparation

    .. [2] Chalian, Majid & Li, Xiaojuan & Guermazi, Ali & Obuchowski,
       Nancy & Carrino, John & Oei, Edwin & Link, Thomas. (2021). The
       QIBA Profile for MRI-based Compositional Imaging of Knee
       Cartilage. Radiology. 301. 204587. 10.1148/radiol.2021204587. 
    """ 

    def __init__(self, index = 2,
        pd = 1,
        chem_shift = 0,
        t1 = 1.24,
        t2 = 36.9e-3,
        t2_star = 24.72e-3,
        dynamic_t1 = np.array([0.7724867, 0.44195728, 0]),
        dynamic_t2 = np.array([0, 0, 24.72e-3]),
        t1_alt = 52.4e-3,
        t2_alt = 24.72e-3,
        dynamic_t1_alt = np.array([1, 0.35737191, 0.03504875]),
        dynamic_t2_alt = np.array([0, 0, 24.72e-3])):
        super().__init__(index, pd, chem_shift, t1, t2, t2_star,
            dynamic_t1, dynamic_t2, t1_alt, t2_alt,
            dynamic_t1_alt, dynamic_t2_alt)
