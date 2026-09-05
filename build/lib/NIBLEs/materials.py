"""
Library of instances of samplegen.Material() classes pre-filled with
relevant material parameters.
"""

import numpy as np

from simcode import samplegen

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
    Instance of Material class, pre-filled with parametres for human
    Grey Matter at 3T. Contains parameters for dynamic relaxation
    through interpolation from 0.05T to 3T.
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
    Instance of Material Class, pre-filled with parametres for human
    White Matter at 3T. Contains parameters for dynamic relaxation
    through interpolation from 0.05T to 3T.
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
    Instance of Material Class, pre-filled with parametres for human 
    CSF at 3T. Contains parameters for dynamic relaxation 
    through interpolation from 0.05T to 3T.
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
    Instance of Material Class, pre-filled with parametres for 
    2% Agarose gel at 3T. Does not contain information for dynamic 
    relaxation. Prefilled with parameters for t1rho relaxation at 
    500Hz. t1 and t2 information calculated from the formula published 
    in Woletz, Medical Physics, 2021. t1rho values from Li, 
    Osteoarthiris and Cartilage, 2015. Synthetic t2star and t2rho.
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
    Instance of Material Class, pre-filled with parametres for 4% 
    Agarose gel at 3T. Does not contain information for dynamic 
    relaxation. Prefilled with parameters for t1rho relaxation at 
    500Hz. t1 and t2 information calculated from the formula published 
    in Woletz, Medical Physics, 2021. t1rho values from Buck, AJR, 
    2011. Synthetic t2star and t2rho.
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
    Instance of Material Class, pre-filled with parametres for 2% 
    Agarose gel at 3T. Contains parameters for dynamic t1rho 
    relaxation. Prefilled with static parameters for t1rho relaxation 
    at 500Hz. All relaxation values taken from CFMM Scans, except t2rho
    which has been set to 1/2 t1rho.
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
    Instance of Material Class, pre-filled with parametres for 4% 
    Agarose gel at 3T. Contains parameters for dynamic t1rho 
    relaxation. Prefilled with static parameters for t1rho relaxation 
    at 500Hz. All relaxation values taken from CFMM Scans, except t2rho
    which has been set to 1/2 t1rho.
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
    Instance of Material Class, pre-filled with parametres for healthy 
    cartilage. Contains information for dynamic relaxation calculations
    at field strengths less than 3T using 
    bottomley_curve_offset_spinlock(). Also contains information for 
    dynamic t1rho under the same conditions. Constant relaxation 
    values are set for B0 = 3T. Relaxation times have been aggregated
    from the literature
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
    Instance of Material Class, pre-filled with parametres for 
    cartilage affected by mild osteoarthritis. Contains information 
    for dynamic relaxation calculations at field strengths less than 3T
    using bottomley_curve_offset_spinlock(). Also contains information 
    for dynamic t1rho under the same conditions. Constant relaxation 
    values are set for B0 = 3T. Relaxation times are equivalent to 
    healthy cartilage +14%, per the diagnostic criteria established by 
    Chalin et al (2021), Radiology
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
