import numpy as np
import simCode.SampleClasses as sCl

# Instances of SampleClasses.Material() classes pre-filled with relevant material parameters
#
# Original version of this code designed and written by John Adams (jadam33@uwo.ca)
#
# This code is published under a GNU General Public License version 3 License. Anyone using this code 
# must give attribution to the original author (John Adams), and must make any derivative code built 
# using this code available under the same licensing terms.


#~~~~~ Version History ~~~~~#
# 1.0  Public Release - July 1 2026 - John Adams

#~~~~~ Test Materials ~~~~~#

class InfiniteRelax(sCl.Material):
     """Instance of Material Class, pre-filled with all relaxation vales set to 1e9 seconds
     except T2star = 1e8 seconds"""
     def __init__(self, ind = 1, PD = 1, chemShift = 0, T1 = 1e9, T2 = 1e9, T2star = 1e8,
                    dynamicT1 = [0], dynamicT2 = [0], T1alt = 1e9, T2alt = 1e9, dynamicT1alt = [0],
                    dynamicT2alt = [0]):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)

#~~~~~ Brain Materials ~~~~~#
class GM_synth(sCl.Material):
    """Instance of Material class, pre-filled with parametres for human Grey Matter at 3T.
    Contains parameters for dynamic relaxation through interpolation from 0.05T to 3T."""

    def __init__(self, ind = 1, PD = 0.825, chemShift = 0, T1 = 1.331, T2 = 110e-3, T2star = 52e-3,
                    dynamicT1 = np.array([[0.05, 0.15, 0.5, 1.5, 3],[0.329, 0.525, 0.815, 1.188, 1.331]]), 
                    dynamicT2 = np.array([[0.05, 0.15, 0.5, 1.5, 3],[0.097, 0.11, 0.11, 0.088, 0.11]]), 
                    T1alt = 8e-3, T2alt = 1e-3, dynamicT1alt = [0],
                    dynamicT2alt = [0]):
        super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                            T1alt, T2alt, dynamicT1alt, dynamicT2alt)
    
class WM_synth(sCl.Material):
    """Instance of Material Class, pre-filled with parametres for human White Matter at 3T.
    Contains parameters for dynamic relaxation through interpolation from 0.05T to 3T."""

    def __init__(self, ind = 2, PD = 0.677, chemShift = 0, T1 = 0.832, T2 = 79.6e-3, T2star = 45e-3,
                    dynamicT1 = np.array([[0.05, 0.15, 0.5, 1.5, 3],[0.273, 0.352, 0.687, 0.656, 0.832]]), 
                    dynamicT2 = np.array([[0.05, 0.15, 0.5, 1.5, 3],[0.092, 0.105, 0.107, 0.084, 0.0796]]), 
                    T1alt = 10e-3, T2alt = 1e-3, dynamicT1alt = [0],
                    dynamicT2alt = [0]):
        super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                            T1alt, T2alt, dynamicT1alt, dynamicT2alt)

class CSF_synth(sCl.Material):
    """Instance of Material Class, pre-filled with parametres for human White Matter at 3T.
    Contains parameters for dynamic relaxation through interpolation from 0.05T to 3T."""

    def __init__(self, ind = 3, PD = 1, chemShift = 0, T1 = 4, T2 = 2.02, T2star = 0.44,
                    dynamicT1 = np.array([[0.05, 0.15, 0.5, 1.5, 3],[3.528, 4.36, 4.22, 4.07, 4.163]]), 
                    dynamicT2 = np.array([[0.05, 0.15, 0.5, 1.5, 3],[1.627, 1.76, 2.19, 2.1, 2.02]]), 
                    T1alt = 10e-3, T2alt = 1e-3, dynamicT1alt = [0],
                    dynamicT2alt = [0]):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)

#~~~~~ Agarose Phantoms ~~~~~#
class Agar2pc(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for 2% Agarose gel at 3T.
    Does not contain information for dynamic relaxation. Prefilled with parameters for
    T1rho relaxation at 500Hz. T1 and T2 information calculated from the formula published 
    in Woletz, Medical Physics, 2021. T1rho values from Li, Osteoarthiris and Cartilage, 2015.
    Synthetic T2star and T2rho."""
     
     def __init__(self, ind = 1, PD = 1, chemShift = 0, T1 = 2.710, T2 = 79.7e-3, T2star = 20e-3,
                    dynamicT1 = [0], dynamicT2 = [0], T1alt = 58.8e-3, T2alt = 20e-3, dynamicT1alt = [0],
                    dynamicT2alt = [0]):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)
            
class Agar4pc(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for 4% Agarose gel at 3T.
    Does not contain information for dynamic relaxation. Prefilled with parameters for
    T1rho relaxation at 500Hz. T1 and T2 information calculated from the formula published 
    in Woletz, Medical Physics, 2021. T1rho values from Buck, AJR, 2011.
    Synthetic T2star and T2rho."""
     
     def __init__(self, ind = 2, PD = 1, chemShift = 0, T1 = 2.270, T2 = 44.9e-3, T2star = 20e-3,
                    dynamicT1 = [0], dynamicT2 = [0], T1alt = 29e-3, T2alt = 20e-3, dynamicT1alt = [0],
                    dynamicT2alt = [0]):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)
            
class Agar2pcDynamic(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for 2% Agarose gel at 3T.
    Does not contain information for dynamic relaxation. Prefilled with parameters for
    T1rho relaxation at 500Hz. T1 and T2 information calculated from the formula published 
    in Woletz, Medical Physics, 2021. T1rho values from Li, Osteoarthiris and Cartilage, 2015.
    Synthetic T2star and T2rho."""
     
     def __init__(self, ind = 1, PD = 1, chemShift = 0, T1 = 2.710, T2 = 79.7e-3, T2star = 40e-3,
                    dynamicT1 =  np.array([
                         [0.117, 1.5, 3, 4.7, 7],
                         [1.122, 2.181, 2.710, 2.799, 2.928]
                    ]),
                    dynamicT2 =  np.array([
                         [0.117, 1.5, 3, 4.7, 7],
                         [56e-3, 60e-3, 79.7e-3, 77.7e-3, 75.1e-3]
                    ]),
                    T1alt = 58.8e-3, T2alt = 20e-3, 
                    dynamicT1alt = [0],
                    dynamicT2alt = [0]):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)
            
class Agar4pcDynamic(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for 4% Agarose gel at 3T.
    Does not contain information for dynamic relaxation. Prefilled with parameters for
    T1rho relaxation at 500Hz. T1 and T2 information calculated from the formula published 
    in Woletz, Medical Physics, 2021. T1rho values from Buck, AJR, 2011.
    Synthetic T2star and T2rho."""
     
     def __init__(self, ind = 2, PD = 1, chemShift = 0, T1 = 2.270, T2 = 44.9e-3, T2star = 22.5e-3,
                    dynamicT1 = np.array([
                         [0.117, 1.5, 3, 4.7, 7],
                         [1, 1839, 2.273, 2.430, 2.680]
                    ]),
                    dynamicT2 = np.array([
                         [0.117, 1.5, 3, 4.7, 7],
                         [23e-3, 36.8e-3, 44.9e-3, 42.4e-3, 39.4e-3]
                    ]),
                    T1alt = 29e-3, T2alt = 20e-3,
                    dynamicT1alt = [0],
                    dynamicT2alt = [0]):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)
            
class CFMM_Agar2pc(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for 2% Agarose gel at 3T.
    Contains parameters for dynamic T1rho relaxation. Prefilled with static parameters for
    T1rho relaxation at 500Hz. All relaxation values taken from CFMM Scans, except T2rho 
    which has been set to 1/2 T1rho."""
     
     def __init__(self, ind = 1, PD = 1, chemShift = 0, T1 = 2.887, T2 = 86.2e-3, T2star = 56.3e-3,
                    dynamicT1 = [0], dynamicT2 = [0], T1alt = 64.1e-3, T2alt = 32e-3, 
                    dynamicT1alt = np.array([
                         [0, 3.131e-6, 3.914e-6],
                         [56.3e-3, 64.4e-3, 64.1e-3]]),
                    dynamicT2alt = np.array([
                         [0, 3.131e-6, 3.914e-6],
                         [28e-3, 32e-3, 32e-3]])):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)
            
class CFMM_Agar4pc(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for 4% Agarose gel at 3T.
    Contains parameters for dynamic T1rho relaxation. Prefilled with static parameters for
    T1rho relaxation at 500Hz. All relaxation values taken from CFMM Scans, except T2rho 
    which has been set to 1/2 T1rho."""
     
     def __init__(self, ind = 2, PD = 1, chemShift = 0, T1 = 2.625, T2 = 52.4e-3, T2star = 31.9e-3,
                    dynamicT1 = [0], dynamicT2 = [0], T1alt = 36.7e-3, T2alt = 18e-3, 
                    dynamicT1alt = np.array([
                         [0, 3.131e-6, 3.914e-6],
                         [31.9e-3, 36.2e-3, 36.7e-3]]),
                    dynamicT2alt = np.array([
                         [0, 3.131e-6, 3.914e-6],
                         [16e-3, 18e-3, 18e-3]])):
            super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)
            
#~~~~~ Cartilage Materials ~~~~~#
            
class CartilageMildOA(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for cartilage affected by 
     mild osteoarthritis. Contains information for dynamic relaxation calculations at field
     strengths less than 3T using BottomleyCurveOffset_SpinLock(). Also contains information
     for dynamic T1rho under the same conditions. Constant relaxation values are set for 
     B0 = 3T. Relaxation times are equivalent to healthy cartilage +14%, per the diagnostic
     criteria established by Chalin et al (2021), Radiology""" 
  
     def __init__(self, ind = 2,
                    PD = 1,
                    chemShift = 0,       
                    T1 = 1.24,
                    T2 = 36.9e-3,
                    T2star = 24.72e-3, 
                    dynamicT1 = np.array([0.7724867, 0.44195728, 0]),
                    dynamicT2 = np.array([0, 0, 24.72e-3]), 
                    T1alt = 52.4e-3, 
                    T2alt = 24.72e-3, 
                    dynamicT1alt = np.array([0.99999289, 0.35737191, 0.03504875]),
                    dynamicT2alt = np.array([0, 0, 24.72e-3])):
          super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)

class CartilageHealthy(sCl.Material):
     """Instance of Material Class, pre-filled with parametres for healthy cartilage. Contains
     information for dynamic relaxation calculations at field strengths less than 3T using 
     BottomleyCurveOffset_SpinLock(). Also contains information for dynamic T1rho under the same
     conditions. Constant relaxation values are set for B0 = 3T. Relaxation times have been 
     aggregated from the literature""" 

     def __init__(self, ind = 1,
                    PD = 1,
                    chemShift = 0,       
                    T1 = 1.24,
                    T2 = 32.37e-3,
                    T2star = 21.7e-3, 
                    dynamicT1 = np.array([0.7724867, 0.44195728, 0]),
                    dynamicT2 = np.array([0, 0, 21.7e-3]), 
                    T1alt = 47.6e-3, 
                    T2alt = 21.7e-3, 
                    dynamicT1alt = np.array([1, 0.36920115, 0.03083421]),
                    dynamicT2alt = np.array([0, 0, 21.7e-3])):
          super().__init__(ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
                        T1alt, T2alt, dynamicT1alt, dynamicT2alt)
