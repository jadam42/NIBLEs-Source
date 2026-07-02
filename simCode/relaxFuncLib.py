import numpy as np

# Functions used to calculate T1 and T2 relaxation times as a function of applied field strength. 
#
#  To be compatible with the NIBLEs solver, relaxation functions must follow the input format 
# func(BLab, Brot, BSolve, ParamT1, ParamT2), where the three B variables are the total applied 
# magnetic field expressed as a 3 element NumPy array in the various reference frames calculated 
# during solving, ParamT1 is a variable containing all other information required to calculate T1, 
# and ParamT2 is a variable containing all other information required to calculate T2. B is 
# calculated automatically by the solver, and does not need to be defined by an end user. Each 
# function should output a pair of values: T1, T2
#
# Original version of this code designed and written by John Adams (jadam33@uwo.ca)
#
# This code is published under a GNU General Public License version 3 License. Anyone using this code 
# must give attribution to the original author (John Adams), and must make any derivative code built 
# using this code available under the same licensing terms.

#~~~~~ Version History ~~~~~#
# 1.0  Public Release - July 1 2026 - John Adams


def Constant(BLab, BRot, BSolve, ParamT1, ParamT2):
    """
    Function returns a known, constant relaxation time

    Parameters:
    BLab = 3-Element Numpy array, containing total applied field in the lab frame
    BRot = 3-Element Numpy array, containing total applied field in the Larmor frame
    BSolve = 3-Element Numpy array, containing total applied field in the final reference frame 
        in which Solve() evolves magnetization
    ParamT1 = Known T1 relaxation time 
    ParamT2 = Known T2 relaxation time

    Returns: 
    T1, T2 = Relaxation times in seconds
    """

    T1 = ParamT1
    T2 = ParamT2

    return T1, T2

def Interpolation(BLab, BRot, BSolve, ParamT1, ParamT2):
    """
    Function calculates relaxation times by interpolating between measured relaxation values 
    on a relaxation dispersion curve. Uses numpy.interp, which linearly interpolates between given 
    data points. If given field values beyond the range of measured data, returns the nearest known 
    relaxation value.

    Parameters:
    BLab = 3-Element Numpy array, containing total applied field in the lab frame
    BRot = 3-Element Numpy array, containing total applied field in the Larmor frame
    BSolve = 3-Element Numpy array, containing total applied field in the final reference frame 
        in which Solve() evolves magnetization
    ParamT1 = 2xn Numpy array, where the bottom row contains measured relaxation times, and the top row 
        contains the corresponding field strengths at which relaxation values were measured.
    ParamT2 = 2xn Numpy array, where the bottom row contains measured relaxation times, and the top row 
        contains the corresponding field strengths at which relaxation values were measured.

    Returns: 
    T1, T2 = Relaxation times in seconds
    """

    magB = np.sqrt(BLab.dot(BLab))
    T1 = np.interp(magB, ParamT1[0,:], ParamT1[1,:])
    T2 = np.interp(magB, ParamT2[0,:], ParamT2[1,:])

    return T1, T2

def Interpolation_SpinLock(BLab, BRot, BSolve, ParamT1, ParamT2):
    """
    Function calculates relaxation times by interpolating between measured relaxation values 
    on a relaxation dispersion curve. Uses numpy.interp, which linearly interpolates between given 
    data points. If given field values beyond the range of measured data, returns the nearest known 
    relaxation value.

    Uses BSolve to calculate relaxation dispersion for spin lock measurements.

    Parameters:
    BLab = 3-Element Numpy array, containing total applied field in the lab frame
    BRot = 3-Element Numpy array, containing total applied field in the Larmor frame
    BSolve = 3-Element Numpy array, containing total applied field in the final reference frame 
        in which Solve() evolves magnetization
    ParamT1 = 2xn Numpy array, where the bottom row contains measured relaxation times, and the top row 
        contains the corresponding field strengths at which relaxation values were measured.
    ParamT2 = 2xn Numpy array, where the bottom row contains measured relaxation times, and the top row 
        contains the corresponding field strengths at which relaxation values were measured.

    Returns: 
    T1, T2 = Relaxation times in seconds

    """

    magB = np.sqrt(BSolve.dot(BSolve))
    T1 = np.interp(magB, ParamT1[0,:], ParamT1[1,:])
    T2 = np.interp(magB, ParamT2[0,:], ParamT2[1,:])

    return T1, T2

def DoubleExponential(BLab, BRot, BSolve, ParamT1, ParamT2):
    """
    Function calculates relaxation times using a fit double exponential relaxation curve:
    y = A*exp(-Bx)+ C*exp(-Dx)

    Parameters:
    BLab = 3-Element Numpy array, containing total applied field in the lab frame
    BRot = 3-Element Numpy array, containing total applied field in the Larmor frame
    BSolve = 3-Element Numpy array, containing total applied field in the final reference frame 
        in which Solve() evolves magnetization
    ParamT1 = 4-Element array containing the coefficients used to define double exponential model 
        for T1 relaxation [A, B, C, D]
    ParamT2 = 4-Element array containing the coefficients used to define double exponential model 
        for T2 relaxation [A, B, C, D]

    Returns: 
    T1, T2 = Relaxation times in seconds
    """

    magB = np.sqrt(BLab.dot(BLab))
    T1 = ParamT1[0]*np.exp(-ParamT1[1]*magB)+ ParamT1[2]*np.exp(-ParamT1[3]*magB)
    T2 = ParamT2[0]*np.exp(-ParamT2[1]*magB)+ ParamT2[2]*np.exp(-ParamT2[3]*magB)

    return T1, T2

def BottomleyCurveOffset(BLab, BRot, BSolve, ParamT1, ParamT2):
    """
    Function calculates relaxation times using a fit double exponential relaxation curve
    as presented in Bottomley (1984):
    y = A*x^B

    Parameters:
    BLab = 3-Element Numpy array, containing total applied field in the lab frame
    BRot = 3-Element Numpy array, containing total applied field in the Larmor frame
    BSolve = 3-Element Numpy array, containing total applied field in the final reference frame 
        in which Solve() evolves magnetization
    ParamT1 = 2-Element array containing the coefficients used to define double exponential model 
        for T1 relaxation [A, B]
    ParamT2 = 2-Element array containing the coefficients used to define double exponential model 
        for T2 relaxation [A, B]

    Returns: 
    T1, T2 = Relaxation times in seconds
    """

    magB = np.sqrt(BLab.dot(BLab))
    T1 = ParamT1[0]*(magB**ParamT1[1])+ ParamT1[2]
    T2 = ParamT2[0]*(magB**ParamT2[1])+ ParamT2[2]

    return T1, T2

def BottomleyCurveOffset_SpinLock(BLab, BRot, BSolve, ParamT1, ParamT2):
    """
    Function calculates relaxation times using a fit double exponential relaxation curve
    as presented in Bottomley (1984) modified to have a finite 0 intercept:
    y = A*x^B+ C

    Parameters:
    BLab = 3-Element Numpy array, containing total applied field in the lab frame
    BRot = 3-Element Numpy array, containing total applied field in the Larmor frame
    BSolve = 3-Element Numpy array, containing total applied field in the final reference frame 
        in which Solve() evolves magnetization
    ParamT1 = 3-Element array containing the coefficients used to define double exponential model 
        for T1 relaxation [A, B]
    ParamT2 = 3-Element array containing the coefficients used to define double exponential model 
        for T2 relaxation [A, B]

    Returns: 
    T1, T2 = Relaxation times in seconds
    """

    magB = np.sqrt(BSolve.dot(BSolve))
    T1 = ParamT1[0]*(magB**ParamT1[1])+ ParamT1[2]
    T2 = ParamT2[0]*(magB**ParamT2[1])+ ParamT2[2]

    return T1, T2
