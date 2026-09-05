"""
Library of functions used to calculate t1 and t2 relaxation times as a
function of applied field strength within the NNIBLEs solver.

To be compatible with the NIBLEs solver, relaxation functions must 
use the input variables b_lab, b_larmor, b_solve, param_t1, param_t2, 
where the three B variables are the total applied magnetic field 
expressed as a 3 element NumPy array in the various reference frames 
calculated during solving, param_t1 is a variable containing all other 
information required to calculate t1, and param_t2 is a variable 
containing all other information required to calculate t2. B is 
calculated automatically by the solver, and does not need to be 
defined by an end user. Each function should output a pair of values: 
t1, t2
"""

import numpy as np

def constant(b_lab, b_larmor, b_solve, param_t1, param_t2):
    """Returns a known, constant relaxation time.

    Parameters
    ----------
    b_lab : array_like
        Total applied field in the lab frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_larmor : array_like
        Total applied field in the Larmor frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_solve : array_like
        Total applied field in the solving frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    param_t1 : float
        Specified T_1 relaxation time.
    param_t2 : float
        Specified T_2 relaxation time.

    Returns
    -------
    t1, t2 : float
        Calulated longitudinal and transverse relaxation times.
    """

    t1 = param_t1
    t2 = param_t2

    return t1, t2

def interpolation(b_lab, b_larmor, b_solve, param_t1, param_t2):
    """Calculates relaxation times by interpolating between known
    relaxation values on a relaxation dispersion curve.

    Linearly interpolates between given points. If given field magnitude
    is beyond the range of measured data, returns the nearest known
    relaxation value.
    
    Parameters
    ----------
    b_lab : array_like
        Total applied field in the lab frame, expressed in Cartesian
        coordinates.
    b_larmor : array_like
        Total applied field in the Larmor frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_solve : array_like
        Total applied field in the solving frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    param_t1 : array_like
        2 row array: the top row contains the field strengths at
        which relaxation values were measured; and the bottom row
        contains measured relaxation times at the corresponding field
        strength.
    param_t1 : array_like
        2 row array: the top row contains the field strengths at
        which relaxation values were measured; and the bottom row
        contains measured relaxation times at the corresponding field
        strength.

    Returns
    -------
    t1, t2 : float
        Calulated longitudinal and transverse relaxation times.
    """

    b_mag = np.sqrt(b_lab.dot(b_lab))
    t1 = np.interp(b_mag, param_t1[0,:], param_t1[1,:])
    t2 = np.interp(b_mag, param_t2[0,:], param_t2[1,:])

    return t1, t2

def interpolation_spinlock(b_lab, b_larmor, b_solve,
                           param_t1, param_t2):
    """Calculates relaxation times by interpolating between known
    relaxation values on a relaxation dispersion curve for T1rho/T2rho
    experiments.

    Linearly interpolates between given points. If given field magnitude
    is beyond the range of measured data, returns the nearest known
    relaxation value. Uses applied field strength in the solving frame
    to accurately calculate effect of rf field on relaxation.
    
    Parameters
    ----------
    b_lab : array_like
        Total applied field in the lab frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_larmor : array_like
        Total applied field in the Larmor frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_solve : array_like
        Total applied field in the solving frame, expressed in Cartesian
        coordinates.
    param_t1 : array_like
        2 row array: the top row contains the field strengths at
        which relaxation values were measured; and the bottom row
        contains measured relaxation times at the corresponding field
        strength.
    param_t1 : array_like
        2 row array: the top row contains the field strengths at
        which relaxation values were measured; and the bottom row
        contains measured relaxation times at the corresponding field
        strength.

    Returns
    -------
    t1, t2 : float
        Calulated longitudinal and transverse relaxation times.
    """

    b_mag = np.sqrt(b_solve.dot(b_solve))
    t1 = np.interp(b_mag, param_t1[0,:], param_t1[1,:])
    t2 = np.interp(b_mag, param_t2[0,:], param_t2[1,:])

    return t1, t2

def double_exponential(b_lab, b_larmor, b_solve, param_t1, param_t2):
    """Calculates relaxation times using a fit double exponential
    relaxation curve.

    Calculates relaxation using the equation:
    y = A*exp(-Bx)+ C*exp(-Dx)
    
    Parameters
    ----------
    b_lab : array_like
        Total applied field in the lab frame, expressed in Cartesian
        coordinates.
    b_larmor : array_like
        Total applied field in the Larmor frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_solve : array_like
        Total applied field in the solving frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    param_t1 : array_like
        4-Element array containing the coefficients used to 
        define double exponential model for T_1 relaxation [A, B, C, D]
    param_t1 : array_like
        4-Element array containing the coefficients used to 
        define double exponential model for T_2 relaxation [A, B, C, D]

    Returns
    -------
    t1, t2 : float
        Calulated longitudinal and transverse relaxation times.
    """

    b_mag = np.sqrt(b_lab.dot(b_lab))
    t1 = (param_t1[0]*np.exp(-param_t1[1]*b_mag)
          + param_t1[2]*np.exp(-param_t1[3]*b_mag))
    t2 = (param_t2[0]*np.exp(-param_t2[1]*b_mag)
          + param_t2[2]*np.exp(-param_t2[3]*b_mag))

    return t1, t2

def bottomly_curve_offset(b_lab, b_larmor, b_solve, param_t1, param_t2):
    """Calculates relaxation times by fitting to the relaxation curves
    presented in Bottomley [1].

    Calculates relaxation using the equation:
    y = A*x^B
    
    Parameters
    ----------
    b_lab : array_like
        Total applied field in the lab frame, expressed in Cartesian
        coordinates.
    b_larmor : array_like
        Total applied field in the Larmor frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_solve : array_like
        Total applied field in the solving frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    param_t1 : array_like
        2-Element array containing the coefficients used to 
        define double exponential model for T_1 relaxation [A, B]
    param_t1 : array_like
        2-Element array containing the coefficients used to 
        define double exponential model for T_2 relaxation [A, B]

    Returns
    -------
    t1, t2 : float
        Calulated longitudinal and transverse relaxation times.

    References
    ----------
    .. [1] Bottomley, P.A., Foster, T.H., Argersinger, R.E. and Pfeifer,
    L.M. (1984), A review of normal tissue hydrogen NMR relaxation times
    and relaxation mechanisms from 1-100 MHz: Dependence on tissue type,
    NMR frequency, temperature, species, excision, and age. Med. Phys.,
    11: 425-448. https://doi.org/10.1118/1.595535
    """

    b_mag = np.sqrt(b_lab.dot(b_lab))
    t1 = param_t1[0]*(b_mag**param_t1[1])+ param_t1[2]
    t2 = param_t2[0]*(b_mag**param_t2[1])+ param_t2[2]

    return t1, t2

def bottomly_curve_offset_spinlock(b_lab, b_larmor, b_solve,
                                   param_t1, param_t2):

    """Calculates relaxation times by fitting to the relaxation curves
    presented in Bottomley [1], modified to have a finite y-intercept
    for T1rho/T2rho reproduction.

    Calculates relaxation using the equation:
    y = A*x^B+ C
    
    Parameters
    ----------
    b_lab : array_like
        Total applied field in the lab frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_larmor : array_like
        Total applied field in the Larmor frame, expressed in Cartesian
        coordinates. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    b_solve : array_like
        Total applied field in the solving frame, expressed in Cartesian
        coordinates.
    param_t1 : array_like
        3-Element array containing the coefficients used to 
        define double exponential model for T_1 relaxation [A, B, C]
    param_t1 : array_like
        3-Element array containing the coefficients used to 
        define double exponential model for T_2 relaxation [A, B, C]

    Returns
    -------
    t1, t2 : float
        Calulated longitudinal and transverse relaxation times.

    References
    ----------
    .. [1] Bottomley, P.A., Foster, T.H., Argersinger, R.E. and Pfeifer,
    L.M. (1984), A review of normal tissue hydrogen NMR relaxation times
    and relaxation mechanisms from 1-100 MHz: Dependence on tissue type,
    NMR frequency, temperature, species, excision, and age. Med. Phys.,
    11: 425-448. https://doi.org/10.1118/1.595535
    """

    b_mag = np.sqrt(b_solve.dot(b_solve))
    t1 = param_t1[0]*(b_mag**param_t1[1])+ param_t1[2]
    t2 = param_t2[0]*(b_mag**param_t2[1])+ param_t2[2]

    return t1, t2
