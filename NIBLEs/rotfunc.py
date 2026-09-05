"""
Library of functions for defining reference frame transformations for
use in the NIBLEs solver.
"""

import numpy as np

def zero(t, *args, **kwargs):
    """Returns zero for any inputs.

    Used within NIBLEs as a placeholder for user-specified functions
    that have not been assigned.
    """
    return 0


def identity(x, *args, **kwargs):
    """Returns the first value it is given regardless of other inputs.
     
    Used within NIBLEs when reference frame rotations away from the
    rotating reference frame are not needed.
    """
    return x


def rotating_reference_frame(field_vec, t, param):
    """Converts magnetic field vectors from the laboratory reference
    frame into the rotating reference frame.

    Used exclusively within the NIBLEs solver.

    Parameters
    ----------
    field_vec : array_like
        Field vector to be rotated in Cartesian coordinates.
    t : float
        Simulation time. Produced by solver and should not be defined by
        end user.
    param : tuple
        Parameter tuple: the first element is the main field strength
        defined as a 1D, 3-element Numpy array; the second element is
        the Larmor frequency of the nucleus being imaged as a float. 
        (b0, Larmor)

    Returns
    -------
    rotated_field_vec : array_like
        field_vec translated into the rotating reference frame.
    """

    # Calculate Bz in rotating reference frame by subtracting nominal
    # B0z from Bz
    rotated_field_vec_z = field_vec[2]- param[0][2]

    # Calculate rotation matrix based on larmor frequency and time
    rotation_matrix = np.array(
                    [[np.cos(-param[1]*t), np.sin(-param[1]*t), 0],
                     [-np.sin(-param[1]*t), np.cos(-param[1]*t), 0],
                     [0, 0, 1]]
                     )

    # Apply rotation matrix and return final field
    rotated_field_vec = np.matmul(rotation_matrix, field_vec)
    rotated_field_vec[2] = rotated_field_vec_z

    return rotated_field_vec


def zero_bz_frame(vector, t, param, tot_b_larmor, dir_fore, if_b):
    """Converts vectors from the rotating reference frame into a
    rotating reference frame where Bz = 0.

    Parameters
    ----------
    vector : array_like
        Field or magnetization vector to be rotated, defined in 
        Cartesian coordinates.
    t : float
        Simulation time. Produced by solver and should not be defined by
        end user.
    param : tuple
        Parameter tuple: the first element is the main field strength
        defined as a 1D, 3-element Numpy array; the second element is
        the Larmor frequency of the nucleus being imaged as a float. 
        (b0, Larmor)
    tot_b_larmor : array_like
        Total magnetic field in the Larmor rotating reference frame.
        Used to identify what transformation will eliminate z-field.
    dir_fore : bool
        Controls direction of transform. If set to True, will convert
        away from Larmor rotating reference frame. If false, will
        convert back to Larmor reference frame.
    if_b : bool
        Identifies if given vector is a magnetic field. If true, will
        adjust the amplitude of the given vector's z-component based on
        the frame transformation. 

    Returns
    -------
    rotated_vector : array_like
        Vector translated to/from a reference frame where Bz is zero.
    """
    # Calculate the frequency difference between the Larmor frequency
    # and the zeroBz frame
    omega = param[1]/param[0][2]*tot_b_larmor[2]

    # Calculate rotation matrix and its inverse
    rotation_matrix = np.array([[np.cos(-omega*t), np.sin(-omega*t), 0],
                     [-np.sin(-omega*t), np.cos(-omega*t), 0],
                     [0, 0, 1]])

    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    # Apply appropriate rotation matrix to vector based on the control
    # variables, adjusting the z component if vector is a magnetic
    # field
    if dir_fore and not if_b:
        rotated_vector = np.matmul(rotation_matrix, vector)
    elif dir_fore and if_b:
        rotated_vector = np.matmul(rotation_matrix, vector)
        rotated_vector[2] = 0
    elif not dir_fore and not if_b:
        rotated_vector = np.matmul(inv_rotation_matrix, vector)
    elif not dir_fore and if_b:
        rotated_vector = np.matmul(inv_rotation_matrix, vector)
        rotated_vector[2] = tot_b_larmor[2]

    return rotated_vector


def spinlock_rotator(vector, t, phi, tot_b_larmor, dir_fore, if_b):
    """Converts vectors from the Larmor reference frame into the
    doubly-rotating reference frame for T1rho/T2rho experiments.

    Parameters
    ----------
    vector : array_like
        Field or magnetization vector to be rotated, defined in 
        Cartesian coordinates.
    t : float
        Simulation time. Produced by solver and should not be defined by
        end user. Not used in calculations, but present as input to
        maintain standard format for supplemental rotation functions.
    phi : float
        Angle in radians indicating the orientation of sample
        magnetization at the beginning of spinlock pulse
    tot_b_larmor : array_like
        Total magnetic field in the Larmor rotating reference frame.
        Not used in calculations, but present as input to maintain
        standard format for supplemental rotation functions.
    dir_fore : bool
        Controls direction of transform. If set to True, will convert
        away from Larmor rotating reference frame. If false, will
        convert back to Larmor reference frame.
    if_b : bool
        Identifies if given vector is a magnetic field. Not used in
        calculations, but present as input to maintain standard format
        for supplemental rotation functions.

    Returns
    -------
    rotated_vector : array_like
        Vector translated to/from a the doubly rotating reference frame.
    """
    rotated_vector = np.zeros(3).reshape(-1,1)

    # Define rotation matrix to rotate RF field to the xz-plane
    rotation_matrix1 = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    # Define rotation matrix to rotate x magnetization to lie along
    # z-axis
    rotation_matrix2 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    # Apply appropriate rotation based on dir_fore
    if dir_fore:
        rotated_vector = np.matmul(rotation_matrix1, vector)
        rotated_vector = -np.matmul(rotation_matrix2, rotated_vector)
    else:
        inv_rotation_matrix1 = np.linalg.inv(rotation_matrix1)
        inv_rotation_matrix2 = np.linalg.inv(rotation_matrix2)
        rotated_vector = -np.matmul(inv_rotation_matrix2, vector)
        rotated_vector = np.matmul(inv_rotation_matrix1, rotated_vector)

    return rotated_vector
