"""
Simulation code for the Numeric Integrator for the Bloch Equations
(NIBLEs)

This file contains the code for evolving magnetization through numeric
solving of the Bloch equations. Numeric solving is performed using
scipy.Solve_IVP, and multi-core parallelization is performed using
Python's concurrent.futures module. Enables solving the Bloch equations
in an arbitrary reference frame; using time and spatially dependent
magnetic fields; and using time and field dependent relaxation values.
"""

import copy
import concurrent.futures

import numpy as np
from scipy.integrate import solve_ivp

import simcode.rotfunc as rot

#~~~~~ Functions for Bloch Solving ~~~~~#

def block(t, mag_solve, index, GAMMA, relax_func, b0_func, brf_func,
          bgrad_func, frame_rot, param_t1, param_t2, param_b0, param_rf,
          param_grad, param_rot, param_rot_plus, equi_m):
    """Implementation of the Bloch Equations in a form which can be
    solved by SciPy's Solve_IVP function.
    
    Uses a variety of input functions and parameters to perform dynamic
    field and relaxation calculations. Solves Bloch equations in the
    rotating reference frame by default, contains additional processing
    step to allow for solving in an arbitrary reference frame.

    Parameters
    ----------
    t : float
        Simulation time. Produced by solver and should not be defined by
        end user.
    mag_solve : array_like
        Magnetization vector to be evolved, defined as a 3 element
        array.
    index : array_like
        List indicating the index in Experiment.sample_mag corresponding
        to the vector being simulated
    GAMMA : float
        Gyromagnetic ratio of nucleus under study.
    relax_func : object
        Function for calculating relaxation times. See relaxfunc.py for
        more details.
    b0_func, brf_func, bgrad_func : object
        Function for calculating applied fields. See fieldfunc.py for
        more details.
    frame_rot : object
        Function for calculating frame rotation. See rotfunc.py for
        more details.
    param_t1, param_t2 : tuple
        Tuples containing parameters used by relax_func to calculate
        relaxation times. See relaxfunc.py for more details.
    param_b0, param_rf, param_grad : tuple
        Tuples containing parameters used by b0_func, brf_func, and
        bgrad_func to calculate applied fields. See fieldfunc.py for
        more details.
    param_rot, param_rot_plus : tuple
        Tuples containing parameters used by rotating_reference_frame ()
        and frame_rot to calculate frame rotations. See fieldfunc.py
        for more details.
    equi_m : array_like
        4D array containing equilibrium magnetization for the sample.

    Returns
    -------
    block : array_like
        Magnetization vector after evolution by the Bloch solver.
        Defined as a 3 element array.
    """

    # Sum input magnetic fields in Lab frame
    b_lab_init = (b0_func(t, index, param_b0)
                      + brf_func(t, index, param_rf)
                      + bgrad_func(t, index, param_grad))
    tot_b_lab = (b_lab_init+ (b_lab_init*param_b0[2][index]*1e-6)
                + param_b0[1][index])
    # Transform input field to Larmor Rotating Reference Frame
    tot_b_larmor = rot.rotating_reference_frame(tot_b_lab, t, param_rot)
    # Transform input field and mangetization into the Solving Reference
    # Frame
    tot_b_solve = frame_rot(tot_b_larmor, t, param_rot_plus,
                        tot_b_larmor = tot_b_larmor, dir_fore = True,
                        if_b = True)
    # Calculate T1 and T2 values based on applied mangetic field
    t1, t2 = relax_func(tot_b_lab, tot_b_larmor, tot_b_solve,
                        param_t1, param_t2)
    # Implementation of the Bloch Equations
    blockx = (GAMMA*(tot_b_solve[2]*mag_solve[1]- tot_b_solve[1]*mag_solve[2])
              - mag_solve[0]/t2)
    blocky = (GAMMA*(tot_b_solve[0]*mag_solve[2]- tot_b_solve[2]*mag_solve[0])
              - mag_solve[1]/t2)
    blockz = (GAMMA*(tot_b_solve[1]*mag_solve[0]- tot_b_solve[0]*mag_solve[1])
              - ((mag_solve[2]-equi_m[2])/t1))
    block = np.array([blockx, blocky, blockz])

    return block


def solver_func(index, GAMMA, int_ends, time_series, sample_mag, relax_func,
                b0_func, brf_func, bgrad_func, rx_func, frame_rot, param_t1,
                param_t2, param_b0, param_rf, param_grad, param_rx, param_rot,
                param_rot_plus, equi_m):
    """Function wrapped around SciPy's solve_ivp function to manage 
    solve_ivp's inputs and outputs. Uses solve_ivp to solve bloch() for 
    the behaviour of a single magnetization vector.

    Parameters
    ----------
    index : array_like
        List indicating the index in Experiment.sample_mag corresponding
        to the vector being simulated
    GAMMA : float
        Gyromagnetic ratio of nucleus under study.
    int_ends : tuple
        Time endpoints of simulation, expressed as a tuple of (0,
        Experiment.duration)
    time_series : array_like
        List of time points at which solve_ivp() should return
        magnetization state
    sample_mag : array_like
        Magnetization vector to be evolved, defined as a 3 element
        array.
    relax_func : object
        Function for calculating relaxation times. See relaxfunc.py for
        more details.
    b0_func, brf_func, bgrad_func : object
        Function for calculating applied fields. See fieldfunc.py for
        more details.
    rx_func : object
        Function for calculating the the spatial sensitivity of the 
        receive coil. See fieldfunc.py for more details.
    frame_rot : object
        Function for calculating frame rotation. See rotfunc.py for
        more details.
    param_t1, param_t2 : tuple
        Tuples containing parameters used by relax_func to calculate
        relaxation times. See relaxfunc.py for more details.
    param_b0, param_rf, param_grad, param_rx : tuple
        Tuples containing parameters used by b0_func, brf_func,
        bgrad_func, and rx_func to calculate applied fields. See
        fieldfunc.py for more details.
    param_rot, param_rot_plus : tuple
        Tuples containing parameters used by rotating_reference_frame()
        and frame_rot to calculate frame rotations. See fieldfunc.py
        for more details.
    equi_m : array_like
        4D array containing equilibrium magnetization for the sample.

    Returns
    -------
    output : array_like
        3-element list. Each element is a numpy array containing: 
            [0] magnetization vector index;
            [1] the final state of the magnetization vector;
            [2] the state of the vector at each timepoint in 
                time_series, arranged such that each column contains the
                results at a single timepoint, and each row contains
                either the x-, y-, or z-component of the vector.
    """

    # Rotate magnetization vector into the reference frame used for
    # solving.
    # Calculate total applied field in the Lab frame
    b_lab_init = (b0_func(int_ends[0], index, param_b0)
                      + brf_func(int_ends[0], index, param_rf)
                      + bgrad_func(int_ends[0], index, param_grad))
    # Add T2star and chemical shift field offsets
    tot_b_lab = (b_lab_init+ (b_lab_init*param_b0[2][index]*1e-6)
               + param_b0[1][index])
    # Rotate to the rotating reference frame
    tot_b_larmor = rot.rotating_reference_frame(tot_b_lab, int_ends[0],
                                                param_rot)
    # Rotate into Solving reference frame
    mag_solve = frame_rot(sample_mag, int_ends[0], param_rot_plus,
                          tot_b_larmor, dir_fore = True, if_b = False)
    # Solve for time evolution using scipy.integrate.solve_ivp
    sol = solve_ivp(block,
                    t_span = int_ends,
                    t_eval = time_series,
                    y0 = mag_solve,
                    args = (index, GAMMA, relax_func, b0_func, brf_func,
                            bgrad_func, frame_rot, param_t1, param_t2,
                            param_b0, param_rf, param_grad, param_rot,
                            param_rot_plus, equi_m),
                    method = 'DOP853',
                    atol = 1e-9,
                    rtol = 1e-9)
    # Return final magnetization state to Larmor frame
    sol_solve = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]])
    sol_larmor_end = frame_rot(sol_solve, int_ends[1], param_rot_plus,
                               tot_b_larmor, dir_fore = False, if_b = False)
    # Return intermediate net magnetization data into Larmor frame
    nrows, ncols = sol.y.shape
    sol_larmor = np.zeros([nrows, ncols])
    for i in range(ncols):
        vec = sol.y[:,i]
        sol_larmor[:, i] = (frame_rot(vec, time_series[i], param_rot_plus,
                                tot_b_larmor, dir_fore = False, if_b = False)
                                * rx_func(time_series[i], index, param_rx))
    output = [index, sol_larmor_end, sol_larmor]

    return output


#~~~~~~~~~~ Multiprocessing Code ~~~~~~~~~~#


def solve(GAMMA, int_ends, time_series, sample_mag, relax_func, b0_func,
          brf_func, bgrad_func, rx_func, frame_rot, param_t1, param_t2,
          param_b0, param_rf, param_grad, param_rx, param_rot, param_rot_plus,
          equi_m, chunksize, max_workers):

    """User callable function which uses the concurrent.futures module
    to parallelize computation of solver_func() to evolve magnetization
    of the entire sample.

    Parameters
    ----------
    GAMMA : float
        Gyromagnetic ratio of nucleus under study.
    int_ends : tuple
        Time endpoints of simulation, expressed as a tuple of (0,
        Experiment.duration)
    time_series : array_like
        List of time points at which solve_ivp() should return
        magnetization state
    sample_mag : array_like
        Magnetization vector to be evolved, defined as a 3 element
        array.
    relax_func : object
        Function for calculating relaxation times. See relaxfunc.py for
        more details.
    b0_func, brf_func, bgrad_func : object
        Function for calculating applied fields. See fieldfunc.py for
        more details.
    rx_func : object
        Function for calculating the the spatial sensitivity of the 
        receive coil. See fieldfunc.py for more details.
    frame_rot : object
        Function for calculating frame rotation. See rotfunc.py for
        more details.
    param_t1, param_t2 : tuple
        Tuples containing parameters used by relax_func to calculate
        relaxation times. See relaxfunc.py for more details.
    param_b0, param_rf, param_grad, param_rx : tuple
        Tuples containing parameters used by b0_func, brf_func,
        bgrad_func, and rx_func to calculate applied fields. See
        fieldfunc.py for more details.
    param_rot, param_rot_plus : tuple
        Tuples containing parameters used by rotating_reference_frame()
        and frame_rot to calculate frame rotations. See fieldfunc.py
        for more details.
    equi_m : array_like
        4D array containing equilibrium magnetization for the sample.
    chunksize : int
        Number of magnetization vectors to be included in each batch of
        data to be distributed to each core.
    max_workers:
        Maximum number of logical cores to be used in parallel
        computing.

    Returns
    -------
    samp_mag_final : array_like
        4D array containing evolved sample magnetization at the end of
        the simulated sequence element.
    vector_sum : array_like
        Array containing net magnetization of the sample at each
        timepoint specified in time_series.
    """

    # Format variables into lists to be parsed by parallelization code
    ind_list = []
    list1, list2, list3, list4, list5, list6 = [], [], [], [], [], []
    list7, list8, list9, list10, list11, list12 = [], [], [], [], [], []
    list13, list14, list15, list16, list17 = [], [], [], [], []
    list18, list19 = [], []
    samp_mag_final = copy.deepcopy(sample_mag)
    vector_sum = np.zeros([3, time_series.size])

    for index in np.ndindex(sample_mag.shape):
        if isinstance(sample_mag[index], np.ndarray) is False:
            continue
        ind_list.append(index)
        list1.append(GAMMA)
        list2.append(int_ends)
        list3.append(time_series)
        list4.append(sample_mag[index])
        list5.append(relax_func)
        list6.append(b0_func)
        list7.append(brf_func)
        list8.append(bgrad_func)
        list9.append(rx_func)
        list10.append(frame_rot)
        list11.append(param_t1[index])
        list12.append(param_t2[index])
        list13.append(param_b0)
        list14.append(param_rf)
        list15.append(param_grad)
        list16.append(param_rx)
        list17.append(param_rot)
        list18.append(param_rot_plus)
        list19.append(equi_m[index])

    # Run solver on each magnetization vector in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers
                                                = max_workers) as executor:
        # Distribute chunksize vectors across assigned CPU cores. As
        # core finish, distribute further chunks until all vectors have
        # been evolved
        for result in executor.map(solver_func, ind_list, list1, list2, list3,
                                   list4, list5, list6, list7, list8, list9,
                                   list10, list11, list12, list13, list14,
                                   list15, list16, list17, list18, list19,
                                   chunksize = chunksize):
            # Save final result for each vector to output M matrix
            samp_mag_final[result[0]] = result[1]
            # Add intermediate data points for each vector together to
            # calculate new magnetization of full sample at intermediate
            # points.
            vector_sum += result[2]

    return samp_mag_final, vector_sum
