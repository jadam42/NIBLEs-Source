"""
Simulation code for the Numeric Integrator for the Bloch Equations
(NIBLEs)

This file defines the Experiment class, which is used for loading in 
sample properties, defining a pulse sequence, and simulating that 
sequence to produce a simulated signal. Methods within Experiment are 
split into five types: 
Methods for setting and manipulating system variables; 
Utility methods used to standardaze common functionality of pulse 
sequence elements; 
Methods which define and simulate the effects of various pulse 
sequence elements; 
Methods for adding noise to images and k-space data; 
Methods for converitng k-space data to images.
The sequence element methods take as input all the nessecary parametres 
for defining that sequence element, then call the solving code to evolve
the magnetization of the sample, then save that evolved magnetization in
the Experiment.sample_mag variable for the next pulse sequence element
to be called.
"""

import os
import copy
import warnings
import datetime as time

import math as mat
import numpy as np
import matplotlib.pyplot as plt

from NIBLEs import solver
import NIBLEs.relaxfunc as relax
import NIBLEs.fieldfunc as field
import NIBLEs.rotfunc as rot


class Experiment():
    """The Experiment class is designed to manage the data inputs and 
    outputs of the NIBLEs solver throughout a simulation. As part of 
    this, the functions an end user can use to simulate common pulse 
    sequence elements are coded as methods within the Experiment class.

    After creating an instance of this class, sample properties are 
    loaded as class variables by the Experiment.sequence_start 
    method. As sequence element methods are called, they simulate the 
    effects of that sequence element on the magnetization of the sample 
    and store the updated magnetization in the Experiment.sample_mag 
    variable.

    All variables should be specified in SI units:
    Length = Metres
    Time = Seconds
    Frequency = Hertz
    Magnetic Field = Tesla
    Angles = Degrees

    Angles are converted to radians within the code.

    Parameters
    ----------
    b0 : array_like
        3-element list defining the static magnetic field as a Cartesian
        vector in the laboratory reference frame.
    GAMMA : float, optional
        Gyromagnetic ratio of nucleus under study. Defaults to hydrogen
        (42.58e6 Hz/T).
 
    Attributes
    ----------
    sample_mag : array_like
        4D array containing the current state or each mangetization
        composing the sample.
    t2_star_bounds : array_like
        2 element list defining the minimum and maximum chi values used
        in t2_star simulation. Must take values between (0, 1).
    t2_star_offsets : array_like
        4D array containing the field offsets for t2_star simulation.
    b90 : float
        Field amplitude that rotates sample magmetization by 90 degrees.
        If set to zero, the rf_pulse method will automatically calculate
        this field mangitude.
    readout : array_like
        1D array containing complex signal calculated by acquisition
        methods.
    signal : array_like
        Array with user defined dimensions, used to store signal across
        multiple acquisitions.
    print_list : array_like
        List of strings, used to record the sequence elements applied to
        the sample and the parameters defining those sequence elements.
    net_mag : array_like
        3 element vector describing net magnetization of the sample in
        the Larmor reference frame.
    """

    def __init__(self, b0, GAMMA=42.57e6):
        self.start_time = time.datetime.now()
        # Chunk Size for parellelized processing
        self.chunksize = 64
        self.max_workers = None
        # Simulation Constants
        self.RNG = np.random.default_rng()
        self.GAMMA = 2*np.pi*GAMMA
        # Main field definition
        self.b0 = np.array(b0)
        self.b0_func = rot.zero
        self.param_b0 = 0
        self.larmor = self.GAMMA* self.b0[2]
        self.relax_func = rot.zero
        self.frame_rot = rot.zero_bz_frame
        self.param_rot = (self.b0, self.larmor)
        self.param_rot_plus = (self.b0, self.GAMMA)
        # Variables for sample properties
        self.sample_mag = [0,0,1]
        self.equi_m = [0,0,1]
        self.m_position = [0,0,0]
        self.size_pixels = 0
        self.size_metric = 0
        self.param_t1 = 0
        self.param_t2 = 0
        self.param_t1_alt = 0
        self.param_t2_alt = 0
        self.chem_shift = 0
        self.size_4d = 0
        self.t2_star_bounds = [0.075, 0.925]
        self.t2_star_offset = 0
        # Variables for defining sequence elements
        self.duration = 0
        self.int_ends = 0
        self.time_series = 0
        self.n_points = 0
        self.brf_func = rot.zero
        self.bgrad_func = rot.zero
        self.rx_func = rot.zero
        self.param_rf = 0
        self.param_grad = 0
        self.param_rx = 0
        self.addtl_params = {}
        # RF field variables
        self.b1 = 0
        self.b90 = 0
        self.phi = 0
        self.flip_angle = 0
        self.flip_angle_array = 0
        self.del_omega = 0
        self.carrier_freq = 0
        self.n_lobes = 0
        # Gradient field variables
        self.grad_amp = 0
        self.rise_time = 0
        # Acquisition Variables
        self.readout = 0
        self.signal = 0
        # Misc. Calculations
        self.count = 0
        self.net_mag = 0
        self.image = 0
        self.noisy_signal = 0
        self.noisy_image = 0
        # Parameter tracking
        self.print_list = []
        self.print_list.extend([f"b0 = {self.b0}", f"gamma = {GAMMA}",
                               "\n"])

    #~~~~~ Utility Methods for data handling and manipulation ~~~~~#-\

    def save_acquisition(self, variable, seq_name, sample_name):
        """Saves print_list and given NumPy array to file, using a file
        name derived from the sequence used and the name of the sample 
        used.

        Parameters
        ----------
        variable : array_like
            NumPy array to be saved.
        seq_name : str
            Name of sequence being simulated.
        sample_name : str
            Name of sammple file being used.
        """
        # Format start_time into Hours/Minutes
        time_str = self.start_time.strftime("%Y-%m-%d_%Hh%Mm")
        # Construct output file name from inputs
        file_name = f"{seq_name}_{sample_name}_{time_str}"
        # Calculate total computation time and save that time to
        # print_list
        compute_time = time.datetime.now() - self.start_time
        self.print_list.extend([f"Total Compute Time = {compute_time}"])
        # Save data and print_list to file
        np.savetxt(f"{file_name}_Output.csv", variable, delimiter=",")
        with open(f"{file_name}_Params.txt", 'w') as f:
            f.write('\n'.join(self.print_list))

    def reset_magnetization(self, mag=None):
        """Resets magnetization to a prior state. 

        If mag is a Numpy array, function will set Experiment.sample_mag
        to mag. For any other input, function will set
        Experiment.sample_mag to Experiment.equi_m (equilibrium
        magnetization).

        Parameters
        ----------
        mag : array_like or None, optional
            Prior state of sample magnetization or dummy variable

        """
        # If mag is not an array, set Experiment.sample_mag
        # == Experiment.equi_m
        if not isinstance(mag, np.ndarray):
            self.sample_mag = copy.deepcopy(self.equi_m)
            self.print_list.extend(["reset_magnetization() used - Equilibrium",
                                   "\n"])
        # If mag is an array, set Experiment.sample_mag == mag
        else:
            self.sample_mag = copy.deepcopy(mag)
            self.print_list.extend(["reset_magnetization() used - Given state",
                                   "\n"])

    def null_transverse_mag(self):
        """Manipulates Experiment.sample_mag, setting sample
        magnetization in the xy-plane to 0.
        """
        # Iterate through all vectors in sample. Ignore empty array
        # elements, set the x- and y-components of extant vectors to
        # zero
        for index in np.ndindex(self.sample_mag.shape):
            if isinstance(self.sample_mag[index], np.ndarray) is False:
                continue
            self.sample_mag[index] = np.array(
                        [0, 0, self.sample_mag[index][2]], dtype = float)

    def rx_set(self, rx_func, **kwargs):
        """Sets parameters defining Rx sensitivity (Experiment.rx_func
        and Experiment.param_rx).

        Parameters
        ----------
        rx_func : object
            Function for defining RX sensitivity over sample volume.
        **kwargs
            Extra arguments that may be required by rx_func. These
            variables are saved as a dictionary in
            Experiment.param_rx[3]
        """
        self.rx_func = rx_func
        self.param_rx = (self.m_position, self.size_metric, self.size_pixels,
                        kwargs)
        self.print_list.extend([f"Rx Function changed to {rx_func}",
                        f"addtl_params = {kwargs}", "\n"])


    def b0_set(self, b0_func, **kwargs):
        """Sets parameters defining b0 field over the sample volume
        (Experiment.b0_func and Experiment.param_b0).

        Parameters
        ----------
        b0_func : object
            Function for defining b0 over sample volume.
        **kwargs
            Extra arguments that may be required by b0_func. These
            variables are saved as a dictionary in
            Experiment.param_b0[3]
        """
        self.b0_func = b0_func
        self.param_b0 = (self.b0, self.t2_star_offset, self.chem_shift,
                        kwargs)
        self.print_list.extend([f"b0 Function changed to {b0_func}",
                        f"addtl_params = {kwargs}", "\n"])

    def sequence_start(self, sample_name, sample_path,
                       relax_func=relax.constant, model_type="Constant"):
        """Imports sample properties from file for use in a 
        simulation, calculates parallel processing, and calculates
        t2_star simulation variables.

        Parameters
        ----------
        sample_name : str
            Name of sample folder.
        sample_path : str
            Path to sample folder.
        relax_func : object, optional
            Function for calculating relaxation for experiment. Defaults
            to constant relaxation.
        model_type : str, optional
            String identifying which relaxation model within each
            material to use. Options are:
                If model_type = “Dynamic”:
                param_t1/param_t2 == Material.DynamicT1/DynamicT2 and
                param_t1_alt/param_t2_alt ==
                Material.DynamicT1alt/DynamicT2alt

                If model_type = “Dynamic_Alt”:
                param_t1/param_t2 == Material.DynamicT1alt/DynamicT2alt
                and param_t1_alt/param_t2_alt ==
                Material.DynamicT1/DynamicT2

                If model_type = “Constant_Alt”:
                param_t1/param_t2 == Material.T1alt/T2alt and
                param_t1_alt/param_t2_alt == Material.T1/T2

                For all other inputs for model_type:
                param_t1/param_t2 == Material.T1/T2 and
                param_t1_alt/param_t2_alt == Material.T1alt/T2alt
        """
        print('Loading Sample...')
        # Load in functions for calculating relaxation times
        self.relax_func = relax_func
        # Set default parameters for Rx field
        self.rx_func = field.rx_uniform
        self.param_rx = ()
        # Load in sample size parameters and equilibrium magnetization
        self.sample_mag = np.load(os.path.join(sample_path, sample_name,
                                      sample_name+ '_M.npy'),
                                      allow_pickle = True)
        self.m_position = np.load(os.path.join(sample_path, sample_name,
                                              sample_name+ '_MPosition.npy'),
                                              allow_pickle = True)
        self.size_4d = np.load(os.path.join(sample_path, sample_name,
                                           sample_name+ '_size4D.npy'),
                                           allow_pickle = True)
        self.size_pixels = self.size_4d[0:3]
        self.size_metric = np.load(os.path.join(sample_path, sample_name,
                                               sample_name+ '_sizeMetric.npy'),
                                               allow_pickle = True)
        # Load in desired relaxation model parameters. Defaults to
        # constant relaxation times.
        if model_type == "Dynamic":
            self.param_t1 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name+ '_dynamicT1.npy'),
                                                allow_pickle = True)
            self.param_t2 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name+ '_dynamicT2.npy'),
                                                allow_pickle = True)
            self.param_t1_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name
                                                   + '_dynamicT1alt.npy'),
                                                   allow_pickle = True)
            self.param_t2_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name
                                                   + '_dynamicT2alt.npy'),
                                                   allow_pickle = True)
        elif model_type == "Dynamic_Alt":
            self.param_t1 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name
                                                + '_dynamicT1alt.npy'),
                                                allow_pickle = True)
            self.param_t2 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name
                                                + '_dynamicT2alt.npy'),
                                                allow_pickle = True)
            self.param_t1_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name
                                                   + '_dynamicT1.npy'),
                                                   allow_pickle = True)
            self.param_t2_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name
                                                   + '_dynamicT2.npy'),
                                                   allow_pickle = True)
        elif model_type == "Constant_Alt":
            self.param_t1 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name+ '_T1alt.npy'),
                                                allow_pickle = True)
            self.param_t2 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name+ '_T2alt.npy'),
                                                allow_pickle = True)
            self.param_t1_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name+ '_T1.npy'),
                                                   allow_pickle = True)
            self.param_t2_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name+ '_T2.npy'),
                                                   allow_pickle = True)
        else:
            self.param_t1 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name+ '_T1.npy'),
                                                allow_pickle = True)
            self.param_t2 = np.load(os.path.join(sample_path, sample_name,
                                                sample_name+ '_T2.npy'),
                                                allow_pickle = True)
            self.param_t1_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name+ '_T1alt.npy'),
                                                   allow_pickle = True)
            self.param_t2_alt = np.load(os.path.join(sample_path, sample_name,
                                                   sample_name+ '_T2alt.npy'),
                                                   allow_pickle = True)
        # Load in remaining parameters from file
        self.chem_shift = np.load(os.path.join(sample_path, sample_name,
                                              sample_name+ '_chemShift.npy'),
                                              allow_pickle = True)
        t2_prime = np.load(os.path.join(sample_path, sample_name,
                                       sample_name+ '_T2prime.npy'),
                                       allow_pickle = True)
        # Pre-define matrix variables
        self.t2_star_offset = np.zeros(self.size_4d)
        self.t2_star_offset = self.t2_star_offset.astype(object)
        # Calculate numer of vectors per pixel, and calculate the Chi
        # values for each vector in T2star simulations
        n = self.size_4d[3]
        pts = np.linspace(self.t2_star_bounds[0],self.t2_star_bounds[1], n)
        # Iterate through sample magnetization, defining the static
        # field in the Lab frame; defining T2star field offsets for each
        # vector; and counting the number of vectors in the sample
        for index in np.ndindex(self.sample_mag.shape):
            if isinstance(self.sample_mag[index], np.ndarray) is False:
                continue
            v_num = index[3]
            do_list = (1/t2_prime[index])*np.tan(np.pi*(pts-0.5))
            for i in range(int(n/2)):
                do_list[n-i-1] = -do_list[i]
            do = do_list[v_num]
            self.t2_star_offset[index] = np.array([0, 0, do/(self.GAMMA)])
            self.count += 1
        # Save Initial sample magnetization
        self.equi_m = copy.deepcopy(self.sample_mag)
        # Automatically set param_b0 to BLab for ideal b0
        self.b0_func = field.ideal_b0
        self.param_b0 = (self.b0, self.t2_star_offset, self.chem_shift)
        # Calculate Chunk Size based on number of vectors in sample
        n_cores = os.cpu_count()
        self.chunksize = mat.ceil(self.count/n_cores)
        # Add Sample information to print_list
        self.print_list.extend([f"Sample Name = {sample_name}",
                               f"Model_Type = {model_type}", "\n"])


    #~~~~~ Internal Methods Used by User Facing Methods ~~~~~#

    def reset_variables(self):
        """Resets shared variables that define sequence elements to
        zero.
        """
        self.duration = 0
        self.b1 = 0
        self.addtl_params = {}
        self.brf_func = rot.zero
        self.bgrad_func = rot.zero
        self.param_rf = 0
        self.param_grad = 0
        self.int_ends = np.array([0])
        self.time_series = np.array([0])
        self.frame_rot = rot.zero_bz_frame
        self.param_rot_plus = (self.b0, self.larmor)

    def net_mag_calc(self, mag):
        """Calculates average magnetization across the sample.

        Stores result in self.net_mag.

        Parameters
        ----------
        mag : array_like
            NumPy array containing magnetization state to be summed.
        """
        # Pre-define variables
        count = 0
        sum_mag = np.array([0,0,0], dtype = float)
        # Iterate through all elements Mag. Ignore empty pixels, sum
        # vectors and count number of vectors
        for index in np.ndindex(mag.shape):
            if isinstance(mag[index], np.ndarray) is False:
                continue
            count += 1
            sum_mag += mag[index]
        # Calculate average magnetization
        self.net_mag = sum_mag/count

    def net_mag_norm(self, vector_sum):
        """Calculates average magnetization across each time point
        output by solve().

        Stores result in self.net_mag.

        Parameters
        ----------
        vector_sum : array_like
            Output of solve() containing array of magnetization at each
            solving time point.
        """
        self.net_mag = vector_sum/self.count

    def variable_flip_calculator(self, n_flip, tr, t1):
        """Calculates a series of flip angles that will provide
        constant transverse magnetization in short tr, spoiled gradient
        echo imaging.

        Maximizes available signal by constraining the final pulse in
        the sequence to flip_angle = 90 degrees. As this function
        depends on the state of magnetization immediately prior to the
        first excitation pulse, it must be called immediately prior to
        that initial pulse. Saves both the list of flip angles in
        self.flip_angle_array.

        Parameters
        ----------
        n_flip : int
            Number of rf pulses in train.
        tr : float
            Repetiton time between excitation pulses in seconds.
        t1 : float
            Sample T1 in seconds.
        """
        # Define output array
        self.flip_angle_array = np.zeros([n_flip])
        mag_xy_arr = np.zeros([n_flip])
        # Calculate relaxation coefficient
        e1 = np.exp(-tr/t1)
        # Calculate current and equilibrium net_mag
        self.net_mag_calc(self.sample_mag)
        m_z_init = self.net_mag[2]
        self.net_mag_calc(self.equi_m)
        m0 = self.net_mag[2]
        # Set up variables for incrementing inital angle
        increment = 1           # 1 degree
        init_angle = 0
        # Iteratively calculate signal excitation from flip angle train
        # until final flip angle needed to maintian constnat transverse
        # magnetization is ~= 90 degrees
        while self.flip_angle_array[-1] < 89.9:
            init_angle += increment
            # Convert guess angle to radians and calculate inital points
            self.flip_angle_array[0] = np.radians(init_angle)
            mag_xy_arr[0] = m_z_init*np.sin(self.flip_angle_array[0])
            m_z = m_z_init*e1*np.cos(self.flip_angle_array[0])+ m0*(1 - e1)
            # Iteratively calulate further flips
            for i in range(1, n_flip):
                # Ignore NaN warning from arcsin when trying to find
                # angles
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                message="invalid value encountered in arcsin")
                    denom = (1 - e1) + m_z*e1*np.cos(self.flip_angle_array[i-1])
                    func = m_z*np.sin(self.flip_angle_array[i-1])/denom
                    self.flip_angle_array[i] = np.arcsin(func)
                    mag_xy_arr[i] = m_z*np.sin(self.flip_angle_array[i])
                    m_z = m_z*e1*np.cos(self.flip_angle_array[i]) + 1 - e1
            self.flip_angle_array = np.degrees(self.flip_angle_array)
            # If final element is nan (due to needing a flip angle >90),
            # decrease angle by increment, decrease increment step size,
            # and set element checked by while loop to zero
            if np.isnan(self.flip_angle_array[-1]) is True:
                init_angle = init_angle - increment
                increment = increment/10
                self.flip_angle_array[-1] = 0
        # Set final pulse to full 90 degrees
        self.flip_angle_array[-1] = 90
        # Save element parameters to text output
        self.print_list.extend(['Variable Flip Angles Calculated:',
                               f"flip_angle_array = {self.flip_angle_array}",
                               f"mag_xy_arr = {mag_xy_arr}", "\n"])
        print("Flip Angles Calculated")

    def rf_calibrate(self):
        """Calculates the RF pulse amplitude that produce a 90 degree
        flip and configures rf pulse parameters based on given
        variables.

        If B1 is specified when rf_pulse() is called, RF amplitude is
        set to the given B1 value.

        If B1 has not been specified and Experiment.b90 has been
        calculated by an earlier instance of rf_calibrate(), RF
        amplitude calculated based on the value stored in Experiment.b90
        and the desited flip angle.

        If B1 has not been specified and Experiment.b90 has not yet been
        calculated, runs a simple search algorithm to determine field
        strength needed to achieve a 90 degree flip angle by maximizing
        transverse magnetization. This value is then saved in
        Experiment.b90, and RF amplitude is calculated based on
        Experiment.b90.
        """

        # Identify if B1 was set by the user when rf_pulse was called
        if self.b1 != 0:
            pass
        # Otherwise, identify if b90 has been calculated by a proeious
        # call to rf_pulse()
        elif self.b90 != 0:
            self.b1 = self.b90*(self.flip_angle/90)
        # Otherwise, run search algorithm to determine b90
        else:
            # Predefine variables
            # Copy of M for calibration simulations
            mag_cal = 0
            # Flag for identifying when maximum Bxy is reached
            max_flag = False
            # RF amplitude
            b90 = 0
            # Result from previous iteration of solver
            prev = 0
            # Result from two iterations prior of solver
            prev2 = 0
            # Initial (maximum) field difference between iterations
            step = 10e-6
            print('Calculating b90 Amplitude...')
            # Iterating through simulations
            while max_flag is False:
                # Add step to b90 from previous simulation
                b90 += step
                # Define RF parameters with new b90
                self.param_rf = (self.duration, b90, self.carrier_freq,
                                 self.phi, self.n_lobes, self.addtl_params)
                # Set inital mangetization equal to equilibrium
                mag_cal = copy.deepcopy(self.equi_m)
                # Solves for state of equi_m after RF Pulse and
                # calculates total magnetization along Bxy
                mag_cal, vector_sum = solver.solve(self.GAMMA,
                            self.int_ends, self.time_series, self.sample_mag,
                            self.relax_func, self.b0_func, self.brf_func,
                            self.bgrad_func, self.rx_func, self.frame_rot,
                            self.param_t1, self.param_t2, self.param_b0,
                            self.param_rf, self.param_grad, self.param_rx,
                            self.param_rot, self.param_rot_plus, mag_cal,
                            self.chunksize, self.max_workers)
                # Calculate net magnetization in the transvese plane
                self.net_mag_norm(vector_sum)
                current = abs(np.linalg.norm(self.net_mag[0:2]))
                # If Bxy from most recent iteraton is larger than
                # previous iteration, increase B1 strength and continue
                # search
                if current > prev:
                    print(f"Transverse Mag = {current} @ b90 = {b90} T")
                    prev2 = prev
                    prev = current
                # If Bxy from most recent iteraton is smaller than
                # previous iteration, and step size is  1e-7 T, stop
                # search
                elif current <= prev and step <= 1e-6:
                    b90 += -step
                    max_flag = True
                    print('Done!')
                # If Bxy from most recent iteraton is smaller than
                # previous iteration, reduce B1 strength to prior value,
                # decrease step size and continue search
                elif current <= prev and max_flag is False:
                    print(f"Transverse Mag = {current} @ b90 = {b90} T")
                    b90 += -2*step
                    step = step/10
                    current = prev2
                    prev = prev2
                    print('Reducing Step Size...')
            # Saving calculated 90-degree pulse amplitude to file
            self.b90 = b90
            # Determine B1 value for the flip angle of the current RF
            # pulse
            self.b1 = self.b90*(self.flip_angle/90)
            print(f'B1 = {self.b1} T')
            # Calculate and output time taken to perform calibrations
            compute_time = time.datetime.now() - self.start_time
            self.print_list.extend(
                [f"b90 Calibrated, Total Compute Time = {compute_time}"])
        # Set param_rf tuple for calculating RF waveform using final RF
        # pulse amplitude
        self.param_rf = (self.duration, self.b1, self.carrier_freq, self.phi,
                        self.n_lobes, self.addtl_params)

    def acquire(self):
        """Method used by acquisition methods that to evolve
        magnetization and to convert magnetization into complex signal.

        Stores resulting signal in the self.readout variable.
        """
        print(f'acquisition: n_points = {self.n_points}')
        # Evolve magnetization
        self.sample_mag, vector_sum = solver.solve(self.GAMMA, self.int_ends,
                            self.time_series, self.sample_mag, self.relax_func,
                            self.b0_func, self.brf_func, self.bgrad_func,
                            self.rx_func, self.frame_rot, self.param_t1,
                            self.param_t2, self.param_b0, self.param_rf,
                            self.param_grad, self.param_rx, self.param_rot,
                            self.param_rot_plus, self.equi_m, self.chunksize,
                            self.max_workers)
        # Calculate net magnetization of the sample at each output
        # timepoint
        self.net_mag_norm(vector_sum)
        # Create a complex time-domain signal where the real component
        # is the x-component of net magnetization, and the imaginary
        # component is the y-component of net magnetization
        readout = []
        for i in range(self.n_points):
            readout.append(complex(self.net_mag[0][i], self.net_mag[1][i]))
        self.readout = np.array(readout)


    #~~~ Methods for Defining and Executing Pulse Sequence Elements ~~~#


    def deadtime(self, duration):
        """Method for evolving sample magnetization under the sole 
        influence of b0.

        Parameters
        ----------
        duration : float
            Length of sequence element in seconds.
        """
        # Load inputs as class variables
        self.duration = duration
        self.int_ends = (0, self.duration)
        self.time_series = np.array([self.duration])
        print(f'Evolving over Deadtime: Dur = {self.duration} s')
        # Evolve magnetization
        self.sample_mag, vector_sum = solver.solve(self.GAMMA, self.int_ends,
                            self.time_series, self.sample_mag, self.relax_func,
                            self.b0_func, self.brf_func, self.bgrad_func,
                            self.rx_func, self.frame_rot, self.param_t1,
                            self.param_t2, self.param_b0, self.param_rf,
                            self.param_grad, self.param_rx, self.param_rot,
                            self.param_rot_plus, self.equi_m, self.chunksize,
                            self.max_workers)
        # Save element parameters to text output
        self.print_list.extend(['Deadtime', f"Duration = {self.duration}",
                               "\n"])
        # Reset variables prior to next sequence element
        self.reset_variables()

    def rf_pulse(self, duration, brf_func, flip_angle, phi=0, del_omeg=0,
                 n_lobes=5, b1=0, **kwargs):
        """Method for evolving sample magnetization under the influence
        of b0 and an RF field.

        Parameters
        ----------
        duration : float
            Length of sequence element in seconds.
        brf_func : object
            Function for calculating the RF field. See fieldfunc.py for
            more details.
        flip_angle : float
            Flip angle of pulse in degrees.
        phi : float, optional
            Phase of RF pulse, determines axis along which pulse is 
            applied. +x-axis is at 0 degrees, +y-axis at 90 degrees. 
            Default value 0.
        del_omega : float, optional
            Frequency offset of RF Pulse. Input in units of PPM
            of Larmor Frequency, stored in Hz. Default value 0.
        n_lobes : int, optional
            Number of lobes in RF waveform. Default value 5.
        b1 : float, optional
            Amplitude of RF waveform in Tesla. Default value 0.
        **kwargs
            Extra arguments that may be required by brf_func. These
            variables are saved as a dictionary in
            Experiment.param_rf[5]
        """

        # Load inputs as class variables
        self.duration = duration
        self.int_ends = (0, self.duration)
        self.time_series = np.array([self.duration])
        self.flip_angle = flip_angle
        self.phi = -np.deg2rad(phi)
        self.n_lobes = n_lobes
        self.del_omega= del_omeg
        self.carrier_freq = self.larmor+ self.del_omega
        self.b1 = b1
        self.brf_func = brf_func
        self.addtl_params = kwargs
        # Set frame_rot to solvein the rotating reference frame
        self.frame_rot = rot.identity
        # Determine RF field amplitude
        self.rf_calibrate()
        print(f'RF Pulse: Angle = {self.flip_angle} deg; {self.duration} s')
        # Evolve magnetization
        self.sample_mag, vector_sum = solver.solve(self.GAMMA, self.int_ends,
                            self.time_series, self.sample_mag, self.relax_func,
                            self.b0_func, self.brf_func, self.bgrad_func,
                            self.rx_func, self.frame_rot, self.param_t1,
                            self.param_t2, self.param_b0, self.param_rf,
                            self.param_grad, self.param_rx, self.param_rot,
                            self.param_rot_plus, self.equi_m, self.chunksize,
                            self.max_workers)
        # Save element parameters to text output
        self.print_list.extend(['rf_pulse', f"Duration = {self.duration}",
                               f"flip_angle = {self.flip_angle}",
                               f"phi = {self.phi}", f"Shape = {self.brf_func}",
                               f"n_lobes = {self.n_lobes}", f"B1 = {self.b1}",
                               f"del_omega= {self.del_omega}",
                               f"addtl_params = {self.addtl_params}",
                               "\n"])
        # Reset variables prior to next sequence element
        self.reset_variables()

    def gradient(self, duration, bgrad_func, grad_amp, rise_time=0, **kwargs):
        """Method for evolving sample magnetization under the influence
        of b0 and a gradient field.

        Parameters
        ----------
        duration : float
            Length of sequence element in seconds.
        bgrad_func : object
            Function for calculating the gradient. See fieldfunc.py for
            more details.
        grad_amp : array_like
            Maximum amplitude of gradient waveform, presented as a.
            3-element list. Each element denotes the maximum amplitude
            of the x-, y-, and z-gradients.
        rise_time : float, optional
            Time over which the gradient ramps from zero to maximum
            value in seconds. Default value 0.
        **kwargs
            Extra arguments that may be required by bgrad_func. These
            variables are saved as a dictionary in
            Experiment.param_grad[5]
        """
        # Load inputs as class variables
        self.duration = duration
        self.int_ends = (0, self.duration)
        self.time_series = np.array([self.duration])
        self.bgrad_func = bgrad_func
        self.grad_amp = np.array(grad_amp, dtype = float)
        self.rise_time = rise_time
        self.addtl_params = kwargs
        self.param_grad = (self.m_position, self.grad_amp, self.size_metric,
                          self.duration, self.rise_time, self.addtl_params)
        print(f'gradient: Amp = {self.grad_amp} T, Dur = {self.duration} s')
        # Evolve magnetization
        self.sample_mag, vector_sum = solver.solve(self.GAMMA, self.int_ends,
                            self.time_series, self.sample_mag, self.relax_func,
                            self.b0_func, self.brf_func, self.bgrad_func,
                            self.rx_func, self.frame_rot, self.param_t1,
                            self.param_t2, self.param_b0, self.param_rf,
                            self.param_grad, self.param_rx, self.param_rot,
                            self.param_rot_plus, self.equi_m, self.chunksize,
                            self.max_workers)
        # Save element parameters to text output
        self.print_list.extend(['gradient', f"Duration = {self.duration}",
                               f"grad_amp = {self.grad_amp}",
                               f"rise_time = {self.rise_time}",
                               f"addtl_params = {self.addtl_params}",
                                "\n"])
        # Reset variables prior to next sequence element
        self.reset_variables()

    def acquisition(self, duration, n_points):
        """Method for evolving sample magnetization and acquiring MR signal
        under the sole influence of b0.

        Parameters
        ----------
        duration : float
            Length of sequence element in seconds.
        n_points : int
            Number of acquired data points.
        """
        # Load inputs as class variables
        self.duration = duration
        self.int_ends = (0, self.duration)
        self.n_points = n_points
        self.time_series = np.linspace(0, self.duration, self.n_points)
        # Evolve magnetization
        self.acquire()
        # Save element parameters to text output
        self.print_list.extend(['acquisition', f"Duration = {self.duration}",
                               f"n_points = {self.n_points}",
                               f"time_series = {self.time_series}", "\n"])
        # Reset variables prior to next sequence element
        self.reset_variables()

    def freq_encode_grad(self, duration, bgrad_func, n_points, grad_amp,
                         rise_time=0, **kwargs):
        """Method for evolving magnetization and acquiring MR signal
        under the influence of a magnetic gradient and b0.

        Parameters
        ----------
        duration : float
            Length of sequence element in seconds.
        bgrad_func : object
            Function for calculating the gradient. See fieldfunc.py for
            more details.
        n_points : int
            Number of acquired data points.
        grad_amp : array_like
            Maximum amplitude of gradient waveform, presented as a.
            3-element list. Each element denotes the maximum amplitude
            of the x-, y-, and z-gradients.
        rise_time : float, optional
            Time over which the gradient ramps from zero to maximum
            value in seconds. Default value 0.
        **kwargs
            Extra arguments that may be required by bgrad_func. These
            variables are saved as a dictionary in
            Experiment.param_grad[5]
        """
        # Load inputs as class variables
        self.duration = duration
        self.int_ends = (0, self.duration)
        self.n_points = n_points
        self.time_series = np.linspace(0, self.duration, self.n_points)
        self.grad_amp = np.array(grad_amp, dtype = float)
        self.bgrad_func = bgrad_func
        self.rise_time = rise_time
        self.addtl_params = kwargs
        self.param_grad = (self.m_position, self.grad_amp, self.size_metric,
                          self.duration, self.rise_time, self.addtl_params)
        print(f'Readout grad: Amp = {self.grad_amp} T, Dur= {self.duration} s')
        # Evolve magnetization
        self.acquire()
        # Save element parameters to text output
        self.print_list.extend(['freq_encode', f"Duration = {self.duration}",
                               f"n_points = {self.n_points}",
                               f"grad_amp = {self.grad_amp}",
                               f"time_series = {self.time_series}",
                               f"rise_time = {self.rise_time}",
                               f"addtl_params = {self.addtl_params}",
                               "\n"])
        # Reset variables prior to next sequence element
        self.reset_variables()

    def slice_select_rf(self, duration, brf_func, bgrad_func, flip_angle,
                      grad_amp, phi=0, del_omega=0, n_lobes=5, b1=0,
                      rise_time=0, **kwargs):
        """Method for evolving magnetization under the influence of
        simultaneous RF fields, magnetic gradients and b0.

        WORK IN PROGRESS NOT TESTED

        Parameters
        ----------
        duration : float
            Length of sequence element in seconds.
        bgrad_func : object
            Function for calculating the gradient. See fieldfunc.py for
            more details.
        brf_func : object
            Function for calculating the RF field. See fieldfunc.py for
            more details.
        flip_angle : float
            Flip angle of pulse in degrees.
        grad_amp : array_like
            Maximum amplitude of gradient waveform, presented as a.
            3-element list. Each element denotes the maximum amplitude
            of the x-, y-, and z-gradients.
        phi : float, optional
            Phase of RF pulse, determines axis along which pulse is 
            applied. +x-axis is at 0 degrees, +y-axis at 90 degrees. 
            Default value 0.
        del_omega : float, optional
            Frequency offset of RF Pulse. Input in units of PPM
            of Larmor Frequency, stored in Hz. Default value 0.
        n_lobes : int, optional
            Number of lobes in RF waveform. Default value 5.
        b1 : float, optional
            Amplitude of RF waveform in Tesla. Default value 0.
        rise_time : float, optional
            Time over which the gradient ramps from zero to maximum
            value in seconds. Default value 0.
        **kwargs
            Extra arguments that may be required. Not currently used.
        """
        # Load RF inputs as class variables
        self.duration = duration
        self.int_ends = (0, self.duration)
        self.time_series = np.array([self.duration])
        self.flip_angle = flip_angle
        self.phi = -np.deg2rad(phi)
        self.n_lobes = n_lobes
        self.del_omega= del_omega
        self.carrier_freq = self.larmor+ self.del_omega
        self.b1 = b1
        self.brf_func = brf_func
        # Set frame_rot to keep solving in the rotating reference frame,
        # increases speed of RF simulation
        self.frame_rot = rot.identity
        # Determine RF field amplitude
        self.rf_calibrate()
        # Load gradient inputs as class variables
        self.grad_amp = np.array(grad_amp, dtype = float)
        self.bgrad_func = bgrad_func
        self.rise_time = rise_time
        self.addtl_params = kwargs
        self.param_grad = (self.m_position, self.grad_amp, self.size_metric,
                          self.duration, self.rise_time,self.addtl_params)
        print(f'SS Pulse: Angle = {self.flip_angle} deg, {self.duration} s')
        # Evolve magnetization
        self.sample_mag, vector_sum = solver.solve(self.GAMMA, self.int_ends,
                            self.time_series, self.sample_mag, self.relax_func,
                            self.b0_func, self.brf_func, self.bgrad_func,
                            self.rx_func, self.frame_rot, self.param_t1,
                            self.param_t2, self.param_b0, self.param_rf,
                            self.param_grad, self.param_rx, self.param_rot,
                            self.param_rot_plus, self.equi_m, self.chunksize,
                            self.max_workers)
        # Save element parameters to text output
        self.print_list.extend(['SliceSelectRF', f"Duration = {self.duration}",
                               f"flip_angle = {self.flip_angle}",
                               f"phi = {self.phi}", f"Shape = {self.brf_func}",
                               f"n_lobes = {self.n_lobes}", f"B1 = {self.b1}",
                               f"del_omega= {self.del_omega}",
                               f"grad_amp = {self.grad_amp}",
                               f"time_series = {self.time_series}",
                               f"rise_time = {self.rise_time}",
                               f"addtl_params = {self.addtl_params}",
                               "\n"])
        # Reset variables prior to next sequence element
        self.reset_variables()

    def spinlock_pulse(self, duration, brf_func, relax_func, phi=0,
                       amp_hz=500, n_lobes=-1, del_omega=0, **kwargs):
        """Method for evolving sample magnetization under the influence
        of b0 and an RF field.

        Parameters
        ----------
        duration : float
            Length of sequence element in seconds.
        brf_func : object
            Function for calculating the RF field. See fieldfunc.py for
            more details.
        flip_angle : float
            Flip angle of pulse in degrees.
        phi : float, optional
            Phase of prior excitation RF pulse. Used to calculate
            orientation of excited magnetization, and determine
            orientation of splin locking field. Default value 0.
        amp_hz : int, optional
            Amplitude of spin locking field in Hz. Default value 500.
        n_lobes : int, optional
            Number of lobes in RF waveform. Default value -1 to produce
            a flat waveform with rf_sinc().
        del_omega : float, optional
            Frequency offset of RF Pulse. Input in units of PPM
            of Larmor Frequency, stored in Hz. Default value 0.
        **kwargs
            Extra arguments that may be required by brf_func. These
            variables are saved as a dictionary in
            Experiment.param_rf[5]
        """

        # Load inputs as class variables
        self.duration = duration
        self.int_ends = (0, self.duration)
        self.time_series = np.array([self.duration])
        self.relax_func = relax_func
        # Offsets given phi so B1 is applied along magnetization
        self.phi = -np.deg2rad(phi- 90)
        # Calulates B1 using amp_hz
        self.b1 = 2*np.pi*amp_hz/self.GAMMA
        self.n_lobes = n_lobes
        self.addtl_params = kwargs
        self.del_omega = del_omega
        self.param_rot_plus = self.phi
        self.brf_func = brf_func
        self.carrier_freq = self.larmor+ self.del_omega
        self.param_rf = (self.duration, self.b1, self.carrier_freq, self.phi,
                        self.n_lobes, self.addtl_params)
        # Tell solver to solve in the double-rotating reference frame
        self.frame_rot = rot.spinlock_rotator
        print(f'SpinLock Pulse: Amp = {amp_hz} Hz, Dur = {self.duration} s')
        # Set new equilibrium magnetization based on B1
        equi_t1rho = copy.deepcopy(self.equi_m)*(self.b1/self.b0[2])
        # Evolve magnetization
        self.sample_mag, vector_sum = solver.solve(self.GAMMA, self.int_ends,
                            self.time_series, self.sample_mag, self.relax_func,
                            self.b0_func, self.brf_func, self.bgrad_func,
                            self.rx_func, self.frame_rot, self.param_t1,
                            self.param_t2, self.param_b0, self.param_rf,
                            self.param_grad, self.param_rx, self.param_rot,
                            self.param_rot_plus, equi_t1rho, self.chunksize,
                            self.max_workers)
        # Save element parameters to text output
        self.print_list.extend(['SpinLock_Pulse', f"Duration = {self.duration}",
                                f"phi = {phi-90}", f"amp_hz = {amp_hz}",
                                f"del_omega= {self.del_omega}", "\n"])
        # Reset variables prior to next sequence element
        self.reset_variables()


    #~~~~~ Noise Calculation Methods ~~~~~#


    def import_image_csv(self, signal_file):
        """Imports image from .csv file, to enable noise addition to
        previously simulated data.

        Image data stored in self.image.

        Parameters
        ----------
        signal_file : str
            Name of file to be imported.
        """
        self.image = np.genfromtxt(signal_file, dtype = complex,
                                   delimiter = ',')

    def noise_calculator_signal_space(self, noise_func, noise_params, n_acq):
        """Adds noise to acquired signal using user-defined noise
        function. 
        
        Noise is calculated for complex k-space signal, then added to
        self.signal and stored in self.noisy_signal.

        Parameters
        ----------
        noise_func : object
            Function to perform noise calculation. See noisefunc.py for
            more details.
        noise_params : tuple
            Tuple containig parameters for use in noise_func. See
            noisefunc.py for more details.
        n_acq : int
            Number of repeat acquisitions in pulse sequence.
        """
        shape = self.signal.shape
        self.noisy_signal = np.zeros(shape, dtype = 'complex')
        noise = noise_func(noise_params, self.signal, self.b0, n_acq,
                           self.GAMMA, self.RNG)
        self.noisy_signal = self.signal+ noise

    def noise_calculator_image_space(self, noise_func, noise_params, n_acq):
        """Adds noise to existing image using user-defined noise
        function.

        Parameters
        ----------
        noise_func : object
            Function to perform noise calculation. See noisefunc.py for
            more details.
        noise_params : tuple
            Tuple containig parameters for use in noise_func. See
            noisefunc.py for more details.
        n_acq : int
            Number of repeat acquisitions in pulse sequence.
        """
        shape = self.image.shape
        self.noisy_image = np.zeros(shape, dtype = 'complex')
        noise = noise_func(noise_params, self.image, self.b0, n_acq,
                          self.GAMMA, self.RNG)
        self.noisy_image = self.image+ noise


    #~~~~~ Methods for converting k-space data to images ~~~~~#


    def import_signal_csv(self, signal_file):
        """Imports signal from .csv file, to enable image analysis of
        previously simulated data.

        Imports data into self.signal.

        Parameters
        ----------
        signal_file : str
            Name of file to be imported.
        """
        self.signal = np.genfromtxt(signal_file, dtype = complex,
                                    delimiter = ',')

    def image_processing(self, n_dim, seq_name, sample_name, save_path,
                        save_flag_csv=False, save_flag_img=False,
                        hann_flag=False, grad_amp=[],
                        noisy_signal=False):
        """Converts 1D or 2D k-space signal to an image using an FFT.

        Optionally, can apply Hanning filtering to the data prior to 
        conversion to image space. Can save images as .png and/or .csv
        files.

        Parameters
        ----------
        n_dim : int
            Dimensionality of the k-sapce signal (1D or 2D).
        seq_name : str
            Name of pulse sequence, used in naming output files.
        sample_name : str
            Name of sample folder, used in naming output files.
        save_path : str
            Path for saving files.
        save_flag_csv : bool, optional
            Controls if image data will be saved as .csv. Default value
            False.
        save_flag_img : bool, optional
            Controls if image data will be saved as .png. Default value
            False.
        hann_flag : bool, optional
            Controls if image data will be hanning filtered. Default
            value False.
        grad_amp : array_like
            Name of file to be imported.
        noisy_signal : bool
            Controls if image will be created from no noise
            (self.signal) or noise added (self.noisy_signal) signal.
        """
        # Construct output file name
        time_str = self.start_time.strftime("%Y-%m-%d_%Hh%Mm")
        file_name = f"{seq_name}_{sample_name}_{time_str}"
        # Identify input variable
        if noisy_signal is True:
            data = self.noisy_signal
        else:
            data = self.signal
        # Calculate and apply Hanning filter to signal
        shape = data.shape
        if hann_flag is True:
            grad0 = np.pi*np.linspace(-grad_amp[0], grad_amp[0], shape[0])
            hann0 = np.square(np.cos(grad0/(2*grad_amp[0])))
            grad1 = np.pi*np.linspace(-grad_amp[1], grad_amp[1], shape[1])
            hann1 = np.square(np.cos(grad1/(2*grad_amp[1])))
            hann = hann0*hann1.reshape(-1,1)
            self.signal = data*hann
            file_name = file_name + "_hannFiltered"
        # Calculate, plot, and save signal magnitude data
        signal_mag = np.abs(data)
        # Apply FFT to data
        if n_dim == 1:
            image_data = np.fft.ifft(self.signal)
            self.image = abs(np.fft.fftshift(image_data))
        elif n_dim == 2:
            image_data = np.fft.ifft2(self.signal)
            self.image = abs(np.fft.fftshift(image_data))
        # Save k-space magnitude data and image as .csv files
        if save_flag_csv is True:
            np.savetxt(save_path+ f"/kMag_{file_name}.csv",
                       signal_mag, delimiter=",")
            np.savetxt(save_path+ f"/Image_{file_name}.csv", self.image,
                       delimiter=",")
        # Save k-space data, k-space magnitude data, and image as .png
        if save_flag_img is True:
            if n_dim == 1:
                # Set figure size
                plt.rcParams["figure.figsize"] = (10,6)
                # Plot Kspace data
                fig = plt.figure(1)
                # Set axes to share scale
                ax1 = fig.add_subplot(1,2,1)
                ax2 = fig.add_subplot(1,2,2, sharex = ax1)
                # Set whichever signal component has greater magnitude
                # to first subplot
                if np.max(np.real(data)) >= np.max(np.imag(data)):
                    ax1.plot(np.real(data))
                    ax2.plot(np.imag(data))
                    ax1.title.set_text('Real')
                    ax2.title.set_text('Imaginary')
                else:
                    ax1.plot(np.imag(data))
                    ax2.plot(np.real(data))
                    ax1.title.set_text('Imaginary')
                    ax2.title.set_text('Real')
                plt.savefig(save_path+ f"/kSpace_{file_name}.png")
                # Plot Kspace magnitude data
                plt.figure(2)
                plt.plot(signal_mag)
                plt.savefig(save_path+ f"/kMag_{file_name}.png")
                # Plot Image
                plt.figure(3)
                plt.plot(self.image)
                plt.savefig(save_path+ f"/Image_{file_name}.png")
            elif n_dim == 2:
                plt.rcParams["figure.figsize"] = (10,6)
                # Plot Kspace data
                plt.figure(1)
                fig, (ax1, ax2) = plt.subplots(1,2)
                # Set axes to share scale
                ax1 = fig.add_subplot(1,2,1)
                ax1.set_axis_off()
                ax2 = fig.add_subplot(1,2,2, sharex = ax1)
                ax2.set_axis_off()
                # Set whichever signal component has greater magnitude
                # to first subplot
                if np.max(np.real(data)) >= np.max(np.imag(data)):
                    ax1.imshow(np.real(data))
                    ax2.imshow(np.imag(data))
                    ax1.title.set_text('Real')
                    ax2.title.set_text('Imaginary')
                else:
                    ax1.imshow(np.imag(data))
                    ax2.imshow(np.real(data))
                    ax1.title.set_text('Imaginary')
                    ax2.title.set_text('Real')
                plt.savefig(save_path+ f"/kSpace_{file_name}.png")
                # Plot Kspace magnitude data
                plt.figure(2)
                plt.imshow(signal_mag)
                plt.savefig(save_path+ f"/kMag_{file_name}.png")
                # Plot Image
                plt.figure(3)
                plt.imshow(self.image, cmap = "gray")
                plt.savefig(save_path+ f"/Image_{file_name}.png")
