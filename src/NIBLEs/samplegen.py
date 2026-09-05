"""
Classes for designing samples for use with the NIBLEs MR simulation
tool.
"""

import os
from pathlib import Path

import numpy as np


class Material:
    """Material is a python class which is used to define the MR
    properties of a single material to be used in sample generation.
    
    The Material class allows for up to four sets of relaxation times to
    be saved to a single material: 2 constant T1/T2 values, and 2 set of
    parameters for calculating field dependent T1/T2 values. This has
    been arranged to allow samples to be used for simulations at
    multiple applied field strengths.
    
    Material additionally calculates the T2` value used in t2_star 
    simulation. T2` is based on the given primary T2 and t2_star values.
    
    Note that an index value of 0 is reserved for empty space within the
    imaging volume.

    Parameters
    ----------
    index : int
        Non-zero index number used in sample mask to indicate material
        location when generating sample.
    pd : float
        Proton density, given as a fraction of water proton density to
        allow for consistent noise calculations.
    chem_shift : float
        Chemical shift of material in PPM.
    t1, t2, t2_star, t1_alt, t2_alt : float
        Static relaxation values for the Material in seconds.
    dynamic_t1, dynamic_t2, dynamic_t1_alt, dynamic_t2_alt : array_like
        Arrays defining the parameters needed for calculating relaxation
        with a desired relaxation function. For further details, 
        please reference the model of choice in relaxfunc.py
    """

    def __init__(self, index, pd, chem_shift, t1, t2, t2_star, dynamic_t1,
                 dynamic_t2, t1_alt, t2_alt, dynamic_t1_alt, dynamic_t2_alt):
        # Assign given parametres to class variables
        self.index = index
        self.pd = pd
        self.chem_shift = chem_shift
        self.t1 = t1
        self.t2 = t2
        self.t2_star = t2_star
        self.t1_alt = t1_alt
        self.t2_alt = t2_alt
        self.dynamic_t1 = dynamic_t1
        self.dynamic_t2 = dynamic_t2
        self.dynamic_t1_alt = dynamic_t1_alt
        self.dynamic_t2_alt = dynamic_t2_alt
    # Calculate T2` from given
        self.t2_prime = (self.t2*self.t2_star)/(self.t2-self.t2_star)

class ImagingVolume:
    """ImagingVolume is a python class which is used to define the
    properties of the imaging volume of the MR system.
    
    To generate a sample, create an instance of the ImagingVolume 
    class, as well as instances of the Material class containing the
    material properties for each material in a sample.

    The ImagingVolume class contains methods for creating simple shapes
    within sample_mask. Once each material has been arrayed within
    sample_mask, use the  construct_sample method to generate arrays of
    MR relevant properties for the sample. Finally, using the
    save_sample method, these arrays can be saved for use in a NIBLEs
    script.

    All output parameters are stored as 4D arrays, where the first 3 
    dimensions are the (x, y, z) co-ordinates of the voxel, and the 4th 
    dimension is the vector number within that voxel 
    (e.g. index (0, 0, 0, 5) denotes the 6th vector in the voxel at the 
    lower leftmost corner of the imaging volume). 

    Parameters
    ----------
    size_metric : array_like
        3-element list defining the dimensions of the sample volume in
        meters.
    size_pixels : array_like
        3-element list defining the dimensions of the sample volume in
        pixels.
    n_vectors : int
        Number of magnetization vectors to be simulated within each
        pixel for t2_star simulation

    Attributes
    ----------
    sample_mask : array_like
        NumPy array with dimensions size_pixels. Spatial map of where
        each material is located within the sample volume.

    """

    def __init__(self, size_metric, size_pixels, n_vectors):
        # Assign inputs to class variables
        self.size_metric = size_metric
        self.size_pixels = size_pixels
        self.n_vectors = n_vectors
        # Pre-define output matrices for sample parameters
        self.sample_mask = np.zeros(self.size_pixels)
        self.size_4d = self.size_pixels + [self.n_vectors]
        self.dis_per_pix = np.array([self.size_metric[0]/self.size_pixels[0],
                                   self.size_metric[1]/self.size_pixels[1],
                                   self.size_metric[2]/self.size_pixels[2]])
        self.sample_mag = np.zeros(self.size_4d)
        self.sample_mag = self.sample_mag.astype(object)
        self.chem_shift = np.zeros(self.size_4d)
        self.m_position = np.zeros(self.size_4d)
        self.m_position = self.m_position.astype(object)
        self.t1 = np.zeros(self.size_4d)
        self.t2 = np.zeros(self.size_4d)
        self.t1_alt = np.zeros(self.size_4d)
        self.t2_alt = np.zeros(self.size_4d)
        self.t2_prime = np.zeros(self.size_4d)
        self.dynamic_t1 = np.zeros(self.size_4d)
        self.dynamic_t1 = self.dynamic_t1.astype(object)
        self.dynamic_t2 = np.zeros(self.size_4d)
        self.dynamic_t2 = self.dynamic_t2.astype(object)
        self.dynamic_t1_alt = np.zeros(self.size_4d)
        self.dynamic_t1_alt = self.dynamic_t1_alt.astype(object)
        self.dynamic_t2_alt = np.zeros(self.size_4d)
        self.dynamic_t2_alt = self.dynamic_t2_alt.astype(object)
        # Create random number generator for vector placement
        # randomization
        self.RNG = np.random.default_rng()

    def construct_sample(self, mater_classes):
        """Assigns MR properties to each magnetization vector in the
        sample.
        
        Takes a list of Material classes and compares the index
        values of the Material classes to the values of each element in
        sample_mask. It then assigns MR properties to each magnetization
        vector in the voxel based on the index. 

        The position of each voxel is defined as the voxel center. 
        Magnetization vectors within a voxel are randomly distributed 
        within a square volume with sides that are 1/10th the length of 
        the full voxel. This distribution is then centered on a random 
        point within the central 15th of the voxel.

        Note: A value of 0 in sample_mask is assumed to be empty space, 
        and is assigned NaN for each MR property.

        Parameters
        ----------
        mater_classes : array_like
            List of Material class objects.

        """

        # Calculate the center of each voxel, calculate a random point
        # within the central 15th of the voxel to serve as the
        # centerpoint for the vector distribution
        center_array_size = self.size_pixels + [3]
        dist_cent_array = 0.5+ (self.RNG.random(center_array_size)-0.5)/15
        # Calculate random offsets for each vector relative to the
        # distribution center
        vector_offset = (self.RNG.random([self.n_vectors])-0.5)/10
        # Iterate through all voxels and vectors in the sample space
        for index in np.ndindex(self.sample_mag.shape):
            index2 = tuple(index[0:3])
            vec_num = index[3]
            # Identify the global position of voxel center
            dist_cent = dist_cent_array[index2]
            # Calculate global position for the current vector
            pos = np.array([index[0]+ dist_cent[0]+ vector_offset[vec_num],
                             index[1]+ dist_cent[1]+ vector_offset[vec_num],
                             index[2]+ dist_cent[2]+ vector_offset[vec_num]])
            # If voxel is empty, assign NaN to all parameters
            if self.sample_mask[index2] == 0:
                self.sample_mag[index] = np.NaN
                self.m_position[index] = np.NaN
                self.chem_shift[index] = np.NaN
                self.t1[index] = np.NaN
                self.t2[index] = np.NaN
                self.t2_prime[index] = np.NaN
                self.dynamic_t1[index] = np.NaN
                self.dynamic_t2[index] = np.NaN
                self.dynamic_t1_alt[index] = np.NaN
                self.dynamic_t2_alt[index] = np.NaN
            # If voxel contains a material, assign vector the parameters
            # defined in the relevant materClass
            for mater_cls in mater_classes:
                if mater_cls.index== self.sample_mask[index2]:
                    self.sample_mag[index] = np.array([0, 0, mater_cls.pd],
                                             dtype = float)
                    self.m_position[index] = np.multiply(pos, self.dis_per_pix)
                    self.chem_shift[index] = mater_cls.chem_shift
                    self.t1[index] = mater_cls.t1
                    self.t2[index] = mater_cls.t2
                    self.t1_alt[index] = mater_cls.t1_alt
                    self.t2_alt[index] = mater_cls.t2_alt
                    self.t2_prime[index] = mater_cls.t2_prime
                    self.dynamic_t1[index] = mater_cls.dynamic_t1
                    self.dynamic_t2[index] = mater_cls.dynamic_t2
                    self.dynamic_t1_alt[index] = mater_cls.dynamic_t1_alt
                    self.dynamic_t2_alt[index] = mater_cls.dynamic_t2_alt

    def save_sample(self, sample_name, sample_path):
        """Saves variables created by Constuct_Sample to file.

        Variables are saved in a folder named sample_name.

        Parameters
        ----------
        sample_name : str
            Desired name of NIBLEs sample.
        sample_path : str
            Path to folder where the NIBLEs sample will be saved

        """

        Path(os.path.join(sample_path, sample_name)
             ).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_M.npy'), self.sample_mag)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_MPosition.npy'), self.m_position)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_chemShift.npy'), self.chem_shift)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_T1.npy'), self.t1)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_T2.npy'), self.t2)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_T1alt.npy'), self.t1_alt)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_T2alt.npy'), self.t2_alt)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_T2prime.npy'), self.t2_prime)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_size4D.npy'), self.size_4d)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_sizeMetric.npy'),
                             self.size_metric)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_dynamicT1.npy'), self.dynamic_t1)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_dynamicT2.npy'), self.dynamic_t2)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_dynamicT1alt.npy'),
                             self.dynamic_t1_alt)
        np.save(os.path.join(sample_path, sample_name,
                             sample_name + '_dynamicT2alt.npy'),
                             self.dynamic_t2_alt)

    def generate_sphere(self, r, mat_index, cent='None'):
        """Creates a sphere of material within ImagingVolume.samp_mask,
        overwriting the indeces within the specified volume.

        Variables are saved in a folder named sample_name.

        Parameters
        ----------
        r : float
            Radius of generated sphere (meters).
        mat_index : int
            Index value of the material that composes the sphere.
        cent : array_like or str, optional
            Element of sample_mask to make the sphere's centre; if 
            given a string of 'None', automatically centers sphere 
            within the sample volume.
        """

        # If located at center of imaging volume, calculate center of
        # imaging volume
        if cent == 'None':
            cent = (self.size_pixels[0]/2, self.size_pixels[1]/2,
                    self.size_pixels[2]/2)
            cent = np.subtract(cent, 0.5)
        # Calculate where each voxel is relative to sphere center
        for index in np.ndindex(self.sample_mask.shape):
            shift = np.subtract(index, cent)
            ind_arr = np.asarray(shift)
            norm = np.linalg.norm(ind_arr)
            # If voxel is within the sphere's radius, assign the given
            # index value to that voxel
            if norm <= r:
                self.sample_mask[index] = mat_index
            else:
                continue

    def generate_square(self, dim, mat_index, cent='None'):
        """Creates a rectancular volume of material within
        ImagingVolume.samp_mask, overwriting the indeces within the
        specified volume.

        Variables are saved in a folder named sample_name.

        Parameters
        ----------
        dim : array_like
            3 element list defining the x, y, and z silde lengths of the
            rectangular volume.
        mat_index : int
            Index value of the material that composes the rectangular
            volume.
        cent : array_like or str, optional
            Element of sample_mask to make the recatangles's centre; if 
            given a string of 'None', automatically centers sphere 
            within the sample volume.
        """

        # If located at center of imaging volume, calculate center of
        # imaging volume
        if cent == 'None':
            cent = (self.size_pixels[0]/2, self.size_pixels[1]/2,
                    self.size_pixels[2]/2)
            cent = np.subtract(cent, 0.5)
        # Calculate where each voxel is relative to square center
        for index in np.ndindex(self.sample_mask.shape):
            upper_x = np.add(cent[0], dim[0]/2)
            lower_x = np.subtract(cent[0], dim[0]/2)
            upper_y = np.add(cent[1], dim[1]/2)
            lower_y = np.subtract(cent[1], dim[1]/2)
            upper_z = np.add(cent[2], dim[2]/2)
            lower_z = np.subtract(cent[2], dim[2]/2)
            # If voxel is within the square, assign the given index
            # value to that voxel
            if (lower_x <= index[0] <= upper_x and lower_y <= index[1] <=
            upper_y and lower_z <= index[2] <= upper_z):
                self.sample_mask[index] = mat_index
            else:
                continue
