import os
import numpy as np

from pathlib import Path

# Classes for designing samples for use with the NIBLEs MR simulation tool.
#
# Original version of this code designed and written by John Adams (jadam33@uwo.ca)
#
# This code is published under a Creative Commons BY-SA 4.0 License. Anyone using this code must give 
# attribution to the original author (John Adams), and must make any derivative code built using this
# code available under the same licensing terms.

#~~~~~ Version History ~~~~~#
# 1.0  Public Release - DATE - John Adams

class Material:
    """
    Material is a python class which is used to store the MR properties of a single material to be used
    in sample generation. The Material class allows for up to four sets of relaxation times to be saved
    to a single material: 2 constant T1/T2 values, and 2 set of parameters for calculating field 
    dependent T1/T2 values. This has been arranged to allow samples to be used for simulations at 
    multiple applied field strengths.

    Material additionally calculates the T2` value used in T2star simulation. T2` is based on the given 
    primary T2 and T2star values.

    Note that an ind value of 0 is reserved for empty space within the imaging volume.

    Parameters:
    ind = Non-zero index number used in sample mask to indicate material location when generating sample
    PD = Proton density, given as a fraction of water proton density to allow for consistent noise 
        calculations
    chemShift = Chemical shift in PPM
    T1, T2, T2star, T1alt, T2alt = Relaxation times in seconds, used in simulation when relaxation times 
        are constant. The alt relaxation times can be used to save the parameters for multiple models 
        into a single sample file.
    dynamicT1, dynamicT2, dynamicT1alt, dynamicT2alt = Parameters for generating field dependent 
        relaxation times. Format as a 1D list. For further details, please reference the model of choice
        in relaxFuncLab.py

    Returns:
    An initialized Material() instance
    """
    def __init__(self, ind, PD, chemShift, T1, T2, T2star, dynamicT1, dynamicT2, 
        T1alt, T2alt, dynamicT1alt, dynamicT2alt):
            
        # Assign given parametres to class variables
        self.ind = ind
        self.PD = PD
        self.chemShift = chemShift
        self.T1 = T1
        self.T2 = T2
        self.T2star = T2star
        self.T1alt = T1alt
        self.T2alt = T2alt
        self.dynamicT1 = dynamicT1
        self.dynamicT2 = dynamicT2
        self.dynamicT1alt = dynamicT1alt
        self.dynamicT2alt = dynamicT2alt

    # Calculate T2` from given
        self.T2prime = (self.T2*self.T2star)/(self.T2-self.T2star)

class Imaging_Volume:
    """
    Class containing methods needed to generate and save a NIBLEs sample.

    To generate a sample, create an instance of the Imaging_Volume class, as well as instances of the 
    material class containing the material properties for each material in a sample.

    The Imaging_Volume.sampleMask variable is an array that acts as spatial map; it identifies where 
    each material exists within the sample space using an index number. The Imaging_Volume class contains
    methods for creating simple shapes within sampleMask. Once each material has been arrayed within 
    sampleMask, use the Construct_Sample method to generate arrays of MR relevant properties for the sample.
    Finally, using the Save_Sample method, these arrays can be saved for use in a NIBLEs script.

    All output parameters are stored in 4D arrays, where the first 3 dimensions are the (x, y, z)
    co-ordinates of the voxel, and the 4th dimension is the vector number within that voxel 
    (e.g. index (0, 0, 0, 5) denotes the 6th vector in the voxel at the lower leftmost corner of the 
    imaging volume). 

    Inputs:
    sizeMetric = Dimensions of sample volume in metres
    sizePixels = Dimensions of sample volume in pixels for simulation
    nVectors = Number of magnetization vectors to be simulated within each pixel for T2star simulation;
        Recommend a minimum value of 128 to ensure accurate T2star in final simulation output

    Returns:
    An initialized Imaging_Volume() instance
    """
    def __init__(self, sizeMetric, sizePixels, nVectors):

        # Assign inputs to class variables
        self.sizeMetric = sizeMetric
        self.sizePixels = sizePixels
        self.nVectors = nVectors

        # Pre-define output matrices for sample parameters
        self.sampleMask = np.zeros(self.sizePixels)
        self.size4D = self.sizePixels + [self.nVectors]
        self.DisPerPix = np.array([self.sizeMetric[0]/self.sizePixels[0],
                                   self.sizeMetric[1]/self.sizePixels[1],
                                   self.sizeMetric[2]/self.sizePixels[2]])

        self.pixVol = self.DisPerPix[0]*self.DisPerPix[1]*self.DisPerPix[2]

        self.M = np.zeros(self.size4D)
        self.M = self.M.astype(object)

        self.chemShift = np.zeros(self.size4D)

        self.MPosition = np.zeros(self.size4D)
        self.MPosition = self.MPosition.astype(object)

        self.T1 = np.zeros(self.size4D)
        self.T2 = np.zeros(self.size4D)
        self.T1alt = np.zeros(self.size4D)
        self.T2alt = np.zeros(self.size4D)
        self.T2prime = np.zeros(self.size4D)

        self.dynamicT1 = np.zeros(self.size4D)
        self.dynamicT1 = self.dynamicT1.astype(object)

        self.dynamicT2 = np.zeros(self.size4D)
        self.dynamicT2 = self.dynamicT2.astype(object)

        self.dynamicT1alt = np.zeros(self.size4D)
        self.dynamicT1alt = self.dynamicT1alt.astype(object)

        self.dynamicT2alt = np.zeros(self.size4D)
        self.dynamicT2alt = self.dynamicT2alt.astype(object)

    def Construct_Sample(self, materClasses):
        """
        This method takes a list of material classes and compares the ind values of the material classes to 
        the values of each element in the sampleMask mask. It then assigns MR properties to each magnetization
        vector in the voxel based on the index. 

        The position of each voxel is defined as the voxel center. Magnetization vectors within a voxel
        are randomly distributed within a square volume with sides that are 1/10th the length of the 
        full voxel. This distribution is then centered on a random point within the central 15th of the voxel.

        Note: A value of 0 in sampleMask is assumed to be empty space, and is assigned NaN for each MR property.

        Inputs:
        materClasses = List of Material class objects
        """
        # Create random number generator for vector placement randomization
        self.rng = np.random.default_rng()

        # Calculate the center of each voxel, calculate a random point within the central 15th of the voxel to
        # serve as the centerpoint for the vector distribution
        centerArraySize = self.sizePixels + [3]
        self.distCenterArray = 0.5+ (self.rng.random(centerArraySize)-0.5)/15

        # Calculate random offsets for each vector relative to the distribution center
        self.vectorOffsets = (self.rng.random([self.nVectors])-0.5)/10

        # Iterate through all voxels and vectors in the sample space
        for index in np.ndindex(self.M.shape):
            index2 = tuple(index[0:3])
            vecNum = index[3]

            # Identify the global position of voxel center
            self.distCenter = self.distCenterArray[index2]
            
            # Calculate global position for the current vector
            PPos = np.array([index[0]+ self.distCenter[0]+ self.vectorOffsets[vecNum],
                index[1]+ self.distCenter[1]+ self.vectorOffsets[vecNum],
                index[2]+ self.distCenter[2]+ self.vectorOffsets[vecNum]])

            # If voxel is empty, assign NaN to all parameters
            if self.sampleMask[index2] == 0:
                self.M[index] = np.NaN
                self.MPosition[index] = np.NaN
                self.chemShift[index] = np.NaN
                self.T1[index] = np.NaN
                self.T2[index] = np.NaN
                self.T2prime[index] = np.NaN
                self.dynamicT1[index] = np.NaN
                self.dynamicT2[index] = np.NaN
                self.dynamicT1alt[index] = np.NaN
                self.dynamicT2alt[index] = np.NaN

            # If voxel contains a material, assign vector the parameters definedd in the relevant materClass
            for materCls in materClasses:
                if materCls.ind == self.sampleMask[index2]:
                    self.M[index] = np.array([0, 0, materCls.PD], dtype = float)
                    self.MPosition[index] = np.multiply(PPos, self.DisPerPix)
                    self.chemShift[index] = materCls.chemShift
                    self.T1[index] = materCls.T1
                    self.T2[index] = materCls.T2
                    self.T1alt[index] = materCls.T1alt
                    self.T2alt[index] = materCls.T2alt
                    self.T2prime[index] = materCls.T2prime
                    self.dynamicT1[index] = materCls.dynamicT1
                    self.dynamicT2[index] = materCls.dynamicT2
                    self.dynamicT1alt[index] = materCls.dynamicT1alt
                    self.dynamicT2alt[index] = materCls.dynamicT2alt

    def Save_Sample(self, sName, fpath):
        """
        This method saves variables created by Constuct_Sample to file. 
        
        Standard practice is to save Samples in a 'Samples' folder in the NIBLEs directory.

        Inputs:
        sName = Name (string) to be used to identify the sample
        fpath = Path where outputs will be saved
        """
        Path(os.path.join(fpath, sName)).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(fpath, sName, sName + '_M.npy'), self.M)
        np.save(os.path.join(fpath, sName, sName + '_MPosition.npy'), self.MPosition)
        np.save(os.path.join(fpath, sName, sName + '_chemShift.npy'), self.chemShift)
        np.save(os.path.join(fpath, sName, sName + '_T1.npy'), self.T1)
        np.save(os.path.join(fpath, sName, sName + '_T2.npy'), self.T2)
        np.save(os.path.join(fpath, sName, sName + '_T1alt.npy'), self.T1alt)
        np.save(os.path.join(fpath, sName, sName + '_T2alt.npy'), self.T2alt)
        np.save(os.path.join(fpath, sName, sName + '_T2prime.npy'), self.T2prime)
        np.save(os.path.join(fpath, sName, sName + '_size4D.npy'), self.size4D)
        np.save(os.path.join(fpath, sName, sName + '_sizeMetric.npy'), self.sizeMetric)
        np.save(os.path.join(fpath, sName, sName + '_dynamicT1.npy'), self.dynamicT1)
        np.save(os.path.join(fpath, sName, sName + '_dynamicT2.npy'), self.dynamicT2)
        np.save(os.path.join(fpath, sName, sName + '_dynamicT1alt.npy'), self.dynamicT1alt)
        np.save(os.path.join(fpath, sName, sName + '_dynamicT2alt.npy'), self.dynamicT2alt)


    def Sphere(self, r, matInd, cent = 'None'):
        """
        Creates a sphere of material within the sample volume, overwrites any existing indeces

        Inputs:
        r = Radius of sphere
        matInd = Material index number to place in sampleMask
        cent = Element of sampleMask to make the sphere's centre; if given a string of 'None',
                automatically centers sphere within the sample volume.
        """
        # If located at center of imaging volume, calculate center of imaging volume
        if cent == 'None':
            cent = (self.sizePixels[0]/2, self.sizePixels[1]/2, self.sizePixels[2]/2)
            cent = np.subtract(cent, 0.5)
        
        # Calculate where each voxel is relative to sphere center
        for index in np.ndindex(self.sampleMask.shape):
            
            shift = np.subtract(index, cent)
            indarr = np.asarray(shift)
            norm = np.linalg.norm(indarr)

            # If voxel is within the sphere's radius, assign the given index value to that voxel
            if norm <= r:
                self.sampleMask[index] = matInd
            else:
                continue

    def Square(self, Dim, matInd, cent = 'None'):
        """
        Creates a rectancular volume of material within the sample volume, overwrites any existing indeces

        Inputs:
        Dim = 3 element list defining the x, y, and z side lengths of the volume
        matInd = Material index number to place in sampleMask
        cent = Element of sampleMask to make the sphere's centre; if given a string of 'None',
                automatically centers sphere within the sample volume.
        """
        # If located at center of imaging volume, calculate center of imaging volume
        if cent == 'None':
            cent = (self.sizePixels[0]/2, self.sizePixels[1]/2, self.sizePixels[2]/2)
            cent = np.subtract(cent, 0.5)

        # Calculate where each voxel is relative to Square center
        for index in np.ndindex(self.sampleMask.shape):
            
            upperX = np.add(cent[0], Dim[0]/2)
            lowerX = np.subtract(cent[0], Dim[0]/2)
            upperY = np.add(cent[1], Dim[1]/2)
            lowerY = np.subtract(cent[1], Dim[1]/2)
            upperZ = np.add(cent[2], Dim[2]/2)
            lowerZ = np.subtract(cent[2], Dim[2]/2)

            # If voxel is within the square, assign the given index value to that voxel
            if lowerX <= index[0] <= upperX and lowerY <= index[1] <= upperY and lowerZ <= index[2] <= upperZ:
                self.sampleMask[index] = matInd
            else:
                continue

    
