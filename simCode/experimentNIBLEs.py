import os
import copy
import warnings
import datetime as time

import math as mat
import numpy as np
import matplotlib.pyplot as plt

import simCode.solverNIBLEs as solver
import simCode.relaxFuncLib as relax
import simCode.fieldFuncLib as field

#~~~~~ Basic Code Layout ~~~~~#
# Simulation code for the Numeric Integrator for the Bloch Equations (NIBLEs)
# 
# This file defines the Experiment class, which is used for loading in sample properties, 
# defining a pulse sequence, and simulating that sequence to produce a simulated signal. Methods 
# within Experiment are split into five types: methods for setting and manipulating system
# variables; utility methods used to standardaze common functionality of pulse sequence elements; 
# methods which define and simulate the effects of various pulse sequence elements; methods for adding
# noise to images and k-space data; and methods for converitng k-space data to images. The sequence
# element methods take as input all the nessecary parametres for defining that sequence element, then 
# call the solving code to evolve the magnetization of the sample, then save that evolved magnetization
# in the Experiment.M variable for the next pulse sequence element to be called.
#
# Original version of this code designed and written by John Adams (jadam33@uwo.ca)
#
# This code is published under a GNU General Public License version 3 License. Anyone using this code 
# must give attribution to the original author (John Adams), and must make any derivative code built 
# using this code available under the same licensing terms.


#~~~~~ Version History ~~~~~#
# 1.0  Public Release - July 1 2026 - John Adams




class Experiment():
	"""
	The Experiment class is designed to manage the data inputs and outputs of the NIBLEs solver throughout a
	simulation. As part of this, the functions an end user can use to simulate common pulse sequence elements
	are coded as methods within the Experiment class.

	The current state of magnetization is stored in Experiment.M. Experiment uses a variable called
	Experiment.printList to keep track of which sequence elements have been applied to the sample and what
	parameters were used.

	After creating an instance of this class, sample properties are loaded as class variables using the
	Experiment.sequenceStart method. As sequence element methods are called, they simulate the effects of that
	sequence element on the magnetization of the sample and store the updated magnetization in the
	Experiment.M variable.

	All variables are in SI units:
	Length = Metres
	Time = Seconds
	Frequency = Hertz
	Magnetic Field = Tesla
	Angles = Degrees

	Parameters:
	B0 = 3 element list describing the static magnetic field along the [x, y, z] axes of the MR system in the
		lab frame. Measured in Tesla. Default value of [0, 0, 0.5]
	GAMMA = Gyromagnetic ratio of nucleus of interest, in Hz/T. Default values of Hydrogen (42.58e6 Hz/T)

	Returns:
	An initialized Experiment() instance
	"""
	def __init__(self, B0, GAMMA = 42.57e6):

		# Saving start time of Experiment run
		self.startTime = time.datetime.now()

		# Chunk Size for parellelized processing
		self.chunksize = 64
		self.max_workers = None                # From concurrent.futures, if None solver will use all 
												# availablbe threads

		# Simulation Constants
		self.GAMMA = 2*np.pi*GAMMA              # Gyromagnetic Ratio (input as Hz/T, stored as rad/Ts)
		self.B0 = np.array(B0)                  # Static Magnetic Field [x, y, z](T)
		self.B0Func = solver.Zero          			# Function for calculating B0 in simulator
		self.ParamB0 = 0
		self.Larmor = self.GAMMA* self.B0[2]    # Larmor Frequency (rad/s)

		# Initialize Random number genreator
		self.rng = np.random.default_rng()

		# Function to calculate relaxation times
		self.relaxFunc = solver.Zero

		# Parameters for reference frame rotations
		self.ParamRot = (self.B0, self.Larmor)      # Parameters for converting from Lab to Rotating frame
		self.ParamRotPlus = (self.B0, self.GAMMA)  # Parameters for rotations away from rotating reference frame

		# Variables for sample properties, see sequenceStart()
		self.M = [0,0,1]
		self.equiM = [0,0,1]
		self.MPosition = [0,0,0]
		self.sizePixels = 0
		self.sizeMetric = 0
		self.ParamT1 = 0
		self.ParamT2 = 0
		self.ParamT1alt = 0
		self.ParamT2alt = 0
		self.count = 0

		# Variables of advanced control of T2star modeling
		self.t2starBounds = [0.075, 0.925]      # Define endpoints of linearly spaced T2star field offsets
												# Must be within range (0, 1)

		# Calculated field strengths required for 90 and 180 degree RF Pulses, see RF_Calibrate()
		self.B90 = 0
		self.B180 = 0

		# Variables for defining sequence elements
		self.Duration = 0                       # Element duration
		self.BrfFunc = solver.Zero                     # Function for calculating RF field waveform in lab frame
		self.BgradFunc = solver.Zero                   # Function for calculating gradient waveform in lab frame
		self.ParamRF = (0)                      # Tuple containing parametres used to define RF field 
		self.ParamGrad = (0)                    # Tuple containing parametres used to define gradient field 
		self.Int_Ends = (0)                     # Tuple defining start and end times for solver
		self.frameRot = solver.zeroBzFrame             # Function to move solving out of the rotating reference frame
		self.RxFunc = solver.Zero                     # Function for calculating Rx sensitivity 
		self.additionalParams = {}                     # Dict for containing additional keyword arguments

		# Variables for Saving Sequence Parametres for later reference
		self.printList = []
		self.printList.extend([f"B0 = {self.B0}", f"gamma = {GAMMA}", "\n"])

	#~~~~~ Utility Methods for data handling and manipulation ~~~~~#

	def Save2CSV(self, var, filename):
		"""
		Saves speficied Numpy array to .csv file. Used as a sub-function of SaveAcquisition().

		Parameters:
		var = Numpy array to be written to file
		filename = String containing name of file to be created
		"""
		np.savetxt(filename, var, delimiter=",")

	def Params2Txt(self, filename):
		"""
		Saves contents of self.printList to file. Used as a sub-function of SaveAcquisition().

		Parameters:
		filename = String containing name of file to be created
		"""
		with open(f"{filename}", 'w') as f:
			f.write('\n'.join(self.printList))

	def SaveAcquisition(self, var, seqName, sampName):
		"""
		Saves printList and given Numpy array to file, using a filename derived from the sequence 
		used and the name of the sample used

		Parameters:
		var = Numpy array to be saved
		seqName = String identifying sequence used to generate final filename
		sampName = String identifying sample used to generate final filename
		"""
		# Format startTime into Hours/Minutes
		timestr = self.startTime.strftime("%Y-%m-%d_%Hh%Mm")

		# Construct output filename from inputs
		fileName = f"{seqName}_{sampName}_{timestr}"

		# Calculate total computation time and save that time to printList
		computeTime = time.datetime.now() - self.startTime
		self.printList.extend([f"Total Compute Time = {computeTime}"])

		# Save data and printList to file
		self.Save2CSV(var, f"{fileName}_Output.csv")
		self.Params2Txt(f"{fileName}_Params.txt")    

	def Reset_Magnetization(self, Mag = None):
		"""
		Method for resetting magnetization to a saved state, allowing a user to 1) reset magnetization
		to equilibrium without simulating full TR or 2) avoid repeated simulation of identical 
		sequence elements. If Mag is a Numpy array, function will set Experiment.M to Mag. For any 
		other input, function will set Experiment.M to Experiment.equiM (equilibrium magnetization).

		Parameters:
		Mag = Saved magnetization state. Default value 'None'
		"""
		# If Mag is not an array, set Experiment.M == Experiment.equiM
		if not isinstance(Mag, np.ndarray):
			self.M = copy.deepcopy(self.equiM)
			self.printList.extend(["Magnetization Reset to Equilibrium using Reset_Magnetization()", "\n"])
		# If Mag is an array, set Experiment.M == Mag
		else:
			self.M = copy.deepcopy(Mag)
			self.printList.extend(["Magnetization Reset to given state using Reset_Magnetization()", "\n"])

	def NullTransverseMag(self):
		"""
		Method which sets magnetization in the xy-plane to 0

		Parameters:
		None
		"""
		# Iterate through all vectors in sample. Ignore empty array elements, set the x- and y-
		# components of extant vectors to zero
		for index in np.ndindex(self.M.shape):
			if isinstance(self.M[index], np.ndarray) is False:
				continue
			else:
				self.M[index] = np.array([0, 0, self.M[index][2]], dtype = float)

	def RxSet(self, RxFunc, **kwargs):
		"""
		Method for setting parameters defining Rx sensitivity. Updates the Experiment.RxFunc and 
		Experiment.ParamRx variables. Any additional kwargs are added to a dictionary at 
		Experiment.ParamRx[3]

		Parameters: 
		RxFunc = Function defining receive coil sensitivity profile
		**kwargs = Additional parameters required by RxFunc
		"""
		self.RxFunc = RxFunc
		self.ParamRx = (self.MPosition, self.sizeMetric, self.sizePixels, kwargs)

		self.printList.extend([f"Rx Function changed to {RxFunc}", 
						f"additionalParams = {kwargs}", "\n"])


	def B0Set(self, B0Func, **kwargs):
		"""
		Method for setting parameters defining B0 field strength. Updates the Experiment.R0Func and 
		Experiment.ParamB0 variables. Any additional kwargs are added to a dictionary at 
		Experiment.ParamB0[3]

		Parameters: 
		RxFunc = Function defining receive coil sensitivity profile
		**kwargs = Additional parameters required by RxFunc
		"""
		self.B0Func = B0Func
		self.ParamB0 = (self.B0, self.t2starOffsets, self.chemShift, kwargs)

		self.printList.extend([f"B0 Function changed to {B0Func}", 
						f"additionalParams = {kwargs}", "\n"])

	def sequenceStart(self, sName, fpath, relaxFunc = relax.Constant, model_type = "Constant"):
		"""
		Method that imports sample properties from file for use in a simulation. In addition to 
		importing data, this method sets key functions for calculating relaxation behaviour, 
		calculates field offsets for T2star reproduction, and calculates chunksize for parallel 
		processing. 

		Parameters:
		sName = Name of sample folder
		fpath = Path to NIBLEs directory
		relaxFunc = Function from relaxFuncLib for calculating T1/T2, defaults to Constant [See 
		Relaxation Time Calculation Functions]
		model_type = String identifying which relaxation model parameters to load from file:

			If model_type = “Dynamic”: ParamT1/ParamT2 == Material.DynamicT1/DynamicT2 and 
			ParamT1alt/ParamT2alt == Material.DynamicT1alt/DynamicT2alt

			If model_type = “Dynamic_Alt”: ParamT1/ParamT2 == Material.DynamicT1alt/DynamicT2alt 
			and ParamT1alt/ParamT2alt == Material.DynamicT1/DynamicT2

			If model_type = “Constant_Alt”: ParamT1/ParamT2 == Material.T1alt/T2alt and 
			ParamT1alt/ParamT2alt == Material.T1/T2

			For all other inputs for model_type: ParamT1/ParamT2 == Material.T1/T2 and 
			ParamT1alt/ParamT2alt == Material.T1alt/T2alt

			Defaults to using Material.T1 and Material.T2 as constant relaxation times.
		"""

		print(f'Loading Sample...')
		# Load in functions for calculating relaxation times 
		self.relaxFunc = relaxFunc
		
		# Set default parameters for Rx field
		self.RxFunc = field.RxUniform
		self.ParamRx = ()

		# Load in sample size parameters and equilibrium magnetization
		self.M = np.load(os.path.join(fpath, sName, sName+ '_M.npy'), allow_pickle = True)
		self.MPosition = np.load(os.path.join(fpath, sName, sName+ '_MPosition.npy'), allow_pickle = True)
		self.size4D = np.load(os.path.join(fpath, sName, sName+ '_size4D.npy'), allow_pickle = True)
		self.sizePixels = self.size4D[0:3]
		self.sizeMetric = np.load(os.path.join(fpath, sName, sName+ '_sizeMetric.npy'), allow_pickle = True)
		
		# Load in desired relaxation model parameters. Defaults to constant relaxation times.
		if model_type == "Dynamic":
			self.ParamT1 = np.load(os.path.join(fpath, sName, sName+ '_dynamicT1.npy'), allow_pickle = True)
			self.ParamT2 = np.load(os.path.join(fpath, sName, sName+ '_dynamicT2.npy'), allow_pickle = True)
			self.ParamT1alt = np.load(os.path.join(fpath, sName, sName+ '_dynamicT1alt.npy'), allow_pickle = True)
			self.ParamT2alt = np.load(os.path.join(fpath, sName, sName+ '_dynamicT2alt.npy'), allow_pickle = True)
		elif model_type == "Dynamic_Alt":
			self.ParamT1 = np.load(os.path.join(fpath, sName, sName+ '_dynamicT1alt.npy'), allow_pickle = True)
			self.ParamT2 = np.load(os.path.join(fpath, sName, sName+ '_dynamicT2alt.npy'), allow_pickle = True)
			self.ParamT1alt = np.load(os.path.join(fpath, sName, sName+ '_dynamicT1.npy'), allow_pickle = True)
			self.ParamT2alt = np.load(os.path.join(fpath, sName, sName+ '_dynamicT2.npy'), allow_pickle = True)
		elif model_type == "Constant_Alt":
			self.ParamT1 = np.load(os.path.join(fpath, sName, sName+ '_T1alt.npy'), allow_pickle = True)
			self.ParamT2 = np.load(os.path.join(fpath, sName, sName+ '_T2alt.npy'), allow_pickle = True)
			self.ParamT1alt = np.load(os.path.join(fpath, sName, sName+ '_T1.npy'), allow_pickle = True)
			self.ParamT2alt = np.load(os.path.join(fpath, sName, sName+ '_T2.npy'), allow_pickle = True)
		else:
			self.ParamT1 = np.load(os.path.join(fpath, sName, sName+ '_T1.npy'), allow_pickle = True)
			self.ParamT2 = np.load(os.path.join(fpath, sName, sName+ '_T2.npy'), allow_pickle = True)
			self.ParamT1alt = np.load(os.path.join(fpath, sName, sName+ '_T1alt.npy'), allow_pickle = True)
			self.ParamT2alt = np.load(os.path.join(fpath, sName, sName+ '_T2alt.npy'), allow_pickle = True)

		# Load in remaining parameters from file
		self.chemShift = np.load(os.path.join(fpath, sName, sName+ '_chemShift.npy'), allow_pickle = True)
		T2prime = np.load(os.path.join(fpath, sName, sName+ '_T2prime.npy'), allow_pickle = True)

		# Pre-define matrix variables
		self.t2starOffsets = np.zeros(self.size4D)
		self.t2starOffsets = self.t2starOffsets.astype(object)
		
		# Calculate numer of vectors per pixel, and calculate the Chi values for each vector in T2star
		# simulations
		n = self.size4D[3]
		pts = np.linspace(self.t2starBounds[0],self.t2starBounds[1], n)

		# Iterate through sample magnetization, defining the static field in the Lab frame; defining 
		# T2star field offsets for each vector; and counting the number of vectors in the sample 
		for index in np.ndindex(self.M.shape):
			if isinstance(self.M[index], np.ndarray) is False:
				continue
			else:
				vNum = index[3]
				dOlist = (1/T2prime[index])*np.tan(np.pi*(pts-0.5))
				for i in range(int(n/2)):
					dOlist[n-i-1] = -dOlist[i]
				dO = dOlist[vNum]
				self.t2starOffsets[index] = np.array([0, 0, dO/(self.GAMMA)])
				self.count += 1

		self.equiM = copy.deepcopy(self.M)                      # Save Initial sample magnetization

		# Automatically set ParamB0 to BLab for ideal B0
		self.B0Func = field.IdealB0
		self.ParamB0 = (self.B0, self.t2starOffsets, self.chemShift)

		# Calculate Chunk Size based on number of vectors in sample
		n_cores = os.cpu_count()
		self.chunksize = mat.ceil(self.count/n_cores)

		# Add Sample information to printList
		self.printList.extend([f"Sample Name = {sName}", f"Model_Type = {model_type}", "\n"])


	#~~~~~ Internal Methods Used by User Facing Methods ~~~~~#

	def Reset_Variables(self):
		"""
		Method for resetting variables that define sequence elements to default values once a sequence 
		element has been simulated.

		Parameters:
		None
		"""
		self.Duration = 0
		self.B1 = 0
		self.additionalParams = {}
		self.BrfFunc = solver.Zero
		self.BgradFunc = solver.Zero
		self.ParamRF = (0)
		self.ParamGrad = (0)
		self.Int_Ends = np.array([0])
		self.timeSeries = np.array([0])
		self.frameRot = solver.zeroBzFrame
		self.ParamRotPlus = (self.B0, self.Larmor)

	def NetMagCalc(self, Mag):
		"""
		Method for calculating the total magnetization across the sample. Updates Experiment.netMag to 
		be the average of all vectors in Mag.

		Parameters: 
		Mag = Numpy array containing magnetization state to be summed.
		"""
		# Pre-define variables
		count = 0
		sumMag = np.array([0,0,0], dtype = float)

		# Iterate through all elements Mag. Ignore empty pixels, sum extant vectors and count number of
		# extant vectors
		for index in np.ndindex(Mag.shape):
			if isinstance(Mag[index], np.ndarray) is False:
				continue
			count += 1
			sumMag += Mag[index]

		# Calculate average magnetization
		self.netMag = sumMag/count

	def NetMagNorm(self, vectorSum):
		"""
		Method for calculating the total magnetization across the sample for many time points from the 
		vectorSum output of Solve(), then saves the resultant array in Experiment.netMag.

		Parameters: 
		vectorSum = Numpy array with dimensions of 3 by length(timeSeries). Contains net magnetization 
		of full sample at each time point specified in timeSeries. Rows respectively contain the x-, 
		y-, and z- components.
		"""
		self.netMag = vectorSum/self.count


	def variableFlipCalculator(self, nflip, TR, T1):
		"""
		Method for calculating a series of flip angles that will provide constant transverse magnetization in 
		short TR, spoiled gradient echo imaging.

		Maximizes available signal by constraining the final pulse in the sequence to flip_angle = 90 degrees.
		As this function depends on the state of magnetization immediately prior to the first excitation pulse,
		it must be called immediately prior to that initial pulse. Saves both the list of flip angles and the
		calculated transverse magnetization after each pulse as class variables.

		Parameters: 
		nflip = Number of flip angles
		TR = Repetiton time between excitation pulses in seconds
		T1 = T1 of the sample in seconds
		"""
		# Define output array
		self.flipAngleArray = np.zeros([nflip])
		self.MxyArray = np.zeros([nflip])

		# Calculate relaxation coefficient
		E1 = np.exp(-TR/T1)

		# Calculate current and equilibrium netMag
		self.NetMagCalc(self.M)
		MzInit = self.netMag[2]
		self.NetMagCalc(self.equiM)
		M0 = self.netMag[2]

		# Set up variables for incrementing inital angle
		increment = 1           # 1 degree
		initAngle = 0

		while self.flipAngleArray[-1] < 89.9:

			initAngle += increment

			# Convert guess angle to radians and calculate inital points
			self.flipAngleArray[0] = np.radians(initAngle)
			self.MxyArray[0] = MzInit*np.sin(self.flipAngleArray[0])
			Mz = MzInit*E1*np.cos(self.flipAngleArray[0])+ M0*(1 - E1)

			# Iteratively calulate further flips
			for i in range(1, nflip):

				# Ignore NaN warning from arcsin when trying to find angles
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", message="invalid value encountered in arcsin")

					denom = (1 - E1) + Mz*E1*np.cos(self.flipAngleArray[i-1])
					func = Mz*np.sin(self.flipAngleArray[i-1])/denom
					self.flipAngleArray[i] = np.arcsin(func)

					self.MxyArray[i] = Mz*np.sin(self.flipAngleArray[i])
					Mz = Mz*E1*np.cos(self.flipAngleArray[i]) + 1 - E1

			self.flipAngleArray = np.degrees(self.flipAngleArray)

			# If final element is nan (due to needing a flip angle >90), decrease angle by increment,
			# decrease increment step size, and set element checked by while loop to zero
			if np.isnan(self.flipAngleArray[-1]) == True:
				initAngle = initAngle - increment
				increment = increment/10
				self.flipAngleArray[-1] = 0

		# Set final pulse to full 90 degrees
		self.flipAngleArray[-1] = 90

		# Save element parameters to text output
		self.printList.extend(['Variable Flip Angles Calculated:', f"flipAngleArray = {self.flipAngleArray}", 
						 f"MxyArray = {self.MxyArray}", "\n"])
		
		# Print status update
		print("Flip Angles Calculated")
		


	def RF_Calibrate(self):
		"""
		Method used by RF_Pulse() and other RF sequence methods to set the amplitude for the current 
		RF pulse:

		If B1 is specified when RF_Pulse() is called, RF amplitude is set to the given B1 value.
		If B1 has not been specified and Experiment.B90 has been calculated by an earlier instance 
			of RF_Pulse(), RF amplitude calculated based on the value stored in Experiment.B90 and 
			the desited flip angle.
		If B1 has not been specified and Experiment.B90 has not yet been calculated, runs a simple 
			search algorithm to determine field strength needed to achieve a 90 degree flip angle by 
			maximizing Bxy. This value is then saved in Experiment.B90, and RF amplitude is calculated 
			based on Experiment.B90.

		Once RF amplitude is determined using one of the above methods, RF_Calibrate() constructs 
		the ParamRF tuple, which is used by the solving code to calculate the waveform of the RF pulse.

		Parameters:
		None
		"""
		# Identify if B1 was set by the user when RF_Pulse was called
		if self.B1 != 0:
			pass

		# Otherwise, identify if B90 has been calculated by a proeious call to RF_Pulse()
		elif self.B90 != 0:
			self.B1 = self.B90*(self.Flip_Angle/90)

		# Otherwise, run search algorithm to determine B90
		else:
			# Predefine variables
			self.M_Cal = 0				# Copy of M for calibration simulations
			maximaFlag = False			# Flag for identifying when maximum Bxy is reached
			B90 = 0						# RF amplitude
			prev = 0					# Result from previous iteration of solver
			prev2 = 0					# Result from two iterations of solver
			step = 10e-6				# Initial (maximum) field difference between iterations

			print('Calculating B90 Amplitude...')

			# Iterating through simulations 
			while maximaFlag is False:
				# Add step to B90 from previous simulation
				B90 += step 
				# Define RF parameters with new B90
				self.ParamRF = (self.Duration, B90, self.freq, self.Phi, self.nLobes, self.additionalParams)
				# Set inital mangetization equal to equilibrium
				self.M_Cal = copy.deepcopy(self.equiM) 
				# Solves for state of equiM after RF Pulse and calculates total magnetization along Bxy
				self.M_Cal, vectorSum = solver.Solve(self.GAMMA, self.Int_Ends, self.timeSeries, self.M,
							self.relaxFunc, self.B0Func, self.BrfFunc, self.BgradFunc, self.RxFunc, self.frameRot,
							self.ParamT1, self.ParamT2, self.ParamB0, self.ParamRF, self.ParamGrad, self.ParamRx,
							self.ParamRot, self.ParamRotPlus, self.equiM, self.chunksize, self.max_workers)

				# Calculate net magnetization in the transvese plane
				self.NetMagNorm(vectorSum)
				current = abs(np.linalg.norm(self.netMag[0:2]))

				# If Bxy from most recent iteraton is larger than previous iteration, increase B1 strength
				# and continue search
				if current > prev:
					print(f"Transverse Mag = {current} @ B90 = {B90} T")
					prev2 = prev
					prev = current

				# If Bxy from most recent iteraton is smaller than previous iteration, and step size is 
				# 1e-7 T, stop search
				elif current <= prev and step <= 1e-6:
					B90 += -step
					maximaFlag = True
					print('Done!')

				# If Bxy from most recent iteraton is smaller than previous iteration, reduce B1 strength to
				# prior value, decrease step size and continue search
				elif current <= prev and maximaFlag is False:
					print(f"Transverse Mag = {current} @ B90 = {B90} T")
					B90 += -2*step
					step = step/10
					current = prev2
					prev = prev2
					print(f'Reducing Step Size....')

			# Saving calculated 90-degree pulse amplitude to file
			self.B90 = B90
			self.B180 = 2*B90

			# Determine B1 value for the flip angle of the current RF pulse
			self.B1 = self.B90*(self.Flip_Angle/90)
			print(f'B1 = {self.B1} T')

			# Calculate and output time taken to perform calibrations
			computeTime = time.datetime.now() - self.startTime
			self.printList.extend([f"B90 Calibrated, Total Compute Time = {computeTime}"])

		# Set ParamRF tuple for calculating RF waveform  using final RF pulse amplitude
		self.ParamRF = (self.Duration, self.B1, self.freq, self.Phi, self.nLobes, self.additionalParams)

	def Acquire(self):
		"""
		Method used by Acquisition() and other methods that simulate signal acquisition to solve for 
		signal and to convert sample magnetization into complex signal. Stores resulting signal in the 
		Experiment.Readout variable.

		Parameters:
		None
		"""
		print(f'Acquisition: nPoints = {self.nPoints}')

		# Evolve magnetization
		self.M, vectorSum = solver.Solve(self.GAMMA, self.Int_Ends, self.timeSeries, self.M,
						self.relaxFunc, self.B0Func, self.BrfFunc, self.BgradFunc, self.RxFunc, self.frameRot,
						self.ParamT1, self.ParamT2, self.ParamB0, self.ParamRF, self.ParamGrad, self.ParamRx,
						self.ParamRot, self.ParamRotPlus, self.equiM, self.chunksize, self.max_workers)
		
		# Calculate net magnetization of the sample at each output timepoint
		self.NetMagNorm(vectorSum)

		# Create a complex time-domain signal where the real component is the x-component 
		# of net magnetization, and the imaginary component is the y-component of net magnetization
		Readout = []
		for i in range(self.nPoints):
			Readout.append(complex(self.netMag[0][i], self.netMag[1][i]))

		self.Readout = np.array(Readout)


	#~~~~~ Methods for Defining and Executing Pulse Sequence Elements ~~~~~#

	def Deadtime(self, Duration):
		"""
		Method for evolving sample magnetization under the sole influence of B0

		Parameters:
		Duration = Length of deadtime (s)
		"""
		# Load inputs as class variables
		self.Duration = Duration
		self.Int_Ends = (0, self.Duration)
		self.timeSeries = np.array([self.Duration])
		print(f'Evolving over Deadtime: Dur = {self.Duration} s')

		# Evolve magnetization
		self.M, vectorSum = solver.Solve(self.GAMMA, self.Int_Ends, self.timeSeries, self.M,
						self.relaxFunc, self.B0Func, self.BrfFunc, self.BgradFunc, self.RxFunc, self.frameRot,
						self.ParamT1, self.ParamT2, self.ParamB0, self.ParamRF, self.ParamGrad, self.ParamRx,
						self.ParamRot, self.ParamRotPlus, self.equiM, self.chunksize, self.max_workers)
		
		# Save element parameters to text output
		self.printList.extend(['Deadtime', f"Duration = {self.Duration}", "\n"])

		# Reset variables prior to next sequence element
		self.Reset_Variables()

	def RF_Pulse(self, Duration, BrfFunc, Flip_Angle, Phi = 0, dOmega = 0, nLobes = 5, B1 = 0, **kwargs):
		"""
		Method for evolving sample magnetization under the sole influence of an RF field and B0.

		Parameters:
		Duration = Length of pulse (s)
		BrfFunc = Function to define RF pulse waveform
		Flip Angle = Desired rotation of sample magnetization (Degrees). Default value 90.
		Phi = Phase factor, determines axis along which pulse is applied. +x-axis is
			at 0 degrees, +y-axis at 90 degrees (Degrees). Default value 0.
		nLobes = Additional variable for waveform definition. Default value 5.
		dOmega = Frequency offset of RF Pulse. Input in units of PPM of Larmor Frequency, stored in Hz. 
			Default value 0.
		B1 = Maximum amplitude of B1 pulse (Tesla). Default value 0.
		"""
		# Load inputs as class variables
		self.Duration = Duration
		self.Int_Ends = (0, self.Duration)
		self.timeSeries = np.array([self.Duration])
		self.Flip_Angle = Flip_Angle
		self.Phi = -np.deg2rad(Phi)
		self.nLobes = nLobes
		self.dOmega = dOmega
		self.freq = self.Larmor+ self.dOmega
		self.B1 = B1
		self.BrfFunc = BrfFunc
		self.additionalParams = kwargs

		# Set frameRot to keep solving in the rotating reference frame, increases speed of RF simulation
		self.frameRot = solver.Identity

		# Determine RF field amplitude
		self.RF_Calibrate()
		print(f'RF Pulse: Angle = {self.Flip_Angle} deg, Dur = {self.Duration} s')

		# Evolve magnetization
		self.M, vectorSum = solver.Solve(self.GAMMA, self.Int_Ends, self.timeSeries, self.M,
						self.relaxFunc, self.B0Func, self.BrfFunc, self.BgradFunc, self.RxFunc, self.frameRot,
						self.ParamT1, self.ParamT2, self.ParamB0, self.ParamRF, self.ParamGrad, self.ParamRx,
						self.ParamRot, self.ParamRotPlus, self.equiM, self.chunksize, self.max_workers)
		
		# Save element parameters to text output
		self.printList.extend(['RF_Pulse', f"Duration = {self.Duration}", f"Flip_Angle = {self.Flip_Angle}",
								f"Phi = {self.Phi}", f"Shape = {self.BrfFunc}", f"nLobes = {self.nLobes}",
								f"B1 = {self.B1}", f"dOmega = {self.dOmega}", 
								f"additionalParams = {self.additionalParams}", "\n"])
		
		# Reset variables prior to next sequence element
		self.Reset_Variables()

	def Gradient(self, Duration, BgradFunc, gradAmp = [0, 0, 0], riseTime = 0, **kwargs): 
		"""
		Method for evolving sample magnetization under the sole influence of magnetic gradients and B0.

		Parameters:
		Duration = Length of gradient (s)
		BgradFunc = Function that defines gradient waveform
		gradAmp = Maximum amplitude of applied gradients in a 3 element list [x, y, z] (T/m). Default 
			value [0, 0, 0].
		riseTime = Time taken for gradient to ramp up to maximum amplitude in T/m/s. Default value 0.
		gradParam1 = Additional variable for waveform definition. Default value 0.
		gradParam2 = Additional variable for waveform definition. Default value 0.
		"""
		# Load inputs as class variables
		self.Duration = Duration
		self.Int_Ends = (0, self.Duration)
		self.timeSeries = np.array([self.Duration])
		self.BgradFunc = BgradFunc
		self.gradAmp = np.array(gradAmp, dtype = float)
		self.riseTime = riseTime
		self.additionalParams = kwargs
		self.ParamGrad = (self.MPosition, self.gradAmp, self.sizeMetric, self.Duration, 
					self.riseTime, self.additionalParams)
		print(f'Gradient: Amp = {self.gradAmp} T, Dur = {self.Duration} s')

		# Evolve magnetization
		self.M, vectorSum = solver.Solve(self.GAMMA, self.Int_Ends, self.timeSeries, self.M,
						self.relaxFunc, self.B0Func, self.BrfFunc, self.BgradFunc, self.RxFunc, self.frameRot,
						self.ParamT1, self.ParamT2, self.ParamB0, self.ParamRF, self.ParamGrad, self.ParamRx,
						self.ParamRot, self.ParamRotPlus, self.equiM, self.chunksize, self.max_workers)
		
		# Save element parameters to text output
		self.printList.extend(['Gradient', f"Duration = {self.Duration}", f"gradAmp = {self.gradAmp}",
								f"riseTime = {self.riseTime}", f"additionalParams = {self.additionalParams}", 
								"\n"])
		
		# Reset variables prior to next sequence element
		self.Reset_Variables()

	def Acquisition(self, Duration, nPoints):
		"""
		Method for evolving sample magnetization and acquiring MR signal under the sole influence of B0.

		Parameters:
		Duration = Length of acquisition window (s)
		nPoints = Number of points in acquisition
		"""
		# Load inputs as class variables
		self.Duration = Duration
		self.Int_Ends = (0, self.Duration)
		self.nPoints = nPoints
		self.timeSeries = np.linspace(0, self.Duration, self.nPoints)

		# Evolve magnetization
		self.Acquire()

		# Save element parameters to text output
		self.printList.extend(['Acquisition', f"Duration = {self.Duration}", f"nPoints = {self.nPoints}",
								f"timeSeries = {self.timeSeries}", "\n"])
		
		# Reset variables prior to next sequence element
		self.Reset_Variables()

	def FreqEncode(self, Duration, BgradFunc, nPoints, gradAmp = [0, 0, 0], riseTime = 0, **kwargs):
		"""
		Method for evolving magnetization and acquiring MR signal under the influence of a magnetic 
		gradient and B0.

		Parameters:
		Duration = Length of gradient (s)
		BgradFunc = Function that defines gradient waveform
		nPoints = Number of points in acquisition
		gradAmp = Maximum amplitude of applied gradients in a 3 element list [x, y, z] (T/m). Default 
			value [10e-3, 0, 0].
		riseTime = Time taken for gradient to ramp up to maximum amplitude. Default value 0.
		gradParam1 = Additional variable for waveform definition. Default value 0.
		gradParam2 = Additional variable for waveform definition. Default value 0.
		"""
		# Load inputs as class variables
		self.Duration = Duration
		self.Int_Ends = (0, self.Duration)
		self.nPoints = nPoints
		self.timeSeries = np.linspace(0, self.Duration, self.nPoints)
		self.gradAmp = np.array(gradAmp, dtype = float)
		self.BgradFunc = BgradFunc
		self.riseTime = riseTime
		self.additionalParams = kwargs
		self.ParamGrad = (self.MPosition, self.gradAmp, self.sizeMetric, self.Duration, self.riseTime,
							self.additionalParams)
		print(f'Readout Gradient: Amp = {self.gradAmp} T, Dur = {self.Duration} s')
		

		# Evolve magnetization
		self.Acquire()

		# Save element parameters to text output
		self.printList.extend(['FreqEncode', f"Duration = {self.Duration}", f"nPoints = {self.nPoints}",
								f"gradAmp = {self.gradAmp}", f"timeSeries = {self.timeSeries}", 
								f"riseTime = {self.riseTime}", f"additionalParams = {self.additionalParams}",
								"\n"])
		
		# Reset variables prior to next sequence element
		self.Reset_Variables()

	def SliceSelectRF(self, Duration, BrfFunc, BgradFunc, Flip_Angle = 90, Phi = 0, nLobes = 5,
				   dOmega = 0, B1 = 0, gradAmp = [0, 0, 0], riseTime = 0, **kwargs):
		"""
		Method for evolving magnetization and acquiring MR signal under the influence of simultaneous RF 
		fields, magnetic gradients and B0.

		*** WIP NOT VALIDATED ***

		Parameters:
		Duration = Length of gradient (s)
		BrfFunc = Function to define RF pulse waveform
		BgradFunc = Function that defines gradient waveform
		Flip Angle = Desired rotation of sample magnetization (Degrees). Default value 90.
		Phi = Phase factor, determines axis along which pulse is applied. +x-axis is at 0 degrees, +y-axis 
			at 90 degrees (Degrees). Default value 0.
		nLobes = Additional variable for waveform definition. Default value 5.
		dOmega = Frequency offset of RF Pulse. Input in units of PPM of Larmor Frequency, stored in Hz. 
			Default value 0.
		B1 = Maximum amplitude of B1 pulse (Tesla). Default value 0.
		gradAmp = Maximum amplitude of applied gradients in a 3 element list [x, y, z] (T/m). Default 
			value [10e-3, 0, 0].
		riseTime = Time taken for gradient to ramp up to maximum amplitude. Default value 0.
		gradParam1 = Additional variable for waveform definition. Default value 0.
		gradParam2 = Additional variable for waveform definition. Default value 0.
		"""
		# Load RF inputs as class variables
		self.Duration = Duration
		self.Int_Ends = (0, self.Duration)
		self.timeSeries = np.array([self.Duration])
		self.Flip_Angle = Flip_Angle
		self.Phi = -np.deg2rad(Phi)
		self.nLobes = nLobes
		self.dOmega = dOmega
		self.freq = self.Larmor+ self.dOmega
		self.B1 = B1
		self.BrfFunc = BrfFunc

		# Set frameRot to keep solving in the rotating reference frame, increases speed of RF simulation
		self.frameRot = solver.Identity
		# Determine RF field amplitude
		self.RF_Calibrate()

		# Load Gradient inputs as class variables
		self.gradAmp = np.array(gradAmp, dtype = float)
		self.BgradFunc = BgradFunc
		self.riseTime = riseTime
		self.additionalParams = kwargs
		self.ParamGrad = (self.MPosition, self.gradAmp, self.sizeMetric, self.Duration, self.riseTime,
							self.additionalParams)
		print(f'Selective RF Pulse: Angle = {self.Flip_Angle} deg, Dur = {self.Duration} s')

		# Evolve magnetization
		self.M, vectorSum = solver.Solve(self.GAMMA, self.Int_Ends, self.timeSeries, self.M,
						self.relaxFunc, self.B0Func, self.BrfFunc, self.BgradFunc, self.RxFunc, self.frameRot,
						self.ParamT1, self.ParamT2, self.ParamB0, self.ParamRF, self.ParamGrad, self.ParamRx,
						self.ParamRot, self.ParamRotPlus, self.equiM, self.chunksize, self.max_workers)
		
		# Save element parameters to text output
		self.printList.extend(['SliceSelectRF', f"Duration = {self.Duration}", f"Flip_Angle = {self.Flip_Angle}",
								f"Phi = {self.Phi}", f"Shape = {self.BrfFunc}", f"nLobes = {self.nLobes}",
								f"B1 = {self.B1}", f"dOmega = {self.dOmega}",
								f"gradAmp = {self.gradAmp}", f"timeSeries = {self.timeSeries}", 
								f"riseTime = {self.riseTime}", f"additionalParams = {self.additionalParams}",
								"\n"])
		
		# Reset variables prior to next sequence element
		self.Reset_Variables()

	def SpinLock_Pulse(self, Duration, BrfFunc, slRelaxFunc, Phi = 0, AmpHz = 500, nLobes = -1, dOmega = 0, **kwargs):
		"""
		Method for evolving sample magnetization under a spin-lock RF pulse. Automatically uses relaxation 
		parameters loaded into Experiment.T1alt and Experiment.T2alt

		Parameters:
		Duration = Length of pulse (s)
		BrfFunc = Function to define RF pulse waveform
		slRelaxFunc = Function for calculating relaxation in the doubly rotating reference frame
		Phi = Phase factor, determines axis along which the spin lock pulse is applied. Assumes spin lock 
			pulse is applied in the same direction as the net transverse magnetization vector, thus should 
			be given the Phi value of the excitation pulse. (Degrees). Default value 0.
		AmpHz = Desired amplitude of spin lock pulse in Hz (gets converted to Tesla for solver). Default 
			value 500.
		nLobes = Additional variable for waveform definition. Default value -1.
		dOmega = Frequency offset of RF Pulse, in PPM of Larmor Frequency. Default value 0.
		"""
		# Load inputs as class variables
		self.Duration = Duration
		self.Int_Ends = (0, self.Duration)
		self.timeSeries = np.array([self.Duration])
		self.slRelaxFunc = slRelaxFunc
		self.Phi = -np.deg2rad(Phi- 90)				# Offsets given Phi so B1 is applied along magnetization
		self.B1 = 2*np.pi*AmpHz/self.GAMMA			# Calulates B1 using AmpHz
		self.nLobes = nLobes
		self.additionalParams = kwargs
		self.dOmega = dOmega
		self.ParamRotPlus = self.Phi
		self.BrfFunc = BrfFunc
		self.freq = self.Larmor+ self.dOmega
		self.ParamRF = (self.Duration, self.B1, self.freq, self.Phi, self.nLobes, self.additionalParams)
		
		# Tell solver to solve in the double-rotating reference frame
		self.frameRot = solver.SpinLock_Rotator
		print(f'SpinLock Pulse: Amp = {AmpHz} Hz, Dur = {self.Duration} s')

		# Set new equilibrium magnetization based on B1
		equilM_T1rho = copy.deepcopy(self.equiM)*(self.B1/self.B0[2])

		# Evolve magnetization
		self.M, vectorSum = solver.Solve(self.GAMMA, self.Int_Ends, self.timeSeries, self.M,
						self.slRelaxFunc, self.B0Func, self.BrfFunc, self.BgradFunc, self.RxFunc, self.frameRot,
						self.ParamT1alt, self.ParamT2alt, self.ParamB0, self.ParamRF, self.ParamGrad, self.ParamRx,
						self.ParamRot, self.ParamRotPlus, equilM_T1rho, self.chunksize, self.max_workers)
		
		# Save element parameters to text output
		self.printList.extend(['SpinLock_Pulse', f"Duration = {self.Duration}",
								f"Phi = {Phi-90}", f"AmpHz = {AmpHz}", f"dOmega = {self.dOmega}", "\n"])
		
		# Reset variables prior to next sequence element
		self.Reset_Variables()



	#~~~~~ Noise Calculation Methods ~~~~~#

	def importImageCSV(self, signalFile):
		"""
		Imports image from .csv file, to enable noise addition to previously simulated data.
		
		Parameters:
		signalFile = String containing name of file to be created
		"""

		self.image = np.genfromtxt(signalFile, dtype = complex, delimiter = ',')


	def noiseCalculatorSignalSpace(self, noiseFunc, noiseParams, nAcq):
		"""
		Adds noise to acquired signal using user-defined noise function. Noise is calculated for complex
		k-space signal.

		Parameters:
		noiseFunc = Function to perform noise calculation
		noiseParams = Tuple containing any additional parameters needed to calculate noise using noiseFunc
		nAcq = Number of acquisitions for noise calculations
		"""
		shape = self.signal.shape
		self.noisySignal = np.zeros(shape, dtype = 'complex')

		noise = noiseFunc(noiseParams, self.signal, shape, self.B0, nAcq, self.GAMMA, self.rng)
		self.noisySignal = self.signal+ noise

	def noiseCalculatorImageSpace(self, noiseFunc, noiseParams, nAcq):
		"""
		Adds noise to existing image using user-defined noise function. Noise is calculated for complex 
		k-space signal.

		Parameters:
		noiseFunc = Function to perform noise calculation
		noiseParams = Tuple containing any additional parameters needed to calculate noise using noiseFunc
		"""
		shape = self.image.shape
		self.noisyImage = np.zeros(shape, dtype = 'complex')

		noise = noiseFunc(noiseParams, self.image, shape, self.B0, nAcq, self.GAMMA, self.rng)
		self.noisyImage = self.image+ noise



	#~~~~~ Methods for converting k-space data to images ~~~~~#

	def importSignalCSV(self, signalFile):
		"""
		Imports simulated signal from file.

		Inputs:
		signalFile = Filename of .csv file to be imported
		"""

		self.signal = np.genfromtxt(signalFile, dtype = complex, delimiter = ',')

	def imageProcessing(self, nDim, seqName, sampName, savePath, saveFlagCSV = False, saveFlagIMG = False,
					hannFlag = False, gradAmp = [], noisySignal = False):
		"""
		Converts 1D or 2D k-space signal to an image using an FFT. Optionally, can apply Hanning filtering
		to the data prior to conversion to image space (see Brown et. Al., Magnetic Resonance Imaging: 
		Physical Principles and Sequence Design, 2nd Ed., Chapter 13).

		Parameters:
		nDim = Dimensions of signal (1 or 2)
		seqName = String identifying sequence used to generate final filename
		sampName = String identifying sample used to generate final filename
		savePath = String containing path to save location
		saveFlagCSV = Control variable, if 'True', saves k-space magnitude data and the final image as .csv 
			files. Default value 'False'.
		saveFlagIMG = Control variable, if 'True', saves k-space data, k-space magnitude data, and the final 
			image to .png files. Default value 'False'.
		HannFlag = Control variable, if 'True', applies a Hanning filter to k-space data prior to FFT. 
			Default value 'False'.
		gradAmp = Maximum amplitude of applied gradients along each axis in a 3 element list [x, y, z] (T/m)
		signalType = Control variable, if 'True' reads k-space data from Experiment.noisySignal instead of 
			Experiment.signal. Default value 'False'.
		"""
		# Construct output file name 
		timestr = self.startTime.strftime("%Y-%m-%d_%Hh%Mm")
		fileName = f"{seqName}_{sampName}_{timestr}"

		# Identify input variable
		if noisySignal == True:
			data = self.noisySignal
		else:
			data = self.signal

		# Calculate and apply Hanning filter to signal
		shape = data.shape
		if hannFlag == True:
			grad0 = np.pi*np.linspace(-gradAmp[0], gradAmp[0], shape[0])
			hann0 = np.square(np.cos(grad0/(2*gradAmp[0])))

			grad1 = np.pi*np.linspace(-gradAmp[1], gradAmp[1], shape[1])
			hann1 = np.square(np.cos(grad1/(2*gradAmp[1])))

			hann = hann0*hann1.reshape(-1,1)
			self.signal = data*hann

			fileName = fileName + "_hannFiltered"

		# Calculate, plot, and save signal magnitude data
		signalMagnitude = np.abs(data)

		# Apply FFT to data
		if nDim == 1:
			self.imageData = np.fft.ifft(self.signal)
			self.image = abs(np.fft.fftshift(self.imageData))
		elif nDim == 2:
			self.imageData = np.fft.ifft2(self.signal)
			self.image = abs(np.fft.fftshift(self.imageData))

		# Save k-space magnitude data and image as .csv files
		if saveFlagCSV == True:
			np.savetxt(savePath+ f"/kMag_{fileName}.csv", signalMagnitude, delimiter=",")
			np.savetxt(savePath+ f"/Image_{fileName}.csv", self.image, delimiter=",")

		# Save k-space data, k-space magnitude data, and image as .png images
		if saveFlagIMG == True:

			if nDim == 1:

				# Set figure size
				plt.rcParams["figure.figsize"] = (10,6)

				# Plot Kspace data
				fig = plt.figure(1)
				# Set axes to share scale
				ax1 = fig.add_subplot(1,2,1)
				ax2 = fig.add_subplot(1,2,2, sharex = ax1)
				# Set whichever signal component has greater magnitude to first subplot
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
				plt.savefig(savePath+ f"/kSpace_{fileName}.png")

				# Plot Kspace magnitude data
				plt.figure(2)
				plt.plot(signalMagnitude)
				plt.savefig(savePath+ f"/kMag_{fileName}.png")

				# Plot Image
				plt.figure(3)
				plt.plot(self.image)
				plt.savefig(savePath+ f"/Image_{fileName}.png")

			elif nDim == 2:

				plt.rcParams["figure.figsize"] = (10,6)

				# Plot Kspace data
				plt.figure(1)
				fig, (ax1, ax2) = plt.subplots(1,2)
				# Set axes to share scale
				ax1 = fig.add_subplot(1,2,1)
				ax1.set_axis_off()
				ax2 = fig.add_subplot(1,2,2, sharex = ax1)
				ax2.set_axis_off()
				# Set whichever signal component has greater magnitude to first subplot
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
				plt.savefig(savePath+ f"/kSpace_{fileName}.png")

				# Plot Kspace magnitude data
				plt.figure(2)
				plt.imshow(signalMagnitude)
				plt.savefig(savePath+ f"/kMag_{fileName}.png")

				# Plot Image
				plt.figure(3)
				plt.imshow(self.image, cmap = "gray")
				plt.savefig(savePath+ f"/Image_{fileName}.png")

				
		



