import copy
import concurrent.futures

import numpy as np

from scipy.integrate import solve_ivp


#~~~~~ Basic Code Layout ~~~~~#
# Simulation code for the Numeric Integrator for the Bloch Equations (NIBLEs)
# 
# This file contains the code for evolving magnetization through numeric solving of the Bloch equations.
# Numeric solving is performed using scipy.Solve_IVP, and multi-core parallelization is performed using
# Python's concurrent.futures module. Enables solving the Bloch equations in an arbitrary reference 
# frame; using time and spatially dependent magnetic fields; and using time and field dependent 
# relaxation values.
#
# Original version of this code designed and written by John Adams (jadam33@uwo.ca)
#
# This code is published under a Creative Commons BY-SA 4.0 License. Anyone using this code must give 
# attribution to the original author (John Adams), and must make any derivative code built using this
# code available under the same licensing terms.

#~~~~~ Version History ~~~~~#
# 1.0  Public Release - DATE - John Adams

#~~~~~ Functions for Magnetic Field Shaping ~~~~~~#

def Zero(t, *args, **kwargs):
	"""
	Function that returns Zero regardless of input, used within NIBLEs as a placeholder for
	user-specified functions that have not been assigned
	"""
	return 0

def Identity(X, *args, **kwargs):
	"""
	Function that returns the first value it is given regardless of other inputs, used within NIBLEs
	when reference frame rotations away from the rotating reference frame are not needed
	"""
	return X

def rotatingReferenceFrame(BVec, t, Param):
	"""
	Function for converting magnetic field vectors from the laboratory reference frame into the rotating 
	reference frame

	Parameters:
	BVec = Vector to be rotated, formatted as a 3-element Numpy array
	t = Simulation time. Produced by solver and should not be defined by end user
	Param == (B0, Larmor)
		Parameter tuple: the first element is the main field strength defined as a 1D, 3-element Numpy
		array; the second element is the Larmor frequency of the nucleus being imaged 
	Returns:
	vOut = vec translated into the rotating reference frame
	"""

	# Calculate Bz in rotating reference frame by subtracting nominal B0z from Bz
	BRotz = BVec[2]- Param[0][2]

	# Calculate rotation matrix based on larmor frequency and time
	Rmat = np.array([[np.cos(-Param[1]*t), np.sin(-Param[1]*t), 0],
					 [-np.sin(-Param[1]*t), np.cos(-Param[1]*t), 0],
					 [0, 0, 1]])
	
	# Apply rotation matrix and return final field
	BRot = np.matmul(Rmat, BVec)
	BRot[2] = BRotz

	return BRot

def zeroBzFrame(Vec, t, Param, TotBRot, dirForeward, ifB):
	"""
	Function for converting vectors from the rotating reference frame into a rotating reference frame where
	Bz = 0.

	Parameters:
	Vec = Vector to be rotated, formatted as a 3-element Numpy array
	t = Simulation time. Produced by solving function and should not be defined by end user
	Param == (B0, Larmor)
		Parameter tuple: the first element is the main field strength defined as a 1D, 3-element Numpy array;
		the second element is the Larmor frequency of the nucleus being imaged 
	TotBRot = Total magnetic field in the Larmor rotating reference frame
	dirForeward = Control variable for direction of transform. If set to True, will convert away from 
		Larmor rotating reference frame. If false, will convert back to Larmor reference frame
	ifB = Control variable to denote if given vector is a magnetic field. If true, will adjust the amplitude 
		of the given vector's z-component based on the frame transformation. 

	Returns:
	rVec = Vec translated to/from the reference frame where Bz is zero
	"""
	# Calculate the frequency difference between the Larmor frequency and the zeroBz frame
	omega = Param[1]/Param[0][2]*TotBRot[2]

	# omega = 2*np.pi*42.58e6*TotBRot[2]

	# Calculate rotation matrix and its inverse
	Rmat = np.array([[np.cos(-omega*t), np.sin(-omega*t), 0],
					 [-np.sin(-omega*t), np.cos(-omega*t), 0],
					 [0, 0, 1]])
	
	invRmat = np.linalg.inv(Rmat)
	
	# Apply appropriate rotation matrix to Vec based on the control variables, adjusting the z component
	# if Vec is a magnetic field
	if dirForeward == True and ifB == False:
		rVec = np.matmul(Rmat, Vec)
	elif dirForeward == True and ifB == True:
		rVec = np.matmul(Rmat, Vec)
		rVec[2] = 0
	elif dirForeward == False and ifB == False:
		rVec = np.matmul(invRmat, Vec)
	elif dirForeward == False and ifB == True:
		rVec = np.matmul(invRmat, Vec)
		rVec[2] = TotBRot[2]

	return rVec

def SpinLock_Rotator(Vec, t, Phi, TotBRot, dirForeward, ifB):
	"""
	Function for converting vectors from the rotating reference frame into the doubly-rotating reference 
	frame for T1rho/T2rho experiments.

	Parameters:
	Vec = Vector to be rotated, formatted as a 3-element Numpy array
	t = Simulation time. Produced by solver and should not be defined by end user
	Param == (B0, Larmor)
		Parameter tuple: the first element is the main field strength defined as a 1D, 3-element Numpy 
		array; the second element is the Larmor frequency of the nucleus being imaged 
	TotBRot = Not used in calculations, but present as input to maintain standard format for supplemental 
	rotation functions.
	dirForeward = Control variable for the direction of the transformation. If set to True, will convert 
	away from Larmor frame. If false, will convert back to Larmor reference frame
	ifB = Not used in calculations, but present as input to maintain standard format for supplemental 
	rotation functions.

	Returns:
	rVec = Vec translated to/from the reference frame where Bz is zero
	"""
	rVec = np.zeros(3).reshape(-1,1)

	# Define rotation matrix to rotate RF field to the xz-plane
	Rz = np.array([[np.cos(Phi), -np.sin(Phi), 0],
					[np.sin(Phi), np.cos(Phi), 0],
					[0, 0, 1]])
	# Define rotation matrix to rotate x magnetization to lie along z-axis
	Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
	
	# Apply appropriate rotation based on dirForeward
	if dirForeward == True:
		rVec = np.matmul(Rz, Vec)
		rVec = -np.matmul(Ry, rVec)
	else:
		invRz = np.linalg.inv(Rz)
		invRy = np.linalg.inv(Ry)
		rVec = -np.matmul(invRy, Vec)
		rVec = np.matmul(invRz, rVec)
	
	return rVec

#~~~~~ Functions for Bloch Solving ~~~~~#


def Bloch(t, MagSolve, index, GAMMA, relaxFunc, B0Func, BrfFunc, BgradFunc, frameRot, 
	ParamT1, ParamT2, ParamB0, ParamRF, ParamGrad, ParamRot, ParamRotPlus, equiM):

	"""
	Implementation of the Bloch Equations in a form which can be solved by SciPy's Solve_IVP function.
	Uses a variety of input functions and parameters to perform dynamic field and relaxation calculations.
	Solves Bloch equations in the rotating reference frame by default, contains additional processing step 
	to allow for solving in an arbitrary reference frame.

	Inputs:
	t = Simulation time (calculated by Solve_IVP, not user input variable)
	Mag = Magnetization vector to be evolved
	index = Index value of Mag in Experiment.M (See Experiment Class for more details)
	GAMMA = Gyromagnetic ratio of nucleus under study
	relaxFunc = Function for calculating relaxation times (See Relaxation Time Calculation Functions
		for more details)
	B0Func, BrfFunc, BgradFunc =  Functions for calculating applied magnetic fields (See Magnetic Field
		Calculation Functions for more details)
	frameRot = Function for calculating transformations away from the rotating reference frame 
	ParamT1, ParamT2, ParamB0, ParamRF, ParamGrad, ParamRot, ParamRotPlus = Tuples containing additional
		information for Relaxation, Applied Field, and Frame Rotation functions. For further details, see
		documentation for the corresponding functions.
	equiM = Equilibrium state for Mag

	Returns:
	Bloch = Mag after being evolved using the Bloch Equations
	"""

	# Sum input magnetic fields in Lab frame
	TotBLabInitial = B0Func(t, index, ParamB0)+ BrfFunc(t, index, ParamRF)+ BgradFunc(t, index, ParamGrad)
	TotBLab = TotBLabInitial+ (TotBLabInitial*ParamB0[2][index]*1e-6)+ ParamB0[1][index]
	# Transform input field to Larmor Rotating Reference Frame
	TotBRot = rotatingReferenceFrame(TotBLab, t, ParamRot)
	# Transform input field and mangetization into the Solving Reference Frame
	TotBSolve = frameRot(TotBRot, t, ParamRotPlus, TotBRot = TotBRot, dirForeward = True, ifB = True)

	# Calculate T1 and T2 values based on applied mangetic field
	T1, T2 = relaxFunc(TotBLab, TotBRot, TotBSolve, ParamT1, ParamT2)

	# Implementation of the Bloch Equations
	Blochx = GAMMA*(TotBSolve[2]*MagSolve[1]- TotBSolve[1]*MagSolve[2])- MagSolve[0]/T2
	Blochy = GAMMA*(TotBSolve[0]*MagSolve[2]- TotBSolve[2]*MagSolve[0])- MagSolve[1]/T2
	Blochz = GAMMA*(TotBSolve[1]*MagSolve[0]- TotBSolve[0]*MagSolve[1])- ((MagSolve[2]-equiM[2])/T1)

	Bloch = np.array([Blochx, Blochy, Blochz])

	return Bloch

def Solver_Func(index, GAMMA, Int_Ends, timeSeries, M, relaxFunc, B0Func, BrfFunc, BgradFunc, RxFunc, frameRot, 
	ParamT1, ParamT2, ParamB0, ParamRF, ParamGrad, ParamRx, ParamRot, ParamRotPlus, equiM):
	"""
	Function wraps around SciPy's Solve_IVP function to manage Solve_IVP's inputs and outputs. Uses Solve_IVP 
	to solve Bloch() for the behaviour of a single magnetization vector.

	Parameters:
	Int_Ends = Time endpoints of simulation, expressed as a tuple of (0, END)
	timeSeries = Numpy array of time points at which Solve_IVP should return magnetization state
	RxFunc = Function that calculates the spatial sensitivity of the receive coil (See Magnetic Field Calculation
	Functions for more details)
	RxParams = Tuple containing parameters needed by RxFunc. For further details, see documentation for the 
	corresponding functions.
	For other Parameters, please see Parameters section of Bloch()

	Returns:
	Output = 3-element list. Each element is a numpy array containing: [0] magnetization vector index; [1] the 
	final state of the magnetization vector; and [2] the state of magnetization at each timepoint in timeSeries, 
	arranged such that each column contains the results at a single timepoint, and each row contains either 
	the x-, y-, or z- component of the vector.
	"""

	# Rotate magnetization vector into the reference frame used for solving
	# Calculate total applied field in the Lab frame
	TotBLabInitial = B0Func(Int_Ends[0], index, ParamB0)+ BrfFunc(Int_Ends[0], index, ParamRF)+ BgradFunc(Int_Ends[0], index, ParamGrad)
	# Add T2star and chemical shift field offsets
	TotBLab = TotBLabInitial+ (TotBLabInitial*ParamB0[2][index]*1e-6)+ ParamB0[1][index]
	# Rotate to the rotating reference frame
	TotBRot = rotatingReferenceFrame(TotBLab, Int_Ends[0], ParamRot)
	# Rotate into Solving reference frame
	MagSolve = frameRot(M, Int_Ends[0], ParamRotPlus, TotBRot, dirForeward = True, ifB = False)

	# Solve for time evolution using scipy.integrate.solve_ivp
	sol = solve_ivp(Bloch,
					t_span = Int_Ends,
					t_eval = timeSeries,
					y0 = MagSolve,
					args = (index, GAMMA, relaxFunc, B0Func, BrfFunc, BgradFunc, frameRot, 
						ParamT1, ParamT2, ParamB0, ParamRF, ParamGrad, ParamRot, 
						ParamRotPlus, equiM),
					method = 'DOP853',
					atol = 1e-9,
					rtol = 1e-9)
	
	# Return final magnetization state to Larmor frame
	solSolve = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]])
	solEndRot = frameRot(solSolve, Int_Ends[1], ParamRotPlus, TotBRot, dirForeward = False, ifB = False)

	# Return intermediate net magnetization data into Larmor frame
	nrows, ncols = sol.y.shape
	solRot = np.zeros([nrows, ncols])
	for i in range(ncols):
		vec = sol.y[:,i]
		solRot[:, i] = frameRot(vec, timeSeries[i], ParamRotPlus, TotBRot, dirForeward = False, ifB = False)* RxFunc(timeSeries[i], index, ParamRx)

	Output = [index, solEndRot, solRot]

	return Output


#~~~~~~~~~~ Multiprocessing Code ~~~~~~~~~~#

def Solve(GAMMA, Int_Ends, timeSeries, M, relaxFunc, B0Func, BrfFunc, BgradFunc, RxFunc, frameRot, 
	ParamT1, ParamT2, ParamB0, ParamRF, ParamGrad, ParamRx, ParamRot, ParamRotPlus, equiM,
	chunksize, max_workers):
	"""
	User callable function which uses the concurrent.futures module to parallelize computation of 
	Solver_Func() to evolve magnetization of the entire sample.

	Parameters:
	M = 4D NumPy array containing all magnetization vectors used to simulate a sample.
	chunksize = Number of magnetization vectors to be assigned to each processing
	max_workers = Maximum number of logical cores to be used
	For other Parameters, please see Parameters section of Bloch() and Solver_Func()

	Returns:
	M_Final = 4D NumPy array containing all magnetization vectors used to simulate a sample, evolved 
	using the Bloch equations
	vectorSum = Numpy array with dimensions of 3 by length(timeSeries). Contains net magnetization of
	full sample at each time point specified in timeSeries. Rows respectively contain the x-, y-, and
	z- components.
	"""

	# Format variables into lists to be parsed by parallelization code
	ind_list = []
	list1, list2, list3, list4, list5, list6 = [], [], [], [], [], []
	list7, list8, list9, list10, list11, list12 = [], [], [], [], [], []
	list13, list14, list15, list16, list17 = [], [], [], [], []
	list18, list19 = [], []
	M_Final = copy.deepcopy(M)
	vectorSum = np.zeros([3, timeSeries.size])

	for index in np.ndindex(M.shape):
		if isinstance(M[index], np.ndarray) is False:
			continue

		ind_list.append(index)
		list1.append(GAMMA)
		list2.append(Int_Ends)
		list3.append(timeSeries)
		list4.append(M[index])
		list5.append(relaxFunc)
		list6.append(B0Func)
		list7.append(BrfFunc)
		list8.append(BgradFunc)
		list9.append(RxFunc)
		list10.append(frameRot)
		list11.append(ParamT1[index])
		list12.append(ParamT2[index])
		list13.append(ParamB0)
		list14.append(ParamRF)
		list15.append(ParamGrad)
		list16.append(ParamRx)
		list17.append(ParamRot)
		list18.append(ParamRotPlus)
		list19.append(equiM[index])

	# Run solver on each magnetization vector in parallel
	with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
		# Distribute chunksize vectors across assigned CPU cores. As core finish, distribute further
		# chunks until all vectors have been evolved
		for result in executor.map(Solver_Func, ind_list, list1, list2, list3, list4, list5,
							list6, list7, list8, list9, list10, list11, list12, 
							list13, list14, list15, list16, list17, list18, list19, 
							chunksize = chunksize):
			
			# Save final result for each vector to output M matrix
			M_Final[result[0]] = result[1]

			# Add intermediate data points for each vector together to calculate new magnetization of 
			# full sample at intermediate points.
			vectorSum += result[2]

	return M_Final, vectorSum