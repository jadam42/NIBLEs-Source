import numpy as np
import math as math

# Functions used to calculate the required gradient parameters to perform an MR imaging experiment. 
#
# Original version of this code designed and written by John Adams (jadam33@uwo.ca)
#
# This code is published under a GNU General Public License version 3 License. Anyone using this code 
# must give attribution to the original author (John Adams), and must make any derivative code built 
# using this code available under the same licensing terms.


#~~~~~ Version History ~~~~~#
# 1.0  Public Release - July 1 2026 - John Adams

GAMMA = 42.58e6 # Hz/T

def freqAmplitudeCalibration(gradDuration, nAcqPoints, FOV, GAMMA = GAMMA):
	"""
	Calculates the required amplitude for a frequency encode gradient with known duration. Assumes 
	ideal square gradient waveform.

	Parameters:
	gradDuration = Duration of applied gradient (s)
	nAcqPoints = Number of data points to be acquired
	FOV = Field of view length in the frequency encode direction (m)
	GAMMA = Gyromagnetic ratio of nucleus under study, in Hz/T: Defaults to hydrogen (42.58e6 Hz/T)

	Returns: 
	Readout gradient amplitude as a floating point number
	"""

	return nAcqPoints/(GAMMA*FOV*gradDuration)

def freqDurAmpCalibration(nAcqPoints, FOV, BW, GAMMA = GAMMA):
	"""
	Calculates duration and amplitude of a frequency encode gradient for known frequency bandwidth. 
	Assumes ideal square gradient waveform.

	Parameters:
	nAcqPoints = Number of data points to be acquired
	FOV = Field of view length in the frequency encode direction (m)
	BW = Desired frequency bandwidth of applied gradient (Hz)
	GAMMA = Gyromagnetic ratio of nucleus under study, in Hz/T: Defaults to hydrogen (42.58e6 Hz/T)

	Returns: 
	readAmp, readDur = Readout gradient amplitude and duration as floating point numbers
	"""

	readAmp = (2*BW)/(GAMMA*FOV)
	readDur = nAcqPoints/(2*BW)

	return readAmp, readDur

def phaseAmplitudeCalibration(gradDuration, nAcqPoints, FOV, GAMMA = GAMMA):
	"""
	Calculates required gradient amplitude for a phase encoding gradient with known duration. Assumes 
	ideal square gradient waveform.

	Parameters:
	gradDuration = Experiment.Duration
	nAcqPoints = Number of data points to be acquired
	FOV = Field of view length in the frequency encode direction (m)
	GAMMA = Gyromagnetic ratio of nucleus under study, in Hz/T: Defaults to hydrogen (42.58e6 Hz/T)

	Returns: 
	Phase encode gradient amplitude as a floating point number
	"""

	return ((nAcqPoints- 1)/2)*(1/(GAMMA*FOV*gradDuration))

def phaseDurationCalibration(nAcqPoints, phaseAmp, FOV, GAMMA = GAMMA):
	"""
	Calculates length of phase encode gradient for given max gradient amplitude. Assumes ideal square 
	gradient waveform.

	Parameters:
	nAcqPoints = Number of data points to be acquired 
	phaseAmp = Max amplitude of phase encode gradient
	FOV = Field of view length in the frequency encode direction (m)
	GAMMA = Gyromagnetic ratio of nucleus under study, in Hz/T: Defaults to hydrogen (42.58e6 Hz/T)

	Returns: 
	phaseDur = Phase encode gradient duration as a floating point number
	"""

	kMax =(nAcqPoints-1)/(2*FOV)
	phaseDur = kMax/(GAMMA*phaseAmp)

	return phaseDur

def phaseOrderingCalculator(nAcqPoints, phaseAmp):
	"""
	Calculates and orders the phase gradient amplitudes to be applied for each iteration of a sequence.
	Phase gradients are ordered such that phaseAmp = 0 is acquired first, and subsequent lines
	are acquired outwards from there.

	Parameters:
	nAcqPoints = Number of data points to be acquired 
	phaseAmp = Max amplitude of phase encode gradient

	Returns: 
	phaseAmpSteps = Numpy array containing the phase gradient amplitudes in Tesla for each repetition of the 
		sequence, ordered from 0 to largest absolute value and alternating between negative and positive 
		amplitude.
	ind = Numpy array containing the k-space line number that corresponds with each gradient in PhaseAmpSteps.
		Use to arrange acquired data in exp.signal.
	"""

	# Calculate index of kspace line where phase gradient = 0
	start = math.ceil(nAcqPoints/2)-1

	# Define index array for output
	ind = np.zeros(nAcqPoints)
	ind[0] = int(start)
	i = 1

	# Generate ordering of kspace line acquisitions, stating with the line where phaseAmp = 0 and acquiring 
	# kspace data outwards from there, alternating between positive and negative gradients
	for k in range(0, nAcqPoints-1):
		if k % 2 == 0:
			ind[k+1] = start + i
		else:
			ind[k+1] = start - i
			i += 1

	# Convert index list to integer data type
	ind = ind.astype(int)

	# Generate list of phase gradient amplitudes, based on number of acquisitions and maximum phaseAmp. 
	# Endpoint argument is used to guarnetee phaseAmp = 0 is part of list
	if nAcqPoints % 2 == 0:
		phaseAmpSteps = np.linspace(-phaseAmp, phaseAmp, nAcqPoints, endpoint = False)
	else:
		phaseAmpSteps = np.linspace(-phaseAmp, phaseAmp, nAcqPoints)

	# Reorder list of gradient amplitudes using indeces
	phaseAmpSteps = phaseAmpSteps[ind]

	return phaseAmpSteps, ind
