"""
Library of functions used to calculate the required gradient parameters
to properly acquire the imaging volume.

Note: the Nyquist criterion is still met if grad_amp/grad_duration
remains constant. Thus, the numbers given by these functions can be
scaled by a constant to meet the demands of a pulse sequence. 
"""

import numpy as np

GAMMA = 42.58e6 # Hz/T

def freq_encode_amp_cal(duration, n_points, fov, GAMMA=GAMMA):
    """Calculates gradient amplitude for a readout gradient of known
    duration.

    Parameters
    ----------
    duration : float
        Duration of applied gradient (seconds).
    n_points : int
        Number of data points to be acquired.
    fov : float
        Field of view length in the frequency encode direction (meters).
    n_acq : int
        Number of acquisitions for noise scaling.
    GAMMA : float, optional
        Gyromagnetic ratio of nucleus under study, in Hz/T: Defaults
        to hydrogen (42.58e6 Hz/T).

    Returns
    -------
    grad_amp : float
        Readout gradient amplitude as a floating point number.
    """

    grad_amp = n_points/(GAMMA*fov*duration)
    return grad_amp

def freq_encode_dur_amp_cal(n_points, fov, bandwidth, GAMMA=GAMMA):
    """Calulates gradient duration and amplitude for a readout gradient
    of known bandwidth.

    Parameters
    ----------
    n_points : int
        Number of data points to be acquired.
    fov : float
        Field of view length in the frequency encode direction (meters).
    bandwidth : float
        Desired receive bandwidth (Hz).
    GAMMA : float, optional
        Gyromagnetic ratio of nucleus under study, in Hz/T: Defaults
        to hydrogen (42.58e6 Hz/T).

    Returns
    -------
    grad_amp : float
        Readout gradient amplitude.
    grad_dur : float
        Readout gradient duration.
    """

    read_amp = (2*bandwidth)/(GAMMA*fov)
    read_dur = n_points/(2*bandwidth)

    return read_amp, read_dur

def phase_encode_amp_cal(duration, n_lines, fov, GAMMA=GAMMA):
    """Calculates phase gradient amplitudes needed for cartesian
    sampling of k-space.

    Automatically orders phase gradients to start at the center of
    k-space, and sample outwards. Provides a list of indeces to allow
    acquired lines to be correctly ordered within k-sapce.

    Parameters
    ----------
    duration : float
        Duration of applied gradient (seconds).
    n_lines : int
        Number of k-space lines to be acquired.
    fov : float
        Field of view length in the frequency encode direction (meters).
    GAMMA : float, optional
        Gyromagnetic ratio of nucleus under study, in Hz/T: Defaults
        to hydrogen (42.58e6 Hz/T).

    Returns
    -------
    phase_amp_steps : array_like
        List of gradient amplitudes, ordered from smallest absolute
        gradient amplitude to largest.
    indeces : array_like
        List of indicies indicating which line of k-space has been
        acquired.
    """

    max_amp = ((n_lines- 1)/2)*(1/(GAMMA*fov*duration))
    phase_amp_steps = np.linspace(-max_amp, max_amp, n_lines, endpoint = False)
    indeces = np.argsort(abs(phase_amp_steps))
    phase_amp_steps = phase_amp_steps[indeces]

    return phase_amp_steps, indeces
