import numpy as np
import matplotlib.pyplot as plt

import NIBLEs.experiment as NIBLEs
import NIBLEs.fieldfunc as field
import NIBLEs.relaxfunc as relax

# Define sequence name, path, and desired relaxation behaviour
sample_name = 'NIBLEsTutorial_NMR_1024Vectors_1x1_1.0cm'
sample_path = '/NIBLEs/samples'
relax_func = relax.constant
model_type = 'Constant'
seq_name = 'CPMG'

# Define sequence timings and parameters
rf_dur = 1.28e-3
te1 = 15e-3
te2 = 30e-3
n_echo = 8
n_points = 100
acq_dur = te2/2

# Calculate Timing Parameters
te1_deadtime = te1 - rf_dur
te2_deadtime = (te2- acq_dur- rf_dur)/2

# Run simulation
exp2 = NIBLEs.Experiment(b0 = [0, 0, 3])
exp2.t2_star_bounds = [0.001, 0.999]
exp2.sequence_start(sample_name = sample_name, sample_path = sample_path,
                    relax_func = relax_func, model_type = model_type)
exp2.signal = np.zeros([n_echo, n_points]).astype(complex)

# Apply Excitation Pulse
exp2.rf_pulse(duration = rf_dur, brf_func = field.rf_sinc, flip_angle = 90,
              phi = 90)
exp2.deadtime(duration = te1_deadtime)

# Apply Echo Train and save resultant signal
for i in range(n_echo):

    if i % 2 == 0:
        exp2.rf_pulse(duration = rf_dur, brf_func = field.rf_sinc,
                      flip_angle = 180, phi = 0)
    else:
        exp2.rf_pulse(duration = rf_dur, brf_func = field.rf_sinc,
                      flip_angle = 180, phi = 180)

    exp2.deadtime(te2_deadtime)
    exp2.acquisition(duration = acq_dur, n_points = n_points)
    exp2.signal[i, :] = exp2.readout
    exp2.save_acquisition(variable = exp2.signal, seq_name = seq_name,
                      sample_name = sample_name)
    exp2.deadtime(te2_deadtime)
