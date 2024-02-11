#%%
import numpy as np
from scipy.io import loadmat
from processing_functions import psd, plot_PSD

# load the mat data
EEG_data = loadmat('/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1d.mat', struct_as_record = True)

# List all the keys in the loaded data
keys = EEG_data.keys()

# Print the keys variables to identify the correct key for EEG data
print(keys)
# %%
# Extract data
markers = EEG_data['mrk']
sfreq = EEG_data['nfo']['fs'][0][0][0][0]
EEGdata   = EEG_data['cnt'].T 
nchannels, nsamples = EEGdata.shape

time_unit = 1 / sfreq
print("Time Unit:", time_unit, "seconds")

chan_names = [s[0] for s in EEG_data['nfo']['clab'][0][0][0]]

event_onsets  = EEG_data['mrk'][0][0][0] # Time points when events occurred
event_codes   = EEG_data['mrk'][0][0][1] # It contains numerical or categorical labels associated with each event.
event_onset_time = event_onsets * time_unit # Seconds

# Creates an array of zeros and then assigns the event codes to the corresponding positions based on the event onsets.
labels = np.zeros((1, nsamples), int)
labels[0, event_onsets] = event_codes

cl_lab = [s[0] for s in EEG_data['nfo']['classes'][0][0][0]]
cl1    = cl_lab[0]
cl2    = cl_lab[1]

# Electrode positions 
xpos = EEG_data['nfo']['xpos']
ypos = EEG_data['nfo']['ypos']

nclasses = len(cl_lab)
nevents = len(event_onsets)

# Print some information
print('Shape of EEG:', EEGdata.shape)
print('Sample rate:', sfreq)
print('Number of channels:', nchannels)
print('Channel names:', chan_names)
print('Number of events (MI movements):', event_onsets.shape[1])
print('Event codes:', np.unique(event_codes))
print('Class labels:', cl_lab)
print('Number of classes:', nclasses)

# %%
# Dictionary to store the trials
trials = {}

# The time window in samples to extract for each trial, here 0.5 -- 4.5 seconds
win = np.arange(int(0.5 * sfreq), int(4.5 * sfreq))

# Length of the time window
nsamples = len(win)
# %%
# Loop over the classes (left vs right hand)
for cl, code in zip(cl_lab, np.unique(event_codes)):

    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]

    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEGdata[:, win + onset]

# Some information about the dimensionality of the data (channels x time x trials)
print('Shape of trials[cl1]:', trials[cl1].shape)
print('Shape of trials[cl2]:', trials[cl2].shape)

#%%
# Compute the PSD
psd_cl1, freqs = psd(trials[cl1], sfreq)
psd_cl2, freqs = psd(trials[cl2], sfreq)
psd_all = {cl1: psd_cl1, cl2: psd_cl2}

# Plot
plot_PSD(psd_all, freqs, chan_names, cl_lab)
    
# %%
# Statistical analysis
from processing_functions import logvar, std, rms
from processing_functions import plot_logvar, plot_std, plot_rms
import matplotlib.pyplot as plt
# Logvar (Log-Variance): Logvar represents the logarithm of the variance of a signal. Variance is a measure of the spread or dispersion of a set of values. By taking the logarithm of the variance, the scale of the values is adjusted, making them more suitable for visualization and analysis.
# Std (Standard Deviation): Std is a measure of the amount of variation or dispersion of a set of values. It is the square root of the variance.
# RMS (Root Mean Square): RMS is a measure of the magnitude of a set of values. It is the square root of the mean of the squares of the values.

# For each channel and class, compute the logvar, std, and rms across trials

# Compute the features
logvar_trials = {cl1: logvar(trials[cl1]),cl2: logvar(trials[cl2])}
std_trials = {cl1: std(trials[cl1]), cl2: std(trials[cl2])}
rms_trials = {cl1: rms(trials[cl1]), cl2: rms(trials[cl2])}

# Bar Plots
plt.figure(figsize=(15, 3))
plot_logvar(logvar_trials, cl_lab, cl1, cl2, nchannels)
plt.figure(figsize=(15, 3))
plot_std(std_trials, cl_lab, cl1, cl2, nchannels)
plt.figure(figsize=(15, 3))
plot_rms(rms_trials, cl_lab, cl1, cl2, nchannels)
plt.show()

# %%
# Scatter Plot of the features
from processing_functions import scatter_logvar, scatter_std, scatter_rms

scatter_logvar(logvar_trials, cl_lab, chan_names)
scatter_std(std_trials, cl_lab, chan_names)
scatter_rms(rms_trials, cl_lab, chan_names)

# %%

