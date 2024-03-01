#%%
import numpy as np
from scipy.io import loadmat
from processing_functions import psd, plot_PSD, logvar, plot_logvar, scatter_logvar, butter_bandpass, cov, whitening, csp, apply_mix
import matplotlib.pyplot as plt

# Define the subject to analyze
subject = 'd'

# load the mat data
EEG_data = loadmat(f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{subject}.mat', struct_as_record = True)

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

chan_names = [s[0] for s in EEG_data['nfo']['clab'][0][0][0]]

event_onsets  = EEG_data['mrk'][0][0][0] # Time points when events occurred
event_codes   = EEG_data['mrk'][0][0][1] # It contains numerical or categorical labels associated with each event.
event_onset_time = event_onsets / sfreq # Seconds

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

#%%
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

# Compute the features
logvar_trials = {cl1: logvar(trials[cl1]),cl2: logvar(trials[cl2])}

# Bar Plots
plt.figure(figsize=(15, 3))
plot_logvar(logvar_trials, cl_lab, cl1, cl2, nchannels)
plt.show()

# Scatter Plot of the features
scatter_logvar(logvar_trials, cl_lab, [0, -1])

#%%
# Band-Pass Filter
lowcut = 8
highcut = 30

trials_filt = {cl1: butter_bandpass(trials[cl1], lowcut, highcut, sfreq, nsamples),
               cl2: butter_bandpass(trials[cl2], lowcut, highcut, sfreq, nsamples)}

#%% 
# Plot the PSD of the filtered signal
psd_cl1, freqs = psd(trials_filt[cl1], sfreq)
psd_cl2, freqs = psd(trials_filt[cl2], sfreq)
psd_all = {cl1: psd_cl1, cl2: psd_cl2}

# Plot
plot_PSD(psd_all, freqs, chan_names, cl_lab)

# Plot the features
logvar_trials_filt = {cl1: logvar(trials_filt[cl1]),cl2: logvar(trials_filt[cl2])}

# Bar Plots
plt.figure(figsize=(15, 3))
plot_logvar(logvar_trials_filt, cl_lab, cl1, cl2, nchannels)
plt.show()

# Scatter Plot of the features
scatter_logvar(logvar_trials_filt, cl_lab, [0, -1])

#%% 
# CSP
W = csp(trials_filt[cl1], trials_filt[cl2], nsamples)

trials_csp = {cl1: apply_mix(W, trials_filt[cl1], nchannels, nsamples),
              cl2: apply_mix(W, trials_filt[cl2], nchannels, nsamples)
            }

print('Shape of trials_csp[cl1]:', trials_csp[cl1].shape)
print('Shape of trials_csp[cl2]:', trials_csp[cl2].shape)
print('Shape of W:', W.shape)

#%%
# Compute the features
logvar_trials_csp = {cl1: logvar(trials_csp[cl1]),cl2: logvar(trials_csp[cl2])}

# Bar Plots
plt.figure(figsize=(15, 3))
plot_logvar(logvar_trials_csp, cl_lab, cl1, cl2, nchannels)
plt.show()

# Scatter Plot of the features 
scatter_logvar(logvar_trials_csp, cl_lab, [0, -1])
# %%