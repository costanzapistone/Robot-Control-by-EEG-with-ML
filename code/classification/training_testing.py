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

scatter_logvar(logvar_trials, cl_lab, [0, -1])
scatter_std(std_trials, cl_lab, [0, -1])
scatter_rms(rms_trials, cl_lab, [0, -1])

#%%
# Band-Pass Filtering

def butter_bandpass(trials, lowcut, highcut, fs, order=5):
    """
    Design a band-pass filter using the Butterworth method.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal for one class.
    lowcut : float
        The lower cut-off frequency of the band-pass filter.
    highcut : float
        The higher cut-off frequency of the band-pass filter.

    fs : float
        The sampling frequency of the signal.
    
    order : int
        The order of the filter.
    
    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The band-pass filtered signal for one class.
    """
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = lfilter(b, a, trials[:,:,i], axis=1)

    return trials_filt

# Band-pass filter the data
lowcut = 8
highcut = 30

trials_filt = {cl1: butter_bandpass(trials[cl1], lowcut, highcut, sfreq),
                    cl2: butter_bandpass(trials[cl2], lowcut, highcut, sfreq)}

# %%
# Common Spatial Patterns (CSP) butterworth 

from numpy import linalg

def cov(trials):
    """
    Calculates the covariance for each trial and return their average.

    """
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T)/ nsamples for i in range(ntrials) ]
    return np.mean(covs, axis=0)

def whitening(sigma):
    """ calculate whitening matrix for covariance matrix sigma. """
    U, l, _ = linalg.svd(sigma)
    return U.dot(np.diag(l ** -0.5))

def csp(trials_r, trials_l):
    """
    Calculates the CSP transformation matrix W.

    Parameters
    ----------
    trials_r, trials_l : 3d-arrays (channels x samples x trials)
        The EEGsignal for right and left hand

    Returns
    -------
    W : mixing matrix (spatial filters that will maximize the variance for one class and minimize the variance for the other)
        The CSP transformation matrix
    """
    cov_r = cov(trials_r)
    cov_l = cov(trials_l)

    P = whitening(cov_r + cov_l)
    B, _, _ = linalg.svd(P.T.dot(cov_l).dot(P))
    
    W = P.dot(B)
    return W

def apply_mix(W, trials):
    """
    Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)
    """
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp


# %%
# Common Spatial Patterns (CSP) 
train_percentage = 0.7

# Calculate the number of trials for each class the above percentage boils down to
ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
ntest_r = trials_filt[cl1].shape[2] - ntrain_r
ntest_f = trials_filt[cl2].shape[2] - ntrain_f

# Splitting the frequency filtered signal into a train and test set
train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
         cl2: trials_filt[cl2][:,:,:ntrain_f]}

test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
        cl2: trials_filt[cl2][:,:,ntrain_f:]}

# Train the CSP on the training set only
W = csp(train[cl1], train[cl2])

# Apply the CSP on both the training and test set
train[cl1] = apply_mix(W, train[cl1])
train[cl2] = apply_mix(W, train[cl2])
test[cl1] = apply_mix(W, test[cl1])
test[cl2] = apply_mix(W, test[cl2])

# Select only the first and last components for classification
comp = np.array([0,-1])
train[cl1] = train[cl1][comp,:,:]
train[cl2] = train[cl2][comp,:,:]
test[cl1] = test[cl1][comp,:,:]
test[cl2] = test[cl2][comp,:,:]

# Calculate the log-var
train[cl1] = logvar(train[cl1])
train[cl2] = logvar(train[cl2])
test[cl1] = logvar(test[cl1])
test[cl2] = logvar(test[cl2])

X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T
X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T
y_train = np.zeros(X_train.shape[0], dtype=int)
y_train[:ntrain_r] = 1
y_test = np.zeros(X_test.shape[0], dtype=int)
y_test[:ntest_r] = 1
#%%
# Classification with different classifiers

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std

# Create a dictionary to store the classifiers
classifiers = {'LDA': LinearDiscriminantAnalysis(),
               'NB': GaussianNB(),
               'SVM': make_pipeline(StandardScaler(), SVC(gamma='auto')),
               'RF': RandomForestClassifier(n_estimators=100),
               'DT': DecisionTreeClassifier(),
               'LR': LogisticRegression()
            }

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# Train and test the classifiers with cross-validation

for classifier in classifiers:
    # Train the classifier
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(classifiers[classifier], X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    print(f'{classifier} Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


    
# %%
# Calibration Plots

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
for i, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram of predicted probabilities for each of the 6 classifier 
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]

clf_list = list(classifiers.items())
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()
plt.show()
# %%
