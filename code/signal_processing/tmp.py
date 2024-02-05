#%%
import numpy as np
from scipy.io import loadmat

#%%
sub = 'a'

# Load the MATLAB file 
EEG_data = loadmat(f"/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{sub}.mat", struct_as_record = True)

# List all the keys in the loaded data
keys = EEG_data.keys()

# Print the keys variables to identify the correct key for EEG data
print(keys)

#%%

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


#%%
# Dictionary to store the trials
trials = {}

# The time window in samples to extract for each trial, here 0.5 -- 4.5 seconds
win = np.arange(int(0.5 * sfreq), int(4.5 * sfreq))

# Length of the time window
nsamples_win = len(win)
# %%
# Loop over the classes (left vs right hand)
for cl, code in zip(cl_lab, np.unique(event_codes)):

    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]

    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples_win, len(cl_onsets)))

    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEGdata[:, win + onset]

# Some information about the dimensionality of the data (channels x time x trials)
print('Shape of trials[cl1]:', trials[cl1].shape)
print('Shape of trials[cl2]:', trials[cl2].shape)
# %%
#FFT

fft_trials = {}

for cl in cl_lab:
    # Get the segmented data for the current class
    trials_cl = trials[cl]

    # Allocate memory for the FFT of the trials
    fft_trials[cl] = np.zeros_like(trials_cl, dtype=complex)

    # Compute the FFT for each trial
    for i in range(trials_cl.shape[2]):
        fft_trials[cl][:,:,i] = np.fft.fft(trials_cl[:,:,i], axis=1)

    # Compute the magnitude of the FFT
    for cl in fft_trials:
        fft_trials[cl] = np.abs(fft_trials[cl])

# Print the shape of the FFT data
print('Shape of fft_trials[cl1]:', fft_trials[cl1].shape)
print('Shape of fft_trials[cl2]:', fft_trials[cl2].shape)

# %%
# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Get the number of trials for each class
ntrials_cl1 = fft_trials[cl1].shape[2]
ntrials_cl2 = fft_trials[cl2].shape[2]
n_features = fft_trials[cl1].shape[0] * fft_trials[cl1].shape[1]

# Reshape the fft_trials data to fit the sklearn LDA
X_cl1 = fft_trials[cl1].reshape(ntrials_cl1, n_features)
X_cl2 = fft_trials[cl2].reshape(ntrials_cl2, n_features)

X = np.concatenate((X_cl1, X_cl2), axis = 0)

# Create the labels for the LDA (cl1 = -1, cl2 = 1)
y = np.concatenate((-np.ones(ntrials_cl1), np.ones(ntrials_cl2)))

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Pipeline with Logistic Regression
model_lr = Pipeline([
    ('lda', LinearDiscriminantAnalysis(n_components=1)),
    ('classifier', LogisticRegression())
])
# evaluate lda with naive bayes algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# # evaluate model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model_lr, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# %%

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
clf_list = [
    (model_lr, "Logistic Regression"),
    # (pipeline_dt, "Decision Tree"),
    # (pipeline_knn, "KNN"),
    # (pipeline_svm, "SVM"),
    # (pipeline_nb, "Naive Bayes"),
]
for i, (clf, name) in enumerate(clf_list):
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
# %%
# Calibrate the model
from sklearn.calibration import CalibratedClassifierCV

# define the model
model_lr = Pipeline([
    ('lda', LinearDiscriminantAnalysis(n_components=1)),
    ('classifier', LogisticRegression())
])

calibration_model_lr = CalibratedClassifierCV(model_lr, method='isotonic', cv=None)

# # evaluate model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(calibration_model_lr, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
clf_list = [
    (model_lr, "Logistic Regression"),
    (calibration_model_lr, "Calibrated Logistic Regression")
    # (pipeline_dt, "Decision Tree"),
    # (pipeline_knn, "KNN"),
    # (pipeline_svm, "SVM"),
    # (pipeline_nb, "Naive Bayes"),
]
for i, (clf, name) in enumerate(clf_list):
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