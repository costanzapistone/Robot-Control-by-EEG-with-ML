#%%
import numpy as np
from scipy.io import loadmat

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
def calculate_rms(trials):
    """
    Calculates the Root Mean Square (RMS) of each channel.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal

    Returns
    -------
    rms_values : 2d-array (channels x trials)
        The RMS values for each channel 
    """
    return np.sqrt(np.mean(trials**2, axis=1))

def calculate_std_dev(trials):
    """
    Calculates the Standard Deviation of each channel.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal

    Returns
    -------
    std_dev_values : 2d-array (channels x trials)
        The standard deviation values for each channel 
    """
    return np.std(trials, axis=1)

rms_trials_cl1 = calculate_rms(trials[cl1])
rms_trials_cl2 = calculate_rms(trials[cl2])
print('Shape of rms_trials_cl1:', rms_trials_cl1.shape)
print('Shape of rms_trials_cl2:', rms_trials_cl2.shape)

#%% 
import matplotlib.pyplot as plt
def plot_rms(trials):

    """
    Plots the RMS features for each channel as a bar chart.

    Parameters
    ----------
    trials : 2-d array (channels x trials)
        The RMS values for each channel

    """
    plt.figure(figsize=(12, 5))

    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('RMS features for each channel')
    plt.xlabel('Channels')
    plt.ylabel('RMS')
    plt.legend(cl_lab) 

rms_trials = {cl1: rms_trials_cl1, cl2: rms_trials_cl2}
plot_rms(rms_trials)


# %%
from sklearn.model_selection import train_test_split
# Concatenate RMS values for each class
rms_features_cl1 = calculate_rms(trials[cl1]).T  # Transpose for trials x channels
rms_features_cl2 = calculate_rms(trials[cl2]).T  # Transpose for trials x channels

X = np.concatenate((rms_features_cl1, rms_features_cl2))
y = np.concatenate((-np.ones(rms_features_cl1.shape[0]), np.ones(rms_features_cl2.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create the classifiers and store in a 
classifiers = {'knn': KNeighborsClassifier(n_neighbors=3),
               'nb': GaussianNB(),
               'lr': LogisticRegression(),
               'dt': DecisionTreeClassifier(),
               'svm': SVC(probability=True),
               'lda': LinearDiscriminantAnalysis()}
            

#evaluate the classifiers
for classifier in classifiers:

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(classifiers[classifier], X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print(f'{classifier} Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



#%%Train the classifiers
trained_models = {}
for classifier in classifiers:
    trained_models[classifier] = classifiers[classifier].fit(X_train, y_train)

# %%# Calibration Curves

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o", "X", "P"]

for i, (name, clf) in enumerate(classifiers.items()):
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
ax_calibration_curve.set_title(f"Calibration Plots  - Before Calibration")

plt.show()

# %%
# Classifiers Calibration
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV

# Train the classifiers with calibration
calibrated_models = {}
for classifier in classifiers:
    # Use CalibratedClassifierCV for calibration
    calibrated_model = CalibratedClassifierCV(classifiers[classifier], method='isotonic', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    calibrated_models[classifier] = calibrated_model

#%%
# Plot the calibration curves
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o", "X"]

for i, (name, clf) in enumerate(calibrated_models.items()):
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
ax_calibration_curve.set_title(f"Calibration Plots - After Calibration")

plt.show()
# %%
