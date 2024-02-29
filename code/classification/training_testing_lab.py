#%%
import numpy as np
from scipy.io import loadmat
from processing_functions import psd, plot_PSD
import pickle
import os

SUBJECT = 'c'

# load the mat data
EEG_data = loadmat(f'/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat', struct_as_record = True)

# List all the keys in the loaded data
keys = EEG_data.keys()
# c,d,e: 'left' 'right'
# a,f :foot,right
# b,g

# Print the keys variables to identify the correct key for EEG data
print(keys)
# %%
# Extract data
markers = EEG_data['mrk']
sfreq = EEG_data['nfo']['fs'][0][0][0][0]
EEGdata   = EEG_data['cnt'].T 
nchannels, nsamples = EEGdata.shape
time_unit = 1 / sfreq
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
nclasses = len(cl_lab)
nevents = len(event_onsets)
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

#%%
# Compute the PSD
psd_cl1, freqs = psd(trials[cl1], sfreq)
psd_cl2, freqs = psd(trials[cl2], sfreq)
psd_all = {cl1: psd_cl1, cl2: psd_cl2}

# Plot
plot_PSD(psd_all, freqs, chan_names, cl_lab)
    
# %%
# Statistical analysis
from processing_functions import logvar, std, rms, plot_logvar, plot_std, plot_rms
import matplotlib.pyplot as plt

# Compute the features
logvar_trials = {cl1: logvar(trials[cl1]),cl2: logvar(trials[cl2])}

# Bar Plots
plt.figure(figsize=(15, 3))
plot_logvar(logvar_trials, cl_lab, cl1, cl2, nchannels)
plt.show()

# %%
# Scatter Plot of the features
from processing_functions import scatter_logvar
scatter_logvar(logvar_trials, cl_lab, [0, -1])

#%%
# Band-Pass Filtering
from processing_functions import butter_bandpass

trials_filt = {cl1: butter_bandpass(trials[cl1], 8, 30, sfreq, nsamples),
               cl2: butter_bandpass(trials[cl2], 8, 30, sfreq, nsamples)}

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
train_percentage = 0.5

# Calculate the number of trials for each class the above percentage boils down to
ntrain_l = int(trials_filt[cl1].shape[2] * train_percentage)
ntrain_r = int(trials_filt[cl2].shape[2] * train_percentage)
ntest_l = trials_filt[cl1].shape[2] - ntrain_l
ntest_r = trials_filt[cl2].shape[2] - ntrain_r

# Splitting the frequency filtered signal into a train and test set
train = {cl1: trials_filt[cl1][:,:,:ntrain_l],
         cl2: trials_filt[cl2][:,:,:ntrain_r]}

test = {cl1: trials_filt[cl1][:,:,ntrain_l:],
        cl2: trials_filt[cl2][:,:,ntrain_r:]}


# Train the CSP on the training set only
W = csp(train[cl1], train[cl2])
print('W shape:', W.shape)

# Save the CSP transformation matrix
# Directory to save it
save_dir_csp_mat = f"/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}_lab/Trained_Models"

# Create the directory if it doesn't exist
os.makedirs(save_dir_csp_mat, exist_ok=True)

filename = os.path.join(save_dir_csp_mat, f"CSP_matrix_W.pkl")
with open(filename, 'wb') as file:
    pickle.dump(W, file)

#%%
# Apply the CSP on both the training and test set
train[cl1] = apply_mix(W, train[cl1])
train[cl2] = apply_mix(W, train[cl2])
test[cl1] = apply_mix(W, test[cl1])
test[cl2] = apply_mix(W, test[cl2])

#%%
# Select only the first and last components for classification
comp = np.array([0,-1])
train[cl1] = train[cl1][comp,:,:]
train[cl2] = train[cl2][comp,:,:]
test[cl1] = test[cl1][comp,:,:]
test[cl2] = test[cl2][comp,:,:]

#%%
# Calculate the log-var
train[cl1] = logvar(train[cl1])
train[cl2] = logvar(train[cl2])
test[cl1] = logvar(test[cl1])
test[cl2] = logvar(test[cl2])

#%%
X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T
y_train = np.concatenate((np.zeros(ntrain_l), np.ones(ntrain_r))) # 1 for right hand, 0 for left hand
X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T
y_test = np.concatenate((np.zeros(ntest_l), np.ones(ntest_r))) # 1 for right hand, 0 for left hand

#%%
##################################### Classification #################################

# Import classifiers from scikit-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import evaluation metrics from scikit-learn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix
# Directory to save trained models
save_dir = f"/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}_lab/Trained_Models"
# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Create a dictionary to store the classifiers
classifiers = {'LDA': LinearDiscriminantAnalysis(),
               'NB': GaussianNB(),
               'SVM': make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True)),
               'RF': RandomForestClassifier(n_estimators=100),
               'DT': DecisionTreeClassifier(),
               'LR': LogisticRegression()
            }

evaluation_metrics = []

# Train and save classifiers
for clf_name, clf in classifiers.items():
    # Train classifier
    clf.fit(X_train, y_train)
    
    # Save the trained classifier to a .pkl file
    clf_filename = os.path.join(save_dir, f"{clf_name}_model.pkl")
    with open(clf_filename, 'wb') as f:
        pickle.dump(clf, f)

    # Predict on test set (if needed for evaluation)
    y_pred = clf.predict(X_test)
    
    # Calculate evaluation metrics (if needed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    specificity = tn / (tn + fp)
    error = 1 - accuracy
    
    # Store evaluation metrics (if needed)
    evaluation_metrics.append({
        'Classifier': clf_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Error': error,
        'Specificity': specificity
    })


# Print the table (if needed)
print("Classifier\tAccuracy\tPrecision\tError\tSpecificity")
for metric in evaluation_metrics:
    print(f"{metric['Classifier']}\t{metric['Accuracy']:.4f}\t{metric['Precision']:.4f}\t{metric['Error']:.4f}\t{metric['Specificity']:.4f}")

# %%
############################################# Reliability Diagrams ######################################

from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o", "X", "P"]
for i, (name, clf) in enumerate(classifiers.items()):
    # clf.fit(X_train, y_train)
    # Retrieve the trained classifier from the dictionary
    trained_clf = classifiers[name]

    display_1 = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display_1

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Reliability Diagram")
ax_calibration_curve.set_xlabel("Mean Predicted Probability")
ax_calibration_curve.set_ylabel("True Positive Rate - Class 1 (Right Hand)") # Fraction of positives

#%%
########################## Reliability diagram of the TRUE NEGATIVE RATE ##########################

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o", "X", "P"]
for i, (name, clf) in enumerate(classifiers.items()):
    # clf.fit(X_train, y_train)
    # Retrieve the trained classifier from the dictionary
    trained_clf = classifiers[name]
   
    display_0 = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        pos_label=0,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display_0

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Reliability Diagram")
ax_calibration_curve.set_xlabel("Mean Predicted Probability")
ax_calibration_curve.set_ylabel("True Negative Rate - Class 0 (Left Hand)") # Fraction of negatives

#%%
# Add histograms for all classifiers
num_classifiers = len(classifiers)
num_rows = (num_classifiers + 1) // 2  # Calculate number of rows needed
num_cols = 2  # Fixed number of columns

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 5))

# Flatten the axes if needed
if num_classifiers > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for i, (name, clf) in enumerate(classifiers.items()):
    ax = axes[i]

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

# Hide any unused subplots
for j in range(num_classifiers, num_rows * num_cols):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


#%%
# Classifiers Calibration
from sklearn.calibration import CalibratedClassifierCV

# Train the classifiers with calibration
calibrated_models = {}
for classifier in classifiers:
    # Use CalibratedClassifierCV for calibration
    calibrated_model = CalibratedClassifierCV(classifiers[classifier], method='sigmoid', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    calibrated_models[classifier] = calibrated_model

#%%
# Plot the calibration curves
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o", "X", "P"]

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
ax_calibration_curve.set_title(f"Reliability Diagram - After Calibration with Platt Scaling Method")

plt.show()
#%% Add histogram of predicted probabilities for each of the 6 classifier 
# Add histograms for all classifiers
num_classifiers = len(classifiers)
num_rows = (num_classifiers + 1) // 2  # Calculate number of rows needed
num_cols = 2  # Fixed number of columns

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 5))

# Flatten the axes if needed
if num_classifiers > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for i, (name, clf) in enumerate(calibrated_models.items()):
    ax = axes[i]

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

# Hide any unused subplots
for j in range(num_classifiers, num_rows * num_cols):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

#%%
# Classifiers Calibration
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
markers = ["^", "v", "s", "o", "X", "P"]

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
ax_calibration_curve.set_title(f"Reliability Diagram - After Calibration with Isotonic Method")

plt.show()
#%% Add histogram of predicted probabilities for each of the 6 classifier 
# Add histograms for all classifiers
num_classifiers = len(classifiers)
num_rows = (num_classifiers + 1) // 2  # Calculate number of rows needed
num_cols = 2  # Fixed number of columns

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 5))

# Flatten the axes if needed
if num_classifiers > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for i, (name, clf) in enumerate(calibrated_models.items()):
    ax = axes[i]

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

# Hide any unused subplots
for j in range(num_classifiers, num_rows * num_cols):
    axes[j].axis('off')

plt.tight_layout()
plt.show()