#%%
import numpy as np
from scipy.io import loadmat
from processing_functions import butter_bandpass, logvar, cov, whitening, csp, apply_mix
import pickle
import os

SUBJECT = 'g'
MATFILE = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat'
MODEL_PATH = f"/home/costanza/Robot-Control-by-EEG-with-ML/models/{SUBJECT}"
TRAIN_PERCENTAGE = 0.6
#%%
# load the mat data
EEG_data = loadmat(MATFILE, struct_as_record = True)

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
# Band-Pass Filtering

trials_filt = {cl1: butter_bandpass(trials[cl1], 8, 30, sfreq, nsamples),
               cl2: butter_bandpass(trials[cl2], 8, 30, sfreq, nsamples)}

# %%
# Common Spatial Patterns (CSP) 
train_percentage = TRAIN_PERCENTAGE

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
W = csp(train[cl1], train[cl2], nsamples)
print('W shape:', W.shape)

# Save the CSP transformation matrix

# Create the directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

filename = os.path.join(MODEL_PATH, f"CSP_matrix_W.pkl")
with open(filename, 'wb') as file:
    pickle.dump(W, file)

print('Train[cl1] shape:', train[cl1].shape)
print('Train[cl2] shape:', train[cl2].shape)
print('Test[cl1] shape:', test[cl1].shape)
print('Test[cl2] shape:', test[cl2].shape)

#%%
# Apply the CSP on both the training and test set
train[cl1] = apply_mix(W, train[cl1], nchannels, nsamples)
train[cl2] = apply_mix(W, train[cl2], nchannels, nsamples)
test[cl1] = apply_mix(W, test[cl1], nchannels, nsamples)
test[cl2] = apply_mix(W, test[cl2], nchannels, nsamples)

print('Train[cl1] shape:', train[cl1].shape)
print('Train[cl2] shape:', train[cl2].shape)
print('Test[cl1] shape:', test[cl1].shape)
print('Test[cl2] shape:', test[cl2].shape)

#%%
# Select only the first and last components for classification
comp = np.array([0,-1])
train[cl1] = train[cl1][comp,:,:]
train[cl2] = train[cl2][comp,:,:]
test[cl1] = test[cl1][comp,:,:]
test[cl2] = test[cl2][comp,:,:]

print('Train[cl1] shape:', train[cl1].shape)
print('Train[cl2] shape:', train[cl2].shape)
print('Test[cl1] shape:', test[cl1].shape)
print('Test[cl2] shape:', test[cl2].shape)

#%%
# Calculate the log-var
train[cl1] = logvar(train[cl1])
train[cl2] = logvar(train[cl2])
test[cl1] = logvar(test[cl1])
test[cl2] = logvar(test[cl2])
print('Train[cl1] shape:', train[cl1].shape)
print('Train[cl2] shape:', train[cl2].shape)
print('Test[cl1] shape:', test[cl1].shape)
print('Test[cl2] shape:', test[cl2].shape)
#%%
X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T
y_train = np.concatenate((np.zeros(ntrain_l), np.ones(ntrain_r)))
X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T
y_test = np.concatenate((np.zeros(ntest_l), np.ones(ntest_r)))

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

# Create the directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

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
    clf_filename = os.path.join(MODEL_PATH, f"{clf_name}_model.pkl")
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
#%%

# Bar Plot of the evaluation metrics

import matplotlib.pyplot as plt
import numpy as np

# Extract data for plotting
classifiers = [metric['Classifier'] for metric in evaluation_metrics]
accuracy = [metric['Accuracy'] for metric in evaluation_metrics]
precision = [metric['Precision'] for metric in evaluation_metrics]
error = [metric['Error'] for metric in evaluation_metrics]
specificity = [metric['Specificity'] for metric in evaluation_metrics]

# Set the width of the bars
bar_width = 0.15

# Set the position of the bars on the x-axis
r1 = np.arange(len(classifiers))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Plotting
plt.bar(r1, accuracy, color='skyblue', width=bar_width, label='Accuracy')
plt.bar(r2, precision, color='orange', width=bar_width, label='Precision')
plt.bar(r3, error, color='lightgreen', width=bar_width, label='Error')
plt.bar(r4, specificity, color='salmon', width=bar_width, label='Specificity')

# Adding labels
plt.xlabel('Classifiers', fontweight='bold')
plt.xticks([r + bar_width * 1.5 for r in range(len(classifiers))], classifiers)
plt.title('Evaluation Metrics ')

# Adjust y-axis ticks to display more values between 0 and 1
plt.yticks(np.arange(0, 1.1, 0.1))

# Move the legend outside the plot area
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Show plot
plt.show()

#%%
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

for clf_name, clf in classifiers.items():
    # Predict probabilities for positive class
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')

# Plot random guessing line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


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
# Draw threshold line
threshold_value = 0.5  # Adjust as needed
ax_calibration_curve.axvline(x=threshold_value, color='r', linestyle='--', label=f'Threshold = {threshold_value:.2f}')
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

# %%
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

# %%
