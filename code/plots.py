#%%
import numpy as np
import pickle
from scipy.io import loadmat
from processing_functions import butter_bandpass, apply_mix, logvar
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import evaluation metrics from scikit-learn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix


# Define constants
SUBJECT = 'g'
MATFILE = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat'
MODEL = 'LR'
CLASSIFIER_FILENAME = f'/home/costanza/Robot-Control-by-EEG-with-ML/models/{SUBJECT}/{MODEL}_model.pkl'
W_FILENAME = f'/home/costanza/Robot-Control-by-EEG-with-ML/models/{SUBJECT}/CSP_matrix_W.pkl'
TRAIN_PERCENTAGE = 0.6
THRESHOLD = 0.2

MODELS = ['LDA', 'NB', 'SVM', 'RF', 'DT', 'LR']
classifiers = {}
for i in range(len(MODELS)):
    MODEL = MODELS[i]
    CLASSIFIER_FILENAME = f'/home/costanza/Robot-Control-by-EEG-with-ML/models/{SUBJECT}/{MODEL}_model.pkl'

    with open(CLASSIFIER_FILENAME, 'rb') as file:
        model = pickle.load(file)

    classifiers[MODEL] = model

# with open(CLASSIFIER_FILENAME, 'rb') as file:
#     model = pickle.load(file)

with open(W_FILENAME, 'rb') as file:
    W = pickle.load(file)

#%%
# load the mat data
EEG_data = loadmat(MATFILE, struct_as_record = True)
keys = EEG_data.keys()

# Extract data
markers = EEG_data['mrk']
sfreq = EEG_data['nfo']['fs'][0][0][0][0]
EEGdata   = EEG_data['cnt'].T 
nchannels, nsamples = EEGdata.shape
chan_names = [s[0] for s in EEG_data['nfo']['clab'][0][0][0]]
event_onsets  = EEG_data['mrk'][0][0][0] # Time points when events occurred
event_codes   = EEG_data['mrk'][0][0][1] # It contains numerical or categorical labels associated with each event.
labels = np.zeros((1, nsamples), int)
labels[0, event_onsets] = event_codes
cl_lab = [s[0] for s in EEG_data['nfo']['classes'][0][0][0]]
cl1    = cl_lab[0]
cl2    = cl_lab[1]
nclasses = len(cl_lab)
nevents = len(event_onsets)

# Dictionary to store the trials
trials = {}
# The time window in samples to extract for each trial, here 0.5 -- 4.5 seconds
win = np.arange(int(0.5 * sfreq), int(4.5 * sfreq))
# Length of the time window
nsamples = len(win)
# Loop over the classes (left vs right hand)
for cl, code in zip(cl_lab, np.unique(event_codes)):

    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]

    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEGdata[:, win + onset]
  
# Band-Pass Filter

trials_filt = {cl1: butter_bandpass(trials[cl1], 8, 30, sfreq, nsamples),
               cl2: butter_bandpass(trials[cl2], 8, 30, sfreq, nsamples)}

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

# Apply the CSP on both the training and test set
train[cl1] = apply_mix(W, train[cl1], nchannels, nsamples)
train[cl2] = apply_mix(W, train[cl2], nchannels, nsamples)
test[cl1] = apply_mix(W, test[cl1], nchannels, nsamples)
test[cl2] = apply_mix(W, test[cl2], nchannels, nsamples)

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

# 1 for right hand, 0 for left hand

X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T
y_train = np.concatenate((np.zeros(ntrain_l), np.ones(ntrain_r)))
X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T
y_test = np.concatenate((np.zeros(ntest_l), np.ones(ntest_r)))



# %%
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


# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming calibration_displays is a dictionary where each entry contains the predicted probabilities for each classifier
# We first prepare the data in a long-format DataFrame suitable for seaborn
data = []
for name, display in calibration_displays.items():
    for prob in display.y_prob:
        data.append({'Classifier': name, 'Predicted Probability': prob})
df = pd.DataFrame(data)

# Now, we plot the density plots for each classifier
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Predicted Probability', hue='Classifier', fill=True, common_norm=False, alpha=0.5)

# plt.title('Density of Predicted Probabilities Across Classifiers')
# plt.xlabel('Predicted Probability')
# plt.ylabel('Density')
# plt.legend(title='Classifier', title_fontsize='13', fontsize='11')
ax = sns.kdeplot(data=df, x='Predicted Probability', hue='Classifier', fill=True, common_norm=False, alpha=0.5)
ax.legend(title='Classifier', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is already created and contains the predicted probabilities for each classifier
# Plotting the violin plots for each classifier
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='Classifier', y='Predicted Probability', palette='Set3', inner='quartile')

plt.title('Violin Plots of Predicted Probabilities by Classifier')
plt.xlabel('Classifier')
plt.ylabel('Predicted Probability')
plt.xticks(rotation=45)  # Rotate classifier names for better readability
plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is already created and contains the predicted probabilities for each classifier
# Plotting the scatter plot with jitter for each classifier
plt.figure(figsize=(12, 6))
sns.stripplot(data=df, x='Classifier', y='Predicted Probability', palette='Set2', jitter=0.2, size=5)

# Write the title in bold and increase the font size
plt.title('Scatter Plot of Predicted Probabilities by Classifier - Subject 1', fontsize=14, fontweight='bold')

plt.xlabel('Classifier', fontsize=13)
plt.ylabel('Predicted Probability', fontsize=13)
plt.xticks(rotation=45)  # Rotate classifier names for better readability
plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is already created and contains the predicted probabilities for each classifier
# Plotting the bee swarm plot for each classifier
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df, x='Classifier', y='Predicted Probability', palette='Spectral')

plt.title('Scatter Plot of Predicted Probabilities by Classifier - Subject 1', fontsize=14, fontweight='bold')
plt.xlabel('Classifier', fontsize=13)
plt.ylabel('Predicted Probability', fontsize=13)
plt.xticks(rotation=45)  # Rotate classifier names for better readability
plt.tight_layout()
plt.show()

# %%
