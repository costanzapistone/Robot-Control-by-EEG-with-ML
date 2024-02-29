#%%
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from processing_functions import butter_bandpass
from processing_functions import logvar
import matplotlib.pyplot as plt
# Define constants
SUBJECT = 'c'
MATFILE = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat'

# Load the models
model = f'/home/costanza/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}/2_Components/Trained_Models/LR_model.pkl'
W = f'/home/costanza/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}/2_Components/Trained_Models/CSP_matrix_W.pkl'

with open(model, 'rb') as file:
    model = pickle.load(file)

with open(W, 'rb') as file:
    W = pickle.load(file)

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


def apply_mix(W, trials):
    """
    Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)
    """
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp

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
# 1 for right hand, 0 for left hand

X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T
y_train = np.concatenate((np.zeros(ntrain_l), np.ones(ntrain_r)))
X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T
y_test = np.concatenate((np.zeros(ntest_l), np.ones(ntest_r)))

# Compute the accuracy of the model

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

#%%
# I want to filter the test set based on the predicted probabilities with a mask
# I have to define a threshold to decide which samples to keep and which to discard
predicted_probs = model.predict_proba(X_test)
# Define thresholds
threshold = 0.1
mask = (predicted_probs[:, 1] > ( 0.5 + threshold)) | (predicted_probs[:, 0] > (0.5 + threshold))

# Apply the mask to the test set
X_test_f = X_test[mask, :]
y_test_f = y_test[mask]

print(X_test_f.shape)
print(y_test_f.shape)

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Define a range of thresholds
threshold_range = np.linspace(0, 0.4, 50)

# Initialize lists to store accuracy scores
accuracy_scores = []

# Iterate over each threshold
for threshold in threshold_range:
    # Create mask
    mask = (predicted_probs[:, 1] > (0.5 + threshold)) | (predicted_probs[:, 0] > (0.5 + threshold))
    
    # Apply mask to test set
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
    
    y_pred = model.predict(X_test_filtered)
    conf_matrix = confusion_matrix(y_test_filtered, y_pred)
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # Append accuracy score to list
    accuracy_scores.append(accuracy)

# Calculate accuracy without any threshold
y_pred= model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

# Plot accuracy vs threshold
plt.plot(threshold_range, accuracy_scores, label='With Threshold')
plt.axhline(y=accuracy, color='r', linestyle='--', label='No Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.xlim(0, 0.4)
plt.title('Accuracy vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Define a range of thresholds
threshold_range = np.linspace(0, 0.4, 50)

# Initialize lists to store accuracy scores and percentage reduction
accuracy_scores = []
percentage_reduction = []

# Total number of samples in the original dataset
total_samples = len(y_test)

# Iterate over each threshold
for threshold in threshold_range:
    # Create mask
    mask = (predicted_probs[:, 1] > (0.5 + threshold)) | (predicted_probs[:, 0] > (0.5 + threshold))
    
    # Calculate the number of samples in the filtered dataset
    filtered_samples = np.sum(mask)
    
    # Calculate the percentage reduction in samples
    reduction_percentage = ((total_samples - filtered_samples) / total_samples) * 100
    
    # Append percentage reduction to the list
    percentage_reduction.append(reduction_percentage)

# Plot percentage reduction vs threshold
plt.plot(threshold_range, percentage_reduction)
plt.xlabel('Threshold')
plt.ylabel('Percentage Reduction')
plt.xlim(0, 0.4)
plt.title('Percentage Reduction of Samples vs Threshold')
plt.grid(True)
plt.show()

# %%
# To quantify the uncertainty of the model i compute the entropy of the predicted probabilities

def entropy(predicted_probs):
    """
    Compute the entropy of predicted probabilities.

    Parameters:
    predicted_probs : array-like, shape (n_samples, n_classes)
        The predicted probabilities for each sample and each class.

    Returns:
    float
        The average entropy value across all samples in the input data.
    """

    # epsilon = 1e-9
    # predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    # Calculate entropy
    entropy_values = -np.sum(predicted_probs * np.log(predicted_probs), axis=1)

    return np.mean(entropy_values)

# Calculate entropy for the original test set
entropy = entropy(predicted_probs)
print('Entropy:', entropy)

# %%
