#%%
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from processing_functions import butter_bandpass
from processing_functions import logvar
from numpy import linalg

# Define constants
SUBJECT = 'e'
MATFILE = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat'

# Load the classifier
model_filename_2 = f'/home/costanza/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}/2_Components/Trained_Models/LDA_model.pkl'
model_filename_all =  f'/home/costanza/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}/All_Components/Trained_Models/LDA_model.pkl'

with open(model_filename_2, 'rb') as file:
    model_2 = pickle.load(file)

with open(model_filename_all, 'rb') as file:
    model_all = pickle.load(file)

# Load the CSP transformation matrix
W_filename_2 = f'/home/costanza/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}/2_Components/Trained_Models/CSP_matrix_W.pkl'

with open(W_filename_2, 'rb') as file:
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
  
# Band-Pass Filtering

lowcut = 8
highcut = 30

trials_filt = {cl1: butter_bandpass(trials[cl1], lowcut, highcut, sfreq, nsamples),
               cl2: butter_bandpass(trials[cl2], lowcut, highcut, sfreq, nsamples)}

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

# Common Spatial Patterns (CSP) 
train_percentage = 0.7

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


X_train = np.concatenate((train[cl1], train[cl2]), axis=1).T
X_test = np.concatenate((test[cl1], test[cl2]), axis=1).T
y_train = np.zeros(X_train.shape[0], dtype=int) # 0 for left hand
y_train[:ntrain_r] = 1
y_test = np.zeros(X_test.shape[0], dtype=int)
y_test[:ntest_r] = 1

predicted_probs = model_2.predict_proba(X_test)
#%%import numpy as np
from sklearn.metrics import confusion_matrix

# Define thresholds
threshold = 0.7

X_test_cl1 = test[cl1].T
X_test_cl2 = test[cl2].T
y_test_cl1 = np.zeros(X_test_cl1.shape[0], dtype=int)
y_test_cl2 = np.ones(X_test_cl2.shape[0], dtype=int)

predicted_probs_cl1 = model_2.predict_proba(X_test_cl1)
predicted_probs_cl2 = model_2.predict_proba(X_test_cl2)

#%%
indices_to_keep_cl1 = []

for i, prob in enumerate(predicted_probs_cl1):
    if (prob[1] > threshold):
        indices_to_keep_cl1.append(i)

X_test_cl1_filtered = X_test_cl1[indices_to_keep_cl1,:]
y_test_cl1_filtered = y_test_cl1[indices_to_keep_cl1]

print('X_test_cl1.shape :',X_test_cl1.shape)
print('y_test_cl1.shape :',y_test_cl1.shape)
print('X_test_cl1_filtered.shape :',X_test_cl1_filtered.shape)
print('y_test_cl1_filtered.shape :',y_test_cl1_filtered.shape)

#%%
indices_to_keep_cl2 = []

for i, prob in enumerate(predicted_probs_cl2):
    if (prob[0] > threshold):
        indices_to_keep_cl2.append(i)

X_test_cl2_filtered = X_test_cl2[indices_to_keep_cl2,:]
y_test_cl2_filtered = y_test_cl2[indices_to_keep_cl2]

print('X_test_cl2.shape :',X_test_cl2.shape)
print('y_test_cl2.shape :',y_test_cl2.shape)
print('X_test_cl2_filtered.shape :',X_test_cl2_filtered.shape)
print('y_test_cl2_filtered.shape :',y_test_cl2_filtered.shape)
#%%
# Concatenate the filtered data
# X_test_f = np.concatenate((X_test_cl1_filtered, X_test_cl2_filtered), axis=0)
# Concatenate the filtered data
X_test_f = np.concatenate((X_test_cl1_filtered, X_test_cl2_filtered), axis=0)
y_test_f = np.zeros(X_test_f.shape[0], dtype=int)
y_test_f[:len(y_test_cl1_filtered)] = 1
# y_test_f = np.concatenate((y_test_cl1_filtered, y_test_cl2_filtered))


print('X_test_f.shape :',X_test_f.shape)
print('y_test_f.shape :',y_test_f.shape)

# Print the original data shape
print('X_test.shape :',X_test.shape)
print('y_test.shape :',y_test.shape)

    
# %%
# Evaluate the confusion matrix and accuracy for both dataset
y_pred = model_2.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print('Accuracy:', accuracy)
print('Confusion matrix:', conf_matrix)

y_pred_f = model_2.predict(X_test_f)
conf_matrix_f = confusion_matrix(y_test_f, y_pred_f)
accuracy_f = np.trace(conf_matrix_f) / np.sum(conf_matrix_f)
print('Accuracy of the filtered dataset:', accuracy_f)
print('Confusion matrix of the filtered dataset:', conf_matrix_f)

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

    epsilon = 1e-9
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    # Calculate entropy
    entropy_values = -np.sum(predicted_probs * np.log(predicted_probs), axis=1)

    return np.mean(entropy_values)

# Calculate entropy for the original test set
entropy = entropy(predicted_probs)
print('Entropy:', entropy)

# %%
