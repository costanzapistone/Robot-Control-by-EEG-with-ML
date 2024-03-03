#%%
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from processing_functions import butter_bandpass, apply_mix, logvar
import matplotlib.pyplot as plt

# Define constants
SUBJECT = 'g'
MATFILE = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat'
MODEL = 'LR'
CLASSIFIER_FILENAME = f'/home/costanza/Robot-Control-by-EEG-with-ML/models/{SUBJECT}/{MODEL}_model.pkl'
W_FILENAME = f'/home/costanza/Robot-Control-by-EEG-with-ML/models/{SUBJECT}/CSP_matrix_W.pkl'
TRAIN_PERCENTAGE = 0.6
THRESHOLD = 0.2

with open(CLASSIFIER_FILENAME, 'rb') as file:
    model = pickle.load(file)

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


y_pred = model.predict(X_test)
print('Predicted Class:', y_pred)

predicted_probs = model.predict_proba(X_test)
# Define thresholds
threshold = 0.2
mask = (predicted_probs[:, 1] > ( 0.5 + threshold)) | (predicted_probs[:, 0] > (0.5 + threshold))

# Apply the mask to the test set
X_test_f = X_test[mask, :]
y_test_f = y_test[mask]

print(X_test_f.shape)
print(y_test_f.shape)

y_pred_f = model.predict(X_test_f)
print('Predicted Class:', y_pred_f)
    

#%%

# random_index_r = np.random.choice(test[cl2].shape[1])
# sample_r = test[cl2][:, random_index_r]
# y_pred_r = model.predict(sample_r.reshape(1,-1))
# pred_proba_r = model.predict_proba(sample_r.reshape(1,-1))

# print('Random Index:', random_index_r)
# print('Sample:', sample_r)
# print('Predicted Probability:', pred_proba_r)
# print('Predicted Class:', y_pred_r)

# #%%

# if (pred_proba_r[0][1] > (0.5 + THRESHOLD)) or (pred_proba_r[0][0] > (0.5 + THRESHOLD)):

#     if y_pred_r == 1:
        
#         print('Predicted Movement: Right')
#         # Move the right robot
#         # key_press_data['right']['TP'] += 1

#     else:
#         print('Predicted Movement: Left')
#         # Move the left robot
#         # key_press_data['right']['FN'] += 1

# else:
#     print('Predicted Movement: No Movement')
#     # Move the left robot
#     # key_press_data['right']['No Action'] += 1

# # %%

# random_index_l = np.random.choice(test[cl1].shape[1])
# sample_l = test[cl1][:, random_index_l]
# y_pred_l = model.predict(sample_l.reshape(1,-1))
# pred_proba_l = model.predict_proba(sample_l.reshape(1,-1))

# print('Random Index:', random_index_l)
# print('Sample:', sample_l)
# print('Predicted Probability:', pred_proba_l)
# print('Predicted Class:', y_pred_l)

# if (pred_proba_l[0][1] > (0.5 + THRESHOLD)) or (pred_proba_l[0][0] > (0.5 + THRESHOLD)):

#     if y_pred_l == 1:

#         print('Predicted Movement: Right')
#         # Move the right 
#         # key_press_data['left']['FP'] += 1                    

#     else:
#         print('Predicted Movement: Left')
#         # Move the left robot
#         # key_press_data['left']['TN'] += 1

# else:
    
#     print('Predicted Movement: No Movement')
#     # Move the left robot
#     # key_press_data['left']['No Action'] += 1

# # %%
