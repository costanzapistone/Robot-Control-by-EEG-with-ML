#%%
import numpy as np
from scipy.io import loadmat

#%%
sub = 'g'

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Save the lda model
import os
import joblib
save_path_folder = "/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/2"
os.makedirs(save_path_folder, exist_ok=True)
model_filename = os.path.join(save_path_folder, f"lda_{sub}.joblib")
joblib.dump(lda, model_filename)

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Create the classifiers and store in a dict
classifiers = {'knn': KNeighborsClassifier(n_neighbors=3),
               'nb': GaussianNB(),
               'lr': LogisticRegression(),
               'dt': DecisionTreeClassifier(),
               'svm': SVC(probability=True)}

# Train the classifiers
trained_models = {}
for classifier in classifiers:
    trained_models[classifier] = classifiers[classifier].fit(X_train_lda, y_train)

    # Save the trained model
    model_filename = os.path.join(save_path_folder, f"{classifier}_{sub}.joblib")
    joblib.dump(trained_models[classifier], model_filename)

# %%
