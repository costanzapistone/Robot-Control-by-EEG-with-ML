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
# Reshape
# Before reshaping the data has a shape (59,400,100), representing 100 trials for each class with dimensions 59x400 which are channels x freq samples
# After reshaping the data has a shape (100, 23600), representing 100 trials for each class with dimensions 59x400 which are channels x freq samples
# Flattening the data like this is a common preprocessing step when using certain machine learning algorithms that expect 2D input rather than 3D input.
# By doing so, the 2D array X can be directly used in algorithms like logistic regression, support vector machines (SVMs), or artificial neural networks.
# Each row in X now represents a single data point (image), and each column represents a feature of that data point (freq intensity).

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# # Get the number of trials for each class
ntrials_cl1 = fft_trials[cl1].shape[2]
ntrials_cl2 = fft_trials[cl2].shape[2]
n_features = fft_trials[cl1].shape[0] * fft_trials[cl1].shape[1]

# # Reshape the fft_trials data to fit the sklearn LDA
# X_cl1 = fft_trials[cl1].reshape(ntrials_cl1, n_features)
# X_cl2 = fft_trials[cl2].reshape(ntrials_cl2, n_features)

X_cl1_t = fft_trials[cl1].transpose(2,1,0)
X_cl2_t = fft_trials[cl2].transpose(2,1,0)
X_cl1 = X_cl1_t.reshape(X_cl1_t.shape[0], -1)
X_cl2 = X_cl2_t.reshape(X_cl2_t.shape[0], -1)

print(X_cl1.shape)
print(X_cl2.shape)

X = np.concatenate((X_cl1, X_cl2), axis = 0)

# Create the labels for the LDA (cl1 = 0, cl2 = 1)
y = np.concatenate((np.zeros(ntrials_cl1), np.ones(ntrials_cl2)))

#%%
# Dimensionality reduction

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from umap import UMAP

#%% 
# PCA
# PCA (Principal Component Analysis) is a widely used linear dimensionality reduction technique that aims to transform high-dimensional data into a lower-dimensional space while preserving as much of the data's variance as possible. The principal components represent the directions of maximum variance in the data, and by choosing a subset of these components, PCA can reduce the number of features in the data.

X_pca = PCA(n_components=2).fit_transform(X)

# Plot the reduced data

plt.figure(figsize=(8,8))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.legend(handles=sc.legend_elements()[0], labels=list(range(2)))#%%
plt.title('Scatter plot of reduced data using PCA')
plt.show()

#%%
# LDA

X_lda = LDA(n_components=1).fit_transform(X, y)

plt.figure(figsize=(8,8))
plt.scatter(X_lda, np.zeros_like(X_lda), c=y)
plt.legend(handles=sc.legend_elements()[0], labels=list(range(2)))
plt.title('Scatter plot of reduced data using LDA')
plt.show()

#%%
# t-SNE
X_tsne = TSNE(n_jobs=-1).fit_transform(X)

plt.figure(figsize=(8,8))
sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.legend(handles=sc.legend_elements()[0], labels=list(range(2)))
plt.title('Scatter plot of reduced data using t-SNE')
plt.show()

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

#%%
# Plot the data before and after LDA in a scatter plot
plt.figure()

plt.scatter(X_cl1_t[:, 0], X_cl1_t[:, 1], label=cl1, color='r')
plt.scatter(X_cl2_t[:, 0], X_cl2_t[:, 1], label=cl2, color='b')

plt.xlabel('')
#%%
# X_cl1_t = np.concatenate([fft_trials[cl1][:, :, i].flatten()[np.newaxis, :] for i in range(ntrials_cl1)], axis=0)
# X_cl2_t = np.concatenate([fft_trials[cl2][:, :, i].flatten()[np.newaxis, :] for i in range(ntrials_cl2)], axis=0)

# X_cl1_t = fft_trials[cl1].transpose(2,1,0)
# X_cl2_t = fft_trials[cl2].transpose(2,1,0)

# X_cl1_reshape = X_cl1_t.reshape(X_cl1_t.shape[0], -1)
# X_cl2_reshape = X_cl2_t.reshape(X_cl2_t.shape[0], -1)

# X_cl1_flatt = X_cl1_t.flatten(start_dim=1)
# X_cl2_flatt = X_cl2_t.flatten(start_dim=1)

# # Check if X_cl1 and X_cl1_t are the same
# print(np.all(X_cl1_flatt == X_cl1_reshape))

# X_cl1_tt = X_cl1_t.transpose(0,2,1)
# X_cl2_tt = X_cl2_t.transpose(0,2,1)

# X_cl1_tt_re = X_cl1_tt.reshape(X_cl1_t.shape[0], -1)
# X_cl2_tt_re = X_cl2_tt.reshape(X_cl2_t.shape[0], -1)

# print(X_cl1_tt_re.shape)

# # Check if they are the same

# print(np.all(X_cl1_tt_re == X_cl1))

#%%
# Save the lda model
import os
import joblib
# save_path_folder = "/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/2"
# os.makedirs(save_path_folder, exist_ok=True)
# model_filename = os.path.join(save_path_folder, f"lda_{sub}.joblib")
# joblib.dump(lda, model_filename)

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.3, random_state=42)


# Create the classifiers and store in a 
classifiers = {'knn': KNeighborsClassifier(n_neighbors=3),
               'nb': GaussianNB(),
               'lr': LogisticRegression(),
               'dt': DecisionTreeClassifier(),
               'svm': SVC(probability=True)}

# Train the classifiers
trained_models = {}
for classifier in classifiers:
    trained_models[classifier] = classifiers[classifier].fit(X_train, y_train)

    # Save the trained model
    # model_filename = os.path.join(save_path_folder, f"{classifier}_{sub}.joblib")
    # joblib.dump(trained_models[classifier], model_filename)

#%%
from sklearn.metrics import accuracy_score

# Evaluate the accuracy of each trained classifier
accuracy_scores = {}
for classifier in trained_models:
    # Predict the labels using the trained classifier
    y_pred = trained_models[classifier].predict(X_test)
    
    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the accuracy score
    accuracy_scores[classifier] = accuracy

# Print the accuracy scores
for classifier, accuracy in accuracy_scores.items():
    print(f'Accuracy of {classifier}: {accuracy:.2f}')
#%%
#evaluate the classifiers
for classifier in classifiers:

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(classifiers[classifier], X_lda, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print(f'{classifier} Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# %%# Calibration Curves

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o", "X"]

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
ax_calibration_curve.set_title(f"Calibration Plots for {sub} - Before Calibration")

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
    calibrated_model.fit(X_train_lda, y_train)
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
        X_test_lda,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title(f"Calibration Plots for {sub} - After Calibration")

plt.show()
# %%
