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
# %%
# Since the features we are looking for (a decrease in mu-activity 8Hz - 12Hz) is a frequency feature
# let's plot the PSD (Power Spectral Density) of the trials to get an idea of the frequency content of the data.

#%%
from matplotlib import mlab

def psd(trials):
    """
    Calculates for each trial the PSD.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal

    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial
    freqs : list of floats
        The frequencies for which the PSD was computed (useful for plotting later)
    """

    ntrials = trials.shape[2]
    # Put 201 if the window is 4 seconds, 101 if the window is 2 seconds.
    # The second dimension corresponds to the number of points in the frequency domain for the PSD.
    trial_PSD = np.zeros((nchannels, 201, ntrials))

    for trial in range(ntrials):
        for ch in range(nchannels):
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sfreq, noverlap=0)
            trial_PSD[ch, :, trial] = PSD.ravel()

    return trial_PSD, freqs


#%%
# Calculate the PSD
psd_r, freqs = psd(trials[cl1])
psd_l, freqs = psd(trials[cl2])
trial_PSD = {cl1: psd_r, cl2: psd_l}
# %%
import matplotlib.pyplot as plt

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


def plot_psd(trial_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    """
    Plots the PSD calculated by the function psd.

    Parameters
    ----------
    trial_PSD : 3d-array
        The PSD for each trial (output from the function psd)
    freqs : list of floats
        The frequencies for which the PSD was computed
    chan_ind : int
        The index of the channel to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    """

    plt.figure(figsize=(12,5))
    
    # Subplots for each channel
    for i, ch_idx in enumerate(chan_ind):
        ax = plt.subplot(1,len(chan_ind),i+1)

        # Plot the PSD for each class
        for cl in trial_PSD.keys():
            plt.plot(freqs, np.mean(trial_PSD[cl][ch_idx,:,:], axis=1), label=cl)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB)')
        plt.xlim(1, 30)

        if chan_lab == None:
            plt.title('Channel %d' % (ch_idx+1))
        else:
            plt.title(chan_lab[i])

        plt.grid()

        if maxy != None:
            plt.ylim(0, maxy)
        plt.legend()
#%%
# Let's plot for 3 channels:
    
# 1. C3: Central, left 
# 2. Cz: Central, central
# 3. C4: Central, right
    
# %%
plot_psd(trial_PSD,
         freqs,
        #  chan_ind = [0,58,-1],
         chan_ind=[chan_names.index(ch) for ch in ['C3','Cz','C4']],
         chan_lab=['C3','Cz','C4'],
         maxy=500
        )
# %%
# Consideration on the PSD plot:
# Con una finestra di 2 secondi (da 0.5 a 2.5) è possibile notare un segnale molto meno rumoroso rispetto a quello della finestra da 0.5 a 4.5.
# Le feature discriminative sono più chiare e delineate.

# Nell'emisfero destro (C4) mu per la mano sinistra dovrebbe essere più basso a causa dell'event related desynchronization (ERD)
# Nell'emisfero sinistro (C3) mu per la mano destra dovrebbe essere più basso a causa dell'event related desynchronization (ERD)
# AL centro l'andamento dovrebbe essere simile per entrambi

# %%
# Next steps:
# 1. We need to quantify the amount of new activity present in each trial
# 2. Make a model that describes the expected values of new activity for each class
# 3. Test the model to unseen data

# %%
import scipy.signal as sg

def bandpass(trials, lo, hi, sfreq):
    """
    Designs and applies a bandpass filter to the signal.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sfreq : float
        Sampling frequency of the signal (in Hz)

    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    """
    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff, but the resulting signal might be shifted in time.
    # Define the filter
    a, b = sg.iirfilter(6, [lo/(sfreq/2.0), hi/(sfreq/2.0)])

    # Apply to each trial
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = sg.filtfilt(a, b, trials[:,:,i], axis=1)

    return trials_filt

# %%
# Apply the bandpass filter
trials_filt = {cl1: bandpass(trials[cl1], 7, 15, sfreq),
               cl2: bandpass(trials[cl2], 7, 15, sfreq)}

# %%
# Plot the PSD of the filtered signal

psd_r, freqs = psd(trials_filt[cl1])
psd_l, freqs = psd(trials_filt[cl2])
filtered_trial_PSD = {cl1: psd_r, cl2: psd_l}

plot_psd(filtered_trial_PSD,
        freqs,
        chan_ind=[chan_names.index(ch) for ch in ['C3','Cz','C4']],
        chan_lab=['C3','Cz','C4'],
        maxy=500
        )

# %%
# Calculate the log(var) of the trials 

def logvar(trials):
    """
    Calculates the log-variance of each channel.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal

    Returns
    -------
    feat : 2d-array (channels x trials)
        The log-variance features for each channel 
    """
    return np.log(np.var(trials, axis=1))

# Apply to the filtered signal
filtered_trials_logvar = {cl1: logvar(trials_filt[cl1]),
                          cl2: logvar(trials_filt[cl2])}

print('shape of filtered_trials_logvar[cl1]:', filtered_trials_logvar[cl1].shape)
# I have now 2-d data (channels x trials) instead of 3-d data (channels x samples x trials) where the second dimension is the log-variance of the signal of the trial.
# %%
# Plot the log-variance features for each channel as a bar chart

def plot_logvar(trials):
    """
    Plots the log-variance features for each channel as a bar chart.
    
    Parameters
    ----------
    trials : 2d-array (channels x trials)
        The EEGsignal
    
    """
    plt.figure(figsize=(12,5))
    
    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('Log-variance features for each channel')
    plt.xlabel('Channels')
    plt.ylabel('Log-variance')
    plt.legend(cl_lab)

#%%
plot_logvar(filtered_trials_logvar)

# rms_trials_cl1 = calculate_rms(trials[cl1])
# rms_trials_cl2 = calculate_rms(trials[cl2])
# print('Shape of rms_trials_cl1:', rms_trials_cl1.shape)
# print('Shape of rms_trials_cl2:', rms_trials_cl2.shape)

# we want to maximize the variance between two classes.

# %%
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

#%%
# X = np.concatenate((trials_filt[cl1], trials_filt[cl2]))
# y = np.concatenate((-np.ones(trials_filt[cl1].shape[2]), np.ones(trials_filt[cl2].shape[2])))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
# Apply the functions 
W = csp(trials_filt[cl1], trials_filt[cl2])

trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
              cl2: apply_mix(W, trials_filt[cl2])}
# %%
# To see the result of the CSP transformation, plot the log-variance features for each channel as a bar chart

trials_logvar_csp = {cl1: logvar(trials_csp[cl1]),
                     cl2: logvar(trials_csp[cl2])}

plot_logvar(trials_logvar_csp)

#%%
# trials_std_csp = {cl1: calculate_std(trials_csp[cl1]), cl2: calculate_std(trials_csp[cl2])}
# plot_logvar(trials_std_csp)

# #%%
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from sklearn.calibration import CalibrationDisplay
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from numpy import mean
# from sklearn.model_selection import train_test_split
# from numpy import std
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# X = np.concatenate((trials_logvar_csp[cl1], trials_logvar_csp[cl2]))
# y = np.concatenate((-np.ones(trials_logvar_csp[cl1].shape[0]), np.ones(trials_logvar_csp[cl2].shape[0])))

# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Create the classifiers and store in a 
# classifiers = {'knn': KNeighborsClassifier(n_neighbors=3),
#                'nb': GaussianNB(),
#                'lr': LogisticRegression(),
#                'dt': DecisionTreeClassifier(),
#                'svm': SVC(probability=True),
#                'lda': LinearDiscriminantAnalysis()}

# #%%Train the classifiers
# trained_models = {}
# for classifier in classifiers:
#     trained_models[classifier] = classifiers[classifier].fit(X_train, y_train)

# #%%
# from sklearn.metrics import accuracy_score

# # Evaluate the accuracy of each trained classifier
# accuracy_scores = {}
# for classifier in trained_models:
#     # Predict the labels using the trained classifier
#     y_pred = trained_models[classifier].predict(X_test)
    
#     # Calculate the accuracy score
#     accuracy = accuracy_score(y_test, y_pred)
    
#     # Store the accuracy score
#     accuracy_scores[classifier] = accuracy

# # Print the accuracy scores
# for classifier, accuracy in accuracy_scores.items():
#     print(f'Accuracy of {classifier}: {accuracy:.2f}')

# #%%    
# #evaluate the classifiers
# for classifier in classifiers:

#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     n_scores = cross_val_score(classifiers[classifier], X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#     # report performance
#     print(f'{classifier} Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# # %%# Calibration Curves

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from sklearn.calibration import CalibrationDisplay

# fig = plt.figure(figsize=(10, 10))
# gs = GridSpec(4, 2)
# colors = plt.get_cmap("Dark2")

# ax_calibration_curve = fig.add_subplot(gs[:2, :2])
# calibration_displays = {}
# markers = ["^", "v", "s", "o", "X", "P"]

# for i, (name, clf) in enumerate(classifiers.items()):
#     display = CalibrationDisplay.from_estimator(
#         clf,
#         X_test,
#         y_test,
#         n_bins=10,
#         name=name,
#         ax=ax_calibration_curve,
#         color=colors(i),
#         marker=markers[i],
#     )
#     calibration_displays[name] = display

# ax_calibration_curve.grid()
# ax_calibration_curve.set_title(f"Calibration Plots  - Before Calibration")

# plt.show()

# %%
# We can see the result also by plotting the PSD of the CSP-filtered signal
psd_r, freqs = psd(trials_csp[cl1])
psd_l, freqs = psd(trials_csp[cl2])
csp_trial_PSD = {cl1: psd_r, cl2: psd_l}

plot_psd(csp_trial_PSD,
         freqs,
         chan_ind=[0,58,-1],
         chan_lab=['First Component','Middle Component','Last Component'],
         maxy=0.75
        )
# %%
# Scatter plot
def plot_scatter(right,left):
    plt.figure()
    plt.scatter(right[0,:], right[-1,:], c='b')
    plt.scatter(left[0,:], left[-1,:], c='r')
    plt.xlabel('First component')
    plt.ylabel('Last component')
    plt.legend(cl_lab)
# %%
# After applying the CSP transformation, the two classes are well separated.
plot_scatter(trials_logvar_csp[cl1], trials_logvar_csp[cl2])
#%%
# Before applying the CSP transformation, the two classes are mixed together.
plot_scatter(filtered_trials_logvar[cl1], filtered_trials_logvar[cl2])
# %%
# Split the data into training and test set
train_percentage = 0.7

# Number of trials for each class for train and test set
ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
ntrain_l = int(trials_filt[cl2].shape[2] * train_percentage)

ntest_r = trials_filt[cl1].shape[2] - ntrain_r
ntest_l = trials_filt[cl2].shape[2] - ntrain_l

train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
         cl2: trials_filt[cl2][:,:,:ntrain_l]}

test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
        cl2: trials_filt[cl2][:,:,ntrain_l:]}
# %%
# Calculate the CSP filters on the training set
W = csp(train[cl1], train[cl2])

# %%
train[cl1] = apply_mix(W, train[cl1])
train[cl2] = apply_mix(W, train[cl2])
test[cl1]  = apply_mix(W, test[cl1])
test[cl2]  = apply_mix(W, test[cl2])

# %%
# Select the first and last component for classification
comp = np.array([0, -1])
train[cl1] = train[cl1][comp,:,:]
train[cl2] = train[cl2][comp,:,:]
test[cl1]  = test[cl1][comp,:,:]
test[cl2]  = test[cl2][comp,:,:]

# %%
# Calculate the log-variance features
train[cl1] = logvar(train[cl1])
train[cl2] = logvar(train[cl2])
test[cl1]  = logvar(test[cl1])
test[cl2]  = logvar(test[cl2])
    
# %%
# Classification with LDA
def train_LDA(class1, class2):
    """ train a LDA classifier
    
    Parameters
    ----------
    class1 = (observations x features) for class 1
    class2 = (observations x features) for class 2

    Returns
    -------
    the projection matrix
    the offset b

    """

    nclasses = 2

    nclass1 = class1.shape[0]
    nclass2 = class2.shape[0]

    prior1 = nclass1 / float(nclass1 + nclass2)
    prior2 = nclass2 / float(nclass1 + nclass1)

    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)

    class1_centered = class1 - mean1
    class2_centered = class2 - mean2

    # Calculate the covariance between the features
    cov1 = class1_centered.T.dot(class1_centered) / (nclass1 - nclasses)
    cov2 = class2_centered.T.dot(class2_centered) / (nclass2 - nclasses)

    # Calculate the projection matrix and offset
    W = (mean2 - mean1).dot(linalg.pinv(prior1 * cov1 + prior2 * cov2))
    b = (prior1 * mean1 + prior2 * mean2).dot(W)

    return (W, b)

def apply_LDA(test, W, b):
    """ apply a trained LDA to data
    
    Parameters
    ----------
    W = projection matrix
    b = offset
    test = array (features x trials) containing the data

    Returns
    -------
    a list containing the class labels for each trial

    """

    ntrials = test.shape[1]

    prediction = []
    for i in range(ntrials):

        result = W.dot(test[:,i]) - b
        if result <= 0:
            prediction.append(1)
        else:
            prediction.append(2)
    
    return np.array(prediction)
# %%
# Train the LDA classifier
W, b = train_LDA(train[cl1].T, train[cl2].T)

print('W:', W)
print('b:', b)
# %%
plot_scatter(train[cl1], train[cl2])
plt.title('Training data')

# Calculate the decision boundary
x = np.arange(-5,1,0.1)
y = (b - W[0] * x) / W[1]

# Plot the decision boundary
plt.plot(x,y, linestyle = '--', color='k', linewidth=2)
plt.xlim(-5,1)
plt.ylim(-2.2,1)
# %%
plot_scatter(test[cl1], test[cl2])
plt.title('Test data')
# Plot the decision boundary
plt.plot(x,y, linestyle = '--', color='k', linewidth=2)
plt.xlim(-5,1)
plt.ylim(-2.2,1)
# %%
# print confusion matrix and accuracy
conf = np.array([
    [(apply_LDA(test[cl1], W, b) == 1).sum(), (apply_LDA(test[cl2], W, b) == 1).sum()],
    [(apply_LDA(test[cl1], W, b) == 2).sum(), (apply_LDA(test[cl2], W, b) == 2).sum()]
])

print('Confusion matrix:')
print(conf)
print('accuracy: %.3f' % (np.sum(np.diag(conf)) / float(np.sum(conf))))
# %%
