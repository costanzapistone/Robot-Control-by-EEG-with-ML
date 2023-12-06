# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


# Import the classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Import the metrics
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

class EEGClass():
    """
    A class for reading the EEG data, pre-processing it, and training the classifiers.

    Parameters:
    - data: the EEG data loaded from the .mat file

    Methods:
    - Extract the data
    - Segment the trials
    - Compute the FFT
    - LDA for dimensionality reduction
    - Train the classifiers
    - Evaluate the classifiers
    - Save the trained best classifier
    - Predict labels for new data

    Metrics:
    - Confusion matrix
    - Accuracy
    - AUC
    """

    def __init__(self, file):
        """
        Initialize the EEGClassification class with the data loaded from the .mat file.
        """
        self.file = file

    def extract_data(self):
        """
        Extract the data from the .mat file.
        """
        # Extract the EEG data from the .mat file
        self.EEGdata = self.file['cnt']
        # Extract the sampling frequency
        self.s_freq = self.file['nfo']['fs'][0][0][0][0]
        # Extract the channel names
        self.chan_names = [chan[0][0] for chan in self.file['nfo']['clab'][0][0][0]]
        # Extract the event onsets
        self.event_onsets = self.file['mrk'][0][0][0]
        # Extract the event codes
        self.event_codes = self.file['mrk'][0][0][1]
        # Extract the class labels
        self.cl_lab = [s[0] for s in self.file['nfo']['classes'][0][0][0]]
        # Extract the first class label
        self.cl1 = self.cl_lab[0]
        # Extract the second class label
        self.cl2 = self.cl_lab[1]
        # Extract the electrode positions
        self.xpos = self.file['nfo']['xpos']
        self.ypos = self.file['nfo']['ypos']

        # Compute the time unit
        self.time_unit = 1 / self.s_freq


        self.EEGdata = self.EEGdata.T
        self.n_channels, self.n_samples = self.EEGdata.shape
        self.n_classes = len(self.cl_lab)
        self.n_events = len(self.event_onsets)

    def segment_trials(self):
        """
        Segment the trials from the EEG data.

        Returns:
        - trials: a dictionary containing the segmented trials for each class
        """
        # Dictionary to store the trials, each class gets an entry
        self.trials = {}

        # The time window (in samples) to extract for each trial, here 0.5 -- 4.5 seconds
        self.win = np.arange(int(0.5 * self.s_freq), int(4.5 * self.s_freq))  # 400 samples
        self.nsamples_win = len(self.win)

        for cl, code in zip(self.cl_lab, np.unique(self.event_codes)):
            # Extract the onsets for the class
            cl_onsets = self.event_onsets[self.event_codes == code]
            self.trials[cl] = np.zeros((self.n_channels, self.nsamples_win, len(cl_onsets)))

            # Extract each trial
            for i, onset in enumerate(cl_onsets):
                self.trials[cl][:, :, i] = self.EEGdata[:, self.win + onset]


    def compute_abs_fft(self, self.trials):
        """
        Compute the absolute value of the FFT of the trials.

        Parameters:
        - trials: a dictionary containing the segmented trials for each class

        Returns:
        - fft_trials: a dictionary containing the FFT trials for each class
        """
        # Dictionary to store FFT trials
        self.fft_trials = {}

        for cl in self.cl_lab:
            # Get the segmented data for the current class
            trials_cl = self.trials[cl]

            # Allocate memory for the FFT trials
            self.fft_trials[cl] = np.zeros_like(trials_cl, dtype=complex)

            # Compute FFT for each trial for selected channels
            for i in range(trials_cl.shape[2]):
                self.fft_trials[cl][:, :, i] = fft(trials_cl[:, :, i], axis=1)

        # Calculate the magnitude of the FFT
        for cl in self.fft_trials:
            self.fft_trials[cl] = np.abs(self.fft_trials[cl])

        return self.fft_trials

        def lda(self.fft_trials, self.cl1, self.cl2):
            """
            Perform LDA for dimensionality reduction.

            Parameters:
            - fft_trials: a dictionary containing the FFT trials for each class
            - cl1: the first class label
            - cl2: the second class label

            Returns:
            - X_lda: the LDA features for the two classes
            """
            # Get the number of trials for each class
            self.n_trials_cl1 = self.fft_trials[self.cl1].shape[2]
            self.n_trials_cl2 = self.fft_trials[self.cl2].shape[2]
            self.n_features = self.fft_trials[self.cl1].shape[0] * self.fft_trials[self.cl1].shape[1]

            # Reshape the FFT trials to fit the sklearn LDA function
            self.X_cl1 = self.fft_trials[self.cl1].reshape(self.n_trials_cl1, self.n_features)
            self.X_cl2 = self.fft_trials[self.cl2].reshape(self.n_trials_cl2, self.n_features)

            self.X = np.concatenate((self.X_cl1, self.X_cl2), axis = 0)

            # Create the labels vector (-1 for class 1, 1 for class 2)
            self.y = np.concatenate((-np.ones(self.n_trials_cl1), np.ones(self.n_trials_cl2)))

            lda = LinearDiscriminantAnalysis(n_components=1)
            self.X_lda = lda.fit_transform(self.X, self.y)

            return self.X_lda

            
           