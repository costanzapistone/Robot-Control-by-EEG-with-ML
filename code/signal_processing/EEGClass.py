# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import joblib


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
        Initialize the EEGClass with the data loaded from the .mat file and extract the data.
        """
        self.file = file
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
        # Extract the number of channels and samples
        self.n_channels, self.n_samples = self.EEGdata.shape
        # Extract the number of classes
        self.n_classes = len(self.cl_lab)
        # Extract the number of events
        self.n_events = len(self.event_onsets)

    def segmentation(self):
        """
        Segment the trials from the EEG data.

        Returns:
        - trials: a dictionary containing the segmented trials for each class
        """
        # Dictionary to store the trials, each class gets an entry
        trials = {}

        # The time window (in samples) to extract for each trial, here 0.5 -- 4.5 seconds
        win = np.arange(int(0.5 * self.s_freq), int(4.5 * self.s_freq))  # 400 samples
        nsamples_win = len(win)

        for cl, code in zip(self.cl_lab, np.unique(self.event_codes)):
            # Extract the onsets for the class
            cl_onsets = self.event_onsets[self.event_codes == code]
            trials[cl] = np.zeros((self.n_channels, nsamples_win, len(cl_onsets)))

            # Extract each trial
            for i, onset in enumerate(cl_onsets):
                trials[cl][:, :, i] = self.EEGdata[:, win + onset]

        return trials

    def sequential_segmentation(self):
        """
        Segment the trials from the EEG data.

        Returns:
        - trials: a dictionary containing all segmented trials, regardless of the class
        """
        # Dictionary to store the trials, each class gets an entry
        trials_all = []

        # The time window (in samples) to extract for each trial, here 0.5 -- 4.5 seconds
        win = np.arange(int(0.5 * self.s_freq), int(4.5 * self.s_freq)) # 400 samples
        
        # Loop over all the events
        for i in range(self.event_onsets.shape[1]):
            # Extract the index of the event_onset
            onset = self.event_onsets[:, i]

            # Append the segmented trial to the list
            trials_all.append(self.EEGdata[:, win + onset])

        return trials_all

    def fft(self, trials):
        """
        Compute the absolute value of the FFT of the trials.

        Parameters:
        - trials: a dictionary containing the segmented trials for each class

        Returns:
        - fft_trials: a dictionary containing the FFT trials for each class
        """
        # Dictionary to store FFT trials
        fft_trials = {}

        for cl in self.cl_lab:
            # Get the segmented data for the current class
            trials_cl = trials[cl]

            # Allocate memory for the FFT trials
            fft_trials[cl] = np.zeros_like(trials_cl, dtype=complex)

            # Compute FFT for each trial for selected channels
            for i in range(trials_cl.shape[2]):
                fft_trials[cl][:, :, i] = fft(trials_cl[:, :, i], axis=1)

        # Calculate the magnitude of the FFT
        for cl in fft_trials:
            fft_trials[cl] = np.abs(fft_trials[cl])

        return fft_trials

    def lda(self, fft_trials):
        """
        Perform LDA for dimensionality reduction. The data is split into training and test sets before performing LDA.

        Parameters:
        - fft_trials: a dictionary containing the FFT trials for each class

        Returns:
        - X_train_lda: LDA features for the training set
        - X_test_lda: LDA features for the test set
        - y_train: labels vector for the training set
        - y_test: labels vector for the test set
        """
        # Get the number of trials for each class
        n_trials_cl1 = fft_trials[self.cl1].shape[2]
        n_trials_cl2 = fft_trials[self.cl2].shape[2]
        n_features = fft_trials[self.cl1].shape[0] * fft_trials[self.cl1].shape[1]

        # Reshape the FFT trials to fit the sklearn LDA function
        X_cl1 = fft_trials[self.cl1].reshape(n_trials_cl1, n_features)
        X_cl2 = fft_trials[self.cl2].reshape(n_trials_cl2, n_features)

        X = np.concatenate((X_cl1, X_cl2), axis = 0)

        # Create the labels vector (-1 for class 1, 1 for class 2)
        y = np.concatenate((-np.ones(n_trials_cl1), np.ones(n_trials_cl2)))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        lda = LinearDiscriminantAnalysis(n_components=1)
        X_train_lda = lda.fit_transform(X_train, y_train)

        X_test_lda = lda.transform(X_test)

        return X_train_lda, X_test_lda, y_train, y_test

    def train_classifiers(self, X_train_lda, y_train):
        """
        Train the classifiers on the training set.

        Parameters:
        - X_train_lda: LDA features for the training set
        - y_train: labels vector for the training set

        Returns:
        - trained_classifiers: a dictionary containing the trained classifiers
        """

        # Store the classifiers in a dictionary
        self.classifier_dict = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'NB': GaussianNB(),
            'LR': LogisticRegression(),
            'DT': DecisionTreeClassifier(),
            'SVM': SVC(probability=True),  # Use probability=True for SVM to enable predict_proba
        }

        trained_classifiers = {}

        for clf_name, clf in self.classifier_dict.items():
            # Train the classifier
            clf.fit(X_train_lda, y_train)
            trained_classifiers[clf_name] = clf

        return trained_classifiers
    
    def evaluate_classifiers(self, trained_classifiers, X_test_lda, y_test):
        """
        Evaluate the trained classifiers on the test set.

        Parameters:
        - trained_classifiers: a dictionary containing the trained classifiers
        - X_test_lda: LDA features for the test set
        - y_test: labels vector for the test set

        Returns:
        - acc_dict: a dictionary containing the accuracy for each classifier
        - auc_dict: a dictionary containing the AUC for each classifier
        """
        acc_dict = {}
        auc_dict = {}

        for clf_name, clf in trained_classifiers.items():
            # Compute the predictions
            y_pred = clf.predict(X_test_lda)

            # Compute the accuracy
            acc = accuracy_score(y_test, y_pred)

            # Compute the AUC
            if hasattr(clf, "predict_proba"):
                proba_class1 = clf.predict_proba(X_test_lda)[:, 1]
            else:
                proba_class1 = clf.decision_function(X_test_lda)

            acc_dict[clf_name] = acc
            auc_dict[clf_name] = roc_auc_score(y_test, proba_class1)

        return acc_dict, auc_dict
    
    def combined_score(self, results_dict):
        """
        Compute the combined score for each classifier.

        Parameters:
        - results_dict: a dictionary containing the accuracy and AUC for each classifier for each subject
        
        Returns:
        - combined_score_dict: a dictionary containing the combined score for each classifier
        """
        # Create dictionaries to store the average accuracy and AUC for each classifier
        acc_avg_dict = {}
        auc_avg_dict = {}

        for classifier in results_dict[0]['acc']:
            acc_avg_dict[classifier] = np.mean([subject['acc'][classifier] for subject in results_dict])
            auc_avg_dict[classifier] = np.mean([subject['auc'][classifier] for subject in results_dict])

        combined_score_dict = {}

        for classifier in results_dict[0]['acc']:
            combined_score_dict[classifier] = 0.5 * acc_avg_dict[classifier] + 0.5 * auc_avg_dict[classifier]

        return combined_score_dict

    def train_and_evaluate_classifiers(self, X_train_lda, X_test_lda, y_train, y_test):
        """
        Train the 5 classifiers and compute the accuracy and AUC for each one.

        Parameters:
        - X_lda: the LDA features for the two classes
        - y: the labels vector (-1 for class 1, 1 for class 2)

        Returns:
        - acc_dict: a dictionary containing the accuracy for each classifier
        - auc_dict: a dictionary containing the AUC for each classifier
        """
        
        # Store the classifiers in a dictionary
        self.classifier_dict = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'NB': GaussianNB(),
            'LR': LogisticRegression(),
            'DT': DecisionTreeClassifier(),
            'SVM': SVC(probability=True),  # Use probability=True for SVM to enable predict_proba
        }

        # Dictionary to store the accuracy and AUC for each classifier
        self.acc_dict = {}
        self.auc_dict = {}

        for clf_name, clf in self.classifier_dict.items():

            # Train the classifier
            clf.fit(X_train_lda, y_train)

            # Compute the predictions
            y_pred = clf.predict(X_test_lda)

            # Compute the accuracy
            self.acc = accuracy_score(y_test, y_pred)
        
            # Compute the AUC
            if hasattr(clf, "predict_proba"):
                proba_class1 = clf.predict_proba(X_test_lda)[:, 1]
            else:
                # use decision function for SVC classifier (SVM doesn't have predict_proba)
                proba_class1 = clf.decision_function(X_test_lda)

            self.acc_dict[clf_name] = accuracy_score(y_test, y_pred)
            self.auc_dict[clf_name] = roc_auc_score(y_test, proba_class1)
        
        return self.acc_dict, self.auc_dict
