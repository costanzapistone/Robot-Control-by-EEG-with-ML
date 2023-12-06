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

        return self.trials


    def compute_abs_fft(self):
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


    def lda(self, fft_trials, cl1, cl2):
        """
        Perform LDA for dimensionality reduction.

        Parameters:
        - fft_trials: a dictionary containing the FFT trials for each class
        - cl1: the first class label
        - cl2: the second class label

        Returns:
        - X_lda: the LDA features for the two classes
        - y: the labels vector (-1 for class 1, 1 for class 2)
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

        return self.X_lda, self.y

    def train_and_evaluate_classifiers(self, X_lda, y):
        """
        Train the 5 classifiers and compute the accuracy and AUC for each one.

        Parameters:
        - X_lda: the LDA features for the two classes
        - y: the labels vector (-1 for class 1, 1 for class 2)

        Returns:
        - acc_dict: a dictionary containing the accuracy for each classifier
        - auc_dict: a dictionary containing the AUC for each classifier
        """
        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_lda, self.y, test_size=0.30, random_state=42)

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
            clf.fit(self.X_train, self.y_train)

            # Compute the predictions
            self.y_pred = clf.predict(self.X_test)

            # Compute the accuracy
            self.acc = accuracy_score(self.y_test, self.y_pred)
        
            # Compute the AUC
            if hasattr(clf, "predict_proba"):
                self.proba_class1 = clf.predict_proba(self.X_test)[:, 1]
            else:
                # use decision function for SVC classifier (SVM doesn't have predict_proba)
                self.proba_class1 = clf.decision_function(self.X_test)

            self.acc_dict[clf_name] = accuracy_score(self.y_test, self.y_pred)
            self.auc_dict[clf_name] = roc_auc_score(self.y_test, self.proba_class1)
        
        return self.acc_dict, self.auc_dict
    
    def save_best_classifier(self, acc_avg_dict, auc_avg_dict):
        """
        Save the best classifier based on the accuracy and AUC results over all the 4 subjects.
        In particular the chosen classifier is the one that has the best combined score of the two average metrics (accuracy and AUC weighted equally).
        The trained classifier is then saved as a .joblib file in the folder 'trained_models'.

        Parameters:
        - acc_avg_dict: a dictionary containing the average accuracy among all the subjects for each classifier.
        - auc_avg_dict: a dictionary containing the average AUC among all the subjects for each classifier.

        Returns:
        - best_clf: the best classifier

        """
        # Dictionary to store the combined score for each classifier
        self.combined_score_dict = {}

        for clf_name in self.classifier_dict.keys():
            self.combined_score_dict[clf_name] = acc_avg_dict[clf_name] + auc_avg_dict[clf_name]

        # Find the best classifier
        self.best_clf = self.classifier_dict[self.best_clf_name]

        # Save the best classifier
        folder_path = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_models'
        file_name = 'trained_best_classifier.joblib'
        file_path = folder_path + '/' + file_name
        joblib.dump(self.best_clf, file_path)

        return self.best_clf
    
    def confusion_matrix(self, y_true, y_pred):
        """
        Compute the confusion matrix.

        Parameters:
        - y_true: the true labels
        - y_pred: the predicted labels

        Returns:
        - cm: the confusion matrix
        """
        self.cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.cm, cmap=plt.cm.Blues,alpha=0.3)
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(x=j, y=i, s=self.cm[i, j], va='center', ha='center', size='xx-large')

        plt.title('Confusion matrix', fontsize=16)
        plt.xlabel('Predicted label', fontsize=14)
        plt.ylabel('True label', fontsize=14)
        plt.show()

        return self.cm
    
    