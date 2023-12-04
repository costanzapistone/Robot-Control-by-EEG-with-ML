#%%
"This script takes the trained model and predicts the labels for the test data, that in this case is the evaluation data."

import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

from analysis_processing_functions import load_and_extract_data, segment_trials, compute_abs_fft, lda, lda_evaluation, segment_trials_seq

#%%
# Load the trained best classifier
folder_path_model = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_models'
file_name_model = 'trained_best_classifier.joblib'
file_path_model = folder_path_model + '/' + file_name_model
best_classifier_instance = joblib.load(file_path_model)

print(f'Best classifier: {best_classifier_instance}')
# %%
  
# Load and preprocess the new EEG evaluation data
file_path = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1a.mat'

# Call the function to load and extract the data
EEGdata, s_freq, chan_names, event_onsets, event_codes, cl_lab, cl1, cl2 = load_and_extract_data(file_path)

# Segmentation
trials = segment_trials(EEGdata, event_onsets, event_codes, cl_lab, segment_length=400)
   
# FFT
fft_trials = compute_abs_fft(trials, cl_lab)

# LDA
X_lda, y_true = lda_evaluation(fft_trials, cl1, cl2)

# Use the loaded classifier to predict labels for the new EEG signal
predicted_labels_new = best_classifier_instance.predict(X_lda)

# Print or use the predicted labels as needed
print("Predicted Labels for New Data:", predicted_labels_new)
# %%
