import numpy as np
import joblib
from scipy.io import loadmat

"""
This file contains the functions used to process the data from the neural signal.
The data is stored in a .mat file. 
Contrary to the EEGClass class, these functions are made to process one single trial at a time.
"""

def fft_one_segment(trial):
    """
    Compute the absolute value of the FFT of a single trial.

    Parameters:
    - trial: a single trial of shape (59, 400, 1)

    Returns:
    - fft_trial: the FFT of the input trial
    """
    # Compute FFT for the trial along the second axis
    fft_trial = np.fft.fft(trial[:, :, 0], axis=1)

    # Calculate the magnitude of the FFT
    fft_trial = np.abs(fft_trial)

    return fft_trial

def lda_one_segment(fft_segment, subject):
    """
    I use the already fitted LDA model.

    Parameters:
    - segment: a segment of EEG data of shape (59, 400, 1)

    Returns:
    - X_test_lda: the dimensionality reduction of the FFT of the segment. Shape expected: (1, 1)
    """

    # Get the number of features
    n_features = fft_segment.shape[0] * fft_segment.shape[1]
    
    # Reshape the fft_segment 
    X = fft_segment.reshape(1, n_features)

    # Trained model filename
    input_filename = f'/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/lda_model_subject_{subject}.joblib'
    # Load the pre-trained LDA model for the selected subject
    lda_trained_model = joblib.load(input_filename)

    # Apply the dimensionality reduction
    X_test_lda = lda_trained_model.transform(X)

    return X_test_lda

def get_random_segment(data, key):
    segments = data[key]
    random_segment_index = np.random.randint(segments.shape[2])
    return segments[:, :, random_segment_index]

