from Learning_from_demonstration import LfD
from scipy.io import loadmat
from EEGClass import EEGClass
import pygame
from pygame.locals import *
import numpy as np
import joblib

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
    input_filename = f'trained_model/lda_model_subject_{subject}.joblib'

    # Load the pre-trained LDA model for the selected subject
    lda_trained_model = joblib.load(input_filename)

    # Apply the dimensionality reduction
    X_test_lda = lda_trained_model.transform(X)

    return X_test_lda

# Create a function that picks a random segment based on the key

def get_random_segment(data, key):
    segments = data[key]
    random_segment_index = np.random.randint(segments.shape[2])
    return segments[:, :, random_segment_index]


# Initialize Pygame
pygame.init()

# Define the subject
subject = 'a'

# Load the .mat file
mat_file = f'data/BCICIV_calib_ds1{subject}.mat'
#mat_file = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{subject}.mat'
data = loadmat(mat_file, struct_as_record=True)

# Create an instance of the class EEGClass
eeg_instance = EEGClass(data)

# Segment the data
trials = eeg_instance.segmentation()

# Compute the FFT
fft_trials = eeg_instance.fft(trials)

# Set up Pygame display (not necessary for non-graphical applications)
width, height = 400, 300
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Key Press Example')

# Loop to interactively process segments
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_LEFT:
                key = list(fft_trials.keys())[0]  # Use the first key
            elif event.key == K_RIGHT:
                key = list(fft_trials.keys())[1]  # Use the second key
            else:
                continue  # Skip processing if the key is not left or right arrow

            # Get a random segment based on the selected key
            segment = get_random_segment(fft_trials, key)

            # Reduce the dimensionality of the segment
            X_test_lda = lda_one_segment(segment, subject)

            # Load the trained model
            trained_model = joblib.load('trained_model/trained_model_best.joblib')

            # Use the trained model to predict the class of the segment
            y_pred = trained_model.predict(X_test_lda)

            # Print the predicted label
            print(y_pred)

# Quit Pygame
pygame.quit()