#%%
from Learning_from_demonstration import LfD
import rospy
from scipy.io import loadmat
from EEGClass import EEGClass
import keyboard #pip install keyboard numpy
import numpy as np
#%%
# Load the .mat file you want to evaluate
mat_file = '/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1a.mat'
data = loadmat(mat_file, struct_as_record=True)

#%%
# Function to process just one segment
def lda_one_segment(fft_segment):
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

    # Load the trained model
    

#%%
# Create a function that picks a random segment based on the key

def get_random_segment(key):
    segments = data[key]
    random_segment_index = np.random.randint(segments.shape[2])
    return segments[:, :, random_segment_index]

