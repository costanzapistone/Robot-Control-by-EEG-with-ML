import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib



def load_and_extract_data(file_path):
    # Load the MATLAB file
    EEG_data = loadmat(file_path, struct_as_record=True)

    # Extract data
    markers = EEG_data['mrk']
    s_freq = EEG_data['nfo']['fs'][0][0][0][0]
    EEGdata = EEG_data['cnt'].T
    n_channels, n_samples = EEGdata.shape

    time_unit = 1 / s_freq
    print("Time Unit:", time_unit, "seconds")

    chan_names = [s[0] for s in EEG_data['nfo']['clab'][0][0][0]]

    event_onsets = EEG_data['mrk'][0][0][0]  # Time points when events occurred
    event_codes = EEG_data['mrk'][0][0][1]  # It contains numerical or categorical labels associated with each event.
    event_onset_time = event_onsets * time_unit  # Seconds

    # Creates an array of zeros and then assigns the event codes to the corresponding positions based on the event onsets.
    labels = np.zeros((1, n_samples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in EEG_data['nfo']['classes'][0][0][0]]
    cl1    = cl_lab[0]
    cl2    = cl_lab[1]

    # Electrode positions
    xpos = EEG_data['nfo']['xpos']
    ypos = EEG_data['nfo']['ypos']

    n_classes = len(cl_lab)
    n_events  = len(event_onsets)

    return EEGdata, s_freq, chan_names, event_onsets, event_codes, cl_lab, cl1, cl2

def segment_trials(EEGdata, event_onsets, event_codes, s_freq, cl_lab):
    # Dictionary to store the trials, each class gets an entry
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 4.5 seconds
    win = np.arange(int(0.5 * s_freq), int(4.5 * s_freq))  # 400 samples
    nsamples_win = len(win)

    for cl, code in zip(cl_lab, np.unique(event_codes)):
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        trials[cl] = np.zeros((len(EEGdata[:, 0]), nsamples_win, len(cl_onsets)))

        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:, :, i] = EEGdata[:, win + onset]

    return trials

def segment_trials_seq(EEGdata, event_onsets, event_codes, cl_lab, segment_length=400):
    # Dictionary to store the trials, each class gets an entry
    trials = {cl: [] for cl in cl_lab}

    # Iterate over each event code
    for code in np.unique(event_codes):
        # Extract the onsets for the current event code
        cl_onsets = event_onsets[event_codes == code]

        # Extract each trial
        for onset in cl_onsets:
            # Ensure that the segment doesn't go beyond the EEGdata length
            if onset + segment_length <= EEGdata.shape[1]:
                # Extract the segment of fixed length
                segment = EEGdata[:, onset:onset + segment_length]

                # Store the segment in the corresponding class entry
                trials[cl_lab[code]].append(segment)

    return trials

def compute_abs_fft(trials, cl_lab):
    # Dictionary to store FFT trials
    fft_trials = {}

    for cl in cl_lab:
        # Get the segmented data for the current class
        segmented_data = trials[cl]

        # Allocate memory for FFT results
        fft_trials[cl] = np.zeros_like(segmented_data, dtype=complex)

        # Compute FFT for each trial for selected channels
        for i in range(segmented_data.shape[2]):
            # Apply FFT along the second axis (axis=1), which represents the time samples
            fft_trials[cl][:, :, i] = fft(segmented_data[:, :, i], axis=1)

    # Calculate the magnitude of the FFT
    for cl in fft_trials:
        fft_trials[cl] = np.abs(fft_trials[cl])

    return fft_trials

def lda(fft_trials, cl1, cl2):
    n_trials_cl1 = fft_trials[cl1].shape[2]
    n_trials_cl2 = fft_trials[cl2].shape[2]
    n_features = fft_trials[cl1].shape[0] * fft_trials[cl1].shape[1]

    # Reshape the FFT data: Flatten each trial into a single row
    X_cl1 = fft_trials[cl1].reshape(n_trials_cl1, n_features)
    X_cl2 = fft_trials[cl2].reshape(n_trials_cl2, n_features)

    X = np.concatenate([X_cl1, X_cl2], axis=0)

    # Create labels: 0 for class 1, 1 for class 2
    y = np.concatenate([np.zeros(n_trials_cl1), np.ones(n_trials_cl2)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(X_train, y_train)

    X_test_lda = lda.transform(X_test)

    return X_train_lda, X_test_lda, y_train, y_test

def lda_evaluation(fft_trials, cl1, cl2):
    n_trials_cl1 = fft_trials[cl1].shape[2]
    n_trials_cl2 = fft_trials[cl2].shape[2]
    n_features = fft_trials[cl1].shape[0] * fft_trials[cl1].shape[1]

    # Reshape the FFT data: Flatten each trial into a single row
    X_cl1 = fft_trials[cl1].reshape(n_trials_cl1, n_features)
    X_cl2 = fft_trials[cl2].reshape(n_trials_cl2, n_features)

    X = np.concatenate([X_cl1, X_cl2], axis=0)

    # Create labels: 0 for class 1, 1 for class 2
    y_true = np.concatenate([np.zeros(n_trials_cl1), np.ones(n_trials_cl2)])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    lda = LinearDiscriminantAnalysis(n_components=1)

    X_lda = lda.fit_transform(X, y_true)
    #X_train_lda = lda.fit_transform(X_train, y_train)

    #X_test_lda = lda.transform(X_test)
    return X_lda, y_true

def predict_labels_from_file(file_path):
    "This function takes as an input the file path of the EEG data-set under evaluation and returns the predicted labels for the new data."
        
    # Load the trained best classifier
    folder_path_model = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_models'
    file_name_model = 'trained_best_classifier.joblib'
    file_path_model = folder_path_model + '/' + file_name_model
    best_classifier_instance = joblib.load(file_path_model)
    
    #print(f'Best classifier: {best_classifier_instance}')

    # Load and preprocess the new EEG evaluation data
    EEGdata, s_freq, chan_names, event_onsets, event_codes, cl_lab, cl1, cl2 = load_and_extract_data(file_path)

    # Segmentation
    trials = segment_trials(EEGdata, event_onsets, event_codes, s_freq, cl_lab)
       
    # FFT
    fft_trials = compute_abs_fft(trials, cl_lab)

    # LDA
    X_lda, y_true = lda_evaluation(fft_trials, cl1, cl2)

    # Use the loaded classifier to predict labels for the new EEG signal
    predicted_labels = best_classifier_instance.predict(X_lda)

    # Print or use the predicted labels as needed
    #print("Predicted Labels for New Data:", predicted_labels_new)

    # Calculate confusion matrix
    #conf_matrix = confusion_matrix(y_true, predicted_labels_new)
    #print("\nConfusion Matrix:")
    #print(conf_matrix)

    # Calculate accuracy
    #accuracy = accuracy_score(y_true, predicted_labels_new)
    #print("\nAccuracy:", accuracy)

    return predicted_labels