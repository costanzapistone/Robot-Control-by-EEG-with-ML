#%% 
import numpy as np
from scipy.io import loadmat
from EEGClass import EEGClass
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the .mat file of 4 subjects
mat_file_a = loadmat('/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1a.mat', struct_as_record=True)
mat_file_f = loadmat('/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1f.mat', struct_as_record=True)
mat_file_b = loadmat('/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1b.mat', struct_as_record=True)
mat_file_g = loadmat('/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1g.mat', struct_as_record=True)

# Subjects names
subjects_names = ['a', 'f', 'b', 'g']

# Create a list of the 4 subjects
mat_files = [mat_file_a, mat_file_f, mat_file_b, mat_file_g]

# Create an istance of the class EEGClass for each subject
a = EEGClass(mat_file_a)
f = EEGClass(mat_file_f)
b = EEGClass(mat_file_b)
g = EEGClass(mat_file_g)

# Create a list of the 4 instances
eeg_instances = [a, f, b, g]
# %%

results_dict = []

for subject, eeg_instance in zip(subjects_names, eeg_instances):
    
    # Segment the data
    trials = eeg_instance.segmentation()
    
    # Compute the FFT
    fft_trials = eeg_instance.fft(trials)
    
    # Compute the LDA for dimensionality reduction
    X_train, X_test, y_train, y_test = eeg_instance.lda(fft_trials, subject)
    
    # Train the classifiers
    trained_models = eeg_instance.train_classifiers(X_train, y_train)

    # Evaluate the classifiers
    acc_dict, auc_dict = eeg_instance.evaluate_classifiers(trained_models, X_test, y_test)

    # Store the results in a dictionary
    results_dict.append({'subject': subject, 'acc': acc_dict, 'auc': auc_dict})
    
    combined_score_dict = eeg_instance.combined_score(results_dict)
#%%
# Extract the classifier with the highest combined score
best_classifier = max(combined_score_dict, key=combined_score_dict.get)
# Save the best classifier in a .joblib file
import joblib
joblib.dump(trained_models[best_classifier], '/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_best.joblib')

#%%
# Print in a table the results

# Prepare data for plotting
classifier_names = list(acc_dict.keys())
data_accuracy = np.zeros((len(results_dict), len(classifier_names))) # 4 subjects, 5 classifiers
data_auc = np.zeros((len(results_dict), len(classifier_names))) # 4 subjects, 5 classifiers

for i, subject in enumerate(results_dict):  
    data_accuracy[i, :] = list(subject['acc'].values()) 
    data_auc[i, :] = list(subject['auc'].values())  

# Decimals to show
decimals = 4

# Create a dataframe for the accuracy  
df_accuracy = pd.DataFrame(data=data_accuracy, index=subjects_names, columns=classifier_names)
df_accuracy = df_accuracy.round(decimals)

# Create a dataframe for the AUC
df_auc = pd.DataFrame(data=data_auc, index=subjects_names, columns=classifier_names)
df_auc = df_auc.round(decimals)

# Calculate average values for each classifier
avg_accuracy = data_accuracy.mean(axis=0)
avg_auc = data_auc.mean(axis=0)

# Add a row for average values
df_accuracy.loc['avg'] = avg_accuracy.round(decimals)
df_auc.loc['avg'] = avg_auc.round(decimals)

# Print the updated dataframes
print("Accuracy:")
print(df_accuracy)

print("\nAUC:")
print(df_auc)
