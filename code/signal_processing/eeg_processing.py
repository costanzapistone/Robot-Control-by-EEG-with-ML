#%% Import packages
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

#%%
# Dictionary to store accuracy and AUC for each subject and each classifier
results_dict = []

for subject, eeg_instance in zip(subjects_names, eeg_instances):
    # Segment the data
    trials = eeg_instance.segmentation()
    
    # Compute the FFT
    fft_trials = eeg_instance.fft(trials)
    
    # Compute the LDA for dimensionality reduction
    X_train, X_test, y_train, y_test = eeg_instance.lda(fft_trials)
    
    # Train and evaluate the classifiers
    acc_dict, auc_dict = eeg_instance.train_and_evaluate_classifiers(X_train, X_test, y_train, y_test)

    # Store the results in a dictionary
    results_dict.append({'subject': subject, 'acc': acc_dict, 'auc': auc_dict})

#%%
# Compute the mean accuracy and AUC for each classifier among the 4 subjects

# Create dictionaries to store the average accuracy and AUC for each classifier
acc_avg_dict = {}
auc_avg_dict = {}

for classifier in acc_dict:
    acc_avg_dict[classifier] = np.mean([subject['acc'][classifier] for subject in results_dict])
    auc_avg_dict[classifier] = np.mean([subject['auc'][classifier] for subject in results_dict])

# Print the results in a table
classifier_names = list(acc_dict.keys())
data_accuracy_avg = np.zeros((1, len(classifier_names))) # 1 row, 5 classifiers
data_auc_avg = np.zeros((1, len(classifier_names))) # 1 row, 5 classifiers

for i, classifier in enumerate(acc_avg_dict):  
    data_accuracy_avg[0, i] = acc_avg_dict[classifier] 
    data_auc_avg[0, i] = auc_avg_dict[classifier]

# Decimals to show
decimals = 4

# Create a dataframe for the accuracy
df_accuracy_avg = pd.DataFrame(data=data_accuracy_avg, index=['Av Acc'], columns=classifier_names)
df_accuracy_avg = df_accuracy_avg.round(decimals)

# Create a dataframe for the AUC
df_auc_avg = pd.DataFrame(data=data_auc_avg, index=['Av AUC'], columns=classifier_names)
df_auc_avg = df_auc_avg.round(decimals)

# Print the dataframes
print(df_accuracy_avg)
print(df_auc_avg)

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

# Print the dataframes
print(df_accuracy)
print(df_auc)

# %%
# Plot the tables
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
# Plot Accuracy table
axs[0].axis('off')
acc_table = axs[0].table(cellText=df_accuracy.values, colLabels=df_accuracy.columns, rowLabels=df_accuracy.index, cellLoc='center', loc='center', colColours=['#f5f5f5']*len(df_accuracy.columns))
acc_table.auto_set_font_size(False)
acc_table.set_fontsize(13)
acc_table.scale(1.8, 1.8)  # Increase cell size
acc_table.auto_set_column_width([0] + list(range(1, len(df_accuracy.columns))))
axs[0].set_title('Accuracy', fontsize=13)

# Plot AUC table
axs[1].axis('off')
auc_table = axs[1].table(cellText=df_auc.values, colLabels=df_auc.columns, rowLabels=df_auc.index, cellLoc='center', loc='center', colColours=['#f5f5f5']*len(df_auc.columns))
auc_table.auto_set_font_size(False)
auc_table.set_fontsize(13)
auc_table.scale(1.8, 1.8)  # Increase cell size
auc_table.auto_set_column_width([0] + list(range(1, len(df_auc.columns))))
axs[1].set_title('AUC', fontsize=13)

plt.show()
# %%

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

# Print the dataframes
print(df_accuracy)
print(df_auc)