#%%

import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

# Functions
from analysis_processing_functions import load_and_extract_data, segment_trials, compute_abs_fft, lda

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# List of subjects
subject1 = ['a', 'f'] # left hand & right foot 
subject2 = ['b', 'g'] # right hand & left hand
subjects = ['a', 'f', 'b', 'g']
# Dictionary to store results for each subject
results_dict = {}

# Dictionary to store average results for each subject
average_results_dict = {}

for subject in subjects:

    # Load the MATLAB file for the current subject
    file_path = f'/home/costanza/thesis/Datasets/BCI_Competition_IV/calib/BCICIV_1_mat/BCICIV_calib_ds1{subject}.mat'
    
    # Call the function to load and extract the data
    EEGdata, s_freq, chan_names, event_onsets, event_codes, cl_lab, cl1, cl2 = load_and_extract_data(file_path)

    # Segmentation
    trials = segment_trials(EEGdata, event_onsets, event_codes, s_freq, cl_lab)
    
    # FFT
    fft_trials = compute_abs_fft(trials, cl_lab)

    # LDA
    X_train_lda, X_test_lda, y_train, y_test = lda(fft_trials, cl1, cl2)

    # Initialize classifiers
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'NB': GaussianNB(),
        'LR': LogisticRegression(),
        'DT': DecisionTreeClassifier(),
        'SVM': SVC(probability=True),  # Use probability=True for SVM to enable predict_proba
    }

    # Dictionary to store results for each classifier
    results_subject = {}

    # Iterate over classifiers and fill the dictionary
    for clf_name, clf in classifiers.items():
        # Train the classifier
        clf.fit(X_train_lda, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test_lda)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate AUC
        if hasattr(clf, 'predict_proba'):
            proba_class_1 = clf.predict_proba(X_test_lda)[:, 1]
        else:  # For SVM, use decision_function
            proba_class_1 = clf.decision_function(X_test_lda)
        auc = roc_auc_score(y_test, proba_class_1)


        # Store results in the dictionary
        results_subject[clf_name] = {'Accuracy': accuracy, 'AUC': auc}

    # Store results for the current subject in the main dictionary
    results_dict[subject] = results_subject

    # Dictionary to store results for each classifier across all subjects
    average_results_classifier = {clf_name: {'Accuracy': 0, 'AUC': 0} for clf_name in classifiers}

    # Iterate over subjects
    for subject, results_subject in results_dict.items():
        for clf_name, results_clf in results_subject.items():
            
            # Accumulate accuracy for the current classifier
            average_results_classifier[clf_name]['Accuracy'] += results_clf['Accuracy']
            average_results_classifier[clf_name]['AUC'] += results_clf['AUC']

    # Calculate average accuracy for each classifier
    num_subjects = len(results_dict)
    for clf_name in classifiers:
        average_results_classifier[clf_name]['Accuracy'] /= num_subjects
        average_results_classifier[clf_name]['AUC'] /= num_subjects


# Print the results
for subject, results_subject in results_dict.items():
    print(f"\nResults for {subject}:")
    print("Classifier\tAccuracy\tAUC")
    for clf_name, results_clf in results_subject.items():
        print(f"{clf_name}\t\t{results_clf['Accuracy']:.4f}\t\t{results_clf['AUC']:.4f}")

# Print average results
print("\nAverage Results Across All Subjects:")
print("Classifier\tAv. Accuracy\tAv. AUC")
for clf_name, avg_results_clf in average_results_classifier.items():
    print(f"{clf_name}\t\t{avg_results_clf['Accuracy']:.4f}\t\t{avg_results_clf['AUC']:.4f}")

#%% Choose the best classifier for each subject 

# Dictionary to store combined scores for each classifier
combined_results_classifier = {clf_name: {'Combined_Score': 0} for clf_name in classifiers}

# Iterate over classifiers
for clf_name in classifiers:
    # Weighted average of accuracy and AUC (adjust weights based on your priorities)
    weight_accuracy = 0.5
    weight_auc = 0.5
    combined_score = (
        weight_accuracy * average_results_classifier[clf_name]['Accuracy'] +
        weight_auc * average_results_classifier[clf_name]['AUC']
    )
    combined_results_classifier[clf_name]['Combined_Score'] = combined_score


# Print the results
print("\nCombined Results Across All Subjects:")
print("Classifier\tCombined Score")
for clf_name, combined_results_clf in combined_results_classifier.items():
    print(f"{clf_name}\t\t{combined_results_clf['Combined_Score']:.4f}")

# Identify the best classifier based on the combined score
best_classifier = max(combined_results_classifier, key=lambda x: combined_results_classifier[x]['Combined_Score'])

# Print the best classifier
print(f"\nBest classifier: {best_classifier}")

#%% Save the trained best classifier

import joblib

best_classifier_instance = classifiers[best_classifier]
best_classifier_instance.fit(X_train_lda, y_train)

# Specify the path of the folder where you want to save the classifier
folder_path_model = '/home/costanza/Robot-Control-by-EEG-with-ML/models'

# Specify the name of the file
file_name_model = f'trained_best_classifier.joblib'

# Create the full file path by concatenating the folder path and the file name 
file_path_model = folder_path_model + '/' + file_name_model

# Save the classifier
joblib.dump(best_classifier_instance, file_path_model)

# Print a message indicating the saved file
print(f"Trained best classifier saved in {file_path_model}")


#%%
# Plot tables with results

import matplotlib.pyplot as plt

# Extract classifier names
classifier_names = list(classifiers.keys())

# Create data for the tables
data_accuracy = [['Subject'] + classifier_names]
data_auc = [['Subject'] + classifier_names]

for subject, results_subject in results_dict.items():
    accuracy_row = [subject]
    auc_row = [subject]

    for clf_name in classifier_names:
        accuracy_row.append(f"{results_subject[clf_name]['Accuracy']:.2f}")
        auc_row.append(f"{results_subject[clf_name]['AUC']:.2f}")

    data_accuracy.append(accuracy_row)
    data_auc.append(auc_row)

# Plot tables
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Table for Accuracy
table_accuracy = ax[0].table(cellText=data_accuracy, loc='center', colLabels=None, cellLoc='center', rowLoc='center')
table_accuracy.auto_set_font_size(False)
table_accuracy.set_fontsize(10)
table_accuracy.scale(1.2, 1.2)  # Adjust the table size

# Table for AUC
table_auc = ax[1].table(cellText=data_auc, loc='center', colLabels=None, cellLoc='center', rowLoc='center')
table_auc.auto_set_font_size(False)
table_auc.set_fontsize(10)
table_auc.scale(1.2, 1.2)  # Adjust the table size

# Hide axes
ax[0].axis('off')
ax[1].axis('off')

plt.show()

#%%
# Create a Bar Plot for the results

# Lista di colori per i soggetti
subject_colors = {'b': 'blue', 'g': 'orange'}

# Creare due subplot affiancati per accuracy e AUC
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Grafico per l'Accuracy
axes[0].set_title('Accuracy')
#axes[0].set_xlabel('Classificatore')
axes[0].set_ylabel('Accuracy')

# Grafico per l'AUC
axes[1].set_title('AUC')
#axes[1].set_xlabel('Classificatore')
axes[1].set_ylabel('AUC')

# Iterate over classifiers
for clf_name, clf_index in zip(classifiers.keys(), range(len(classifiers))):
    # Lista di valori di accuracy e AUC per i soggetti
    accuracy_values = []
    auc_values = []

    # Iterate over subjects (using subject_colors.keys())
    for subject, color in subject_colors.items():
        # Recuperare i risultati per il classificatore e il soggetto correnti
        results_subject = results_dict[subject][clf_name]
        accuracy_values.append(results_subject['Accuracy'])
        auc_values.append(results_subject['AUC'])

    x = np.arange(len(subject_colors)) + clf_index * 0.2 + np.array([0, 0.1])

    # Barre per l'Accuracy
    axes[0].bar(x, accuracy_values, width=0.2, label=clf_name)

    # Barre per l'AUC
    axes[1].bar(x, auc_values, width=0.2, label=clf_name)

# Impostare la legenda
axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1),title='Classifiers')
axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1),title='Classifiers')

# Impostare le etichette degli assi per i soggetti
axes[0].set_xticks(np.arange(len(subject_colors)) + np.array([0.4, 0.5]))
axes[0].set_xticklabels(subject_colors.keys())
axes[1].set_xticks(np.arange(len(subject_colors)) + np.array([0.4, 0.5]))
axes[1].set_xticklabels(subject_colors.keys())

fig.suptitle('Right Hand & Left Foot (subjects b & g)', fontsize=16)
# Mostrare i grafici
plt.tight_layout()
plt.show()

