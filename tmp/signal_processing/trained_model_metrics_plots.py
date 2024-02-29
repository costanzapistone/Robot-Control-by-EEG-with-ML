#%%
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.io import loadmat
from EEGClass import EEGClass

# Load the trained model
model = joblib.load('/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_best.joblib')

# Load the data
data = loadmat('/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1a.mat', struct_as_record=True)
data_instance = EEGClass(data)
trials = data_instance.segmentation()
fft_trials = data_instance.fft(trials)
X_train, X_test, y_train, y_test = data_instance.lda(fft_trials)

# Predict the labels of the test set
y_pred = model.predict(X_test)

#%%
# Compute the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
import seaborn as sns
sns.set()
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
# %%
# ROC curve
import sklearn
import scikitplot as skplt

# Compute the probabilities of the test set
y_probas = model.predict_proba(X_test)

# Plot the ROC curve
skplt.metrics.plot_roc_curve(y_test, y_probas, title='ROC curve', figsize=(10, 6))

# %%
# Precision-recall curve

# Plot the precision-recall curve
skplt.metrics.plot_precision_recall_curve(y_test, y_probas, title='Precision-recall curve', figsize=(10, 6))



# %%
