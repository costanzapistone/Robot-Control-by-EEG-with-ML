# Robot-Control-by-EEG-with-ML

## Overview

## Data
The dataset used in this project is sourced from Dataset I of BCI Competition IV, which is publicly available online. It consists of EEG signals recorded from 59 channels while healthy subjects performed motor imagery tasks involving movements of the right hand and left hand, or left hand and right foot.
For more detailed information about the dataset, please refer to the official competition website: https://bbci.de/competition/iv/desc_1.html

The total number of subjects are 7:
subject a - MI movements: 'foot', 'right' - real 
subject b - MI movements: 'left', 'right' - real
subject c - MI movements: 'left', 'right' - artificial 
subject d - MI movements: 'left', 'right' - artificial
subject e - MI movements: 'left', 'right' - artificial
subject f - MI movements: 'foot', 'right' - real
subject g - MI movements: 'left', 'right' - real

You can find the dataset in the following directory within this repository: 

/Robot-Control-by-EEG-with-ML/data/

---

## ML Algorithm
Pre-processing data steps:
- Segmentation
- Fast Fourier Transform (FFT)
- Linear Discriminant Analysis (LDA) for dimesnionality reduction

Classificiers used:
- Support Vector Machine
- K-Nearest Neighbors (K-NN)
- Logistic Regression (LR)
- Decision Tree (DT)
- Naive Bayes (NB)

The choice of the best classifier for robot control purposes has been done using a combined score. The score is derived by a simple arithmetic sum of AUC (Area Under the Curve) and ACC (Accuracy), with each metric contribuiting equally to the final result. 


---

## Robot Control
The robot used for the simulation is the robotic arm Franka Emika.
