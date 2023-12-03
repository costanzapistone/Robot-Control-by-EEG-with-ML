# Robot-Control-by-EEG-with-ML

## Overview

## Data
The Dataset used is the public Dataset I from BCI Competition IV available online. EEG signals from 4 different subject performing the motor imagery task (right hand and left hand or left hand and right foot)have been recorded from 59 channels.
More detailed description can be found here: https://bbci.de/competition/iv/desc_1.html

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

The metrics used to compare the classifiers are: Accuracy and AUC. 

---

## Robot Control
The robot used for the simulation is the robotic arm Franka Emika.
