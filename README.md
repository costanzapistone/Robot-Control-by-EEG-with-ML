# Neural Signal-Based Robot Control with Machine Learning: Uncertainty Analysis for Improved Decision-Making

## Overview
This repository contains code for a Master's thesis project on "Robot Control by EEG with Machine Learning: Uncertainty Analysis for Improved Decision-Making." The project explores the use of neural signals obtained through EEG (Electroencephalography) signals for controlling robots, employing machine learning techniques for classification, and analyzing uncertainty to enhance decision-making processes in the field of HRI.

## Data
The dataset used in this project is sourced from Dataset I of BCI Competition IV, which is publicly available online. It consists of EEG signals recorded from 59 channels while healthy subjects performed motor imagery tasks involving movements of the right hand and left hand, or left hand and right foot.
For more detailed information about the dataset, please refer to [BCI Competition IV Description](https://bbci.de/competition/iv/desc_1.html).

### Subjects

The total number of subjects are 7:  
subject a - 'foot', 'right' - real  
subject b - 'left', 'right' - real  
subject c - 'left', 'right' - artificial  
subject d - 'left', 'right' - artificial  
subject e - 'left', 'right' - artificial  
subject f - 'foot', 'right' - real  
subject g - 'left', 'right' - real  


### Location

You can find the dataset in the following directory within this repository: 

/Robot-Control-by-EEG-with-ML/data/

---

## ML Algorithm
Pre-processing data steps:
- Segmentation
- Fast Fourier Transform (FFT)
- Linear Discriminant Analysis (LDA) for dimesnionality reduction

## Classification
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
