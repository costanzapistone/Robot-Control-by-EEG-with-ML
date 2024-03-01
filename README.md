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
The preprocessing follows these steps:  
Segmentation: The EEG data is segmented into trials based on event markers, dividing them into distinct epochs for analysis.  
Filtering: A band-pass filter is applied to the EEG trials to isolate frequency components relevant to the task at hand, enhancing the signal-to-noise ratio.  
Common Spatial Patterns (CSP): The CSP algorithm is utilized to extract spatial filters that maximize the variance for one class of trials while minimizing it for another. This step enhances the discriminative power of the EEG signals.

### Location

You can find the implementation and the functions used respectively in the following directories within this repository:  

/Robot-Control-by-EEG-with-ML/code/data_processing.py  
/Robot-Control-by-EEG-with-ML/code/processing_functions.py


## Classification
Classificiers used:
- Support Vector Machine (SVM)
- Linear Discriminant Analysis (LDA)
- Logistic Regression (LR)
- Decision Tree (DT)
- Naive Bayes (NB)
- Random Forest (RF)

---

## Robot Control
The robot used for the simulation is the robotic arm Franka Emika.
