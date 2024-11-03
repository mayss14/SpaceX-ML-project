# SpaceX Falcon 9 First Stage Landing Prediction
This project uses machine learning to predict the landing outcome of SpaceX Falcon 9's first stage. By analyzing flight data and various features, we aim to develop a model that predicts whether a Falcon 9 booster successfully lands or not.

## Table of Contents
Project Overview
Dataset
Requirements
Steps
Usage
Results
Acknowledgements

## Project Overview
SpaceX has been making strides in reusable rocket technology by successfully landing its Falcon 9 boosters. Predicting the success of these landings can provide insights for operational improvements. This project involves data preprocessing, feature scaling, model training, and evaluation to predict the landing success of Falcon 9 rockets.

## Dataset
The data used in this notebook is sourced from two CSV files:

Dataset Part 2: Contains features for each flight.
Dataset Part 3: Contains additional features specifically relevant to landing prediction.

Features:

Numerical and categorical variables including payload mass, launch site, booster version, orbit type, etc.
Target Variable: Class (indicates whether the booster landed successfully).

## Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
Install the requirements using:

pip install pandas numpy matplotlib seaborn scikit-learn

Steps
Data Loading: Import datasets from provided URLs.
Data Preprocessing: Perform scaling and handle categorical variables to prepare features.
Model Selection and Training: Train multiple classification models, including:
Logistic Regression
Support Vector Machine (SVM)
Decision Trees
K-Nearest Neighbors (KNN)
Hyperparameter Tuning: Use GridSearchCV to optimize model parameters.
Evaluation: Assess models using accuracy scores and confusion matrices.

## Usage
Run the cells in the notebook sequentially to train and evaluate the models. The notebook uses standard machine learning evaluation methods and includes visualizations to better understand performance.

## Results
The notebook compares various models and identifies the best-performing one based on accuracy. The best model can predict the likelihood of a successful booster landing based on flight features, and the results are visualized with confusion matrices.

### Acknowledgements
This project is part of the IBM Data Science Professional Certificate and uses data from the IBM Skills Network.
