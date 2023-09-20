---

# House Price Prediction Model - README

## Overview

This repository contains the code and documentation for my submission to the Kaggle competition "House Prices: Advanced Regression Techniques." In this competition, the goal is to predict house prices based on a set of input features.

## Model Description

### 1. Data Preprocessing

- The training data (`train.csv`) and test data (`test.csv`) were loaded into Pandas DataFrames.
- Missing values were handled using the K-nearest neighbors imputation technique.
- Categorical features were one-hot encoded to create dummy variables.
- Numerical features were standardized to have a mean of 0 and a standard deviation of 1.

### 2. Model Training

- A Random Forest Regressor was chosen as the predictive model.
- The training data was split into a training set and a validation set (70% training, 30% validation).
- The model was trained using the training set.
- Hyperparameter tuning was performed to optimize model performance.

### 3. Model Evaluation

- The following evaluation metrics were used:
  - R-squared (R2 Score): Measures the goodness of fit of the model.
  - Root Mean Squared Error (RMSE): Measures the average prediction error.
  - Mean Absolute Error (MAE): Measures the average absolute prediction error.

### 4. Model Prediction

- The trained model was used to make predictions on the test data.
- Predictions were scaled back to the original data range.

## Usage

1. Clone this repository to your local machine:

   https://github.com/Ismat-Samadov/house_pricing_kaggle.git

2. Install the required libraries:

  matplotlib
  pandas
  numpy
  sklearn

3. Run the Jupyter Notebook or Python script to reproduce the model training and prediction.

4. The final predictions are saved in a CSV file named `RF_house_price.csv`.

## Dependencies

- Python 3.7+
- Libraries listed in `requirements.txt`

## Files and Directories

- `train.csv`: Training data
- `test.csv`: Test data
- `README.md`: This README file
- `house_price_prediction.ipynb`: Jupyter Notebook containing the code
- `RF_house_price.csv`: Predicted house prices for the test data

## Results

- The model achieved an R2 score of [Your R2 Score], an RMSE of [Your RMSE], and an MAE of [Your MAE].

## Acknowledgments

- Kaggle for hosting the competition and providing the dataset.
- Scikit-learn, Pandas, and other open-source libraries for their valuable tools and resources.

## Author

- Ismat Samadov
- ismetsemedov@gmail.com
