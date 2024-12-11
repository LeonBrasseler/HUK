# Data Science Coding Challenge

This project involves training a Generalized Linear Model (GLM) and a simple Neural Network (NN) model to predict the 
annual claim amount based on provided input features. The dataset is highly imbalanced, with most contracts not 
resulting in a claim, making the data sparse. Additionally, some claims are extremely high, leading to skewed data.

XGBoost to predict chance combined with the Gamma distribution to predict the annual claim amount currently works "best".

## Files

### `train.py`
Runs both models sequentially and prints their Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

### `glm.py`
Trains two GLM models to predict the annual claim amount based on the provided input features.
The model is split into a frequency model trained using a Poisson distribution and a log link function and a severity
model using a Gamma distribution. The final prediction is the product of the frequency and severity models.

### `nn.py`
Trains two simple NN models to predict the annual claim amount based on the provided input features.
Similar to the GLM model, the NN model is split into a frequency model predicting the probability of a claim and a 
regression model predicting the severity.

### `XGBoost.py`
Trains an XGBoost model to predict the chance of a claim and then combines that with a Gamma distribution to predict the
annual claim amount.

### `data_load.py`
Loads the data from the provided ARFF file and converts data types.

### `data_vis.py`
Visualizes the data to provide a better understanding through simple correlation and distribution plots.