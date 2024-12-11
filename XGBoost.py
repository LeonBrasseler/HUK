import xgboost as xgb
import pandas as pd
import numpy as np
import torch
import statsmodels.api as sm

from NN import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def XGBoost():
    """
    Train a combined model using XGBoost to predict the probability of having a claim and a GLM to predict the claim
    amount given that a claim has been made. Combine the predictions to get the expected claim amount and evaluate the
    model using the MAE and RMSE.
    """
    # Load and preprocess data
    from data_load import load_data

    data = load_data()
    train_data, test_data = preprocess(data, categorical_type='category')

    # Define features and target for claim prediction
    y_train = train_data['HasClaim']
    y_test = test_data['HasClaim']
    X_train = train_data.drop(columns=['LogClaimAmount', 'ClaimAmount', 'IDpol', 'HasClaim', 'ClaimNb', 'Exposure'])
    X_test = test_data.drop(columns=['LogClaimAmount', 'ClaimAmount', 'IDpol', 'HasClaim', 'ClaimNb', 'Exposure'])

    # Train the XGBoost model to predict the probability of having a claim
    # 20x scale_pos_weight to account for the class imbalance
    model = xgb.XGBRegressor(objective='binary:logistic', n_estimators=150, learning_rate=0.1, max_depth=6,
                             enable_categorical=True, early_stopping_rounds=10, scale_pos_weight=20)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Make predictions
    y_pred_prob = model.predict(X_test)

    # Train a GLM to predict the claim amount
    glm_train_data = train_data[train_data['HasClaim'] == 1]
    glm_X_train = glm_train_data.drop(
        columns=['LogClaimAmount', 'ClaimAmount', 'IDpol', 'HasClaim', 'ClaimNb', 'Exposure'])
    glm_y_train = glm_train_data['ClaimAmount']

    # Dummify categorical variables
    glm_X_train = pd.get_dummies(glm_X_train, drop_first=True,
                                 columns=['Area', 'VehBrand', 'VehGas', 'Region'], dtype=float)
    glm_model = sm.GLM(glm_y_train, sm.add_constant(glm_X_train),
                       family=sm.families.Gamma(link=sm.families.links.log()))
    glm_results = glm_model.fit()

    # Make predictions for the claim amount
    glm_X_test = test_data.drop(columns=['LogClaimAmount', 'ClaimAmount', 'IDpol', 'HasClaim', 'ClaimNb', 'Exposure'])
    glm_X_test = pd.get_dummies(glm_X_test, drop_first=True,
                                columns=['Area', 'VehBrand', 'VehGas', 'Region'], dtype=float)
    glm_y_pred = glm_results.predict(sm.add_constant(glm_X_test))

    # Combine the predictions
    expected_claim_amount = y_pred_prob * glm_y_pred

    # Evaluate the combined model
    actual_claim_amount = test_data['ClaimAmount'].values
    mae = np.mean(np.abs(expected_claim_amount - actual_claim_amount))
    rmse = np.sqrt(np.mean((expected_claim_amount - actual_claim_amount) ** 2))
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')

    return {'MAE': mae, 'RMSE': rmse}


if __name__ == '__main__':
    XGBoost()
