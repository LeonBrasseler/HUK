import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import pandas as pd
import torch

from data_load import load_data


def preprocess_data_glm(data_full):
    """
    Preprocess the data for the GLM model

    :param data_full: the full dataset
    :return:
    """
    # Map Area to continuous values
    area_mapping = {"'A'": 1, "'B'": 2, "'C'": 3, "'D'": 4, "'E'": 5, "'F'": 6}
    data_full['Area'] = data_full['Area'].map(area_mapping)

    # Merge VehPower groups
    # data_full['VehPower'] = data_full['VehPower'].apply(lambda x: x if x < 9 else 9)

    # Create categorical classes for VehAge
    # data_full['VehAge'] = pd.cut(data_full['VehAge'], bins=[np.min(data_full['VehAge'])-1, 1, 10, np.inf],
    #                              labels=[0, 1, 2])

    # Create categorical classes for DrivAge
    # data_full['DrivAge'] = pd.cut(data_full['DrivAge'], bins=[np.min(data_full['DrivAge'])-1, 21, 26, 31, 41, 51, 71, np.inf],
    #                               labels=[0, 1, 2, 3, 4, 5, 6])

    # Scale numerical variables
    data_full['VehPower'] = (data_full['VehPower'] - data_full['VehPower'].mean()) / data_full['VehPower'].std()
    data_full['VehAge'] = (data_full['VehAge'] - data_full['VehAge'].mean()) / data_full['VehAge'].std()
    data_full['DrivAge'] = (data_full['DrivAge'] - data_full['DrivAge'].mean()) / data_full['DrivAge'].std()

    # Cap BonusMalus at 150
    data_full['BonusMalus'] = data_full['BonusMalus'].clip(upper=150)

    # Log-transform Density
    data_full['Density'] = np.log1p(data_full['Density'])

    return data_full


def GLM_model(data_):
    """
    Fit a GLM model to the data
    :param data_: the full dataset
    :return: None
    """
    data_full = data_.copy()
    data_full = data_full[data_full['Exposure'] > 0]

    # Model the frequency and severity separately to handle the zero-inflation
    # For frequency, we use Poisson distribution as the target variable is a count
    model_freq = smf.glm(
        'ClaimNb ~ DrivAge + VehPower + VehAge + BonusMalus + VehBrand + VehGas + Region + Area + Density',
        # '+ (DrivAge^2) + (DrivAge^3) + (VehPower^2) + (VehPower^3) + (VehAge^2) + (VehAge^3)',
        data=data_full, family=sm.families.Poisson(),
        offset=np.log(data_full['Exposure'])
    ).fit()
    print(model_freq.summary())

    # For severity, we use the Gamma distribution as the target variable is continuous
    data_full['LogClaimAmount'] = np.log1p(data_full['ClaimAmount'])
    severity_data = data_full[data_full['ClaimAmount'] > 0]  # Filter positive claims
    data_full['NormalizedClaimAmount'] = data_full['ClaimAmount'] / data_full['Exposure']
    model_sev = smf.glm(
        'LogClaimAmount ~ DrivAge + VehPower + VehAge + BonusMalus + VehBrand + VehGas + Region + Area + Density',
        data=severity_data, family=sm.families.Gamma()
    ).fit()
    print(model_sev.summary())

    # Frequency prediction: For the entire dataset
    freq_pred = model_freq.predict()

    # Severity prediction: Only for rows with ClaimAmount > 0
    sev_pred = np.zeros(data_full.shape[0])  # Initialize with zeros
    sev_pred[data_full['ClaimAmount'] > 0] = np.expm1(model_sev.predict())  # Fill positive claims

    # Calculate Expected Claim Cost
    data_full['ExpectedClaimCost'] = freq_pred * sev_pred * data_full['Exposure']
    return data_full[['IDpol', 'NormalizedClaimAmount', 'ExpectedClaimCost']], model_freq, model_sev


def evaluate_glm(glm_freq, glm_sev, test_data):
    """
    Evaluate the model on the test data
    :param glm_freq: the frequency model
    :param glm_sev: the severity model
    :param test_data: the test data
    :return: metric values
    """
    prediction = glm_freq.predict(test_data) * np.expm1(glm_sev.predict(test_data)) * test_data['Exposure']

    MAE = torch.mean(
        torch.abs(torch.Tensor(np.array(prediction)) - torch.Tensor(np.array(test_data['ClaimAmount'])))).item()

    print(f"GLM MAE: {MAE}")

    return MAE


def train_glm():
    data = load_data()
    data = preprocess_data_glm(data)

    # Split data
    data_train = data.sample(frac=0.8, random_state=0)
    data_test = data.drop(data_train.index)

    glm_pred, glm_freq, glm_sev = GLM_model(data_train)
    return glm_pred, glm_freq, glm_sev, data_test
