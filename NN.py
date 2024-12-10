import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from data_load import load_data


GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")


def preprocess(data_):
    """
    Preprocess the data for the neural network model
    One hot encodes the categorical variables
    :param data_:
    :return:
    """
    data_full = data_.copy()
    data_full = data_full[data_full['Exposure'] > 0]

    # Map Area to continuous values
    # area_mapping = {"'A'": 1, "'B'": 2, "'C'": 3, "'D'": 4, "'E'": 5, "'F'": 6}
    # data_full['Area'] = data_full['Area'].map(area_mapping)

    # Merge VehPower groups
    # data_full['VehPower'] = data_full['VehPower'].apply(lambda x: x if x < 9 else 9)

    # Create categorical classes for VehAge
    # data_full['VehAge'] = pd.cut(data_full['VehAge'], bins=[np.min(data_full['VehAge']) - 1, 1, 10, np.inf],
    #                              labels=[0, 1, 2])

    # Create categorical classes for DrivAge
    # data_full['DrivAge'] = pd.cut(data_full['DrivAge'],
    #                               bins=[np.min(data_full['DrivAge']) - 1, 21, 26, 31, 41, 51, 71, np.inf],
    #                               labels=[0, 1, 2, 3, 4, 5, 6])

    # Cap BonusMalus at 150
    # data_full['BonusMalus'] = data_full['BonusMalus'].clip(upper=150)

    # Log-transform Density
    data_full['Density'] = np.log1p(data_full['Density'])

    # Scale numerical variables
    # VehPower, VehAge, DrivAge, BonusMalus, Density
    data_full['VehPower'] = (data_full['VehPower'] - data_full['VehPower'].mean()) / data_full['VehPower'].std()
    data_full['VehAge'] = (data_full['VehAge'] - data_full['VehAge'].mean()) / data_full['VehAge'].std()
    data_full['DrivAge'] = (data_full['DrivAge'] - data_full['DrivAge'].mean()) / data_full['DrivAge'].std()
    data_full['BonusMalus'] = (data_full['BonusMalus'] - data_full['BonusMalus'].mean()) / data_full['BonusMalus'].std()
    data_full['Density'] = (data_full['Density'] - data_full['Density'].mean()) / data_full['Density'].std()

    # Transform Vehicle Brand/Vehicle Gas/Region/Area to one-hot encoding
    data_full = pd.get_dummies(data_full, columns=['VehBrand'], dtype=int)
    data_full = pd.get_dummies(data_full, columns=['VehGas'], dtype=int)
    data_full = pd.get_dummies(data_full, columns=['Region'], dtype=int)
    data_full = pd.get_dummies(data_full, columns=['Area'], dtype=int)

    # Create binary target for zero vs. non-zero claims
    data_full['HasClaim'] = (data_full['ClaimAmount'] > 0).astype(int)

    # Scale target variable with log transformation as the distribution is skewed
    data_full['LogClaimAmount'] = np.log1p(data_full['ClaimAmount'])

    # Split data
    data_train = data_full.sample(frac=0.8, random_state=0)
    data_test = data_full.drop(data_train.index)

    return data_train, data_test


def train_classifier(train_data):
    """
    Train a classifier to predict whether a claim will be made
    Outputs the probability of a claim being made
    :param train_data:
    :return: trained model
    """
    model = nn.Sequential(
        nn.Linear(train_data.shape[1] - 4, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    has_claim = torch.Tensor(train_data['HasClaim'].values)
    train_data = torch.Tensor(train_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values)
    batch_size = 256
    train_data = torch.split(train_data.to(device), batch_size)
    has_claim = torch.split(has_claim.to(device), batch_size)
    model.to(device)
    for epoch in range(50):
        epoch_loss = 0
        for x, y in zip(train_data, has_claim):
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    return model


def train_regressor(train_data):
    """
    Train a regressor to predict the claim amount
    Only trains on the rows with a claim
    :param train_data: Only rows with a claim from the train dataset
    :return: model to predict the claim amount
    """
    model = nn.Sequential(
        nn.Linear(train_data.shape[1] - 4, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1)
    )
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    LogClaimAmount = torch.Tensor(train_data['LogClaimAmount'].values).to(device)
    train_data = torch.Tensor(
                train_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values
        ).to(device)
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = loss_fn(outputs.squeeze(), LogClaimAmount)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model


def train_NN_model():
    """
    Train the neural network model, e.g. the classifier and regressor
    :return: classifier model, regressor model, test_data
    """
    data_ = load_data()
    train_data, test_data = preprocess(data_)

    classifier = train_classifier(train_data)
    regressor = train_regressor(train_data[train_data['HasClaim'] == 1])

    return classifier, regressor, test_data


def evaluate_model(classifier, regressor, test_data):
    classifier.eval()
    regressor.eval()

    with torch.no_grad():
        classifier_outputs = classifier(torch.Tensor(
            test_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values).to(device))
        regressor_outputs = regressor(torch.Tensor(
            test_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values).to(device))

        # Predicted claim amount is the product of the classifier output (chance of having a claim) and
        # the regressor output (amount of the claim)
        predictions = classifier_outputs.squeeze().to('cpu').numpy()
        predictions *= regressor_outputs.squeeze().to('cpu').numpy()

        mae = np.mean(np.abs(predictions - test_data['ClaimAmount'].values))
        print(f"Test MAE: {mae}")

    return mae


if __name__ == '__main__':
    classifier, regressor, test_data = train_NN_model()
    MAE = evaluate_model(classifier, regressor, test_data)
