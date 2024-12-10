import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from data_load import load_data


def preprocess(data_):
    data_full = data_.copy()
    data_full = data_full[data_full['Exposure'] > 0]

    # Map Area to continuous values
    # area_mapping = {"'A'": 1, "'B'": 2, "'C'": 3, "'D'": 4, "'E'": 5, "'F'": 6}
    # data_full['Area'] = data_full['Area'].map(area_mapping)

    # Merge VehPower groups
    data_full['VehPower'] = data_full['VehPower'].apply(lambda x: x if x < 9 else 9)

    # Create categorical classes for VehAge
    # data_full['VehAge'] = pd.cut(data_full['VehAge'], bins=[np.min(data_full['VehAge']) - 1, 1, 10, np.inf],
    #                              labels=[0, 1, 2])

    # Create categorical classes for DrivAge
    # data_full['DrivAge'] = pd.cut(data_full['DrivAge'],
    #                               bins=[np.min(data_full['DrivAge']) - 1, 21, 26, 31, 41, 51, 71, np.inf],
    #                               labels=[0, 1, 2, 3, 4, 5, 6])

    # Cap BonusMalus at 150
    data_full['BonusMalus'] = data_full['BonusMalus'].clip(upper=150)

    # Log-transform Density
    data_full['Density'] = np.log1p(data_full['Density'])

    # Transform Vehicle Brand/Vehicle Gas/Region/Area to one-hot encoding
    data_full = pd.get_dummies(data_full, columns=['VehBrand'], dtype=int)
    data_full = pd.get_dummies(data_full, columns=['VehGas'], dtype=int)
    data_full = pd.get_dummies(data_full, columns=['Region'], dtype=int)
    data_full = pd.get_dummies(data_full, columns=['Area'], dtype=int)

    # Create binary target for zero vs. non-zero claims
    data_full['HasClaim'] = (data_full['ClaimAmount'] > 0).astype(int)

    # Scale target variable
    data_full['LogClaimAmount'] = np.log1p(data_full['ClaimAmount'])

    # Split data
    data_train = data_full.sample(frac=0.8, random_state=0)
    data_test = data_full.drop(data_train.index)

    return data_train, data_test


def train_classifier(train_data):
    model = nn.Sequential(
        nn.Linear(train_data.shape[1] - 4, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    has_claim = torch.Tensor(train_data['HasClaim'].values)
    train_data = torch.Tensor(train_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values)
    batch_size = 256
    train_data = torch.split(train_data, batch_size)
    has_claim = torch.split(has_claim, batch_size)
    for epoch in range(25):
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
    model = nn.Sequential(
        nn.Linear(train_data.shape[1] - 4, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(25):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.Tensor(train_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values))
        loss = loss_fn(outputs.squeeze(), torch.Tensor(train_data['LogClaimAmount'].values))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model


def train_NN_model():
    data_ = load_data()
    train_data, test_data = preprocess(data_)

    classifier = train_classifier(train_data)
    regressor = train_regressor(train_data[train_data['HasClaim'] == 1])

    return classifier, regressor, test_data


def evaluate_model(classifier, regressor, test_data):
    classifier.eval()
    regressor.eval()

    with torch.no_grad():
        classifier_outputs = classifier(torch.Tensor(test_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values))
        regressor_outputs = regressor(torch.Tensor(test_data.drop(columns=['IDpol', 'ClaimAmount', 'LogClaimAmount', 'HasClaim']).values))

        predictions = classifier_outputs.squeeze().numpy()
        predictions *= regressor_outputs.squeeze().numpy()

        mae = np.mean(np.abs(predictions - test_data['ClaimAmount'].values))
        print(f"Test MAE: {mae}")

    return mae


if __name__ == '__main__':
    classifier, regressor, test_data = train_NN_model()
    MAE = evaluate_model(classifier, regressor, test_data)
