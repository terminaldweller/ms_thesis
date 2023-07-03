#!/usr/bin/env -S docker exec -it 83047ce42523 python3 /opt/app/data/model.py
import numpy as np  # for array
import pandas as pd  # for csv files and dataframe
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # plotting
from scipy import stats
from tqdm import tqdm  # Progress bar

import pickle  # To load data int disk

# from prettytable import PrettyTable  # To print in tabular format

import warnings

warnings.filterwarnings("ignore")

# from sklearn.preprocessing import StandardScaler  # Standardizer
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # One hot Encoder
from scipy.sparse import csr_matrix  # For sparse matrix

from sklearn.model_selection import train_test_split

# Different Models
# from sklearn.linear_model import LogisticRegression, SGDClassifier  # LR
# from sklearn.svm import LinearSVC  # SVM
# from sklearn.tree import DecisionTreeClassifier  # DT
# from sklearn.ensemble import RandomForestClassifier  # RF

# import xgboost as xgb  # XGB

# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     make_scorer,
# )  # Scoring functions

# from sklearn.metrics import auc, f1_score, roc_curve, roc_auc_score  # Scoring fns

# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # Cross validation

file_path = "/opt/app/data/"
# Train and Test data
x_train, y_train = pickle.load(open(file_path + "/final_train.pkl", "rb"))
x_test, y_test = pickle.load(open(file_path + "/final_test.pkl", "rb"))
# print(x_train.shape, x_test.shape)

# Dictionaries
saved_dict = pickle.load(open(file_path + "/saved_dict.pkl", "rb"))
mode_dict = pickle.load(open(file_path + "/mode_dict.pkl", "rb"))

# Standard scaler
scaler = pickle.load(open(file_path + "/scaler.pkl", "rb"))

# Onehot encoders
ohe_proto = pickle.load(open(file_path + "/ohe_proto.pkl", "rb"))
ohe_service = pickle.load(open(file_path + "/ohe_service.pkl", "rb"))
ohe_state = pickle.load(open(file_path + "/ohe_state.pkl", "rb"))


# Making the train data sparse matrix
x_train_csr = csr_matrix(x_train.values)

col = x_train.columns

# Creating sparse dataframe with x_train sparse matrix
x_train = pd.DataFrame.sparse.from_spmatrix(x_train_csr, columns=col)


# Saving it to disk to use later
pickle.dump((x_train, y_train), open(file_path + "/train_sparse.pkl", "wb"))

# Loading sparse data
x_train, y_train = pickle.load(open(file_path + "/train_sparse.pkl", "rb"))

# ------------------------------------------------------------------------------------------
# Data Cleaning
# ------------------------------------------------------------------------------------------
def clean_data(data):
    """
    Cleans given raw data. Performs various cleaning, removes Null and wrong values.
    Check for columns datatype and fix them.
    """
    numerical_col = data.select_dtypes(
        include=np.number
    ).columns  # All the numerical columns list
    categorical_col = data.select_dtypes(
        exclude=np.number
    ).columns  # All the categorical columns list

    # Cleaning the data
    for col in data.columns:
        val = mode_dict[col]  # Mode value of the column in train data
        data[col] = data[col].fillna(value=val)
        data[col] = data[col].replace(" ", value=val)
        data[col] = data[col].apply(lambda x: "None" if x == "-" else x)

        # Fixing binary columns
        if col in saved_dict["binary_col"]:
            data[col] = np.where(data[col] > 1, val, data[col])

    # Fixing datatype of columns
    bad_dtypes = list(set(categorical_col) - set(saved_dict["cat_col"]))
    for bad_col in bad_dtypes:
        data[col] = data[col].astype(float)

    return data


# ------------------------------------------------------------------------------------------
# Feature Engineering: Apply log1p
# ------------------------------------------------------------------------------------------
def apply_log1p(data):
    """
    Performs FE on the data. Apply log1p on the specified columns create new column and remove those original columns.
    """
    for col in saved_dict["log1p_col"]:
        new_col = col + "_log1p"  # New col name
        data[new_col] = data[col].apply(
            np.log1p
        )  # Creating new column on transformed data
        data.drop(col, axis=1, inplace=True)  # Removing old columns
    return data


# ------------------------------------------------------------------------------------------
# Standardizing: Mean centering an d varience scaling
# ------------------------------------------------------------------------------------------
def standardize(data):
    """
    Stanardize the given data. Performs mean centering and varience scaling.
    Using stanardscaler object trained on train data.
    """
    data[saved_dict["num_col"]] = scaler.transform(data[saved_dict["num_col"]])
    return data


# ------------------------------------------------------------------------------------------
# Onehot encoding of categorical columns
# ------------------------------------------------------------------------------------------
def ohencoding(data):
    """
    Onehot encoding the categoricla columns.
    Add the ohe columns with the data and removes categorical columns.
    Using Onehotencoder objects trained on train data.
    """
    # Onehot encoding cat col using onehotencoder objects
    X = ohe_service.transform(data["service"].values.reshape(-1, 1))
    Xm = ohe_proto.transform(data["proto"].values.reshape(-1, 1))
    Xmm = ohe_state.transform(data["state"].values.reshape(-1, 1))

    # Adding encoding data to original data
    data = pd.concat(
        [
            data,
            pd.DataFrame(
                Xm.toarray(), columns=["proto_" + i for i in ohe_proto.categories_[0]]
            ),
            pd.DataFrame(
                X.toarray(),
                columns=["service_" + i for i in ohe_service.categories_[0]],
            ),
            pd.DataFrame(
                Xmm.toarray(), columns=["state_" + i for i in ohe_state.categories_[0]]
            ),
        ],
        axis=1,
    )

    # Removing cat columns
    data.drop(["proto", "service", "state"], axis=1, inplace=True)
    return data


def get_final_data(data, saved_dict=saved_dict, mode_dict=mode_dict):
    """
    This functions takes raw input and convert that to model required output.
    """
    data.reset_index(drop=True, inplace=True)
    data.columns = saved_dict["columns"]

    data["network_bytes"] = data["dbytes"] + data["sbytes"]

    dropable_col = saved_dict["to_drop"] + saved_dict["corr_col"]
    data.drop(columns=dropable_col, inplace=True)

    data = clean_data(data)
    data = apply_log1p(data)
    data = standardize(data)
    data = ohencoding(data)

    return data


# Using pipeline to prepare test data
# x_train.shape, x_test.shape
# x_test = get_final_data(x_test)
x_test = get_final_data(x_test)
x_train.shape, x_test.shape

import torch

x_test = x_test.sample(frac=0.1)
y_test = y_test.sample(frac=0.1)
X = torch.from_numpy(x_test.values).type(torch.float)
Y = torch.from_numpy(y_test.values).type(torch.float)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# my cpu is better than my gpu
device = torch.device("cpu")


# class NIDSCNN(nn.Module):
#     def __init__(self):
#         super(NIDSCNN, self).__init__()
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
#         # self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
#         # self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)
#         self.fc1 = nn.Linear(21, 128)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 1)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x.unsqueeze(1)
#         x = self.conv1(x)
#         # x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         # x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x


class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(64 * ((input_size - 2) // 2 - 2), 64)
        self.fc1 = nn.Linear(3008, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (batch_size, channels, input_size)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# model = NIDSCNN().to(device)
model = CNNModel(197).to(device)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = X_train
Y_train = Y_train
Y_test = Y_test
len(X_train), len(Y_train), len(X_test), len(Y_test)


class UNSW_Dataset(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.label = Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y


train_dataset = UNSW_Dataset(X_train, Y_train)
test_dataset = UNSW_Dataset(X_test, Y_test)

batch_size = 64
num_epochs = 30
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # inputs = inputs.unsqueeze(dim=1).to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(inputs.shape)

        # Zero the gradients

        # Forward pass
        outputs = model(inputs)
        # print(outputs)

        # Calculate loss
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print training progress
        # if (i + 1) % 100 == 0:
        #     print(
        #         f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
        #     )
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
