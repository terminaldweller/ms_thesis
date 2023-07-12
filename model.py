#!/usr/bin/env -S docker exec -it 83047ce42523 python3 /opt/app/data/model.py
import torch
import numpy as np  # for array
import pandas as pd  # for csv files and dataframe
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # plotting
from scipy import stats
from tqdm import tqdm  # Progress bar

import pickle  # To load data int disk

# from prettytable import PrettyTable  # To print in tabular format

import warnings

# warnings.filterwarnings("ignore")

# from sklearn.preprocessing import StandardScaler  # Standardizer
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # One hot Encoder
from scipy.sparse import csr_matrix  # For sparse matrix

from sklearn.model_selection import train_test_split

torch.autograd.set_detect_anomaly(True)

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
# x_train.shape, x_test.shape

x_test = x_test.sample(frac=0.1)
y_test = y_test.sample(frac=0.1)
# x_test = x_train.sample(frac=0.1)
# y_test = y_train.sample(frac=0.1)
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

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
print(f"X_train_shape: {X_train.shape} -- Y_train_shape: {Y_train.shape}")
print(f"X_test_shape: {X_test.shape} -- Y_test_shape: {Y_test.shape}")


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


class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.rnn = nn.RNN(64, 64, 5, batch_first=True)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x, _ = self.rnn(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        # x = self.sigmoid(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class Die(nn.Module):
    def __init__(self, input_size):
        super(Die, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        # self.rnn = nn.RNN(128, 128, 5, batch_first=True)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # torch.nn.init.uniform_(self.fc1.weight)
        # torch.nn.init.uniform_(self.fc2.weight)
        # torch.nn.init.uniform_(self.fc3.weight)
        # torch.nn.init.uniform_(self.fc4.weight)
        # torch.nn.init.uniform_(self.fc5.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x, _ = self.rnn(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x


# model = NIDSCNN().to(device)
# model = CNNModel(197).to(device)
# model = DNNModel(5).to(device)
# model = RNNModel(197, 128, 5)
model = Die(197)
# model = Dummy(197)


# class UNSW_Dataset(Dataset):
#     def __init__(self, X, Y):
#         self.data = X
#         self.label = Y

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.label[index]
#         return x, y


# train_dataset = UNSW_Dataset(X_train, Y_train)
# test_dataset = UNSW_Dataset(X_test, Y_test)
# print(X_train.unsqueeze(1).shape)
# print(Y_train.unsqueeze(1).shape)
# use for everything else
# train_dataset = torch.utils.data.TensorDataset(
#     X_train.unsqueeze(1), Y_train.unsqueeze(1)
# )
# use for rnn
# print(X_train)
# print(Y_train)
# turn half the results to 1
# mask = torch.rand(Y_train.shape) < 0.5
# Y_train[mask] = 1
# print(X_train.shape)
# X_train = X_train[:, :5]
# print(X_train.shape)

# X_train = torch.randn(15000, 1, 197)
# Y_train = torch.randint(0, 2, (15000, 1)).float()
# X_test = torch.randn(15000, 1, 197)
# Y_test = torch.randint(0, 2, (15000, 1)).float()


# from torchsampler import ImbalancedDatasetSampler

# num_samples = len(train_dataset)
# num_classes = 2
# class_counts = torch.zeros(num_classes)
# for i in range(num_samples):
#     _, label = train_dataset[i]
#     if i % 100 == 0:
#         print(train_dataset[i])
#         print(label)
#     class_counts[label.long()] += 1

# class_weights = 1.0 / class_counts
# print(class_counts)
# print(class_weights)

# sampler = ImbalancedDatasetSampler(train_dataset, class_weights)


class BalancedDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

        # Separate the inputs and labels based on class
        class_0_indices = torch.where(labels == 0)[0]
        class_1_indices = torch.where(labels == 1)[0]

        # Determine the size of the smaller class
        min_class_size = min(len(class_0_indices), len(class_1_indices))

        # Sample equal number of samples from each class
        balanced_indices = torch.cat(
            [
                torch.randperm(len(class_0_indices))[:min_class_size],
                torch.randperm(len(class_1_indices))[:min_class_size],
            ]
        )

        # Update the inputs and labels with balanced data
        self.inputs = self.inputs[balanced_indices]
        self.labels = self.labels[balanced_indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = self.inputs[index]
        label = self.labels[index]
        return input_data, label


train_dataset = BalancedDataset(X_train, Y_train)
# train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = BalancedDataset(X_test, Y_test)
# test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

# batch_size = 64
batch_size = 64
num_epochs = 400
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.Adagrad(model.parameters(), lr=0.001)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# model.train()
# num_epochs = 10
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for inputs, labels in data_loader:
#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels.unsqueeze(1))

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     # Print the average loss for the epoch
#     avg_loss = running_loss / len(data_loader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in data_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        if torch.isinf(outputs).any() or torch.isnan(outputs).any():
            print("Model returned inf or nan values during evaluation.")

        loss = criterion(outputs, labels.unsqueeze(1))
        # loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

torch.save(model.state_dict(), "/opt/app/data/Oracle_CNN_SGD.pt")

model.eval()
total_corrects = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        if torch.isinf(outputs).any() or torch.isnan(outputs).any():
            print("Model returned inf or nan values during evaluation.")
        predicted_labels = torch.round(outputs)
        # print(f"input_shape: {inputs.shape} -- output_shape: {outputs.shape}")
        # print(f"predicted_shape: {predicted_labels.shape} -- lables_shape: {labels.shape}")
        correct = torch.sum((predicted_labels == labels)[:, 0])
        sample_count = labels.size(0)
        total_corrects += correct
        total_samples += sample_count
        # print(f"correct: {correct} -- samples: {sample_count}")

print(f"total_corrects: {total_corrects}")
print(f"total_samples: {total_samples}")
accuracy = total_corrects / total_samples
print(f"Evaluation Accuracy: {accuracy:.4f}")
