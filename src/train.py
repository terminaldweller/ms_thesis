#!/usr/bin/env -S docker exec -it 83047ce42523 python3 /opt/app/data/train.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

import pickle

from prettytable import PrettyTable

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict

saved_dict = {}

dfs = []
for i in range(1, 5):
    path = "/opt/app/data/UNSW-NB15_{}.csv"
    dfs.append(pd.read_csv(path.format(i), header=None))
all_data = pd.concat(dfs).reset_index(drop=True)

df_col = pd.read_csv("/opt/app/data/NUSW-NB15_features.csv", encoding="ISO-8859-1")
df_col["Name"] = df_col["Name"].apply(lambda x: x.strip().replace(" ", "").lower())
all_data.columns = df_col["Name"]
saved_dict["columns"] = df_col["Name"][df_col["Name"] != "label"].tolist()
del df_col

print(all_data.shape)
print(all_data.head())

train, test = train_test_split(all_data, test_size=0.3, random_state=16)
del all_data

print(train.shape, "\n", test.shape)

print(train.isnull().sum())

print(train["attack_cat"].value_counts())
train["attack_cat"] = train.attack_cat.fillna(value="normal").apply(
    lambda x: x.strip().lower()
)
print(train["attack_cat"].value_counts())

train["ct_flw_http_mthd"] = train.ct_flw_http_mthd.fillna(value=0)
print(train["is_ftp_login"].value_counts())

print(train.isnull().sum().sum())

print(train.columns)

train_0, train_1 = train["label"].value_counts()[0] / len(train.index), train[
    "label"
].value_counts()[1] / len(train.index)
test_0, test_1 = test["label"].value_counts()[0] / len(test.index), test[
    "label"
].value_counts()[1] / len(test.index)

print(
    "In Train: there are {} % of class 0 and {} % of class 1".format(train_0, train_1)
)
print("In Test: there are {} % of class 0 and {} % of class 1".format(test_0, test_1))

plt.figure()
plt.title("class distribution of train and test dataset")
train["label"].value_counts().plot(kind="bar", color="b", label="train")
test["label"].value_counts().plot(kind="bar", color="orange", label="test")
plt.xlabel("Class")
plt.ylabel("Count")
plt.legend()
plt.savefig("/opt/app/data/data_distrib.png")
plt.show()

print(train["ct_ftp_cmd"].unique())
train["ct_ftp_cmd"] = train["ct_ftp_cmd"].replace(to_replace=" ", value=0).astype(int)

saved_dict["binary_col"] = ["is_sm_ips_ports", "is_ftp_login"]

for col in "is_sm_ips_ports", "is_ftp_login":
    print(train[col].value_counts())
    print()


train["is_ftp_login"] = np.where(train["is_ftp_login"] > 1, 1, train["is_ftp_login"])
train["service"] = train["service"].apply(lambda x: "None" if x == "-" else x)
train["attack_cat"] = (
    train["attack_cat"]
    .replace("backdoors", "backdoor", regex=True)
    .apply(lambda x: x.strip().lower())
)

train.to_csv("/opt/app/data/train_alldata_EDA.csv", index=False)
test.to_csv("/opt/app/data/test_alldata_EDA.csv", index=False)
