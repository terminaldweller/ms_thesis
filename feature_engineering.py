#!/usr/bin/env -S docker exec -it 83047ce42523 python3 /opt/app/data/feature_engineering.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


train = pd.read_csv("/opt/app/data/train_alldata_EDA.csv")
test = pd.read_csv("/opt/app/data/test_alldata_EDA.csv")


def multi_corr(col1, col2="label", df=train):
    """
    This function returns correlation between 2 given features.
    Also gives corr of the given features with "label" afetr applying log1p to it.
    """
    corr = df[[col1, col2]].corr().iloc[0, 1]
    log_corr = df[col1].apply(np.log1p).corr(df[col2])

    print("Correlation : {}\nlog_Correlation: {}".format(corr, log_corr))


def corr(col1, col2="label", df=train):
    """
    This function returns correlation between 2 given features
    """
    return df[[col1, col2]].corr().iloc[0, 1]


# Selecting all the features with high correlation values with other features
# Refer: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
corr_matrix = train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# We don't want to use these features for plotting because these are having high corr
# And most likely have same kind of plots with already plotted feature
print("to_drop:", to_drop)


saved_dict = {
    "columns": [
        "srcip",
        "sport",
        "dstip",
        "dsport",
        "proto",
        "state",
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "service",
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "swin",
        "dwin",
        "stcpb",
        "dtcpb",
        "smeansz",
        "dmeansz",
        "trans_depth",
        "res_bdy_len",
        "sjit",
        "djit",
        "stime",
        "ltime",
        "sintpkt",
        "dintpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "is_sm_ips_ports",
        "ct_state_ttl",
        "ct_flw_http_mthd",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_srv_src",
        "ct_srv_dst",
        "ct_dst_ltm",
        "ct_src_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "attack_cat",
    ],
    "binary_col": ["is_sm_ips_ports", "is_ftp_login"],
}

saved_dict["corr_col"] = to_drop
# removing the features from train and test data
train.drop(columns=to_drop, inplace=True)

print(train.shape, test.shape)

train["network_bytes"] = train["sbytes"] + train["dbytes"]
train.drop(["srcip", "sport", "dstip", "dsport", "attack_cat"], axis=1, inplace=True)

saved_dict["to_drop"] = ["srcip", "sport", "dstip", "dsport", "attack_cat"]

log1p_col = [
    "dur",
    "sbytes",
    "dbytes",
    "sload",
    "dload",
    "spkts",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "sjit",
    "djit",
    "network_bytes",
]

saved_dict["log1p_col"] = log1p_col

mode_dict = train.mode().iloc[0].to_dict()


def log1p_transform(col, df=train):
    """
    Apply log1p on given column.
    Remove the original cola and keep log1p applied col
    """
    new_col = col + "_log1p"
    df[new_col] = df[col].apply(np.log1p)
    df.drop(col, axis=1, inplace=True)


for col in log1p_col:
    log1p_transform(col, df=train)


print(train.shape)
print(train.columns)

x_train, y_train = train.drop(columns=["label"]), train["label"]
x_test, y_test = test.drop(columns=["label"]), test["label"]

pickle.dump((x_train, y_train), open("/opt/app/data/final_train.pkl", "wb"))
pickle.dump((x_test, y_test), open("/opt/app/data/final_test.pkl", "wb"))

cat_col = ["proto", "service", "state"]
num_col = list(set(x_train.columns) - set(cat_col))

saved_dict["cat_col"] = cat_col
saved_dict["num_col"] = num_col

print(x_train.head())

scaler = StandardScaler()
scaler = scaler.fit(x_train[num_col])

x_train[num_col] = scaler.transform(x_train[num_col])

print(x_train.head())

service_ = OneHotEncoder()
proto_ = OneHotEncoder()
state_ = OneHotEncoder()
ohe_service = service_.fit(x_train.service.values.reshape(-1, 1))
ohe_proto = proto_.fit(x_train.proto.values.reshape(-1, 1))
ohe_state = state_.fit(x_train.state.values.reshape(-1, 1))


for col, ohe in zip(["proto", "service", "state"], [ohe_proto, ohe_service, ohe_state]):
    x = ohe.transform(x_train[col].values.reshape(-1, 1))
    tmp_df = pd.DataFrame(
        x.todense(), columns=[col + "_" + i for i in ohe.categories_[0]]
    )
    x_train = pd.concat([x_train.drop(col, axis=1), tmp_df], axis=1)

print(x_train.head())

file_path = "/opt/app/data/"
pickle.dump(scaler, open(file_path + "scaler.pkl", "wb"))  # Standard scaler
pickle.dump(
    saved_dict, open(file_path + "saved_dict.pkl", "wb")
)  # Dictionary with important parameters
pickle.dump(
    mode_dict, open(file_path + "mode_dict.pkl", "wb")
)  #  Dictionary with most frequent values of columns

pickle.dump(ohe_proto, open(file_path + "ohe_proto.pkl", "wb"))
pickle.dump(ohe_service, open(file_path + "ohe_service.pkl", "wb"))
pickle.dump(ohe_state, open(file_path + "ohe_state.pkl", "wb"))

pickle.dump((x_train, y_train), open(file_path + "final_train.pkl", "wb"))


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


def standardize(data):
    """
    Stanardize the given data. Performs mean centering and varience scaling.
    Using stanardscaler object trained on train data.
    """
    data[saved_dict["num_col"]] = scaler.transform(data[saved_dict["num_col"]])
    return data


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
