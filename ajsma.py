#!/usr/bin/env -S docker exec -it 83047ce42523 python3 /opt/app/data/ajsma.py
# https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/jsma.html
import torch
import numpy as np
import pickle
import torch.nn as nn
import typing
import matplotlib.pyplot as plt  # for plotting
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.autograd.functional as autograd_func

# import argparse

from torchattacks.attack import Attack

torch.autograd.set_detect_anomaly(True)


# class Argparser:
#     """Argparser class."""

#     def __init__(self):
#         self.parser = argparse.ArgumentParser()
#         self.parser.add_argument(
#             "--rho",
#             "-r",
#             type=float,
#             help="number of iterations we will do for the JSMA",
#             default=10,
#         )
#         self.parser.add_argument(
#             "--theta",
#             "-t",
#             type=float,
#             help="the amount of disturbance to add",
#             default=1,
#         )
#         self.parser.add_argument(
#             "--magnitude",
#             "-m",
#             type=float,
#             help="lambda in JSMA",
#             default=0.05,
#         )
#         self.args = self.parser.parse_args()


cols = [
    "sttl",
    "dttl",
    "swin",
    "trans_depth" "res_bdy_len",
    "stime",
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
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_dst_sport_ltm",
    "dur_log1p",
    "sbytes_log1p",
    "dbytes_log1p",
    "sload_log1p",
    "dload_log1p",
    "spkts_log1p",
    "stcpb_log1p",
    "dtcpb_log1p",
    "smeansz_log1p",
    "dmeansz_log1p",
    "sjit_log1p",
    "djit_log1p",
    "network_bytes_log1p",
    "proto_3pc",
    "proto_a/n",
    "proto_aes-sp3-d",
    "proto_any",
    "proto_argus",
    "proto_aris",
    "proto_arp",
    "proto_ax.25",
    "proto_bbn-rcc",
    "proto_bna",
    "proto_br-sat-mon",
    "proto_cbt",
    "proto_cftp",
    "proto_chaos",
    "proto_compaq-peer",
    "proto_cphb",
    "proto_cpnx",
    "proto_crtp",
    "proto_crudp",
    "proto_dcn",
    "proto_ddp",
    "proto_ddx",
    "proto_dgp",
    "proto_egp",
    "proto_eigrp",
    "proto_emcon",
    "proto_encap",
    "proto_esp",
    "proto_etherip",
    "proto_fc",
    "proto_fire",
    "proto_ggp",
    "proto_gmtp",
    "proto_gre",
    "proto_hmp",
    "proto_i-nlsp",
    "proto_iatp",
    "proto_ib",
    "proto_icmp",
    "proto_idpr",
    "proto_idpr-cmtp",
    "proto_idrp",
    "proto_ifmp",
    "proto_igmp",
    "proto_igp",
    "proto_il",
    "proto_ip",
    "proto_ipcomp",
    "proto_ipcv",
    "proto_ipip",
    "proto_iplt",
    "proto_ipnip",
    "proto_ippc",
    "proto_ipv6",
    "proto_ipv6-frag",
    "proto_ipv6-no",
    "proto_ipv6-opts",
    "proto_ipv6-route",
    "proto_ipx-n-ip",
    "proto_irtp",
    "proto_isis",
    "proto_iso-ip",
    "proto_iso-tp4",
    "proto_kryptolan",
    "proto_l2tp",
    "proto_larp",
    "proto_leaf-1",
    "proto_leaf-2",
    "proto_merit-inp",
    "proto_mfe-nsp",
    "proto_mhrp",
    "proto_micp",
    "proto_mobile",
    "proto_mtp",
    "proto_mux",
    "proto_narp",
    "proto_netblt",
    "proto_nsfnet-igp",
    "proto_nvp",
    "proto_ospf",
    "proto_pgm",
    "proto_pim",
    "proto_pipe",
    "proto_pnni",
    "proto_pri-enc",
    "proto_prm",
    "proto_ptp",
    "proto_pup",
    "proto_pvp",
    "proto_qnx",
    "proto_rdp",
    "proto_rsvp",
    "proto_rtp",
    "proto_rvd",
    "proto_sat-expak",
    "proto_sat-mon",
    "proto_sccopmce",
    "proto_scps",
    "proto_sctp",
    "proto_sdrp",
    "proto_secure-vmtp",
    "proto_sep",
    "proto_skip",
    "proto_sm",
    "proto_smp",
    "proto_snp",
    "proto_sprite-rpc",
    "proto_sps",
    "proto_srp",
    "proto_st2",
    "proto_stp",
    "proto_sun-nd",
    "proto_swipe",
    "proto_tcf",
    "proto_tcp",
    "proto_tlsp",
    "proto_tp++",
    "proto_trunk-1",
    "proto_trunk-2",
    "proto_ttp",
    "proto_udp",
    "proto_udt",
    "proto_unas",
    "proto_uti",
    "proto_vines",
    "proto_visa",
    "proto_vmtp",
    "proto_vrrp",
    "proto_wb-expak",
    "proto_wb-mon",
    "proto_wsn",
    "proto_xnet",
    "proto_xns-idp",
    "proto_xtp",
    "proto_zero",
    "service_None",
    "service_dhcp",
    "service_dns",
    "service_ftp",
    "service_ftp-data",
    "service_http",
    "service_irc",
    "service_pop3",
    "service_radius",
    "service_smtp",
    "service_snmp",
    "service_ssh",
    "service_ssl",
    "state_ACC",
    "state_CLO",
    "state_CON",
    "state_ECO",
    "state_ECR",
    "state_FIN",
    "state_INT",
    "state_MAS",
    "state_PAR",
    "state_REQ",
    "state_RST",
    "state_TST",
    "state_TXD",
    "state_URH",
    "state_URN",
    "state_no",
]

allowed_cols = [
    0,
    3,
    4,
    5,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    25,
    27,
    28,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
]

# for i, col in enumerate(cols):
#     print(f"{col}: {i}")


def augment(inputs, model, magnitude=0.05):
    xp = inputs.clone().requires_grad_()
    jacobian = autograd_func.jacobian(model, xp)
    print(f"jacobian_shape: {jacobian.shape}")

    abs_gradients = torch.abs(jacobian)
    scaled_gradients = magnitude * abs_gradients
    print(
        f"inputs_shape: {inputs.shape} -- grad_shape: {scaled_gradients.squeeze(1).shape}"
    )
    augmented_inputs = inputs + scaled_gradients.squeeze(1)
    return augmented_inputs


class Oracle(nn.Module):
    def __init__(self, input_size):
        super(Oracle, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.sigmoid(x)
        return x


class Substitute_Model(nn.Module):
    def __init__(self, input_size):
        super(Substitute_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
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


file_path = "/opt/app/data"

oracle = Oracle(197)
oracle.load_state_dict(torch.load("/opt/app/data/Oracle_Adam.pt"))
oracle.eval()
file_path = "/opt/app/data/"


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


# procedure
# first we train an oracle. it could be any deep learning model so this
# step is just a classical training step
# after training the oracle we can do the attack
# first we give the oracle some inputs and get the lables for those
# we use those to train the model
# then we use the JBDSA to get new data points
# then we give these new data points to the model
# and then train the model
# after we have this enough times, we use the substitute model
# to craft adversarial examples


batch_size = 1


def train_loop(X, Y, num_epochs):
    sub_model = Substitute_Model(197)
    # sub_model = RNNModel(197, 32, 5)
    # sub_model = CNNModel(197)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(sub_model.parameters(), lr=0.001)
    train_dataset = BalancedDataset(X, Y)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_losses: typing.List[float] = []
    valid_losses: typing.List[float] = []
    # fig, ax = plt.subplots()
    # Z = torch.zeros(X.shape[0])
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        running_loss = 0.0

        sub_model.train()
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            # inputs.requires_grad_()

            outputs = sub_model(inputs)
            if torch.isinf(outputs).any() or torch.isnan(outputs).any():
                print("Model returned inf or nan values during evaluation.")

            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # ax.plot(range(1, epoch + 2), train_losses, label="Train Loss")
        # ax.set_xlabel("Epoch")
        # ax.set_ylabel("Loss")
        # ax.set_title("Training Progress")
        # ax.legend()
        # plt.pause(0.1)

    # plt.savefig("/opt/app/data/model_train_progress.png")
    # plt.show()
    # plt.close()
    return sub_model


x, y = pickle.load(open(file_path + "/oracle_data.pkl", "rb"))
x = x.sample(frac=0.003)
y = y.sample(frac=0.003)
print(x.shape, y.shape)
X = torch.from_numpy(x.values).type(torch.float)
Y = torch.from_numpy(y.values).type(torch.float)
# X = torch.tensor(x.values, requires_grad=True).type(torch.float)
# Y = torch.tensor(y.values).type(torch.float)


def jacobian_based_augmentation(X, rho, theta, magnitude, ax):
    adversarial_counts = []
    adversarial_counts_percentages = []
    model = Substitute_Model(197)
    for _ in range(0, rho):
        oracle_predictions = oracle(X)
        if (
            torch.isinf(oracle_predictions).any()
            or torch.isnan(oracle_predictions).any()
        ):
            print("Model returned inf or nan values during evaluation.")

        model = train_loop(X, Y, 10)
        # gradients = X.grad
        X_grad = X.detach().requires_grad_()
        # X.requires_grad_()
        outputs = model(X_grad)
        jacobian = torch.zeros(X.shape[0], 197)
        for i in range(X.shape[0]):
            model.zero_grad()
            output_element = outputs.flatten()[i]
            output_element.backward(retain_graph=True)
            jacobian[i, :] = X_grad.grad.flatten(1)[i, :]
            # print(f"jacobian_shape {jacobian.shape}")
            # Z = torch.cat((inputs, inputs + magnitude * torch.sign(jacobian)))
        # X_new = augment(X, model, magnitude=0.05)
        grads = X_grad.grad
        print(f"grads_shape: {grads.shape}")
        abs_grads = torch.abs(grads)
        norm = torch.norm(abs_grads, p=1)
        print(f"abs_grads_shape: {abs_grads.shape}")
        max_grads, _ = torch.max(abs_grads, 0)
        print(f"max_grads_shape: {max_grads.shape}")
        # normalized_grads = abs_grads / max_grads
        normalized_grads = abs_grads / norm
        print(f"normalized_grads_shape: {normalized_grads.shape}")
        saliency_map = normalized_grads.squeeze()
        print(f"saliency_map: {saliency_map.shape}")
        Z = X + magnitude * torch.sign(jacobian)

        saliency_map_limited = torch.index_select(
            saliency_map, dim=1, index=torch.tensor(allowed_cols)
        )
        print(f"saliency_map_limited_shape: {saliency_map_limited.shape}")
        salient_features = torch.topk(saliency_map_limited, 2, dim=1, largest=True)

        adversarial_count = 0
        for i in range(0, X.shape[0]):
            # print(salient_features[1][i].item)
            feature_to_disturb_1 = salient_features[1][i][0].item()
            feature_to_disturb_2 = salient_features[1][i][1].item()

            out = oracle(X[i, :])
            purturbed_input = X[i, :].unsqueeze(0)
            # print(f"perturbation_single_shape: {purturbed_input.shape}")
            purturbed_input[0, feature_to_disturb_1] = (
                purturbed_input[0, feature_to_disturb_1] + theta
            )
            purturbed_input[0, feature_to_disturb_2] = (
                purturbed_input[0, feature_to_disturb_2] + theta
            )
            out_perturbed = oracle(purturbed_input)

            # print(out_perturbed > 0.5, out > 0.5)
            if (
                torch.sigmoid(out > 0.5).item()
                != torch.sigmoid(out_perturbed > 0.5).item()
            ):
                if (
                    abs(
                        torch.norm(X[i, :].unsqueeze(0), p=0)
                        - torch.norm(purturbed_input, p=0)
                    )
                    <= 5
                ):
                    adversarial_count += 1
        print(f"adversarial_count: {adversarial_count}")
        print(f"percent: {adversarial_count/X.shape[0]}")
        adversarial_counts.append(adversarial_count)
        adversarial_counts_percentages.append(adversarial_count / X.shape[0])

        # print(f"X_shape: {X.shape} -- X_new_shape: {X_new.shape}")
        X = torch.cat((X, Z), 0)
        print(f"X_shape: {X.shape}")

    ax.plot(
        range(1, rho + 1),
        adversarial_counts_percentages,
        label=f"theta={theta}--lambda={magnitude}",
    )
    ax.legend()
    plt.pause(0.01)

    return model


def main() -> None:
    # argparser = Argparser()
    # rho = 5
    # theta = 1
    # magnitude = 0.05

    fig, ax = plt.subplots()
    ax.set_xlabel("rho")
    ax.set_ylabel("Adversarial Example Count %")
    ax.set_title(f"lambda = 0.02")
    ax.legend()
    _ = jacobian_based_augmentation(X, 5, 1, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.95, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.85, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.75, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.70, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.65, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.15, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.1, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.05, 0.02, ax)
    _ = jacobian_based_augmentation(X, 5, 0.001, 0.02, ax)
    plt.savefig(f"/opt/app/data/adv_count_MLP_02.png")
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel("rho")
    ax.set_ylabel("Adversarial Example Count")
    ax.set_title(f"lambda = 0.05")
    ax.legend()
    _ = jacobian_based_augmentation(X, 5, 1, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.95, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.85, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.75, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.70, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.65, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.15, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.1, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.05, 0.05, ax)
    _ = jacobian_based_augmentation(X, 5, 0.001, 0.05, ax)
    plt.savefig(f"/opt/app/data/adv_count_MLP_05.png")
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel("rho")
    ax.set_ylabel("Adversarial Example Count")
    ax.set_title(f"lambda = 0.1")
    ax.legend()
    _ = jacobian_based_augmentation(X, 5, 1, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.95, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.85, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.75, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.70, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.65, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.15, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.1, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.05, 0.1, ax)
    _ = jacobian_based_augmentation(X, 5, 0.001, 0.1, ax)
    plt.savefig(f"/opt/app/data/adv_count_MLP_1.png")
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel("rho")
    ax.set_ylabel("Adversarial Example Count")
    ax.set_title(f"lambda = 0.2")
    ax.legend()
    _ = jacobian_based_augmentation(X, 5, 1, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.95, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.85, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.75, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.70, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.65, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.15, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.1, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.05, 0.2, ax)
    _ = jacobian_based_augmentation(X, 5, 0.001, 0.2, ax)
    plt.savefig(f"/opt/app/data/adv_count_MLP_2.png")
    plt.show()
    plt.close()

    # torch.save(substitute_model.state_dict(), "/opt/app/data/Substitute_Model.pt")


if __name__ == "__main__":
    main()
