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

from torchattacks.attack import Attack


class JSMA(Attack):
    r"""
    Jacobian Saliency Map Attack in the paper 'The Limitations of Deep Learning in Adversarial Settings'
    [https://arxiv.org/abs/1511.07528v1]

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        theta (float): perturb length, range is either [theta, 0], [0, theta]
        gamma (float): highest percentage of pixels can be modified

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.JSMA(model, theta=1.0, gamma=0.1)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, theta=1.0, gamma=0.1):
        super().__init__("JSMA", model)
        self.theta = theta
        self.gamma = gamma
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            # Because the JSMA algorithm does not use any loss function,
            # it cannot perform untargeted attacks indeed
            # (we have no control over the convergence of the attack to a data point that is NOT equal to the original class),
            # so we make the default setting of the target label is right circular shift
            # to make attack work if user didn't set target label.
            target_labels = (labels + 1) % 10

        adv_images = None
        for im, tl in zip(images, target_labels):
            # Since the attack uses the Jacobian-matrix,
            # if we input a large number of images directly into it,
            # the processing will be very complicated,
            # here, in order to simplify the processing,
            # we only process one image at a time.
            # Shape of MNIST is [-1, 1, 28, 28],
            # and shape of CIFAR10 is [-1, 3, 32, 32].
            pert_image = self.perturbation_single(
                torch.unsqueeze(im, 0), torch.unsqueeze(tl, 0)
            )
            try:
                adv_images = torch.cat((adv_images, pert_image), 0)
            except Exception:
                adv_images = pert_image

        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images

    def compute_jacobian(self, image):
        var_image = image.clone().detach()
        var_image.requires_grad = True
        output = self.get_logits(var_image)

        num_features = int(np.prod(var_image.shape[1:]))
        jacobian = torch.zeros([output.shape[1], num_features])
        for i in range(output.shape[1]):
            if var_image.grad is not None:
                var_image.grad.zero_()
            output[0][i].backward(retain_graph=True)
            # Copy the derivative to the target place
            jacobian[i] = (
                var_image.grad.squeeze().view(-1, num_features).clone()
            )  # nopep8

        return jacobian.to(self.device)

    @torch.no_grad()
    def saliency_map(
        self, jacobian, target_label, increasing, search_space, nb_features
    ):
        # The search domain
        domain = torch.eq(search_space, 1).float()
        # The sum of all features' derivative with respect to each class
        all_sum = torch.sum(jacobian, dim=0, keepdim=True)
        # The forward derivative of the target class
        target_grad = jacobian[target_label]
        # The sum of forward derivative of other classes
        others_grad = all_sum - target_grad

        # This list blanks out those that are not in the search domain
        if increasing:
            increase_coef = 2 * (torch.eq(domain, 0)).float().to(self.device)
        else:
            increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(self.device)
        increase_coef = increase_coef.view(-1, nb_features)

        # Calculate sum of target forward derivative of any 2 features.
        target_tmp = target_grad.clone()
        target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
        # PyTorch will automatically extend the dimensions
        alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(
            -1, nb_features, 1
        )
        # Calculate sum of other forward derivative of any 2 features.
        others_tmp = others_grad.clone()
        others_tmp += increase_coef * torch.max(torch.abs(others_grad))
        beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

        # Zero out the situation where a feature sums with itself
        tmp = np.ones((nb_features, nb_features), int)
        np.fill_diagonal(tmp, 0)
        zero_diagonal = torch.from_numpy(tmp).byte().to(self.device)

        # According to the definition of saliency map in the paper (formulas 8 and 9),
        # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
        if increasing:
            mask1 = torch.gt(alpha, 0.0)
            mask2 = torch.lt(beta, 0.0)
        else:
            mask1 = torch.lt(alpha, 0.0)
            mask2 = torch.gt(beta, 0.0)

        # Apply the mask to the saliency map
        mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
        # Do the multiplication according to formula 10 in the paper
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
        # Get the most significant two pixels
        max_idx = torch.argmax(saliency_map.view(-1, nb_features * nb_features), dim=1)
        # p = max_idx // nb_features
        p = torch.div(max_idx, nb_features, rounding_mode="floor")
        # q = max_idx % nb_features
        q = max_idx - p * nb_features
        return p, q

    def perturbation_single(self, image, target_label):
        """
        image: only one element
        label: only one element
        """
        var_image = image
        var_label = target_label
        var_image = var_image.to(self.device)
        var_label = var_label.to(self.device)

        if self.theta > 0:
            increasing = True
        else:
            increasing = False

        num_features = int(np.prod(var_image.shape[1:]))
        shape = var_image.shape

        # Perturb two pixels in one iteration, thus max_iters is divided by 2
        max_iters = int(np.ceil(num_features * self.gamma / 2.0))

        # Masked search domain, if the pixel has already reached the top or bottom, we don't bother to modify it
        if increasing:
            search_domain = torch.lt(var_image, 0.99)
        else:
            search_domain = torch.gt(var_image, 0.01)

        search_domain = search_domain.view(num_features)
        output = self.get_logits(var_image)
        current_pred = torch.argmax(output.data, 1)

        iter = 0
        while (
            (iter < max_iters)
            and (current_pred != target_label)
            and (search_domain.sum() != 0)
        ):
            # Calculate Jacobian matrix of forward derivative
            jacobian = self.compute_jacobian(var_image)
            # Get the saliency map and calculate the two pixels that have the greatest influence
            p1, p2 = self.saliency_map(
                jacobian, var_label, increasing, search_domain, num_features
            )
            # Apply modifications
            # var_sample_flatten = var_image.view(-1, num_features).clone().detach_()
            var_sample_flatten = var_image.view(-1, num_features)
            var_sample_flatten[0, p1] += self.theta
            var_sample_flatten[0, p2] += self.theta

            new_image = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
            new_image = new_image.view(shape)
            search_domain[p1] = 0
            search_domain[p2] = 0
            # var_image = new_image.clone().detach().to(self.device)
            var_image = new_image.to(self.device)

            output = self.get_logits(var_image)
            current_pred = torch.argmax(output.data, 1)
            iter += 1

        adv_image = var_image
        return adv_image


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


def augment(inputs, gradients, magnitude=0.05):
    # FIXME-we probably should change this for AJSMA
    abs_gradients = torch.abs(gradients)
    scaled_gradients = magnitude * abs_gradients
    augmented_inputs = input_data + scaled_gradients
    return augmented_inputs


def train_loop(X, Y, num_epochs):
    sub_model = Substitute_Model(197)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(sub_model.parameters(), lr=0.001)
    train_dataset = BalancedDataset(X, Y)
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    train_losses: typing.List[float] = []
    valid_losses: typing.List[float] = []
    fig, ax = plt.subplots()
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        running_loss = 0.0
        sub_model.train()
        for inputs, labels in data_loader:
            optimizer.zero_grad()

            outputs = sub_model(inputs)
            if torch.isinf(outputs).any() or torch.isnan(outputs).any():
                print("Model returned inf or nan values during evaluation.")

            loss = criterion(outputs, labels.unsqueeze(1))
            # loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # sub_model.eval()
        # with torch.no_grad():
        #     for inputs, lables in test_loader:
        #         outputs = sub_model(inputs)

        #         loss = criterion(outputs, lables.unsqueeze(1))

        #         valid_loss += loss.item() * inputs.size(0)

        #     valid_loss /= len(test_dataset)
        #     valid_losses.append(valid_loss)

        ax.plot(range(1, epoch + 2), train_losses, label="Train Loss")
        # ax.plot(range(1, epoch + 2), valid_losses, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress")
        ax.legend()
        plt.pause(0.1)

    plt.savefig("/opt/app/data/model_train_progress.png")
    plt.show()
    return model


x, y = pickle.load(open(file_path + "/oracle_data.pkl", "rb"))
x = x.sample(frac=0.003)
y = y.sample(frac=0.003)
print(x.shape, y.shape)
# X = torch.from_numpy(x.values).type(torch.float)
# Y = torch.from_numpy(y.values).type(torch.float)
X = torch.tensor(x.values, requires_grad=True).type(torch.float)
Y = torch.tensor(y.values).type(torch.float)

rho = 10
magnitude = 0.05


def jacobian_based_augmentation(X):
    model = Substitute_Model(197)
    for _ in range(0, rho):
        oracle_predictions = oracle(X)
        if (
            torch.isinf(oracle_predictions).any()
            or torch.isnan(oracle_predictions).any()
        ):
            print("Model returned inf or nan values during evaluation.")

        model = train_loop(X, Y, 10)
        gradients = X.grad

        X_new = augment(X, gradients, magnitude=0.05)

        X = torch.cat((X, X_new), 0)

    return model


substitute_model = jacobian_based_augmentation(X)

# model = YourModel()
# input_data = torch.randn(
#     batch_size, input_channels, input_height, input_width, requires_grad=True
# )
# output = model(input_data)
# output.sum().backward()
# gradients = input_data.grad
# transformed_data = apply_transformations(input_data, gradients)
# augmented_samples = transformed_data.detach()
