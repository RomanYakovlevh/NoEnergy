import time

import read_buildings_data
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from plots import plot_named_parameter_with_gaussian_filter
from read_buildings_data import BuildingsWorkbook
workbook = BuildingsWorkbook()
class BuildingsDataset(Dataset):
    def __init__(self, test_idx, train):

        #self.X_weather = workbook.weather_archive_sheet.numeric_features_only # shape = 8755,7

        timestamps_months = workbook.electricity_sheet.timestamps_numpy.astype('datetime64[M]').astype(int) % 12

        test_idx_bool = np.zeros(10).astype(bool)
        test_idx_bool[test_idx] = True

        if train:
            self.X_electricity = workbook.electricity_sheet.all_buildings[np.logical_or(timestamps_months == 0, timestamps_months == 1), :] # shape = 1416, 10
            self.X_electricity = self.X_electricity[:, ~test_idx_bool]
            self.Y_electricity = workbook.electricity_sheet.all_buildings[np.logical_and(timestamps_months != 0, timestamps_months != 1), :] # shape = 7339, 10
            self.Y_electricity = np.append(self.Y_electricity, [np.repeat(0, 10)],
                                           axis=0)  # extending 7339 to 7340
            self.Y_electricity = self.Y_electricity[:, ~test_idx_bool]
        else:
            self.X_electricity = workbook.electricity_sheet.all_buildings[np.logical_or(timestamps_months == 0, timestamps_months == 1), test_idx_bool][:, np.newaxis,] # shape = 1416, 10
            self.Y_electricity = workbook.electricity_sheet.all_buildings[np.logical_and(timestamps_months != 0, timestamps_months != 1), test_idx_bool][:, np.newaxis] # shape = 7339, 10
            self.Y_electricity = np.append(self.Y_electricity, [[0]],
                                           axis=0)  # extending 7339 to 7340
        print()

    def __len__(self):
        return self.X_electricity.shape[1]

    def __getitem__(self, idx):
        return self.X_electricity[:, idx].astype("float32"), self.Y_electricity[:, idx].astype("float32")


class InferBuildingCNN(nn.Module):
    def __init__(self):
        super(InferBuildingCNN, self).__init__()

        # # Weather dataset seize is 8755 by 7, and every one of the 7 features needs to be treated separately
        #
        # IN_CHANNELS = 7
        # LAYER_1_OUT = 7 * 16 # 112
        # LAYER_2_OUT = 7 * 32 # 224
        # LAYER_3_OUT = 7 * 32 # 224
        # LAYER_4_OUT = 7 * 8  # 56
        #
        # # The output size will be 7x8755, but with 7*16 channels -> 112x8755
        # self.conv1_weather = nn.Conv1d(IN_CHANNELS, LAYER_1_OUT, 15, padding=7)
        #
        # # Pooling layer with kernel size 2 and stride 2, so the output size will be -> 16x1751
        # self.pool1_weather = nn.MaxPool1d(5, 5, padding=1)
        #
        # # The output size will be 112x1751, but with 32*7 channels -> 224x1751
        # self.conv2_weather = nn.Conv1d(LAYER_1_OUT, LAYER_2_OUT, 5, padding=2)
        #
        # # Pooling layer with kernel size 2 and stride 2, so the output size will be -> 224x351
        # self.pool2_weather = nn.MaxPool1d(5, 5, padding=1)
        #
        # # As we use padding 2, the output size will be 224x351, but with 224 channels -> 224x351
        # self.conv3_weather = nn.Conv1d(LAYER_2_OUT, LAYER_3_OUT, 5, padding=2)
        #
        # # Pooling layer with kernel size 2 and stride 2, so the output size will be -> 224x71
        # self.pool3_weather = nn.MaxPool1d(5, 5, padding=1)
        #
        # # As we use padding 2, the output size will be 224x71, but with 56 channels -> 56x71
        # self.conv4_weather = nn.Conv1d(LAYER_3_OUT, LAYER_4_OUT, 5, padding=2)
        #
        # # Pooling layer with kernel size 2 and stride 2, so the output size will be -> 56x15
        # self.pool4_weather = nn.MaxPool1d(5, 5, padding=1)

        pass

        # 2 MONTHS ELECTRICITY DATA

        # Input size is 1416, with 16 channels -> 16x1416
        self.conv1 = nn.Conv1d(1, 64, 15, padding=7)

        # Pooling layer with kernel size 5 and stride 5, so the output size will be -> 16x284
        self.pool1 = nn.MaxPool1d(5, 5, padding=1)

        # Input size is 284, with 32 channels -> 32x284
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)

        # Pooling layer with kernel size 5 and stride 5, so the output size will be -> 32x57
        self.pool2= nn.MaxPool1d(5, 5, padding=1)

        # Input size is 284, with 32 channels -> 32x57
        self.conv3 = nn.Conv1d(128, 128, 5, padding=2)

        # That is where we flatten the output to 32 * 57 = 1824
        self.fc1 = nn.Linear(128 * 57, 5000)
        self.fc2 = nn.Linear(5000, 2936)

        # Time to reconstruct 500 features into time-series 7340 (7340 / 2 -> 3670 / 2 -> 1835 / 5 -> 367)
        # 367 * 5 = 1835 -> 32x1835
        self.reconv1 = nn.ConvTranspose1d(8, 128, 15, padding=5, stride=5)

        # 1835 * 2 = 3670 -> 16x3670
        self.reconv2 = nn.ConvTranspose1d(128, 64, 6, padding=2, stride=2)

        # 3670 * 2 = 7340 -> 1x7340
        self.reconv3 = nn.ConvTranspose1d(64, 1, 6, padding=2, stride=2)


    def forward(self, x):
        # Pass through Conv 1 -> ReLU -> Pooling
        x = self.pool1(torch.relu(self.conv1(x)))

        # Pass through Conv 2 -> ReLU -> Pooling
        x = self.pool2(torch.relu(self.conv2(x)))

        x = torch.relu(self.conv3(x))

        # Pass through the fully connected layers
        x = x.view(-1, 128 * 57)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = x.view(8, 367)
        x = torch.relu(self.reconv1(x))
        x = torch.relu(self.reconv2(x))

        x = self.reconv3(x)

        return x




if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = InferBuildingCNN().to(device)
    print(model)

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    test_indx = 7

    # Download and load the training data
    # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(BuildingsDataset(test_indx, True), batch_size=1, shuffle=True)

    # Download and load the test data
    # testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(BuildingsDataset(test_indx, False), batch_size=1, shuffle=False)

    # Choose the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the network
    epochs = 300

    best_loss = 200

    best_model_name = f'model_infer_building_cnn-{time.time()}.pth'

    for epoch in range(epochs):
        running_loss = 0.0

        # Iterate over the training data
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")


        if (running_loss / len(trainloader)) < best_loss:
            best_loss = (running_loss / len(trainloader))
            print(f"Saved new best model, best loss: {best_loss}")
            torch.save(model.state_dict(), best_model_name)

    if best_loss == 200:
        torch.save(model.state_dict(), best_model_name)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.data

            plot_named_parameter_with_gaussian_filter("Actual", labels.view(-1).cpu().numpy()[:-1],
                                                      workbook.electricity_sheet.timestamps[1416:], sigma=50)
            plot_named_parameter_with_gaussian_filter("Predicted", predicted.view(-1).cpu().numpy()[:-1],
                                                      workbook.electricity_sheet.timestamps[1416:], sigma=50)
            plt.legend()
            plt.show()

            pml = (predicted - labels)
            correct += np.median(pml)

    print(f"Accuracy on test set: {100 * correct:.2f}%")