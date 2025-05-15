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

        self.X_weather = workbook.weather_archive_sheet.numeric_features_only[:, 0] # shape = 8755,1

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
        return self.X_electricity[:, idx].astype("float32"),  self.X_weather.astype("float32"), self.Y_electricity[:, idx].astype("float32")


class InferBuildingCNN(nn.Module):
    def __init__(self):
        super(InferBuildingCNN, self).__init__()

        # 32x8755
        self.conv1_weather = nn.Conv1d(1, 32, 15, padding=7)

        # 32x351
        self.pool1_weather = nn.MaxPool1d(25, 25, padding=1)

        # 64x351
        self.conv2_weather = nn.Conv1d(32, 64, 5, padding=2)

        # 64x36
        self.pool2_weather = nn.MaxPool1d(10, 10, padding=1)


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
        self.fc1 = nn.Linear(128 * 57 + 2240, 5000)
        self.fc2 = nn.Linear(5000, 2936)

        # Time to reconstruct 500 features into time-series 7340 (7340 / 2 -> 3670 / 2 -> 1835 / 5 -> 367)
        # 367 * 5 = 1835 -> 32x1835
        self.reconv1 = nn.ConvTranspose1d(8, 128, 15, padding=5, stride=5)

        # 1835 * 2 = 3670 -> 16x3670
        self.reconv2 = nn.ConvTranspose1d(128, 64, 6, padding=2, stride=2)

        # 3670 * 2 = 7340 -> 1x7340
        self.reconv3 = nn.ConvTranspose1d(64, 1, 6, padding=2, stride=2)


    def forward(self, x):
        x1, x2 = x
        # Pass through Conv 1 -> ReLU -> Pooling
        x1 = self.pool1(torch.relu(self.conv1(x1)))

        # Pass through Conv 2 -> ReLU -> Pooling
        x1 = self.pool2(torch.relu(self.conv2(x1)))

        x1 = torch.relu(self.conv3(x1))

        x2 = self.pool1_weather(torch.relu(self.conv1_weather(x2)))

        x2 = self.pool2_weather(torch.relu(self.conv2_weather(x2)))

        # Pass through the fully connected layers
        x =  torch.cat((x1.view(-1, 128 * 57), x2.view(-1, 2240)), 1)
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
        for electer, weather, labels in trainloader:
            electer, weather, labels = electer.to(device), weather.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model((electer, weather))
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
        for electer, weather, labels in testloader:
            electer, weather, labels = electer.to(device), weather.to(device), labels.to(device)
            # Forward pass
            outputs = model((electer, weather))

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