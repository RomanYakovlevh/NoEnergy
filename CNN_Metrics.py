import numpy as np
import torch
import matplotlib.pyplot as plt

from CNN import BuildingsDataset, InferBuildingCNN

from plots import plot_named_parameter_with_gaussian_filter

from read_buildings_data import BuildingsWorkbook, workbook

book = BuildingsWorkbook()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testloader = torch.utils.data.DataLoader(BuildingsDataset(7, False), batch_size=1, shuffle=False)

model = InferBuildingCNN().to(device)

model.load_state_dict(torch.load('model_infer_building_cnn-1747285988.128024_test_building_7.pth'))
model.eval()  # Set to evaluation mode

correct = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.data

        plot_named_parameter_with_gaussian_filter("Actual", labels.view(-1).cpu().numpy()[:-1], book.electricity_sheet.timestamps[1416:], sigma=50)
        plot_named_parameter_with_gaussian_filter("Predicted", predicted.view(-1).cpu().numpy()[:-1], book.electricity_sheet.timestamps[1416:], sigma=50)
        plt.legend()
        plt.show()

        pml = (predicted - labels)
        correct += np.median(pml)

print(f"Accuracy on test set: {100 * correct:.2f}%")