import numpy as np
import matplotlib.pyplot as plt
from read_buildings_data import ElectricitySheet
from scipy.ndimage import gaussian_filter1d

electricity_sheet = ElectricitySheet()

def plot_electricity_usage():
    time_axis = electricity_sheet.timestamps

    def plot_building_kwh_consumption(building_name: str):
        building_data = electricity_sheet.building[building_name]
        smooth_data = gaussian_filter1d(building_data, sigma=100)
        sd_mean = np.mean(smooth_data)
        smooth_data -= sd_mean
        smooth_data *= np.std(building_data) / np.std(smooth_data)
        smooth_data += sd_mean

        plt.plot(time_axis, smooth_data, label=building_name)

    for building_name in electricity_sheet.building.keys():
        plot_building_kwh_consumption(building_name)

    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_electricity_usage()