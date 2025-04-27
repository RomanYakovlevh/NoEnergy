import numpy as np
import matplotlib.pyplot as plt
from read_buildings_data import ElectricitySheet, WeatherArchiveSheet, BuildingsWorkbook
from scipy.ndimage import gaussian_filter1d

workbook = BuildingsWorkbook()


def plot_named_parameter_with_gaussian_filter(name: str, dictionary, time_axis, sigma):
    data = dictionary[name]
    smooth_data = gaussian_filter1d(data, sigma=sigma)
    sd_mean = np.mean(smooth_data)
    smooth_data -= sd_mean
    smooth_data *= np.std(data) / np.std(smooth_data)
    smooth_data += sd_mean
    plt.plot(time_axis, smooth_data, label=name)


def plot_all_named_params(dictionary, time_axis, ncol=2, sigma=50):
    for name in dictionary.keys():
        plot_named_parameter_with_gaussian_filter(name, dictionary, time_axis, sigma)

    plt.legend(ncol=ncol)
    plt.grid(True)
    plt.show()


def plot_electricity_usage():
    electricity_sheet = workbook.electricity_sheet
    plt.plot(electricity_sheet.timestamps, electricity_sheet.building['ICT'], label="ICT Raw")
    plot_all_named_params(electricity_sheet.building, electricity_sheet.timestamps, ncol=3, sigma=100)


def plot_weather_archive():
    weather_archive_sheet = workbook.weather_archive_sheet

    temperature_only = dict()
    temperature_only['air temperature'] = weather_archive_sheet.numeric_feature['air temperature']
    temperature_only['dewpoint temperature'] = weather_archive_sheet.numeric_feature['dewpoint temperature']
    plot_all_named_params(temperature_only, weather_archive_sheet.timestamps)

    pressure_only = dict()
    pressure_only['Atm pressure mm of mercury'] = weather_archive_sheet.numeric_feature['Atm pressure mm of mercury']
    pressure_only['atm pressure to sea level'] = weather_archive_sheet.numeric_feature['atm pressure to sea level']
    plot_all_named_params(pressure_only, weather_archive_sheet.timestamps)

    humidity = {'Relative humidity (%)': weather_archive_sheet.numeric_feature['Relative humidity (%)']}
    plot_all_named_params(humidity, weather_archive_sheet.timestamps)

    mean_wind_speed = {'Mean wind speed': weather_archive_sheet.numeric_feature['Mean wind speed']}
    plot_all_named_params(mean_wind_speed, weather_archive_sheet.timestamps)

    visibility = {'visibility': weather_archive_sheet.numeric_feature['visibility']}
    plot_all_named_params(visibility, weather_archive_sheet.timestamps)


if __name__ == '__main__':
    plot_electricity_usage()
    plot_weather_archive()
