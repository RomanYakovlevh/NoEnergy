from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from read_buildings_data import BuildingsWorkbook
from plots import plot_named_parameter_with_gaussian_filter


def predict_one_building():
    workbook = BuildingsWorkbook()
    for building_name in workbook.electricity_sheet.building.keys():
        building = workbook.electricity_sheet.building[building_name]
        numeric_features = workbook.weather_archive_sheet.numeric_features_only
        timestamps = workbook.electricity_sheet.timestamps
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            numeric_features, building, timestamps, test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)

        r_squared = model.score(X_test, y_test)
        print(
            f"model.score: {r_squared}, model.coef_: {model.coef_}, model.intercept_ {model.intercept_}")

        y_predict = model.predict(X_test)

        t_train_sorted, indexes_train = np.unique(t_train, return_index=True)
        t_test_sorted, indexes_test = np.unique(t_test, return_index=True)
        plot_named_parameter_with_gaussian_filter("y_train", y_train[indexes_train], t_train_sorted, 25)
        plot_named_parameter_with_gaussian_filter("y_predict", y_predict[indexes_test], t_test_sorted, 25)

        plt.legend(ncol=2)
        plt.title(f"Linear correlation between weather and electricity \n consumption for building {building_name}")
        plt.figtext(0.03, 0.03, f"R^2: {r_squared}")
        plt.grid(True)
        plt.show()


def predict_all_buildings_by_area():
    workbook = BuildingsWorkbook()

    buildings_electricity_dict = workbook.electricity_sheet.building
    building_areas_dict = workbook.areas_sheet.building
    numeric_features = workbook.weather_archive_sheet.numeric_features_only
    timestamps = workbook.electricity_sheet.timestamps

    # cross validation
    for test_building_name in buildings_electricity_dict.keys():
        test_building_electricity = buildings_electricity_dict[test_building_name]
        test_building_weather_and_area = np.column_stack(
            (np.repeat(building_areas_dict[test_building_name], numeric_features.shape[0]), numeric_features))

        train_buildings_electricity = np.append([],
            [buildings_electricity_dict[building_name] for building_name in buildings_electricity_dict.keys() if
             building_name != test_building_name])
        train_buildings_weather_and_area = np.vstack(
            [np.column_stack(
                (np.repeat(building_areas_dict[building_name], numeric_features.shape[0]), numeric_features)) for
             building_name in buildings_electricity_dict.keys() if
             building_name != test_building_name])

        indexes = np.array(range(train_buildings_electricity.shape[0]))
        np.random.shuffle(indexes)

        model = LinearRegression().fit(train_buildings_weather_and_area[indexes,:], train_buildings_electricity[indexes])

        r_squared = model.score(test_building_weather_and_area, test_building_electricity)
        print(
            f"Test building: {test_building_name}, model.score: {r_squared}")

        y_predicted = model.predict(test_building_weather_and_area)
        plot_named_parameter_with_gaussian_filter("y_test",test_building_electricity, timestamps, 25)
        plot_named_parameter_with_gaussian_filter("y_predicted",y_predicted, timestamps, 25)


        plt.legend(ncol=2)
        plt.title(f"Using area as influencing parameter. Test Building: {test_building_name}")
        plt.figtext(0.03, 0.03, f"R^2: {r_squared}")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    predict_one_building()
    predict_all_buildings_by_area()
