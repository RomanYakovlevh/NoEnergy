import datetime
import openpyxl
import numpy as np
from openpyxl.workbook import Workbook

workbook = openpyxl.load_workbook("./data/Buildings_el.xlsx")


class BuildingsWorkbook:
    def __init__(self, leave_only_matching_entries=True, save_new=True):
        self.electricity_sheet = ElectricitySheet()
        self.weather_archive_sheet = WeatherArchiveSheet()
        self.areas_sheet = AreasSheet()

        # If 'leave_only_matching_entries' is True, this piece of code will filter out hourly entries in
        # weather_archive_sheet that do not have matching hourly entries (by timestamp) in electricity_sheet.
        # It will also apply same thing to electricity_sheet, so only those hourly entries will be left
        # that have matching entries in weather_archive_sheet.
        #
        # This is needed so we can just take an entry from weather_archive_sheet and find corresponding
        # entry in electricity_sheet by index, without having to parse their timestamps.
        if leave_only_matching_entries:
            was_timestamps_h = self.weather_archive_sheet.timestamps_numpy.astype('datetime64[h]')
            es_timestamps_h = self.electricity_sheet.timestamps_numpy.astype('datetime64[h]')
            # print(f"weather_archive_sheet: {was_timestamps_h.shape[0]}, electricity_sheet: {es_timestamps_h.shape[0]}")

            was_es_time_intersect = np.intersect1d(was_timestamps_h, es_timestamps_h, return_indices=True)
            # print(f"weather_archive_and_electricity_time_intersect: {was_es_time_intersect[0].shape[0]}")

            self.electricity_sheet.apply_array_indexing(was_es_time_intersect[2])
            self.weather_archive_sheet.apply_array_indexing(was_es_time_intersect[1])

        if save_new:
            wb = Workbook()

            es = wb.create_sheet('Electricity kWh')
            es['A2'] = 'Timestamps'
            for i, key in enumerate(self.electricity_sheet.building.keys()):
                es.cell(row=2, column=2 + i, value=key)
            for i, timestamp in enumerate(self.electricity_sheet.timestamps):
                es.cell(row=3 + i, column=1, value=timestamp)
            for i in range(self.electricity_sheet.all_buildings.shape[0]):
                for j in range(self.electricity_sheet.all_buildings.shape[1]):
                    es.cell(row=3 + i, column=j + 2, value=self.electricity_sheet.all_buildings[i][j])
                    # es['B3':f'K{len(self.electricity_sheet.timestamps) + 3}'] = self.electricity_sheet.all_buildings

            was = wb.create_sheet('Weather Archive')
            was['A2'] = 'Timestamps'
            col_titles = [
                "air temperature", "Atm pressure mm of mercury", "atm pressure to sea level", "Relative humidity (%)",
                "Mean wind direction",
                "Mean wind speed", "Max gust value", "weather phenomena", "weather phenomena 2", "total cloud cover",
                "visibility", "dewpoint temperature"]
            for i, key in enumerate(col_titles):
                was.cell(row=2, column=2 + i, value=key)
            for i, timestamp in enumerate(self.weather_archive_sheet.timestamps):
                was.cell(row=3 + i, column=1, value=timestamp)

            for i, key in enumerate(self.weather_archive_sheet.categorical_feature.keys()):
                was.cell(row=3 + i, column=2 + len(self.weather_archive_sheet.numeric_feature.keys()) + i, value=key)

            # was['B2':'I2'] = self.weather_archive_sheet.numeric_features_only.keys()
            # was['J2':'M2'] = self.weather_archive_sheet.categorical_feature.values()
            for i in range(self.weather_archive_sheet.all_parameters.shape[0]):
                for j in range(self.weather_archive_sheet.all_parameters.shape[1]):
                    v = self.weather_archive_sheet.all_parameters[i][j]
                    v = v if v is not None else 0
                    was.cell(row=3 + i, column=j + 2, value=v)

            wb.save('data/Buildings_aligned.xlsx')


class ElectricitySheet:
    def __init__(self):
        self.sheet = workbook['Electricity kWh']

        self.max_row = self.sheet.max_row

        self.timestamps = [x[0].value for x in self.sheet['A3': f'A{self.max_row}']]
        self.timestamps_numpy = np.array(self.timestamps, dtype='datetime64[m]')

        self.all_buildings = np.array([[x.value for x in row] for row in self.sheet['B3': f'K{self.max_row}']])

        self.ict = self.all_buildings[:, 0]
        self.u06_u06a_u05b = self.all_buildings[:, 1]
        self.obs = self.all_buildings[:, 2]
        self.u05_u04_u04b_geo = self.all_buildings[:, 3]
        self.teg = self.all_buildings[:, 4]
        self.lib = self.all_buildings[:, 5]
        self.mek = self.all_buildings[:, 6]
        self.soc = self.all_buildings[:, 7]
        self.s01 = self.all_buildings[:, 8]
        self.d04 = self.all_buildings[:, 9]

        self.building = dict()
        self.set_dicts()

    def set_fields(self):
        self.ict = self.all_buildings[:, 0]
        self.u06_u06a_u05b = self.all_buildings[:, 1]
        self.obs = self.all_buildings[:, 2]
        self.u05_u04_u04b_geo = self.all_buildings[:, 3]
        self.teg = self.all_buildings[:, 4]
        self.lib = self.all_buildings[:, 5]
        self.mek = self.all_buildings[:, 6]
        self.soc = self.all_buildings[:, 7]
        self.s01 = self.all_buildings[:, 8]
        self.d04 = self.all_buildings[:, 9]

    def set_dicts(self):
        self.building["ICT"] = self.ict
        self.building["U06, U06A, U05B"] = self.u06_u06a_u05b
        self.building["OBS"] = self.obs
        self.building["U05, U04, U04B, GEO"] = self.u05_u04_u04b_geo
        self.building["TEG"] = self.teg
        self.building["LIB"] = self.lib
        self.building["MEK"] = self.mek
        self.building["SOC"] = self.soc
        self.building["S01"] = self.s01
        self.building["D04"] = self.d04

    def apply_array_indexing(self, indexes: np.ndarray):
        self.max_row = indexes.shape[0]
        self.timestamps = [x for i, x in enumerate(self.timestamps) if i in indexes]
        self.timestamps_numpy = self.timestamps_numpy[indexes]
        self.all_buildings = self.all_buildings[indexes, :]
        self.set_fields()
        self.set_dicts()
        # print(f"len(self.building['SOC']): {self.building["SOC"].shape[0]}, len(self.soc): {self.soc.shape[0]}")


class WeatherArchiveSheet:
    def __init__(self):
        self.sheet = workbook['Weather archive']

        self.max_row = self.sheet.max_row

        # reversed because in 'Weather archive' all data is in descending order for some reason
        self.timestamps = np.flip([datetime.datetime.strptime(x[0].value, '%d.%m.%Y %H:%M') for x in
                                   self.sheet['A4': f'A{self.max_row}']])
        self.timestamps_numpy = np.array(self.timestamps, dtype='datetime64[m]')

        self.all_parameters = np.array([[x.value for x in row] for row in self.sheet['B4': f'M{self.max_row}']],
                                       dtype=object)
        self.all_parameters = np.flip(self.all_parameters, axis=0)

        self.air_temperature = None
        self.atm_pressure_mm_mercury = None
        self.atm_pressure_sea_level = None
        self.relative_humidity = None
        self.mean_wind_direction = None
        self.mean_wind_speed = None
        self.max_gust_value = None
        self.weather_phenomena = None
        self.weather_phenomena_2 = None
        self.total_cloud_cover = None
        self.visibility = None
        self.dewpoint_temperature = None

        self.numeric_features_only = None

        self.set_fields()
        self.numeric_feature = dict()
        self.categorical_feature = dict()
        self.set_dicts()

    def fill_missing_data(self):
        # TODO: Come up with a better way of dealing with missing data
        self.air_temperature[np.isnan(self.air_temperature)] = 0
        self.atm_pressure_mm_mercury[np.isnan(self.atm_pressure_mm_mercury)] = 0
        self.atm_pressure_sea_level[np.isnan(self.atm_pressure_sea_level)] = 0
        self.relative_humidity[np.isnan(self.relative_humidity)] = 0
        self.mean_wind_speed[np.isnan(self.mean_wind_speed)] = 0
        self.dewpoint_temperature[np.isnan(self.dewpoint_temperature)] = 0

    def set_fields(self):
        self.air_temperature = self.all_parameters[:, 0].astype('float64')
        self.atm_pressure_mm_mercury = self.all_parameters[:, 1].astype('float64')
        self.atm_pressure_sea_level = self.all_parameters[:, 2].astype('float64')
        self.relative_humidity = self.all_parameters[:, 3].astype('float64')

        # TODO: Figure out a nice way to pre-process labels in mean_wind_direction
        self.mean_wind_direction = self.all_parameters[:, 4].astype('<U37')

        self.mean_wind_speed = self.all_parameters[:, 5].astype('float64')

        # TODO: How to deal with 'None' values in max_gust_value?
        self.max_gust_value = self.all_parameters[:, 6]

        # TODO: Same problem as in max_gust_value - field weather_phenomena contains a lot of 'None'
        self.weather_phenomena = self.all_parameters[:, 7]

        # TODO: What data in weather_phenomena_2 is supposed to mean?
        self.weather_phenomena_2 = self.all_parameters[:, 8]

        # TODO: Omg how are we even supposed to work with this
        self.total_cloud_cover = self.all_parameters[:, 9]

        # I think we can agree that value '10.0 and more' is just 10.0
        self.visibility = self.all_parameters[:, 10]
        self.visibility[self.visibility == '10.0 and more'] = 10.0
        self.visibility = self.visibility.astype('float64')

        self.dewpoint_temperature = self.all_parameters[:, 11].astype('float64')

        self.fill_missing_data()

        self.numeric_features_only = np.column_stack((self.air_temperature, self.atm_pressure_mm_mercury,
                                                      self.atm_pressure_sea_level, self.relative_humidity,
                                                      self.mean_wind_speed, self.visibility,
                                                      self.dewpoint_temperature))

    def set_dicts(self):
        self.numeric_feature['air temperature'] = self.air_temperature
        self.numeric_feature['Atm pressure mm of mercury'] = self.atm_pressure_mm_mercury
        self.numeric_feature['atm pressure to sea level'] = self.atm_pressure_sea_level
        self.numeric_feature['Relative humidity (%)'] = self.relative_humidity
        self.numeric_feature['Mean wind speed'] = self.mean_wind_speed
        # self.numeric_feature['Max gust value'] = self.max_gust_value # Uncomment once we figure out 'None'-s
        self.numeric_feature['visibility'] = self.visibility
        self.numeric_feature['dewpoint temperature'] = self.dewpoint_temperature

        self.categorical_feature['Mean wind direction'] = self.mean_wind_direction
        self.categorical_feature['weather phenomena WW'] = self.weather_phenomena
        self.categorical_feature["weather phenomena W'W'"] = self.weather_phenomena_2
        self.categorical_feature['total cloud cover'] = self.total_cloud_cover

    def apply_array_indexing(self, indexes: np.ndarray):
        self.max_row = indexes.shape[0]
        self.timestamps = [x for i, x in enumerate(self.timestamps) if i in indexes]
        self.timestamps_numpy = self.timestamps_numpy[indexes]
        self.all_parameters = self.all_parameters[indexes, :]
        self.set_fields()
        self.set_dicts()
        # print(f"len(self.building['Atm pressure mm of mercury']): {self.numeric_feature["Atm pressure mm of mercury"].shape[0]}, len(self.atm_pressure_mm_mercury): {self.atm_pressure_mm_mercury.shape[0]}")


class AreasSheet:
    def __init__(self):
        self.sheet = workbook['Areas']

        self.max_row = self.sheet.max_row

        self.building_id = [x[0].value for x in self.sheet['A2': f'A{self.max_row}']]
        self.area_m2 = np.array([x[0].value for x in self.sheet['B2': f'B{self.max_row}']], dtype='float64')

        self.building = {self.building_id[i]: self.area_m2[i] for i in range(len(self.building_id))}


if __name__ == '__main__':
    BuildingsWorkbook()
