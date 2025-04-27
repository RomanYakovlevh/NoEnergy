import datetime
import openpyxl
import numpy as np

workbook = openpyxl.load_workbook("./data/Buildings_el.xlsx")


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


class WeatherArchiveSheet:
    def __init__(self):
        self.sheet = workbook['Weather archive']

        self.max_row = self.sheet.max_row

        self.timestamps = [datetime.datetime.strptime(x[0].value, '%d.%m.%Y %H:%M') for x in
                           self.sheet['A4': f'A{self.max_row}']]
        self.timestamps_numpy = np.array(self.timestamps, dtype='datetime64[m]')

        self.air_temperature = np.array([x[0].value for x in self.sheet['B4': f'B{self.max_row}']], dtype='float64')
        self.atm_pressure_mm_mercury = np.array([x[0].value for x in self.sheet['C4': f'C{self.max_row}']],
                                                dtype='float64')
        self.atm_pressure_sea_level = np.array([x[0].value for x in self.sheet['D4': f'D{self.max_row}']],
                                               dtype='float64')
        self.relative_humidity = np.array([x[0].value for x in self.sheet['E4': f'E{self.max_row}']], dtype='float64')

        # TODO: Figure out a nice way to pre-process labels in mean_wind_direction
        self.mean_wind_direction = np.array([x[0].value for x in self.sheet['F4': f'F{self.max_row}']], dtype='<U37')

        self.mean_wind_speed = np.array([x[0].value for x in self.sheet['G4': f'G{self.max_row}']], dtype='float64')

        # TODO: How to deal with 'None' values in max_gust_value?
        self.max_gust_value = np.array([x[0].value for x in self.sheet['H4': f'H{self.max_row}']], dtype=object)

        # TODO: Same problem as in max_gust_value - field weather_phenomena contains a lot of 'None'
        self.weather_phenomena = np.array([x[0].value for x in self.sheet['I4': f'I{self.max_row}']], dtype=object)

        # TODO: What data in weather_phenomena_2 is supposed to mean?
        self.weather_phenomena_2 = np.array([x[0].value for x in self.sheet['J4': f'J{self.max_row}']], dtype=object)

        # TODO: Omg how are we even supposed to work with this
        self.total_cloud_cover = np.array([x[0].value for x in self.sheet['K4': f'K{self.max_row}']], dtype=object)

        # I think we can agree that value '10.0 and more' is just 10.0
        self.visibility = np.array(
            [x[0].value if x[0].value != '10.0 and more' else 10.0 for x in self.sheet['L4': f'L{self.max_row}']],
            dtype='float64')

        self.dewpoint_temperature = np.array([x[0].value for x in self.sheet['M4': f'M{self.max_row}']],
                                             dtype='float64')

        # TODO: Come up with a better way of dealing with missing data
        self.air_temperature[np.isnan(self.air_temperature)] = 0
        self.atm_pressure_mm_mercury[np.isnan(self.atm_pressure_mm_mercury)] = 0
        self.atm_pressure_sea_level[np.isnan(self.atm_pressure_sea_level)] = 0
        self.relative_humidity[np.isnan(self.relative_humidity)] = 0
        self.mean_wind_speed[np.isnan(self.mean_wind_speed)] = 0
        self.dewpoint_temperature[np.isnan(self.dewpoint_temperature)] = 0

        self.numeric_feature = dict()
        self.numeric_feature['air temperature'] = self.air_temperature
        self.numeric_feature['Atm pressure mm of mercury'] = self.atm_pressure_mm_mercury
        self.numeric_feature['atm pressure to sea level'] = self.atm_pressure_sea_level
        self.numeric_feature['Relative humidity (%)'] = self.relative_humidity
        self.numeric_feature['Mean wind speed'] = self.mean_wind_speed
        # self.numeric_feature['Max gust value'] = self.max_gust_value # Uncomment once we figure out 'None'-s
        self.numeric_feature['visibility'] = self.visibility
        self.numeric_feature['dewpoint temperature'] = self.dewpoint_temperature

        self.categorical_feature = dict()
        self.categorical_feature['Mean wind direction'] = self.mean_wind_direction
        self.categorical_feature['weather phenomena WW'] = self.weather_phenomena
        self.categorical_feature["weather phenomena W'W'"] = self.weather_phenomena_2
        self.categorical_feature['total cloud cover'] = self.total_cloud_cover


class AreasSheet:
    def __init__(self):
        self.sheet = workbook['Areas']

        self.max_row = self.sheet.max_row

        self.building_id = [x[0].value for x in self.sheet['A2': f'A{self.max_row}']]
        self.area_m2 = np.array([x[0].value for x in self.sheet['A2': f'A{self.max_row}']])
