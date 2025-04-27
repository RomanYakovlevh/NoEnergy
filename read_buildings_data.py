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

        self.ict              = self.all_buildings[:,0]
        self.u06_u06a_u05b    = self.all_buildings[:,1]
        self.obs              = self.all_buildings[:,2]
        self.u05_u04_u04b_geo = self.all_buildings[:,3]
        self.teg              = self.all_buildings[:,4]
        self.lib              = self.all_buildings[:,5]
        self.mek              = self.all_buildings[:,6]
        self.soc              = self.all_buildings[:,7]
        self.s01              = self.all_buildings[:,8]
        self.d04              = self.all_buildings[:,9]

        self.building                     = dict()
        self.building["ict"]              = self.ict
        self.building["u06_u06a_u05b"]    = self.u06_u06a_u05b
        self.building["obs"]              = self.obs
        self.building["u05_u04_u04b_geo"] = self.u05_u04_u04b_geo
        self.building["teg"]              = self.teg
        self.building["lib"]              = self.lib
        self.building["mek"]              = self.mek
        self.building["soc"]              = self.soc
        self.building["s01"]              = self.s01
        self.building["d04"]              = self.d04


class WeatherArchiveSheet:
    def __init__(self):
        self.sheet = workbook['Weather archive']

        self.max_row = self.sheet.max_row

class AreasSheet:
    def __init__(self):
        self.sheet = workbook['Areas']

        self.max_row = self.sheet.max_row