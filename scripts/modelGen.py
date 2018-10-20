import json
import pandas as pd
from utils import utilsVector
from utils import utilsRaster
import numpy as np
import math
import glob


class DawuapMonteCarlo(object):

    def __init__(self, param_files, rand_size):

        self.param_files = param_files
        self.rand_size = rand_size
        self.run_number = 0
        self.rand_array = None
        self.rast_params = None
        self.stream_params = None
        self.basin_params = None

    def _get_raster_values(self):

        out_df = pd.DataFrame()

        with open(self.param_files) as json_data:
            dat = json.load(json_data)

        for feat in dat.values():

            arr = utilsRaster.RasterParameterIO("../params/" + feat).array
            val = np.unique(arr.squeeze())
            by_val = val/2.

            out_values = np.random.uniform(float(val) - float(by_val), float(val) + float(by_val), self.rand_size)
            out_values = pd.DataFrame({feat: out_values})

            out_df = pd.concat([out_df, out_values], axis=1)

        return out_df

    def _get_basin_values(self):

        out_df = pd.DataFrame()

        dat = utilsVector.VectorParameterIO("../params/subsout.shp")
        hbv_ck0 = np.zeros(1)
        hbv_ck1 = np.zeros(1)
        hbv_ck2 = np.zeros(1)
        hbv_hl1 = np.zeros(1)
        hbv_perc = np.zeros(1)
        hbv_pbase = np.zeros(1)

        for feat in dat.read_features():
            hbv_ck0 = np.append(hbv_ck0, feat['properties']['hbv_ck0'])
            hbv_ck1 = np.append(hbv_ck1, feat['properties']['hbv_ck1'])
            hbv_ck2 = np.append(hbv_ck2, feat['properties']['hbv_ck2'])
            hbv_hl1 = np.append(hbv_hl1, feat['properties']['hbv_hl1'])
            hbv_perc = np.append(hbv_perc, feat['properties']['hbv_perc'])
            hbv_pbase = np.append(hbv_pbase, feat['properties']['hbv_pbase'])

        hbv_ck0 = np.unique(hbv_ck0[hbv_ck0 != 0.])
        hbv_ck1 = np.unique(hbv_ck1[hbv_ck1 != 0.])
        hbv_ck2 = np.unique(hbv_ck2[hbv_ck2 != 0.])
        hbv_hl1 = np.unique(hbv_hl1[hbv_hl1 != 0.])
        hbv_perc = np.unique(hbv_perc[hbv_perc != 0.])
        hbv_pbase = np.unique(hbv_pbase[hbv_pbase != 0.])

        dat = {"hbv_ck0": hbv_ck0,
               "hbv_ck1": hbv_ck1,
               "hbv_ck2": hbv_ck2,
               "hbv_hl1": hbv_hl1,
               "hbv_perc": hbv_perc}

        for val in dat:

            by_val = dat[val] / 2.

            out_values = np.random.uniform(float(dat[val]) - float(by_val), float(dat[val]) + float(by_val), self.rand_size)
            out_values = pd.DataFrame({val: out_values})

            out_df = pd.concat([out_df, out_values], axis=1)

        by_val = hbv_pbase / 2.
        by_val = math.ceil(by_val)

        out_values = np.random.randint(int(hbv_pbase) - int(by_val), int(hbv_pbase) + int(by_val), self.rand_size)
        out_values = pd.DataFrame({'hbv_pbase': out_values})

        return pd.concat([out_df, out_values], axis=1)

    def _get_river_values(self):

        # These values might change. Keep in mind for later

        e = np.random.uniform(0, 0.5, self.rand_size)
        ks = np.random.uniform(41200., 123600., self.rand_size)

        return pd.DataFrame({'e': e,
                             'ks': ks})

    def set_rand_array(self):

        self.rand_array = pd.concat([self._get_river_values(),
                                     self._get_basin_values(),
                                     self._get_raster_values()], axis=1)


    def _create_raster_parameters(self):

        with open(self.param_files) as json_data:
            dat = json.load(json_data)

        params = {}

        for feat in dat:

            value = self.rand_array[dat[feat]].iloc[self.run_number]
            rast = utilsRaster.RasterParameterIO('../params/' + dat[feat])
            rast.array.fill(value)
            params[feat] = rast


    def _create_basin_parameters(self):

        sub_vec = utilsVector.VectorParameterIO('../params/subsout.shp')

        lstDict = []
        for feat in sub_vec.read_features():
            feat['properties']['hbv_ck0'] = self.rand_array['hbv_ck0'].iloc[self.run_number]
            feat['properties']['hbv_ck1'] = self.rand_array['hbv_ck1'].iloc[self.run_number]
            feat['properties']['hbv_ck2'] = self.rand_array['hbv_ck2'].iloc[self.run_number]
            feat['properties']['hbv_hl1'] = self.rand_array['hbv_hl1'].iloc[self.run_number]
            feat['properties']['hbv_perc'] = self.rand_array['hbv_perc'].iloc[self.run_number]
            feat['properties']['hbv_pbase'] = self.rand_array['hbv_pbase'].iloc[self.run_number]

            lstDict.append(feat)

        #sub_vec.write_dataset('../temp/temp_basin.shp', params=lstDict)


        return None

    def _create_river_parameters(self):

        riv_vec = utilsVector.VectorParameterIO('../params/rivout.shp')

        lstDict = []
        for feat in riv_vec.read_features():

            feat['properties']['e'] = self.rand_array['e'].iloc[self.run_number]
            feat['properties']['ks'] = self.rand_array['ks'].iloc[self.run_number]

            lstDict.append(feat)

        riv_vec.write_dataset('..temp/temp_riv.shp', params=lstDict)

        return None



test = DawuapMonteCarlo("../params/param_files_test.json", 1000)

test.set_rand_array()

test._create_basin_parameters()
