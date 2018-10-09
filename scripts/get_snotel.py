#"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultipleStationReport/daily/start_of_period/1040:CO:SNTL|619:OR:SNTL|1048:NM:SNTL|name/POR_BEGIN,POR_END/stationId,WTEQ::value,state.code,network.code"
import pandas as pd
import requests

# function that reads the .json of snotel stations in MT and creates a string that can be used to retrieve the data from
# the snotel api.
# json_file should be the path to the swecoords.json on your machine.
def read_snotel_stations(json_file):

    df = pd.read_json(json_file).transpose().reset_index()
    df['index'] = df['index'].apply(str)
    df['name'] = df['index'] + ":MT:SNTL"

    names = df['name'].values.tolist()

    return names


pd.set_option('display.max_columns', 500)
# reads each of the stations from the function above and then gives daily SWE for the date range provided
# dates should be formatted as strings (e.g. start_date="2017-01-01", end_date="2018-01-01")
#
def get_snotel(start_date, end_date, json_file, out_name="../"):

    stations = read_snotel_stations(json_file)
    base_str = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultipleStationReport/daily/start_of_period/"
    df = pd.DataFrame()

    for station in stations:

        new_str = base_str + station + "|name/" + start_date + "," + end_date + "/stationId,WTEQ::value"
        test = pd.read_csv(new_str, header=53, names=['date', 'id', 'swe'])
        df = df.append(test, ignore_index=True)

        print(df.tail(10))

    return df


print(get_snotel("../data/swecoords.json", "2016-09-01", "2017-08-31"))


