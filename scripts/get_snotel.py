import pandas as pd


def __read_snotel_stations(json_file):
    """
    Reads the swecoord.json and creates a list of station IDs that will be used to retrieve data from the SNOTEL api.
    :param json_file: a file that contains SNOTEL station IDs and their coordinates, string
    :return: list of SNOTEL station IDs
    """

    df = pd.read_json(json_file).transpose().reset_index()
    df['index'] = df['index'].apply(str)
    df['name'] = df['index'] + ":MT:SNTL"

    names = df['name'].values.tolist()

    return names


def get_snotel(start_date, end_date, json_file="../data/swecoords.json", out_name="../data/snotel_swe.csv"):
    """
    Retrieves data from the SNOTEL api and returns a pandas data frame of SWE values for all SNOTEL stations in Montana
    for a given date range
    :param start_date: start date of data retrieval, string
    :param end_date: end date of data retrieval, string
    :param json_file: a file that contains SNOTEL station IDs and their coordinates, string
    :param out_name: path on local machine to save the snotel data to, string
    :return: None
    """

    stations = __read_snotel_stations(json_file)
    base_str = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultipleStationReport/daily/start_of_period/"

    swe_vals = pd.DataFrame()

    for station in stations:

        station_number = station.split(":")[0]

        new_str = base_str + station + "|name/" + start_date + "," + end_date + "/stationId,WTEQ::value"
        dat = pd.read_csv(new_str, header=53, names=['date', 'station_id', station_number],
                          usecols=['date', station_number],
                          date_parser=pd.to_datetime, index_col=0)

        swe_vals = pd.concat([swe_vals, dat], axis=1)

    swe_vals = swe_vals * 25.4
    swe_vals.to_csv(out_name)


# example to run code:
get_snotel("2016-08-31", "2017-08-30", "../data/swecoords.json", "../data/snotel_swe.csv")


