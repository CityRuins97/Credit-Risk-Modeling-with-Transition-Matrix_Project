"""
 This file is solely created to retrieve unemployment data from BLS data analytics api

 Course Code: FIM601
 Project Name: Deep Neural Network in State Transition Model

 last modified date: 02.16.2023
 last modified by: Colson

"""
import requests
import json
import numpy as np
import pandas as pd
import gc


def requestBLS(seriesid):
    headers = {'Content-type': 'application/json'}
    data = json.dumps({
        "seriesid": seriesid,
        "startyear": "2020",
        "endyear": "2022",
        "registrationkey": "0ae44aa40b314106ac8e0c2292797a7e"
    })
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)
    if len(json_data['Results']) > 0:
        year = []
        period = []
        value = []
        id = []
        for series in json_data['Results']['series']:
            if len(series['data']) > 0:
                for item in series['data']:
                    year.append(item['year'])
                    period.append(item['period'][1:])
                    value.append(item['value'])
                    id.append(series['seriesID'])
        df = pd.DataFrame()
        df['Year'] = year
        df['Month'] = period
        df['URate'] = value
        df['id'] = id
        return df, True
    else:
        print("Failed")
        print(json_data)
        return pd.DataFrame(), False


# Deprecated
def getGeocode(input_path):
    names = ["CBSA", "CBSA_Title", "State_Name", "FIPS"]
    dtype = {0: 'string', 3: 'string', 8: 'string', 9: 'string'}
    usecols = [0, 3, 8, 9]
    geo_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype)
    geo_data.dropna(how='all', inplace=True)
    st_data = geo_data["CBSA_Title"].str.split(pat=",", expand=True)
    st_data.drop(columns=[0], inplace=True)
    st_data.rename(columns={1: "State"}, inplace=True)
    st_data["FIPS"] = geo_data["FIPS"].astype(int)
    st_data["State"] = st_data["State"].str.strip()
    st_data.drop_duplicates(inplace=True)
    st_data = st_data[~st_data["State"].str.contains("-")]
    st_data.reset_index(drop=True, inplace=True)
    print("Read Interest Rate Completed!!")
    return st_data


def getAreaCode(input_path):
    names = ["type", "code"]
    dtype = {0: 'string', 1: 'string'}
    usecols = [0, 1]
    area_data = pd.read_csv(input_path, sep="\t", header=0, names=names, usecols=usecols, dtype=dtype)
    mask = (area_data["type"] == "A")
    st_series = area_data.loc[mask, "code"]
    mask = (area_data["type"] == "B")
    msa_series = area_data.loc[mask, "code"]
    return st_series, msa_series


if __name__ == "__main__":
    # Deprecated
    # input_path = "RawData/GeoCode.csv"
    # geo_data = getMSA(input_path)
    # print(geo_data)

    input_path = "RawData/areacode.txt"
    st_series, msa_series = getAreaCode(input_path)
    print(st_series.describe())
    print(msa_series.describe())

    series_list = []
    for index, id in enumerate(msa_series):
        id = "LAU" + id + "03"
        series_list.append(id)

    ur_list = []
    for i in range(0, len(series_list), 50):
        print(str(i), "---------------------------------------")
        df, flag = requestBLS(series_list[i: i+50])
        print(df.describe())
        ur_list.append(df)

    ur_data = pd.concat(ur_list)
    print(ur_data.describe())
    ur_data.to_csv('RawData/URateDataMSA.csv', float_format='%.1f', index=False, mode='a')

    series_list = []
    for index, id in enumerate(st_series):
        id = "LAS" + id + "03"
        series_list.append(id)

    ur_list = []
    for i in range(0, len(series_list), 50):
        print(str(i), "---------------------------------------")
        df, flag = requestBLS(series_list[i: i + 50])
        print(df.describe())
        ur_list.append(df)

    ur_data = pd.concat(ur_list)
    print(ur_data.describe())
    ur_data.to_csv('RawData/URateDataState.csv', float_format='%.1f', index=False, mode='a')
