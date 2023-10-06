"""
 This file is solely created to preprocess the economic data and merge then into performance data

 Course Code: FIM601
 Project Name: Deep Neural Network in State Transition Model

 last modified date: 03.09.2023
 last modified by: Colson

"""

# Header File Here
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import UnemploymentRetrieve


# Interest Rate(Spread) Data
def getInterest(input_path):
    names = ["Date", "30_Year", "15_Year"]
    dtype = {0: 'string', 1:'float64', 2:'float64'}
    usecols = [0, 1, 2]
    interest_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype)
    interest_data.dropna(how='all', inplace=True)
    interest_data["Date"] = pd.to_datetime(interest_data["Date"], format="%d/%m/%Y")
    interest_data["Year"] = interest_data["Date"].dt.year
    interest_data["Month"] = interest_data["Date"].dt.month
    interest_data = interest_data.groupby(['Year', 'Month']).tail(1)
    interest_data.drop(columns=['Date'], inplace=True)
    gc.collect()
    print("Read Interest Rate Completed!!")
    return interest_data


# HPI Data
def getHPI(input_path):
    names = ["Zip", "Year", "Quarter", "HPI"]
    dtype = {0: 'string', 1: "int64", 2: "int64", 3: "float64"}
    usecols = [0, 1, 2, 3]
    hpi_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype)
    print("Read HPI ZIP Completed!!")
    return hpi_data


# HPI State Level Data
def getHPISt(input_path):
    names = ["State", "Year", "Quarter", "HPI"]
    dtype = {0: 'string', 1: "int64", 2: "int64", 3: "float64"}
    usecols = [0, 1, 2, 3]
    hpi_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype)
    print("Read HPI State Completed!!")
    return hpi_data


# MSA level Unemployment Data
def getURate(input_path, flag=True):
    names = ["Year", "Month", "URate", "Series"]
    dtype = {0: 'string', 1: 'string', 2: 'float64', 3: 'string'}
    usecols = [0, 1, 2, 3]
    ur_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype, na_values='-')
    ur_data.dropna(how='all', inplace=True)
    ur_data['MSA'] = ur_data['Series'].str.slice(start=7, stop=12)
    ur_data['FIPS'] = ur_data['Series'].str.slice(start=5, stop=7)
    if flag:
        ur_data.drop(columns=['Series', 'FIPS'], inplace=True)
    ur_data["Month"] = ur_data["Month"].astype(int)
    ur_data["Year"] = ur_data["Year"].astype(int)
    gc.collect()
    print("Read Unemployment Rate Completed!!")
    return ur_data


# State level Unemployment data
# path 1 for state level data and path 2 for geocoding
def getURateSt(input_path1, input_path2):
    geo_data = UnemploymentRetrieve.getGeocode(input_path2)
    ur_data = getURate(input_path1, False)
    ur_data.drop(columns=["Series", "MSA"], inplace=True)
    ur_data["FIPS"] = ur_data["FIPS"].astype(int)
    ur_data = ur_data.merge(geo_data, how='left', right_on=["FIPS"], left_on=["FIPS"])
    return ur_data


# UR data from Arch
def getURateArch(input_path):
    names = ["FIPS", "Date", "URate", "StateAbb"]
    dtype = {0: 'string', 1: 'string', 2: 'float64', 3: 'string'}
    usecols = [0, 1, 2, 3]
    ur_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype, na_values='-')
    ur_data.dropna(subset="URate", inplace=True)
    ur_data["Date"] = pd.to_datetime(ur_data["Date"], format="%m/%d/%Y")
    ur_data["Month"] = ur_data["Date"].dt.month
    ur_data["Year"] = ur_data["Date"].dt.year
    print(ur_data["Year"].describe())
    ur_data.drop(columns=["Date"], inplace=True)
    return ur_data


# State level Judicial Information
def getJudicial(input_path):
    names = ["State", "Full", "Judicial"]
    dtype = {0: 'string', 1: "string", 2: "int64"}
    usecols = [0, 2]
    jd_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype)
    print("Read State Judicial Completed!!")
    return jd_data


# 10yr 2yr spread as macro economic indicator
def getMacro(input_path):
    names = ["Date", "Spread"]
    dtype = {0: 'string', 1: "float64"}
    usecols = [0, 1]
    macro_data = pd.read_csv(input_path, header=0, names=names, usecols=usecols, dtype=dtype)
    macro_data["Date"] = pd.to_datetime(macro_data["Date"], format="%d/%m/%Y")
    macro_data["Month"] = macro_data["Date"].dt.month
    macro_data["Year"] = macro_data["Date"].dt.year
    macro_data.drop(columns=["Date"], inplace=True)
    print("Read Treasury Spread Completed!!")
    return macro_data


if __name__ == "__main__":
    print("This is Utility module")
    getURateArch("RawData/ur_3q22.csv")
