"""
 This file is solely created to run multiple files so that we could chain up the whole process

 Course Code: FIM601
 Project Name: Deep Neural Network in State Transition Model

 last modified date: 03.09.2023
 last modified by: Colson

"""

# Header File Here
import numpy as np
import pandas as pd
import Preprocess
import EconomicsData
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from pathlib import Path

# Global Parameters
chunk_size = 10 ** 6
output_df_list = []

# Main Thread here
if __name__ == "__main__":
    print("This is main thread")

    # timer for overall running time
    tic_0 = time.perf_counter()

    # Clean current working directory
    input_path = "ProcessedFile/"
    Preprocess.emptyCurrentFolder(input_path)

    # Get Economic Indicators
    input_path = "RawData/HPI_AT_ZIP.csv"
    hpi_data = EconomicsData.getHPI(input_path)

    input_path = "RawData/HPI_AT_state.csv"
    hpist_data = EconomicsData.getHPISt(input_path)

    input_path = "RawData/Mortgage_Rate.csv"
    interest_data = EconomicsData.getInterest(input_path)

    # input_path = "RawData/URateDataMSA.csv"
    # urate_data = EconomicsData.getURate(input_path)
    #
    # input_path = "RawData/URateDataState.csv"
    # uratest_data = EconomicsData.getURateSt(input_path, "RawData/GeoCode.csv")

    input_path = "RawData/ur_3q22.csv"
    urate_data_arch = EconomicsData.getURateArch(input_path)

    input_path = "RawData/Judicial_Table.csv"
    jd_data = EconomicsData.getJudicial(input_path)

    input_path = "RawData/T10Y2Y.csv"
    macro_data = EconomicsData.getMacro(input_path)

    # Get File Structure
    input_path = "RawData/Variable_Params.csv"
    meta_data = pd.read_csv(input_path, sep=',', header=0,
                            dtype={0: "int64", 1: "string", 2: "string", 3: "string"})
    meta_data['Abbreviation'] = meta_data['Abbreviation'].str.rstrip()
    selected = meta_data.loc[meta_data["Indicator"] == "1"]
    print("----------Read in meta completed!----------")

    # Get File List
    input_path = "RawData/Data_Files_Name.csv"
    file_list = pd.read_csv(input_path, header=0)
    first_flag = True
    timer = []
    for index, row in file_list.iterrows():
        print("file index: " + str(index + 1), ". file name: " + row[0])
        input_path = "RawData/Performance_All/" + row[0]
        Preprocess.getInputByChunk(input_path=input_path,
                                   meta=selected,
                                   chunk_size=chunk_size,
                                   hpi_data=hpi_data,
                                   hpist_data=hpist_data,
                                   interest_data=interest_data,
                                   urate_data=urate_data_arch,
                                   # uratest_data=uratest_data,
                                   jd_data=jd_data,
                                   macro_data=macro_data,
                                   first_flag=first_flag,
                                   timer=timer)
        first_flag = False
        # Preprocess.distCheck(raw_data)
        # Preprocess.writeCurrentDataset(raw_data, 'ProcessedFile/Full1.csv')
        # Preprocess.splitDataset(raw_data)

    # check frequence of each variable to handle missingness and uniqueness
    # for col in raw_data.columns:
    #     print(raw_data[col].describe())

    print("----------All File Preprocessed Complete!----------")

    # timer for overall process
    toc_0 = time.perf_counter()
    print("Total Running Time: ", str(toc_0-tic_0), " s.")
    print("Total Number of Chunk: ", str(len(timer)), " chunks.")
    avg = sum(timer)/len(timer)
    print("Average Running Time for each block: ", str(avg), " s.")
