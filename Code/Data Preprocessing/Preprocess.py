"""
 This file is solely created to preprocess the data so that we have intermediate dataset that could be used later

 Course Code: FIM601
 Project Name: Deep Neural Network in State Transition Model

 last modified date: 03.02.2023
 last modified by: Colson

"""

# Header File Here
import gc
import time
import pandas as pd
import Utility
import os
import csv
from pathlib import Path
from amortization.amount import calculate_amortization_amount
from scipy import stats


# Read file by chunk and calling processing function
def getInputByChunk(input_path, meta, chunk_size, hpi_data, hpist_data, interest_data, urate_data, jd_data,
                    macro_data, first_flag,  timer):
    listed_input = []

    # with pd.read_csv(input_path, sep='|', header=None, names=meta["Abbreviation"], usecols=meta["ID"],
    #                  dtype=meta["dType"].to_dict(), chunksize=chunk_size,
    #                  nrows=2000000) as reader:
    with pd.read_csv(input_path, sep='|', header=None, names=meta["Abbreviation"], usecols=meta["ID"],
                     dtype=meta["dType"].to_dict(), chunksize=chunk_size) as reader:
        chunk_counter = 1
        for chunk in reader:
            tic = time.perf_counter()
            print("Chunk: " + str(chunk_counter) + "Read Completed!!")
            chunk.insert(0, "Time_Stamp", input_path[24:30])

            formatting(chunk)
            filtering(chunk)
            missingHandle(chunk)
            transformation(chunk)
            variableCreation(chunk)
            chunk = combineHPI(chunk, hpi_data, hpist_data)
            chunk = combineInterest(chunk, interest_data)
            chunk = combineURate(chunk, urate_data)
            chunk = combineJd(chunk, jd_data)
            chunk = combineMacro(chunk, macro_data)
            missingHandle(chunk, True)
            chunk = featureEngineering(chunk)
            cleanUp(chunk)

            # Create structure for output file, only need to be called once
            if chunk_counter == 1 and first_flag:
                updateColumnName(chunk, 'ProcessedFile/')
                Utility.writeMeta(chunk, 'ProcessedFile/metadata.csv')
                first_flag = False

            splitDataset(chunk)
            print("Chunk: " + str(chunk_counter) + "Process Completed!!")
            chunk_counter += 1
            toc = time.perf_counter()
            timer.append(toc-tic)
        # raw_data = pd.concat(listed_input)
        print("Successfully Process " + input_path[24:30] + ".csv")
        print("--------------------EOF--------------------\n")
        gc.collect()
        return chunk


# DF renaming and formatting
def formatting(df):
    df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'])
    df['Act_Period'] = pd.to_datetime(df['Act_Period'])
    df['Orig_Date'] = pd.to_datetime(df['Orig_Date'])
    df['Maturity_Date'] = pd.to_datetime(df['Maturity_Date'])
    df['First_Pay'] = pd.to_datetime(df['First_Pay'])


# Impute the first few entry for each loam base on fixed rate amortization
def imputeAmortization(df):
    df.reset_index(drop=True, inplace=True)
    index = df[df['Curr_UPB'] == 0].index
    first_entry = []
    for i in index:
        if (i == 0) or (df.loc[i - 1, "Curr_UPB"] != 0):
            first_entry.append(i)
    for i in index:
        if i in first_entry:
            df.loc[i, "Curr_UPB"] = df.loc[i, "Orig_UPB"] - \
                                    (calculate_amortization_amount(df.loc[i, "Orig_UPB"],
                                                                   df.loc[i, "Orig_Rate"] / 100,
                                                                   df.loc[i, "Orig_Term"]) - \
                                     df.loc[i, "Orig_Rate"] / 100 / 12 * df.loc[i, "Orig_UPB"])
        else:
            df.loc[i, "Curr_UPB"] = df.loc[i - 1, "Curr_UPB"] - \
                                    (calculate_amortization_amount(df.loc[i, "Orig_UPB"],
                                                                   df.loc[i, "Orig_Rate"] / 100,
                                                                   df.loc[i, "Orig_Term"]) - \
                                     df.loc[i, "Orig_Rate"] / 100 / 12 * df.loc[i - 1, "Curr_UPB"])


# Missing value check & imputation
def missingHandle(df, flag=False):
    if flag:
        df.dropna(subset=["Curr_URate", "Curr_HPI", "Benchmark_Rate", "Serv_Ind", "Next_Status", "First_Flag"], inplace=True)
    else:
        df.dropna(how='all', inplace=True)
        df.fillna({'Mi_Pct': "0", 'Mi_Type': "0", "Interest_Only_Loan_Indicator": "Y"}, inplace=True)
        df["DTI"].fillna(df["DTI"].mean(), inplace=True)
        df["Cscore_B"].fillna(df["Cscore_B"].mean(), inplace=True)


# Filtering for unwanted data
def filtering(df):
    drop_list = []

    # HomeReady Program filter
    mask = df['HomeReady_Program_Indicator'] == '7'
    df["HomeReady_Program_Indicator"] = "Y"
    df.loc[mask, "HomeReady_Program_Indicator"] = "N"

    # FRM only filter
    index = df[df['Product'] == 'ARM'].index
    df.drop(index, inplace=True)
    drop_list.append('Product')

    # Modification filter
    index = df[df['Mod_Flag'] == 'Y'].index
    df.drop(index, inplace=True)
    drop_list.append('Mod_Flag')

    # Deprecated
    # Interest only filter
    # index = df[df['Interest_Only_Loan_Indicator'] == 'Y'].index
    # df.drop(index, inplace=True)
    # drop_list.append('Interest_Only_Loan_Indicator')

    # Deprecated
    # Relocation filter
    # index = df[df['Relocation_Mortgage_Indicator'] == 'Y'].index
    # df.drop(index, inplace=True)
    # drop_list.append('Relocation_Mortgage_Indicator')

    drop_list.append("Adj_Rem_Months")

    # Drop column that only used for filtering
    df.drop(columns=drop_list, inplace=True)


# Transformation of existing variables
def transformation(df):
    # Delinquent Record + Zero Balance Code => Status for our project
    # Default detection
    mask = (df["Current_Loan_Delinquency_Status"] == "XX")
    df.loc[mask, "Current_Loan_Delinquency_Status"] = str(999)

    # Prepay detection
    df["Rem_Months"] = df.groupby('Loan_ID')["Rem_Months"].ffill()
    mask = (df["Zero_Balance_Code"] == 1)
    df.loc[mask, "Current_Loan_Delinquency_Status"] = str(-1)

    # Zero Balance Code filtering
    mask = (df["Current_Loan_Delinquency_Status"] == "XX") & (df["Zero_Balance_Code"].isin([2, 3, 9]))
    df.loc[mask, "Current_Loan_Delinquency_Status"] = str(7)

    # SDA for loan age?
    df["Age_Prop"] = df["Loan_Age"] / df["Orig_Term"]


# New field creation
def variableCreation(df):
    # Forward one period status as target field for each row
    df["Current_Status"] = pd.to_numeric(df["Current_Loan_Delinquency_Status"], errors='coerce')
    df["Next_Status"] = df.groupby("Loan_ID")["Current_Status"].shift(periods=-1)
    df["Cum_Max_Status"] = df.groupby("Loan_ID")["Current_Status"].cummax()
    df["Cum_Count"] = df.groupby("Loan_ID")["Loan_ID"].cumcount()

    # Re-filter the dataset to drop all records after first default
    index = df[df["Cum_Max_Status"] >= 7].index
    df.drop(index, inplace=True)

    # Re-filter the dataset to drop all first record for each group
    index = df[df["Cum_Count"] == 0].index
    df.drop(index, inplace=True)

    # Re-filter the dataset to drop the last record for prepay/mature
    index = df[df["Current_Status"] == -1].index
    df.drop(index, inplace=True)

    # Re-filter the dataset to drop the last record for abnormal closing (other zero balance code)
    index = df[df["Next_Status"] == 999].index
    df.drop(index, inplace=True)

    df["Current_Status"] = df["Current_Status"].astype("category")
    df["Next_Status"] = df["Next_Status"].astype("category")

    # Deprecated
    # First few records without current upb after encoding
    # index = df[df['Curr_UPB'] == 0].index
    # df.drop(index, inplace=True)
    # Use imputed value instead
    # imputeAmortization(df)
    df.dropna(subset=["Curr_UPB"], inplace=True)

    # Drop utility column after encoding
    df.drop(columns=["Cum_Max_Status", "Cum_Count",
                     "Zero_Balance_Code"], inplace=True)


# MTM-related Economic covariate creation
def combineHPI(orig_df, hpi_df, hpist_df):
    # create time column for merging
    orig_df["act_year"] = orig_df["Act_Period"].dt.year
    orig_df["act_quarter"] = orig_df["Act_Period"].dt.quarter
    orig_df["orig_year"] = orig_df["Orig_Date"].dt.year
    orig_df["orig_quarter"] = orig_df["Orig_Date"].dt.quarter
    orig_df = orig_df.merge(hpi_df, how='left', left_on=['ZIP', 'act_year', 'act_quarter'],
                            right_on=['Zip', 'Year', 'Quarter'])
    orig_df = orig_df.merge(hpi_df, how='left', left_on=['ZIP', 'orig_year', 'orig_quarter'],
                            right_on=['Zip', 'Year', 'Quarter'])

    # clean up redundant columns
    orig_df.rename(columns={'HPI_x': 'Curr_HPI', 'HPI_y': 'Orig_HPI'}, inplace=True)

    # fill in state level data if necessary
    orig_df = orig_df.merge(hpist_df, how='left', left_on=['State', 'act_year', 'act_quarter'],
                            right_on=['State', 'Year', 'Quarter'])
    orig_df = orig_df.merge(hpist_df, how='left', left_on=['State', 'orig_year', 'orig_quarter'],
                            right_on=['State', 'Year', 'Quarter'])
    orig_df["Curr_HPI"].fillna(orig_df["HPI_x"], inplace=True)
    orig_df["Orig_HPI"].fillna(orig_df["HPI_y"], inplace=True)

    orig_df.drop(columns=['act_quarter', 'act_year', 'orig_quarter', 'orig_year', 'HPI_x', 'HPI_y',
                          'Zip_x', 'Year_x', 'Quarter_x', 'Zip_y', 'Year_y', 'Quarter_y'], inplace=True)

    orig_df["HPI_Adjust_Factor"] = orig_df["Curr_HPI"] / orig_df["Orig_HPI"]
    orig_df["MTMLTV"] = orig_df['Curr_UPB'] / (orig_df['Orig_UPB'] / orig_df['OLTV'] * orig_df['HPI_Adjust_Factor'])

    return orig_df


# Interest-related
def combineInterest(orig_df, interest_df):
    orig_df["act_year"] = orig_df["Act_Period"].dt.year
    orig_df["act_month"] = orig_df["Act_Period"].dt.month
    orig_df = orig_df.merge(interest_df, how='left', left_on=['act_year', 'act_month'],
                            right_on=['Year', 'Month'])

    # figure out rate for corresponding maturity
    mask = (orig_df["Orig_Term"] == 360)
    orig_df.loc[mask, "15_Year"] = orig_df.loc[mask, "30_Year"]
    orig_df.rename(columns={'15_Year': 'Benchmark_Rate'}, inplace=True)
    orig_df["Refinance Indicator"] = 100 * orig_df["Orig_Rate"] / orig_df["Benchmark_Rate"]

    # clean up redundant columns
    orig_df.drop(columns=['act_year', 'act_month', 'Year', 'Month', '30_Year'], inplace=True)

    # Original Spread (SATO)
    orig_df["orig_year"] = orig_df["Orig_Date"].dt.year
    orig_df["orig_month"] = orig_df["Orig_Date"].dt.month
    orig_df = orig_df.merge(interest_df, how='left', left_on=['orig_year', 'orig_month'],
                            right_on=['Year', 'Month'])

    # figure out rate for corresponding maturity
    orig_df.loc[mask, "15_Year"] = orig_df.loc[mask, "30_Year"]
    orig_df["SATO"] = orig_df["Orig_Rate"] - orig_df["15_Year"]

    # # clean up redundant columns
    orig_df.drop(columns=['orig_year', 'orig_month', 'Year', 'Month', '30_Year', '15_Year'], inplace=True)

    return orig_df


# Unemployment Rate related
def combineURate(orig_df, urate_df):
    # create time column for merging
    orig_df["act_year"] = orig_df["Act_Period"].dt.year
    orig_df["act_month"] = orig_df["Act_Period"].dt.month
    orig_df["orig_year"] = orig_df["Orig_Date"].dt.year
    orig_df["orig_month"] = orig_df["Orig_Date"].dt.month
    orig_df = orig_df.merge(urate_df, how='left', left_on=['MSA', 'act_year', 'act_month'],
                            right_on=['FIPS', 'Year', 'Month'])
    orig_df = orig_df.merge(urate_df, how='left', left_on=['MSA', 'orig_year', 'orig_month'],
                            right_on=['FIPS', 'Year', 'Month'])

    # clean up redundant columns
    orig_df.rename(columns={'URate_x': 'Curr_URate', 'URate_y': 'Orig_URate'}, inplace=True)

    # fill in state level data if necessary
    orig_df = orig_df.merge(urate_df, how='left', left_on=['State', 'act_year', 'act_month'],
                            right_on=['StateAbb', 'Year', 'Month'])
    orig_df = orig_df.merge(urate_df, how='left', left_on=['State', 'orig_year', 'orig_month'],
                            right_on=['StateAbb', 'Year', 'Month'])
    orig_df["Curr_URate"].fillna(orig_df["URate_x"], inplace=True)
    orig_df["Orig_URate"].fillna(orig_df["URate_y"], inplace=True)

    orig_df.drop(columns=['act_month', 'act_year', 'orig_month', 'orig_year', 'Year_x', 'Month_x', 'Year_y', 'Month_y',
                          'FIPS_x', 'FIPS_y', 'StateAbb_x', 'StateAbb_y', 'URate_x', 'URate_y'], inplace=True)
    orig_df["URate_Change"] = orig_df["Curr_URate"] - orig_df["Orig_URate"]

    return orig_df


# Combine Judicial Information
def combineJd(orig_df, jd_df):
    orig_df = orig_df.merge(jd_df, how="left", left_on="State", right_on="State")
    return orig_df


def combineMacro(orig_df, macro_df):
    orig_df["act_year"] = orig_df["Act_Period"].dt.year
    orig_df["act_month"] = orig_df["Act_Period"].dt.month
    orig_df = orig_df.merge(macro_df, how='left', left_on=['act_year', 'act_month'],
                            right_on=['Year', 'Month'])
    orig_df.rename(columns={'Spread': '10-2Spread'}, inplace=True)
    orig_df.drop(columns=['act_month', 'act_year', 'Month', 'Year'], inplace=True)
    return orig_df


# Feature engineering
def featureEngineering(df):
    # Orig_Year for vintage effects
    df["Orig_Year"] = df["Orig_Date"].dt.year
    df["Vintage_Year"] = pd.cut(df["Orig_Year"], [0, 2004, 2006, 2008, 2010, 2020])
    df = pd.get_dummies(df, prefix=["Vintage"], columns=["Vintage_Year"], drop_first=True)
    df.drop(columns=["Orig_Year"], inplace=True)

    # Seasonality effects
    df["Act_Month"] = df["Act_Period"].dt.month
    df = pd.get_dummies(df, prefix=["Month"], columns=["Act_Month"], drop_first=True)

    # First order difference for some variables
    # Spline/Discretization transformation
    return df


# Outlier detection and handling
def distCheck(df):
    Utility.Univariate(df)
    # for col in df.columns:
    #     print(df[col].describe())


# Update column name after completing the data portion
def updateColumnName(df, output_path):
    current_status_list = [0, 1, 2, 3, 4, 5, 6]
    next_status_list = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
    for current_status in current_status_list:
        for next_status in next_status_list:
            if next_status <= (current_status + 1) or next_status == 7:
                header = list(df.columns)
                with open(output_path + str(current_status) + "-" + str(next_status) + ".csv", 'w') as file:
                    dw = csv.DictWriter(file, delimiter=',',
                                        fieldnames=header)
                    dw.writeheader()
    print("Output Column Name Construction Completed!!")


# static clean up funciton
def cleanUp(df):
    drop_vars = ["Curr_Rate", "Orig_HPI", "Orig_Date",  "First_Pay", "Loan_Age", "Rem_Months", "Maturity_Date", "OCLTV", "State",
                 "Current_Loan_Delinquency_Status",   "Orig_URate"]
    continuous_vars = ["Curr_URate", "Curr_HPI", "Orig_Rate", "Orig_UPB", "Curr_UPB", "Num_Bo", "DTI", "Cscore_B", "Mi_Pct", "HPI_Adjust_Factor",
                       "MTMLTV", "Benchmark_Rate", "Interest_Spread", "URate_Change"]
    categorical_vars = ["Channel", "Orig_Term", "First_Flag", "Purpose", "Prop", "No_Units", "Occ_Stat",
                        "Prepaymnet_Penalty_Indicator", "Interest_Only_Loan_Indicator", "Mod_Flag", "Mi_Type", "Serv_Ind", "MSA", "ZIP",
                        "HomeReady_Program_Indicator", "Relocation_Mortgage_Indicator"]
    utility_vars = ["Time_Stamp", "Loan_ID", "Act_Period", "Current_Status", "Next_Status"]
    df.drop(columns=drop_vars, inplace=True)


# Split dataset base on initial status
def splitDataset(df):
    current_status_list = [0, 1, 2, 3, 4, 5, 6]
    next_status_list = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
    for current_status in current_status_list:
        for next_status in next_status_list:
            # if not(current_status == 0 and next_status == 0):
            if next_status <= (current_status + 1) or next_status == 7:
                mask = (df["Current_Status"] == current_status) & (df["Next_Status"] == next_status)
                if current_status == 0:
                    out_df = df.loc[mask, :].sample(frac=0.0005, replace=False, random_state=601)
                elif current_status == 1:
                    out_df = df.loc[mask, :].sample(frac=0.05, replace=False, random_state=601)
                else:
                    out_df = df.loc[mask, :].sample(frac=0.2, replace=False, random_state=601)
                writeCurrentDataset(out_df, ("ProcessedFile/" + str(current_status) + "-" + str(next_status) + ".csv"))


# Write out processed dataset
def writeCurrentDataset(df, output_path, frame=False, append=True):
    real_path = Path(output_path)
    real_path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        df.to_csv(real_path, index=frame, float_format='%.4f', mode='a', header=frame)
    else:
        df.to_csv(real_path, index=frame, float_format='%.4f', header=frame)
    # print("Write Completed!")


# Clean the csv file in current folder
def emptyCurrentFolder(input_path):
    for folder, subfolders, files in os.walk(input_path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.png'):
                path = os.path.join(folder, file)
                print('Deleted: ', path)
                os.remove(path)


if __name__ == "__main__":
    print("This is Cleaning module")