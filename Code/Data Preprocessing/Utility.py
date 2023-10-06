"""
 This file is solely created to store any utility function for simple use(e.g. visualization)

 Course Code: FIM601
 Project Name: Deep Neural Network in State Transition Model

 last modified date: 03.09.2023
 last modified by: Colson

"""
# Header File Here
import matplotlib.pyplot as plt
import pandas as pd
import Preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Geo-Coding Conversion


# Univariate Plotting function
def Univariate(df):
    for col in df.columns:
        if df.dtypes[col] == "float64" or df.dtypes[col] == "int64":
            df.hist(column=col, bins=25)
            plt.savefig(fname=("Output/" + col + "_hist.png"))
            plt.close()
            df.boxplot(column=col)
            plt.savefig(fname=("Output/" + col + "_boxplot.png"))
            plt.close()
        elif df.dtypes[col] != "datetime64[ns]":
            df.groupby(col).size().plot(kind='pie')
            plt.savefig(fname=("Output/" + col + "_pie.png"))
            plt.close()


# Combining Data File
def combineAll(output_path):
    initial_state = [0, 1, 2, 3, 4, 5, 6]
    df_list = []

    meta_data = pd.read_csv("ProcessedFile/metadata.csv", sep=',', header=0, usecols=[1, 2], dtype={1: "string", 2: "string"})
    mask = ~((meta_data["dtype"] == "float64") | (meta_data["dtype"] == "uint8"))
    meta_data.loc[mask, "dtype"] = "string"
    mask = meta_data["Column_Name"] == "Mi_Pct"
    meta_data.loc[mask, "dtype"] = "float64"
    mask = meta_data["Column_Name"] == "Num_Bo"
    meta_data.loc[mask, "dtype"] = "string"
    mask = meta_data["Column_Name"] == "No_Units"
    meta_data.loc[mask, "dtype"] = "string"
    mask = meta_data["Column_Name"] == "Judicial"
    meta_data.loc[mask, "dtype"] = "uint8"
    dtype = meta_data.to_dict()['dtype']
    mask = meta_data["dtype"] == "float64"

    for initial in initial_state:
        possible_state = [-1] + [*set([i for i in range(0, initial + 2)] + [7])]
        for possible in possible_state:
            temp = pd.read_csv("ProcessedFile/" + str(initial) + "-" + str(possible) + ".csv")
            df_list.append(temp)
    out_df = pd.concat(df_list)
    out_df.dropna(inplace=True)
    Preprocess.writeCurrentDataset(out_df, output_path, frame=True, append=False)
    print("Write Combine Completed")


# Meta-data Generate
def writeMeta(df, output_path):
    meta = pd.DataFrame(df.columns)
    meta.rename(columns={0: 'Column_Name'}, inplace=True)
    meta['dtype'] = list(df.dtypes)
    Preprocess.writeCurrentDataset(meta, output_path, True)
    print("Metadata write completed!!")


# Write full timeseries of abnormal entries
def writeAbnormal():
    # Get Target List
    target_data = pd.read_csv("ProcessedFile/0-7.csv", header=0, usecols=[0, 1])
    target_list = target_data["Loan_ID"].astype(str).to_list()

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

    # Allocate output place
    result = []

    for index, row in file_list.iterrows():
        print("file index: " + str(index + 1), ". file name: " + row[0])
        input_path = "RawData/" + row[0]
        with pd.read_csv(input_path, sep='|', header=None, names=meta_data["Abbreviation"], usecols=meta_data["ID"],
                         dtype=meta_data["dType"].to_dict(), chunksize=1000000,
                         nrows=200000) as reader:
            for chunk in reader:
                chunk.insert(0, "Time_Stamp", input_path[8:14])
                Preprocess.formatting(chunk)
                for index, row in chunk.iterrows():
                    if row["Loan_ID"] in target_list:
                        result.append(row.to_list())

        # raw_data = pd.concat(listed_input)
        print("Successfully Process " + input_path[8:14] + ".csv")
        print("--------------------EOF--------------------\n")
    result = pd.DataFrame(result)
    Preprocess.writeCurrentDataset(result, "ProcessedFile/Outliers.csv")


# General Training Preparation
def train_prepare(initial_state, file_path="ProcessedFile/", meta_path="ProcessedFile/metadata.csv",
                  onehot=False, returncat=False, test_size=0.2, returnFull=False):
    # Read in the meta data and change the data type
    meta_data = pd.read_csv(meta_path, sep=',', header=0, usecols=[1, 2], dtype={1: "string", 2: "string"})
    mask = ~((meta_data["dtype"] == "float64") | (meta_data["dtype"] == "uint8"))
    meta_data.loc[mask, "dtype"] = "string"
    mask = meta_data["Column_Name"] == "Mi_Pct"
    meta_data.loc[mask, "dtype"] = "float64"
    mask = meta_data["Column_Name"] == "Num_Bo"
    meta_data.loc[mask, "dtype"] = "string"
    mask = meta_data["Column_Name"] == "No_Units"
    meta_data.loc[mask, "dtype"] = "string"
    mask = meta_data["Column_Name"] == "Judicial"
    meta_data.loc[mask, "dtype"] = "uint8"
    dtype = meta_data.to_dict()['dtype']
    mask = meta_data["dtype"] == "float64"
    numeric_features = meta_data.loc[mask, "Column_Name"].tolist()
    dtype[52] = 'string'
    dtype[53] = 'string'

    # recombining the possible target file
    possible_state = [-1] + [*set([i for i in range(0, initial_state + 2)] + [7])]
    merged_data = []
    for i in range(0, len(possible_state)):
        raw_data = pd.read_csv(file_path + str(initial_state) + '-' + str(possible_state[i])
                               + '.csv', dtype=dtype)
        merged_data.append(raw_data)
    merged_data = pd.concat(merged_data)
    merged_data.dropna(inplace=True)
    full_data = merged_data.copy(deep=True)
    target = merged_data["Next_Status"]
    merged_data.drop(columns=["Time_Stamp", "Loan_ID", "Act_Period", "Next_Status", "Current_Status", "Curr_HPI",
                              "Benchmark_Rate", "Curr_URate", "Interest_Only_Loan_Indicator", "MSA", "ZIP"], inplace=True)
    numeric_features.remove("Curr_HPI")
    numeric_features.remove("Benchmark_Rate")
    numeric_features.remove("Curr_URate")

    if onehot:
        mask = meta_data["dtype"] == "string"
        categorical_features = meta_data.loc[mask, "Column_Name"].tolist()[3:-2]
        merged_data = pd.get_dummies(merged_data, columns=categorical_features, drop_first=True)

    categorical_index = merged_data.dtypes.reset_index()[0]
    categorical_index = categorical_index[categorical_index == "string"].index.tolist()
    x_train, x_test, y_train, y_test = train_test_split(merged_data, target, test_size=test_size, random_state=123)
    scaler = StandardScaler()
    scaler.fit(merged_data[numeric_features])
    x_train[numeric_features] = scaler.transform(x_train[numeric_features])
    x_test[numeric_features] = scaler.transform(x_test[numeric_features])
    print('Data Preparation is Complete!')
    if returncat:
        if returnFull:
            return x_train, x_test, y_train, y_test, categorical_index, full_data
        else:
            return x_train, x_test, y_train, y_test, categorical_index
    else:
        if returnFull:
            return x_train, x_test, y_train, y_test, full_data
        else:
            return x_train, x_test, y_train, y_test


def recordCount(states):
    train_sum = 0
    test_sum = 0
    for state in states:
        df = pd.read_csv("Visualization/train_" + str(state) + ".csv")
        train_sum += df.shape[0]
        df = pd.read_csv("Visualization/Pred" + str(state) + ".csv")
        test_sum += df.shape[0]
    print(train_sum)
    print(test_sum)
    print(train_sum+test_sum)


# Main Thread here
if __name__ == "__main__":
    print("This is Utility module")
    # combineAll("ProcessedFile/combineAll.csv")
    recordCount([0, 1, 2, 3, 4, 5, 6])