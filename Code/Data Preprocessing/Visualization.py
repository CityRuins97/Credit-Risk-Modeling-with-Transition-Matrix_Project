import matplotlib.pyplot as plt
import pandas as pd
import Preprocess


def cleanPlot(input_path):
    df = pd.read_csv(input_path, header=0)
    df.dropna(how='any', inplace=True)
    plt.hist(df['Cscore_B'], bins=20, range=(500, 850), rwidth=0.8)
    plt.xlabel("Credit Score", fontsize=16)
    plt.ylabel("Frequency Count", fontsize=16)
    plt.title("Histogram of Credit Score of Borrower", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig("Visualization/Credit_Score_Hist.png")
    return df


def boxPlot(df):
    plt.clf()
    df1 = df[df['Next_Status'] == -1]
    df2 = df[df['Next_Status'] == 0]
    df3 = df[(df['Next_Status'] > 0) & (df['Next_Status'] < 7)]
    df4 = df[df['Next_Status'] == 7]
    data = [df1['Cscore_B'], df2['Cscore_B'], df3['Cscore_B'], df4['Cscore_B']]
    plt.xticks([])
    plt.ylabel("Credit Score", fontsize=16)
    plt.title("Credit Score aggregated on Next Status", fontsize=20)
    plt.boxplot(data)
    plt.show()


if __name__ == "__main__":
    input_path = "ProcessedFile/combineAll.csv"
    clean_data = cleanPlot(input_path)
    print(clean_data.shape)
    boxPlot(clean_data)