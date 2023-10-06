import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from yellowbrick.classifier import ROCAUC
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import cross_val_score



file_path='ProcessedFile/'
merged_data=pd.DataFrame()
numeric_features = ['Orig_Rate', 'Curr_Rate', 'Curr_UPB', 'OLTV', 'DTI', 'Cscore_B', 'Curr_HPI', 'Orig_HPI',
                    'HPI_Adjust_Factor', 'MTMLTV', 'Benchmark_Rate', 'Interest_Spread', 'Rem_Months']
categorical_features = ['Channel', 'Loan_Age', 'Purpose', 'Prop', 'Occ_Stat', 'First_Flag']
target = ['Next_Status']
raw_data=pd.read_csv('ProcessedFile/0-0.csv')
merged_data=pd.concat([merged_data,raw_data])
defected_features=['Mi_Pct','Mi_Type','Cscore_C']
merged_data=merged_data.drop(defected_features,axis=1)
clean_data=merged_data.dropna()
x_train, x_test, y_train, y_test = train_test_split(clean_data[numeric_features + categorical_features],
                                                    clean_data[target],
                                                    test_size=0.3, random_state=123)
x_train = pd.concat([x_train.drop(categorical_features, axis=1), pd.get_dummies(x_train[categorical_features])],
                    axis=1)
x_test = pd.concat([x_test.drop(categorical_features, axis=1), pd.get_dummies(x_test[categorical_features])],
                   axis=1)
scaler = StandardScaler()
scaler.fit(clean_data[numeric_features])
x_train[numeric_features] = scaler.transform(x_train[numeric_features])
x_test[numeric_features] = scaler.transform(x_test[numeric_features])

print(x_train.shape)

merged_data1=pd.DataFrame()
numeric_features = ['Orig_Rate', 'Curr_Rate', 'Curr_UPB', 'OLTV', 'DTI', 'Cscore_B', 'Curr_HPI', 'Orig_HPI',
                    'HPI_Adjust_Factor', 'MTMLTV', 'Benchmark_Rate', 'Interest_Spread', 'Rem_Months']
categorical_features = ['Channel', 'Loan_Age', 'Purpose', 'Prop', 'Occ_Stat', 'First_Flag']
target = ['Next_Status']
raw_data1=pd.read_csv('ProcessedFile/0-7.csv')
merged_data1=pd.concat([merged_data1,raw_data1])
defected_features=['Mi_Pct','Mi_Type','Cscore_C']
merged_data1=merged_data1.drop(defected_features,axis=1)
clean_data1=merged_data1.dropna()
x_train1, x_test1, y_train1, y_test1 = train_test_split(clean_data1[numeric_features + categorical_features],
                                                    clean_data1[target],
                                                    test_size=0.3, random_state=123)
x_train1 = pd.concat([x_train1.drop(categorical_features, axis=1), pd.get_dummies(x_train1[categorical_features])],
                    axis=1)
x_test1 = pd.concat([x_test1.drop(categorical_features, axis=1), pd.get_dummies(x_test1[categorical_features])],
                   axis=1)
scaler = StandardScaler()
scaler.fit(clean_data1[numeric_features])
x_train1[numeric_features] = scaler.transform(x_train1[numeric_features])
x_test1[numeric_features] = scaler.transform(x_test1[numeric_features])

x_train_real = pd.concat([x_train1,x_train])
y_train_real = pd.concat([y_train1,y_train])
merged_real = pd.concat([x_train_real,y_train_real],axis=1)
merged_real = merged_real.dropna()
y_train_real = merged_real[target]
x_train_real = merged_real.drop(target, axis=1)

print(y_train_real.shape)
print(x_train_real.shape)

def feature_sel(x_train, y_train):
    x_train_svc, x_test_svc, y_train_svc, y_test_svc = train_test_split(x_train, y_train, test_size=0.2,random_state=123)
    svc = SVC()
    forward_fs_best = sfs(estimator=svc, n_features_to_select='auto')
    forward_fs_best.fit(x_train_svc, np.ravel(y_train_svc))
    sfs_indices=forward_fs_best.support_
    x_train_svc = x_train_svc.loc[:, sfs_indices]
    fs_sel_feal = x_train_svc.columns
    return fs_sel_feal

sel_feat = feature_sel(x_train_real, y_train_real)
xexam = x_train[sel_feat]
print(xexam.shape)



