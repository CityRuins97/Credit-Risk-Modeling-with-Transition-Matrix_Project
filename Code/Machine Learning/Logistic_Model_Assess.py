import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from yellowbrick.classifier import ROCAUC
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
from sklearn.feature_selection import SequentialFeatureSelector as sfs

def prepare_data(initial_state,file_path='ProcessedFile/'):
    possible_state = [-1] + [*set([i for i in range(0, initial_state + 2)] + [7])]
    print(possible_state)
    merged_data = pd.DataFrame()
    numeric_features = ['Orig_Rate','Orig_UPB','Curr_UPB', 'Orig_Term','OLTV', 'DTI', 'Cscore_B', 'Curr_HPI','HPI_Adjust_Factor',
                        'MTMLTV', 'Benchmark_Rate', 'Age_Prop', 'Refinance Indicator','SATO','URate_Change','10-2Spread','Mi_Pct']
    categorical_features = ['Channel', 'Purpose', 'Prop',  'Occ_Stat', 'First_Flag','Num_Bo','Interest_Only_Loan_Indicator',
                            'Mod_Flag','Mi_Type','HomeReady_Program_Indicator','Relocation_Mortgage_Indicator','Judicial']
    target = ['Next_Status']
    for i in range(0, len(possible_state)):
        raw_data = pd.read_csv(file_path + str(initial_state) + '-' + str(possible_state[i])
                               + '.csv')

    clean_data = merged_data.dropna()
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
    print('Data Preparation is Complete!')
    return x_train, x_test, y_train, y_test


def feature_sel(x_train, y_train):
    rfc = RandomForestClassifier(n_estimators=50, random_state=123)
    print("1")
    x_train_rfc, x_test_rfc, y_train_rfc, y_test_rfc = train_test_split(x_train, y_train,
                                                                        test_size=0.2, random_state=123)
    print("2")
    while True:
        rfc.fit(x_train_rfc, np.ravel(y_train_rfc))
        crit = min(0.1, 0.1/ x_train_rfc.shape[1])
        sel_feat = rfc.feature_importances_ > crit
        if sel_feat.all():
            #feat_importance = rfc.feature_importances_
            break
        else:
            x_train_rfc = x_train_rfc.loc[:, sel_feat]
    sel_features = x_train_rfc.columns
    print('Feature Selection is Complete!')
    return sel_features



# def feature_sel(x_train, y_train):
#     x_train_svc, x_test_svc, y_train_svc, y_test_svc = train_test_split(x_train, y_train, test_size=0.2,random_state=123)
#     print("1")
#     forward_fs_best = sfs(estimator=LogisticRegression(), n_features_to_select='auto',direction='forward')
#     print("1")
#     forward_fs_best.fit(x_train_svc, np.ravel(y_train_svc))
#     print("1")
#     sfs_indices=forward_fs_best.support_
#     print("1")
#     x_train_svc = x_train_svc.loc[:, sfs_indices]
#     print("1")
#     fs_sel_feal = x_train_svc.columns
#     print("1")
#     print(fs_sel_feal)
#     print("1")
#     print('Feature Selection is Complete!')
#     print("1")
#     return fs_sel_feal


def logistic_model(x_train,y_train,sel_feat):
    log_model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10000,C=1)
    log_model.fit(x_train[sel_feat], np.ravel(y_train))
    print('Logistic Model is Complete!')
    return log_model


def calibration_multi(y_test, y_pred_prob, multi_class):
    prob_dict = defaultdict()
    for i in range(0, len(multi_class)):
        binary_y = list(y_test.Next_Status == multi_class[i])
        y_pred_prob_i = [row[i] for row in y_pred_prob]
        prob_true, prob_pred = calibration_curve(binary_y, y_pred_prob_i, pos_label=1, n_bins=5,
                                                 strategy='quantile')
        prob_dict['true' + str(multi_class[i])] = prob_true
        prob_dict['pred' + str(multi_class[i])] = prob_pred
    return prob_dict


def calibration_plot(y_test, y_pre_prob, poss_state):
    calibration_dict = calibration_multi(y_test, y_pre_prob, poss_state)
    for i in range(0, len(poss_state)):
        plt.scatter(calibration_dict['pred' + str(poss_state[i])],
                    calibration_dict['true' + str(poss_state[i])],
                    label='calibration point of state ' + str(poss_state[i]))
    plt.legend()
    plt.grid(False)
    print('Calibration Plot is Complete!')


def roc_plot(x_train,y_train,x_test,y_test,sel_features):
    visualizer = ROCAUC(LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10000,C=1), micro=False, per_class = False)
    visualizer.fit(x_train[sel_features], np.ravel(y_train))
    visualizer.score(x_test[sel_features], y_test)
    print('AUROC Plot is Complete!')
    return visualizer

def prepdata(initial_state, file_path = '/Users/xudongchen/Desktop/NCSU/Financial Math/2022Fall/Project/FIM601/data/drive-download-20230412T172454Z-001的副本/'):
    numeric_features = ['Orig_Rate','Orig_UPB','Curr_UPB', 'Orig_Term','OLTV', 'DTI', 'Cscore_B', 'Curr_HPI','HPI_Adjust_Factor',
                        'MTMLTV', 'Benchmark_Rate', 'Age_Prop', 'Refinance Indicator','SATO','URate_Change','10-2Spread','Mi_Pct']
    categorical_features = ['Channel', 'Purpose', 'Prop',  'Occ_Stat', 'First_Flag','Num_Bo','Interest_Only_Loan_Indicator',
                            'Mi_Type','HomeReady_Program_Indicator','Relocation_Mortgage_Indicator','Judicial']
    target = ['Next_Status']
    x_train = pd.read_csv(file_path + 'train' + '_' + str(initial_state) + '.csv')
    x_train = x_train[numeric_features + categorical_features]
    x_train = pd.concat([x_train.drop(categorical_features, axis=1), pd.get_dummies(x_train[categorical_features])],
                        axis=1)
    print(x_train.head())
    x_test = pd.read_csv(file_path + 'Pred' + str(initial_state)  + '.csv')
    x_test = x_test[numeric_features + categorical_features]
    x_test = pd.concat([x_test.drop(categorical_features, axis=1), pd.get_dummies(x_test[categorical_features])],
                        axis=1)
    print(x_test.head())
    y_train = pd.read_csv(file_path + 'train' + '_' + str(initial_state)  + '.csv')
    y_train = y_train[target]
    print(y_train.head())
    y_test = pd.read_csv(file_path + 'Pred' + str(initial_state)  + '.csv')
    y_test = y_train[target]
    print(y_test.head())
    print('Data Preparation 2 is Complete!')
    return x_train, x_test, y_train, y_test

def logistic_assess(int_state):

    # Prepare Data
    x_train, x_test, y_train, y_test = prepdata(int_state)
    print('Step 1 is Complete!')
    # Feature Selection
    sel_features = feature_sel(x_train, y_train)
    print('Step 2 is Complete!')
    # Model Building
    log_model = logistic_model(x_train,y_train,sel_features)
    print('Step 3 is Complete!')
    # Model Evaluation
    y_pre_prob = log_model.predict_proba(x_test[sel_features])
    y_pre_prob = pd.DataFrame(y_pre_prob,columns=['-1' , '0', '1' ])
    y_pre_prob.to_csv('log_result_prob.csv')
    y_pred = log_model.predict(x_test[sel_features])
    y_pred = pd.DataFrame(y_pred, columns=['Truth'])
    y_pred.to_csv('log_result_turth.csv')




def main():
    tic_0 = time.perf_counter()

    # Initial Status of Interest
    int_state = 0
    logistic_assess(int_state)

    toc_0 = time.perf_counter()
    print("Total Running Time: ", str(toc_0-tic_0), " s.")


if __name__ == '__main__':
    # All errors raised can be fixed by increasing the data size
    main()