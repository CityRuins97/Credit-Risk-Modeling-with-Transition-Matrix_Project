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


def prepare_data(initial_state,possible_state,file_path='ProcessedFile/'):
    merged_data=pd.DataFrame()
    numeric_features = ['Orig_Rate','Orig_UPB','Curr_UPB', 'Orig_Term','OLTV', 'DTI', 'Cscore_B', 'Curr_HPI','HPI_Adjust_Factor',
                        'MTMLTV', 'Benchmark_Rate', 'Age_Prop', 'Refinance Indicator','SATO','URate_Change','10-2Spread','Mi_Pct']
    categorical_features = ['Channel', 'Purpose', 'Prop',  'Occ_Stat', 'First_Flag','Num_Bo','Interest_Only_Loan_Indicator',
                            'Mod_Flag','Mi_Type','HomeReady_Program_Indicator','Relocation_Mortgage_Indicator','Judicial']
    target = ['Next_Status']
    for i in range(0,len(possible_state)):
        raw_data=pd.read_csv(file_path+str(initial_state)+'-'+str(possible_state[i])
                             +'.csv')
        merged_data=pd.concat([merged_data,raw_data])
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
    print('Data Preparation is Complete')
    return x_train,x_test,y_train,y_test


# def feature_sel(x_train,y_train):
#     rfc=RandomForestClassifier(n_estimators=50,random_state=100)
#     x_train_rfc,x_test_rfc,y_train_rfc,y_test_rfc=train_test_split(x_train,y_train,
#                                                                    test_size=0.2,random_state=123)
#     while True:
#         rfc.fit(x_train_rfc, np.ravel(y_train_rfc))
#         crit = min(0.1, 1/ x_train_rfc.shape[1])
#         sel_feat = rfc.feature_importances_ > crit
#         if sel_feat.all():
#             feat_importance=rfc.feature_importances_
#             break
#         else:
#             x_train_rfc = x_train_rfc.loc[:, sel_feat]
#     sel_features=x_train_rfc.columns
#     return sel_features


# def feature_sel(x_train, y_train):
#     x_train_svc, x_test_svc, y_train_svc, y_test_svc = train_test_split(x_train, y_train,
#                                                                         test_size=0.2, random_state=123)
def feature_sel(x_train, y_train):
    x_train_svc, x_test_svc, y_train_svc, y_test_svc = train_test_split(x_train, y_train, test_size=0.2,random_state=123)
    svc = SVC()
    forward_fs_best = sfs(estimator=svc, n_features_to_select='auto')
    forward_fs_best.fit(x_train_svc, np.ravel(y_train_svc))
    sfs_indices=forward_fs_best.support_
    x_train_svc = x_train_svc.loc[:, sfs_indices]
    fs_sel_feal = x_train_svc.columns
    return fs_sel_feal

def svc_model(int_state,poss_state):
    x_train, x_test, y_train, y_test = prepare_data(int_state, poss_state)
    sel_feat = feature_sel(x_train, y_train)
    svc_model=SVC(C=1,class_weight='balanced',probability=True)
    svc_model.fit(x_train[sel_feat],np.ravel(y_train))
    return svc_model

def logistic_model(x_train,y_train,sel_feat):
    log_model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10000,C=1)
    log_model.fit(x_train[sel_feat], np.ravel(y_train))
    print('Logistic Model is Complete!')
    return log_model

# def svc_model(int_state,poss_state):
#     x_train, x_test, y_train, y_test = prepare_data(int_state, poss_state)
#     sel_feat = feature_sel(x_train, y_train)
#
#     kernel_SVC = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
#     gamma_SVC = ["scale","auto"]
#     mean_pred = []
#     for i in range(0, len(kernel_SVC)):
#         if i == 1:
#             for g in range(0,2):
#                 mean_temp = []
#                 svc_model = SVC(kernel=kernel_SVC[i], gamma = gamma_SVC[g], C=1, class_weight='balanced', probability=True)
#                 scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
#                 mean_temp.append(np.mean(scores))
#                 if g == 2:
#                     mean_pred.append(np.mean(mean_temp))
#         elif i == 2:
#             for g in range(0,2):
#                 mean_temp = []
#                 svc_model = SVC(kernel=kernel_SVC[i], gamma = gamma_SVC[g], C=1, class_weight='balanced', probability=True)
#                 scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
#                 mean_temp.append(np.mean(scores))
#                 if g == 2:
#                     mean_pred.append(np.mean(mean_temp))
#         elif i == 3:
#             for g in range(0,2):
#                 mean_temp = []
#                 svc_model = SVC(kernel=kernel_SVC[i], gamma = gamma_SVC[g], C=1, class_weight='balanced', probability=True)
#                 scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
#                 mean_temp.append(np.mean(scores))
#                 if g == 2:
#                     mean_pred.append(np.mean(mean_temp))
#         svc_model = SVC(kernel=kernel_SVC[i], C=1, class_weight='balanced', probability=True)
#         scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
#         mean_pred.append(np.mean(scores))
#     ind_SVC = max(mean_pred)
#     index_finding = kernel_SVC.index(ind_SVC)
#     svc_model = SVC(kernel=kernel_SVC[index_finding], C=1, class_weight='balanced', probability=True)
#     svc_model.fit(x_train[sel_feat],np.ravel(y_train))
#     return svc_model



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
    visualizer = ROCAUC(LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10000,C=1),
                        micro=False)
    visualizer.fit(x_train[sel_features], np.ravel(y_train))
    visualizer.score(x_test[sel_features], y_test)
    print('AUROC Plot is Complete!')
    return visualizer


def logistic_assess(int_state, poss_state):
    # Prepare Data
    x_train,x_test,y_train,y_test=prepare_data(int_state,poss_state)

    # Feature Selection
    sel_features = feature_sel(x_train, y_train)

    # Model Building
    svc_model1=svc_model(int_state,poss_state)
    # Model Evaluation
    y_pre_prob = svc_model1.predict_proba(x_test[sel_features])
    auroc = roc_auc_score(y_test, y_pre_prob, multi_class='ovr')
    visualizer = ROCAUC(SVC(C=1,probability=True,class_weight='balanced'))
    visualizer.fit(x_train[sel_features], np.ravel(y_train))
    visualizer.score(x_test[sel_features], y_test)
    calibration_plot(y_test, y_pre_prob, poss_state)
    plt.legend()
    plt.grid(False)
    plt.show()
    print('The AUC score of the Logistic Model for the initial state ' + str(int_state) + ' is ' + str(auroc))



def main():
    int_state=1
    poss_state = [-1, 0, 1,2]
    logistic_assess(int_state,poss_state)



if __name__ == '__main__':
    main()
