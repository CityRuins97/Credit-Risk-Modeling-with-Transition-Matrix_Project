
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
    numeric_features = ['Orig_Rate', 'Curr_Rate', 'Curr_UPB', 'OLTV', 'DTI', 'Cscore_B', 'Curr_HPI', 'Orig_HPI',
                        'HPI_Adjust_Factor', 'MTMLTV', 'Benchmark_Rate', 'Interest_Spread', 'Rem_Months']
    categorical_features = ['Channel', 'Loan_Age', 'Purpose', 'Prop', 'State', 'Occ_Stat', 'First_Flag']
    target = ['Next_Status']
    for i in range(0,len(possible_state)):
        raw_data=pd.read_csv(file_path+str(initial_state)+'-'+str(possible_state[i])
                             +'.csv')
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
    return x_train,x_test,y_train,y_test


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



def calibration_multi(y_test,y_pred_prob,multi_class):
    prob_dict=defaultdict()
    for i in range(0,len(multi_class)):
        binary_y=list(y_test.Next_Status==multi_class[i])
        y_pred_prob_i=[row[i] for row in y_pred_prob]
        prob_true,prob_pred=calibration_curve(binary_y,y_pred_prob_i,pos_label=1,n_bins=5)
        prob_dict['true'+str(multi_class[i])]=prob_true
        prob_dict['pred'+str(multi_class[i])]=prob_pred
    return prob_dict


def calibration_plot(y_test, y_pre_prob, poss_state):
    calibration_dict=calibration_multi(y_test, y_pre_prob, poss_state)
    for i in range(0,len(poss_state)):
        plt.scatter(calibration_dict['pred'+str(poss_state[i])],
                    calibration_dict['true'+str(poss_state[i])],label='calibration of status'+str(poss_state[i]))


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
    visualizer = ROCAUC(SVC(C=1,class_weight='balanced',probability=True))
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