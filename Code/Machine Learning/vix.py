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

def feature_sel(x_train, y_train):
    rfc = RandomForestClassifier(n_estimators=50, random_state=123)
    x_train_rfc, x_test_rfc, y_train_rfc, y_test_rfc = train_test_split(x_train, y_train,
                                                                        test_size=0.2, random_state=123)
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

def svc_model(int_state,poss_state):
    x_train, x_test, y_train, y_test = prepare_data(int_state, poss_state)
    sel_feat = feature_sel(x_train, y_train)

    kernel_SVC = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    gamma_SVC = ["scale","auto"]
    mean_pred = []
    for i in range(0, len(kernel_SVC)):
        if i == 1:
            for g in range(0,2):
                mean_temp = []
                svc_model = SVC(kernel=kernel_SVC[i], gamma = gamma_SVC[g], C=1, class_weight='balanced', probability=True)
                scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
                mean_temp.append(np.mean(scores))
                if g == 2:
                    mean_pred.append(np.mean(mean_temp))
        elif i == 2:
            for g in range(0,2):
                mean_temp = []
                svc_model = SVC(kernel=kernel_SVC[i], gamma = gamma_SVC[g], C=1, class_weight='balanced', probability=True)
                scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
                mean_temp.append(np.mean(scores))
                if g == 2:
                    mean_pred.append(np.mean(mean_temp))
        elif i == 3:
            for g in range(0,2):
                mean_temp = []
                svc_model = SVC(kernel=kernel_SVC[i], gamma = gamma_SVC[g], C=1, class_weight='balanced', probability=True)
                scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
                mean_temp.append(np.mean(scores))
                if g == 2:
                    mean_pred.append(np.mean(mean_temp))
        svc_model = SVC(kernel=kernel_SVC[i], C=1, class_weight='balanced', probability=True)
        scores = cross_val_score(svc_model, x_train[sel_feat], np.ravel(y_train), cv=5)
        mean_pred.append(np.mean(scores))
    ind_SVC = max(mean_pred)
    index_finding = kernel_SVC.index(ind_SVC)
    svc_model = SVC(kernel=kernel_SVC[index_finding], C=1, class_weight='balanced', probability=True)
    svc_model.fit(x_train[sel_feat],np.ravel(y_train))
    return svc_model

