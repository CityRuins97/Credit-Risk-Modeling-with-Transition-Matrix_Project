import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Utility import train_prepare
from catboost import Pool, CatBoostClassifier, CatBoost
from catboost.utils import get_roc_curve
import matplotlib.pyplot as plt
from Preprocess import writeCurrentDataset, emptyCurrentFolder


def catBoost_tuning(state):
    x_train, x_test, y_train, y_test, cat_features = train_prepare(state, returncat=True)
    train_data = x_train.values.tolist()
    train_label = y_train.astype('str').tolist()
    train_dataset = Pool(data=train_data, label=train_label, cat_features=cat_features)
    model = CatBoost()
    grid = {'learning_rate': [0.3, 0.5, 0.8, 1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 10]}
    grid_search_result = model.grid_search(grid, X=train_dataset, plot=False)
    print(grid_search_result["params"])
    print(pd.DataFrame(grid_search_result["cv_results"]))


# catBoost classifier
def catBoost_assess(initial_state):
    x_train, x_test, y_train, y_test, cat_features = train_prepare(initial_state, returncat=True)
    train_data = x_train.values.tolist()
    eval_data = x_test.values.tolist()

    possible_state = [-1] + [*set([i for i in range(0, initial_state + 2)] + [7])]
    score_list = []
    fi_list = []
    prob_list = []
    for possible in possible_state:
        train_label = y_train.astype('str').tolist()
        train_label = [1 if float(label) == float(possible) else 0 for label in train_label]
        eval_label = y_test.astype('str').tolist()
        eval_label = [1 if float(label) == float(possible) else 0 for label in eval_label]
        if 1 in train_label and 1 in eval_label:
            train_dataset = Pool(data=train_data, label=train_label, cat_features=cat_features)
            eval_dataset = Pool(data=eval_data, label=eval_label, cat_features=cat_features)
            model = CatBoostClassifier(iterations=30, learning_rate=0.3, depth=3, l2_leaf_reg=10, loss_function='Logloss',
                                       custom_metric=['Precision', 'Recall', 'AUC:hints=skip_train~false', 'PRAUC', 'BrierScore'])

            # model.fit(train_dataset)
            model.fit(X=train_dataset, eval_set=eval_dataset, verbose=False)
            model.save_model(fname="Models/model" + str(initial_state) + "-" + str(possible))
            prob_df = pd.DataFrame(model.predict_proba(eval_dataset))
            prob_df.rename(columns={0: "base", 1: str(possible)}, inplace=True)
            prob_df.drop(columns=["base"], inplace=True)
            prob_list.append(prob_df)
            (fpr, tpr, thresholds) = get_roc_curve(model, eval_dataset, plot=False)
            plt.plot(fpr, tpr, 'r-', linewidth=1)
            plt.plot(fpr, fpr, 'b--', linewidth=1)
            plt.title("ROC: From " + str(initial_state) + " to " + str(possible))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.grid(visible=True, which='both')
            plt.savefig(fname=("Output/"+str(initial_state) + "_" + str(possible) + ".png"))
            plt.clf()
            score_list.append(pd.DataFrame(model.get_best_score()))
            new_df = pd.DataFrame(zip(x_train.columns, model.get_feature_importance()))
            new_df.rename(columns={0: 'feature', 1: 'importance'}, inplace=True)
            new_df.set_index('feature', inplace=True)
            fi_list.append(new_df)
        else:
            print("initial: " + str(initial_state) + ", next: " + str(possible) + " failed!")
            continue

    final_score = pd.concat(score_list, axis=1)
    final_fi = pd.concat(fi_list, axis=1)
    final_prob = pd.concat(prob_list, axis=1)
    final_prob["row_sum"] = final_prob.sum(axis=1)
    for column in final_prob.columns:
        if column != "row_sum":
            final_prob[column] = final_prob[column]/final_prob["row_sum"]
    final_prob.drop(columns=["row_sum"], inplace=True)
    return final_score, final_fi, final_prob, x_test, eval_label


if __name__ == '__main__':
    # All errors raised can be fixed by increasing the data size
    initial_states = [0, 1, 2, 3, 4, 5, 6]
    emptyCurrentFolder("Output/")
    for state in initial_states:
        print("Current States: ", str(state))
        score, fi, preds_proba, eval_data, eval_label = catBoost_assess(state)
        writeCurrentDataset(score, "Output/Metrics for initial state " + str(state) + ".csv", True, False)
        writeCurrentDataset(fi, "Output/Feature Importance for initial state " + str(state) + ".csv", True, False)
        print(eval_data)
        print(pd.DataFrame(preds_proba))
