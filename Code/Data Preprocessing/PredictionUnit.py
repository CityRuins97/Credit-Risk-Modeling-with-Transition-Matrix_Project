import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import Preprocess
from Utility import train_prepare
from catboost import Pool, CatBoostClassifier, CatBoost
from catboost.utils import get_roc_curve
import matplotlib.pyplot as plt
from Preprocess import writeCurrentDataset, emptyCurrentFolder


def predict(initial_state):
    x_train, x_test, y_train, y_test, cat_features, full_data = train_prepare(initial_state,
                                                                   returncat=True, returnFull=True)
    eval_data = x_test.values.tolist()
    possible_state = [-1] + [*set([i for i in range(0, initial_state + 2)] + [7])]
    prob_list = []
    for possible in possible_state:
        eval_label = y_test.astype('str').tolist()
        eval_label = [1 if float(label) == float(possible) else 0 for label in eval_label]
        if 1 in eval_label:
            eval_dataset = Pool(data=eval_data, label=eval_label, cat_features=cat_features)
            model = CatBoostClassifier()
            model.load_model("Models/model" + str(initial_state) + "-" + str(possible))
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
            # plt.savefig(fname=("Output/" + str(initial_state) + "_" + str(possible) + ".png"))
            plt.clf()
        else:
            print("initial: " + str(initial_state) + ", next: " + str(possible) + " failed!")
            continue
    final_prob = pd.concat(prob_list, axis=1)
    final_prob["row_sum"] = final_prob.sum(axis=1)
    for column in final_prob.columns:
        if column != "row_sum":
            final_prob[column] = final_prob[column] / final_prob["row_sum"]
    final_prob.drop(columns=["row_sum"], inplace=True)
    final_prob["Truth"] = y_test.astype('str').tolist()

    target = full_data["Next_Status"]
    x_train, x_test_full, y_train, y_test = train_test_split(full_data, target, test_size=0.2, random_state=123)
    final_df = pd.concat([x_test_full.reset_index(), final_prob.reset_index()], axis=1)
    final_df.drop(columns=["index"], inplace=True)
    test_df = pd.concat([x_train.reset_index(), y_train.reset_index()], axis=1)
    test_df.drop(columns=["index"], inplace=True)
    writeCurrentDataset(test_df, "Visualization/train_" + str(initial_state) + ".csv", frame=True, append=False)
    print(test_df.shape)
    print(final_df.shape)
    print(full_data.shape)
    return final_df


if __name__ == "__main__":
    initial_states = [0, 1, 2, 3, 4, 5, 6]
    for state in initial_states:
        print("Current States: ", str(state))
        final_df = predict(state)
        writeCurrentDataset(final_df, "Visualization/Pred" + str(state) + ".csv", frame=True, append=False)
