
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from keras.losses import binary_crossentropy, categorical_crossentropy
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# data 2004,2006,2008

def get_train_test_numpy_arrays(df):

    drop_cols = ['Loan_ID', 'Current_Status', 'Next_Status']
    target = 'Target'

    df_numeric = df.select_dtypes(include='number')
    df_numeric = df_numeric.drop(drop_cols, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_numeric.drop(target, axis=1), df_numeric[target], test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train.values)
    X_test = sc.fit_transform(X_test.values)
    y_train = y_train.values
    y_test = y_test.values
    return X_train, y_train, X_test, y_test

def neural_network_create_model():

    # The paper states that we need to take 272 features with 5 hidden layers
    # First hidden layer has about 200 units and rest 4 have 148 units each
    input_features = 36
    # first_hidden_layer_units = 50
    # other_hidden_layer_units = 30



    # Sequence of ANN
    classifier_model = Sequential()

    # initalize layers and stack them sequentially
    classifier_model.add(Dense(units=input_features, kernel_initializer='uniform', activation='relu', input_dim=input_features))

    #Add dropout layer with 20% probability at every layer
    classifier_model.add(Dropout(0.2))

    # Add the input layer and corresponding first hidden layer
    classifier_model.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))

    #Add dropout layer with 20% probability at every layer
    classifier_model.add(Dropout(0.2))

    # Add the second layer and previous layer as input nodes
    classifier_model.add(Dense(units=45, kernel_initializer='uniform', activation='relu'))

    #Add dropout layer with 20% probability at every layer
    classifier_model.add(Dropout(0.2))

    # Add the second layer and previous layer as input nodes
    classifier_model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))

    #Add dropout layer with 20% probability at every layer
    classifier_model.add(Dropout(0.2))

    # Add the second layer and previous layer as input nodes
    classifier_model.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

    #Add dropout layer with 20% probability at every layer
    classifier_model.add(Dropout(0.2))

    # Add the second layer and previous layer as input nodes
    classifier_model.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))

    # Output layer Current, Delinquent, Default and Prepaid and softmax as it's multiclass NN
    # classifier_model.add(Dense(units=4, kernel_initializer='uniform', activation='softmax'))

    classifier_model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    adam = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adam"
    )


    # Compile the model in memory
    classifier_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier_model


def fetch_data(path):
    df = pd.read_csv(path)
    return df


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, batch_size = 128, epochs = 1000)
    classifier = model.to_json()
    with open("SavedModels/model_current_current.json", "w") as json_file:
        json_file.write(classifier)
    model.save_weights("SavedModels/model_current_current.h5")
    print("Saved model to disk")

def load_model_predict(path_json, path_weights, X_test, y_test):
    json_file = open(path_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])
    y_pred = loaded_model.predict(X_test)
    y_pred = np.argmax(y_pred>0.5,axis=1)
    cm = confusion_matrix(y_test, y_pred)
    return cm

def prepare_data_for_current_current():
    df00 = fetch_data('ProcessedFile/0-0.csv')
    df01 = fetch_data('ProcessedFile/0-1.csv')
    df07 = fetch_data('ProcessedFile/0-7.csv')
    df0P = fetch_data('ProcessedFile/0--1.csv')

    df00['Target'] = 1
    df01['Target'] = 0
    df0P['Target'] = 0
    df07['Target'] = 0

    df = df00.append(df01, ignore_index=True)
    df = df.append(df0P, ignore_index=True)
    df = df.append(df07, ignore_index=True)

    return df

if __name__ == '__main__':
    clf_model = neural_network_create_model()
    df = prepare_data_for_current_current()
    X_train, y_train, X_test, y_test = get_train_test_numpy_arrays(df)
    train_model(clf_model, X_train, y_train)
    cm = load_model_predict('SavedModels/model.json', 'SavedModels/model.h5', X_train,y_train)





