# Import Libraries

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer


# Data Preparation

def data_preparation (filename):
    """This function imports the data from the csv file and performs data preparation
        Args:
            file_name (str): The filename
        Return:
            data (pandas.DataFrame): The pandas Dataframe
    """
    # Load the data
    data = pd.read_csv("credit_data.csv")
    
    # Cleaning the names of the columns
    data.columns = data.columns.str.lower()
    
    # Mapping variables from int to more intuitive attributes
    status_values = {
        1: "ok",
        2: "default",
        0: "unk"
        }
    data.status = data.status.map(status_values)

    home_values = {
        1: "rent",
        2: "owner",
        3: "private",
        4: "ignore",
        5: "parents",
        6: "other",
        0: "unk"
    }
    data.home = data.home.map(home_values)

    marital_values = {
        1: "single",
        2: "married",
        3: "widow",
        4: "separated",
        5: "divorced",
        0: "unk"
    }
    data.marital = data.marital.map(marital_values)

    records_values = {
        1: "no",
        2: "yes",
        0: "unk"
    }
    data.records = data.records.map(records_values)

    job_values = {
        1: "fixed",
        2: "partime",
        3: "freelance",
        4: "others",
        0: "unk"
    } 
    data.job = data.job.map(job_values)

    # Filling missing values
    for c in ["income", "assets", "debt"]:
        data[c] = data[c].replace(to_replace=99999999, value=np.nan)

    # Removing rows which the objective variable is unkown
    data = data[data.status != "unk"].reset_index()

    return data


def split_train_val_test(data):
    """This functions splits the dataset between train, validation and test
        Args:
            data (pandas.DataFrame): list that contains the explanatory variables and objective variable
            
        Return:
            set_used (list): list that contains x_train, x_val, x_test, y_train, y_val, y_test, x_full_train, y_full_train
    """
    # Create x variable with the explanatory variables and y variable with the objective variable
    x = data.loc[:,data.columns!="status"]
    y = (data.status == "default").astype("int")

    # Split dataset into full train (train and validation) - 80% - and test set - 20%.
    x_full_train, x_test, y_full_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Split the full train dataset into train - 75% - and validation - 35%.
    x_train, x_val, y_train, y_val = train_test_split(x_full_train, y_full_train, test_size=0.25, random_state=1)

    #Reset the index
    set_used = [x_train, x_val, x_test, y_train, y_val, y_test, x_full_train, y_full_train]
    for i in set_used:
        i = i.reset_index(drop=True)

    return set_used


def one_hot_enconding(set_used_x, set_used_y):
    """This function performs one hot encoding to categorical variables
    Args:
        set_used_x (list) : list with explanatory variables
        set_used_y (list) : list with objective variable
        
    Return:
        X_train (Numpy Array) : Array that contains the explanatory variables one hot encoded for train dataset
        X_val (Numpy Array) : Array that contains the explanatory variables one hot encoded for validation dataset
        dv (sklearn.feature_extraction._dict_vectorizer.DictVectorizer) : method for one hot encoding
    """
    # For train dataset
    train_dicts = set_used_x.fillna(0).to_dict(orient="records") # filling na values with 0
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    # For validation dataset
    val_dicts = set_used_y.fillna(0).to_dict(orient="records")
    X_val = dv.transform(val_dicts) # Validation datased is not fitted because it was already fitted for train dataset

    return X_train, X_val, dv


def xgb_model (X_train, Y_train, features, X_val, Y_val):
    """This function trains this XGBoost model and get the predictions

        Args:
            X_train (Numpy Array) : Array that contains the explanatory variables one hot encoded for train dataset
            Y_train (list) : list that contains the objective variable for train dataset
            features (Numpy Array) : Array that contains the name of the features used
            X_val (Numpy Array) : Array that contains the explanatory variables one hot encoded for validation dataset
            Y_val (list) : list that contains the objective variable for validation dataset


        Returns:
            model ('xgboost.core.Booster') : XGBoost model trained
            y_pred (Numpy Array) : array that contains the predictions obtained from the model
            auc (integer) : integer that corresponds to the area under the curve value of the model
    """

    dtrain = xgb.DMatrix(X_train, label=Y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=Y_val, feature_names=features)

    xgb_params = {
        'eta': 0.3,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'nthread': 8, 
        'seed': 1,
        'verbosity': 1, 
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=200)

    # Predict (Validation Dataset)
    y_pred = model.predict(dval) 

    # AUC
    auc = roc_auc_score(Y_val, y_pred)

    return model, y_pred, auc


def parse_arguments():
    """This function parses the argument(s) of this model
        Args:
            file_name: name of the command line field to insert on the runtime

        Return:
            args: Stores the extracted data from the parser run
    """
    parser = argparse.ArgumentParser(description="Process all the arguments for this model")
    parser.add_argument("file_name", help="The csv file name")
    args = parser.parse_args()

    return args


def main():
    """This is the main function of this Random Forest model"""

    args = parse_arguments()

    data = data_preparation(args.file_name)

    set_used = split_train_val_test(data)

    X_train, X_val, dv = one_hot_enconding(set_used[0], set_used[1])

    # Learning in Train data and Evaluates in Val Data and Depth Parameter Tunning
    features = dv.get_feature_names_out()
    model, y_pred, auc =  xgb_model(X_train, set_used[3], features, X_val, set_used[4])

    # AUC
    print ("auc train:", auc)

    
if __name__ == "__main__":
    main()