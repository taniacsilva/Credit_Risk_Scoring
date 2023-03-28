# Import Libraries

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import contextlib
import io

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


def xgb_model (X_train, Y_train, features, X_val, Y_val, eta, md, ch):
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
            output_string () : 
    """

    dtrain = xgb.DMatrix(X_train, label=Y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=Y_val, feature_names=features)

    watchlist = [(dtrain, 'train'), (dval, 'dval')]

    scores={}
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        xgb_params = {
            'eta': eta,
            'max_depth': md,
            'min_child_weight': ch,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 8, 
            'seed': 1,
            'verbosity': 1, 
        }
        model = xgb.train(xgb_params, dtrain, evals=watchlist, verbose_eval=5, num_boost_round=200)

    # Predict (Validation Dataset)
    y_pred = model.predict(dval) 

    # AUC
    auc = roc_auc_score(Y_val, y_pred)

    # Parameters Tuning
    key_eta = 'eta=%s' % (xgb_params['eta'])
    key_md = 'md=%s' % (xgb_params['max_depth'])
    key_ch = 'ch=%s' % (xgb_params['min_child_weight'])

    output_string = captured_output.getvalue()
    
    return model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch


def parse_xgb_output(output_string):
    """This function parses the output_string obtained.
    
        Args:
            output_string () : 
    """
    results = []
    tree = []
    aucs_train = []
    aucs_val = []
    for line in output_string.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')
    
        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it,train, val))

    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    
    return df_results
    

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
    
    eta = 0.3
    md = 6
    ch = 1
    model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch =  xgb_model(X_train, set_used[3], features, X_val, set_used[4], eta, md, ch)

    # Obtain the output of our XGBoost model
    df_score = parse_xgb_output(output_string)
    
    print (df_score)
    

    # Plot the output of XGBoost, namely auc for training and validation dataset
    plt.plot(df_score.num_iter, df_score.train_auc, label='train')
    plt.plot(df_score.num_iter, df_score.val_auc, label='val')
    plt.legend()
    plt.show()

    # AUC
    print ("auc train:", auc)
    

    # Parameters Tuning

    # ETA - learning rate
    scores_save = {}
    etas = [0.3, 0.1, 1]
    for eta in etas:
        model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch =  xgb_model(X_train, set_used[3], features, X_val, set_used[4], eta, md, ch)
        scores[key_eta] = parse_xgb_output(output_string)
        scores_save.update(scores)

    etas_strings = ['eta=0.3', 'eta=0.1','eta=1']
    for eta in etas_strings:
        df_score=scores_save[eta]
        plt.plot(df_score.num_iter, df_score.val_auc, label=eta)
    plt.legend()
    plt.show()

    
    # Max_Depth

    scores_save = {}
    eta = 0.1 # selected eta
    mds = [3, 4, 6, 10]

    for md in mds:
        model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch =  xgb_model(X_train, set_used[3], features, X_val, set_used[4], eta, md, ch)
        scores[key_md] = parse_xgb_output(output_string)
        scores_save.update(scores)

    md_strings = ['md=3', 'md=4','md=6']#, 'md=10']
    for md in md_strings:
        df_score=scores_save[md]
        plt.plot(df_score.num_iter, df_score.val_auc, label=md)
    plt.legend()
    plt.ylim(0.8, 0.84)
    plt.show()


    # Min Child Weight

    scores_save = {}
    eta = 0.1
    md = 3 # selected max depth
    chs = [1, 10, 30]

    for ch in chs:
        model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch =  xgb_model(X_train, set_used[3], features, X_val, set_used[4], eta, md, ch)
        scores[key_ch] = parse_xgb_output(output_string)
        scores_save.update(scores)

    ch_strings = ['ch=1', 'ch=10','ch=30']
    for ch in ch_strings:
        df_score=scores_save[ch]
        plt.plot(df_score.num_iter, df_score.val_auc, label=ch)
    plt.legend()
    plt.show()

    # Select the best model
    ch = 6
    model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch =  xgb_model(X_train, set_used[3], features, X_val, set_used[4], eta, md, ch)
    scores[key_ch] = parse_xgb_output(output_string)
    print ("auc train - after parameter tuning:", auc)

    # Train the model using full train dataset and evaluate the model (using test dataset)
    X_full_train, X_test, dv = one_hot_enconding(set_used[6], set_used[2])
    model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch =  xgb_model(X_full_train, set_used[7], features, X_test, set_used[5], eta, md, ch)
    print ("auc test", auc)


if __name__ == "__main__":
    main()