# Import Libraries

import pandas as pd
import numpy as np
import argparse


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
    
    # Mapping target variable from 0, 1 and 2 to more intuitive attributes
    status_values = {
        1: "ok",
        2: "default",
        0: "unk"
        }
    data.status = data.status.map(status_values)

    home_values = {
        1: 'rent',
        2: 'owner',
        3: 'private',
        4: 'ignore',
        5: 'parents',
        6: 'other',
        0: 'unk'
    }
    data.home = data.home.map(home_values)

    marital_values = {
        1: 'single',
        2: 'married',
        3: 'widow',
        4: 'separated',
        5: 'divorced',
        0: 'unk'
    }
    data.marital = data.marital.map(marital_values)

    records_values = {
        1: 'no',
        2: 'yes',
        0: 'unk'
    }
    data.records = data.records.map(records_values)

    job_values = {
        1: 'fixed',
        2: 'partime',
        3: 'freelance',
        4: 'others',
        0: 'unk'
    } 
    data.job = data.job.map(job_values)

    # Filling missing values
    
    return data


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
    """This is the main function of this Linear Model Regression Implementation model"""

    args = parse_arguments()

    data = data_preparation(args.file_name)



    if __name__ == '__main__':
        main()