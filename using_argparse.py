# import libraries
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(args):
    # read data
    df = get_data(args.training_data)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)
    
    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

    # run main function
    main(args)


# run the script
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py --training_data diabetes.csv",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
    )

# in the terminal
python train.py --training_data diabetes.csv
