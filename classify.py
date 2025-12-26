import pandas as pd
import numpy as np
from StackingEnsemble import StackingEnsemble
import os





main_folder = "mallorn-astronomical-classification-challenge"
data_folder = "Data"
data_path = os.path.join(os.getcwd(), data_folder)

path = os.path.join(os.getcwd(), main_folder)

def main():

    Stackingmodel: StackingEnsemble = StackingEnsemble.load_or_create()

    data_folder = "Data"
    filename = "MALLORN-data_test.csv"
    data_path = os.path.join(os.path.join(os.getcwd(), data_folder), filename)

    
    data = pd.read_csv(data_path, sep=',')

    # sample_sub

    



def predict(Stackingmodel: StackingEnsemble):

    Stackingmodel.predict()



if __name__ == "__main__":
    pass