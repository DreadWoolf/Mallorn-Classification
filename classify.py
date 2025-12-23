import pandas as pd
import numpy as np
from StackingEnsemble import StackingEnsemble





def main():

    Stackingmodel: StackingEnsemble = StackingEnsemble.load_or_create()

    Stackingmodel.predict()




if __name__ == "__main__":
    pass