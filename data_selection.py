from time import sleep
import nbformat
import pandas as pd
from constants import LEARNING_DATA_FILE
from nbconvert.preprocessors import ExecutePreprocessor


def getNotebookOutput():
    if not LEARNING_DATA_FILE.exists():
        nb = nbformat.read("./data_selection.ipynb", as_version=4)
        ep = ExecutePreprocessor(timeout=None, kernel_name="python3")

        resultNb, _ = ep.preprocess(nb)
        pass

    # wait for maximun 5*2 seconds
    for _ in range(5):
        if LEARNING_DATA_FILE.exists():
            break
        else:
            sleep(2)
    else:
        raise IOError(LEARNING_DATA_FILE.__str__() + " took too much time to write.")

    return pd.read_csv(LEARNING_DATA_FILE)