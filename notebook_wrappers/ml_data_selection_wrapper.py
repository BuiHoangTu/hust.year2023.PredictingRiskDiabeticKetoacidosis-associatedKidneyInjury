import pandas as pd
from constants import TEMP_PATH

from notebook_wrapper import NotebookWrapper


def getNotebookOutput():
    LEARNING_DATA_FILE = TEMP_PATH / "learning_data.csv"
    
    if LEARNING_DATA_FILE.exists():
        return pd.read_csv(LEARNING_DATA_FILE)
    else:
        dfData: pd.DataFrame = NotebookWrapper("ml_data_selection.ipynb", None, "dfData").run()  # type: ignore
        dfData.to_csv(LEARNING_DATA_FILE)
        return dfData
