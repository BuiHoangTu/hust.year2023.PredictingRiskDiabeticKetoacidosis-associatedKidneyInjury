import patients
from variables_charateristics_diabetes import macroangiopathy, microangiopathy

def getDiabeteType():
    """Get patient diabete type: 1, 2, 0 - Others

    Returns:
        pandas.DataFrame: ["stay_id", "dka_type"]
    """
    df = patients.getNotebookOutput()
    return df[["stay_id", "dka_type"]]

def getMacroangiopathy():
    return macroangiopathy.get()


def getMicroangiopathy():
    return microangiopathy.get()
