from notebook_wrappers import target_patients_wrapper
from variables.charateristics_diabetes import macroangiopathy
from variables.charateristics_diabetes import microangiopathy

def getDiabeteType():
    """Get patient diabete type: 1, 2, 0 - Others

    Returns:
        pandas.DataFrame: ["stay_id", "dka_type"]
    """
    df = target_patients_wrapper.getNotebookOutput()
    return df[["stay_id", "dka_type"]]

def getMacroangiopathy():
    return macroangiopathy.get()


def getMicroangiopathy():
    return microangiopathy.get()
