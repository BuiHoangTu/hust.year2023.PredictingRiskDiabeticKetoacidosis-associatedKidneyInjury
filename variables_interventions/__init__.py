from variables_interventions import mechanical_ventilation as mv
from variables_interventions import use_crrt as crrt
from variables_interventions import use_nahco3 as nahco3


def extractInterventions():
    """Deprecated

    Returns:
        pandas.DataFrame: consists of mv, crrt, nahco3
    """
    df1 = mv.get()
    df2 = crrt.get()
    df3 = nahco3.get()

    return df1.merge(df2, "outer").merge(df3, "outer")


def getMV():
    """mechanical ventilation

    Returns:
        pandas.DataFrame: ["stay_id", "mechanical_ventilation"]
    """

    return mv.get()


def getCrrt():
    """continuous renal replacement therapy

    Returns:
        pandas.DataFrame: ["stay_id", "use_crrt"]
    """

    return crrt.get()


def getNaHCO3():
    """use of NaHCO3

    Returns:
        pandas.DataFrame: ["stay_id", "use_NaHCO3"]
    """
    return nahco3.get()
