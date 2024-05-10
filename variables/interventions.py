from utils.extract_mesurements import extractInputEvents
from mimic_sql import crrt, ventilation
from utils.reduce_mesurements import reduceByStayId


def getMV():
    """mechanical ventilation

    Returns:
        pandas.DataFrame: ["stay_id", "mechanical_ventilation"]
    """

    dfMV = ventilation.extractVentilation()
    dfMV["mechanical_ventilation"] = dfMV["ventilation_status"].isin(
        [
            "Tracheostomy",
            "InvasiveVent",
        ]
    )
    dfMerged = reduceByStayId(dfMV, "starttime", "endtime")

    dfMerged = dfMerged[["stay_id", "mechanical_ventilation", "starttime"]]
    dfMerged = dfMerged[dfMerged["mechanical_ventilation"]]

    return dfMerged.rename(columns={"starttime": "time"})


def getCrrt():
    """continuous renal replacement therapy

    Returns:
        pandas.DataFrame: ["stay_id", "use_crrt"]
    """

    dfCrrt = crrt.runSql()

    dfMerged = reduceByStayId(dfCrrt)

    return dfMerged[["stay_id", "use_crrt", "charttime"]].rename({"charttime": "time"})


def getNaHCO3():
    """use of NaHCO3

    Returns:
        pandas.DataFrame: ["stay_id", "use_NaHCO3"]
    """

    dfNahco3 = extractInputEvents([220995, 221211, 227533], "use_nahco3.csv")

    dfReduced = reduceByStayId(dfNahco3, "starttime", "endtime")

    dfReduced["use_NaHCO3"] = True

    return dfReduced[["stay_id", "use_NaHCO3", "starttime"]].rename(columns={"starttime": "time"})
