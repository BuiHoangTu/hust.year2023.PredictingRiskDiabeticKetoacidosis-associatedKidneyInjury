from middle_query import crrt
from reduce_mesurements import reduceByStayId


def get():
    dfCrrt = crrt.runSql()
    dfCrrt = dfCrrt.drop_duplicates("stay_id")
    dfCrrt["use_crrt"] = True

    dfMerged = reduceByStayId(dfCrrt)

    return dfMerged[["stay_id", "use_crrt"]]
