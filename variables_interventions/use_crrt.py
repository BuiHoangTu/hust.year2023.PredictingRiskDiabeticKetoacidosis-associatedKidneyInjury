from middle_query.crrt import extractCrrt
from reduce_mesurements import reduceByStayId


def extractUseCrrt():
    dfCrrt = extractCrrt()
    dfCrrt = dfCrrt.drop_duplicates("stay_id")
    dfCrrt["use_crrt"] = True

    dfMerged = reduceByStayId(dfCrrt)

    return dfMerged[["stay_id", "use_crrt"]]
