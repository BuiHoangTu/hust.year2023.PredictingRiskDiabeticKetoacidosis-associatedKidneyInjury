from reduce_mesurements import reduceByCharttime
from middle_query.ventilation import (
    extractVentilation,
)


def extractMechVent():
    dfMV = extractVentilation()
    dfMV["mechanical_ventilation"] = dfMV["ventilation_status"].isin(
        [
            "Tracheostomy",
            "InvasiveVent",
        ]
    )
    dfMerged = reduceByCharttime(dfMV, "starttime", "endtime")

    dfMerged = dfMerged[["stay_id", "mechanical_ventilation"]]
    dfMerged = dfMerged[dfMerged["mechanical_ventilation"]]

    return dfMerged.drop_duplicates("stay_id")
