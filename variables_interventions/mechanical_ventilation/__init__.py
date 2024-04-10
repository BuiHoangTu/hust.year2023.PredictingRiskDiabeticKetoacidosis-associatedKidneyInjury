from variables_interventions.mechanical_ventilation.ventilation import extractVentilation


def extractMechVent():
    df = extractVentilation()
    df["mechanical_ventilation"] = df["ventilation_status"].isin(
        [
            "Tracheostomy",
            "InvasiveVent",
        ]
    )
    
    return df
