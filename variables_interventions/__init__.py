from variables_interventions.mechanical_ventilation import extractMechVent
from variables_interventions.use_crrt import extractUseCrrt
from variables_interventions.use_nahco3 import extractUseOfNaHCO3


def extractInterventions() :
    df1 = extractUseCrrt()
    df2 = extractMechVent()
    df3 = extractUseOfNaHCO3()
    
    return df1.merge(df2, "outer").merge(df3, "outer")
    