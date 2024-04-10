from variables_interventions.mechanical_ventilation import extractMechVent
from variables_interventions.use_crrt import extractUseCrrt


def extractInterventions() :
    extractUseCrrt()
    extractMechVent()
    