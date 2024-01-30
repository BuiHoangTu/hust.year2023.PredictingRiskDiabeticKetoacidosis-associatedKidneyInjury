# mimic PATH (in which there are hosp and icu data)
from pathlib import Path


MIMIC_PATH = Path("../mimiciv2.2/")

# temporary path
TEMP_PATH = Path("tmp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)

IMPORTANT_MESUREMENTS_ICU = {
    227519: "urine_output",
    224639: "weight",
    227457: "plt",
    220615: "creatinine"
}

IMPORTANT_MESUREMENTS_LABEVENT = {
    51006: "bun"    
}