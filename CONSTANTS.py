# mimic PATH (in which there are hosp and icu data)
from pathlib import Path


MIMIC_PATH = Path("../mimiciv2.2/")

# temporary path
TEMP_PATH = Path("tmp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)