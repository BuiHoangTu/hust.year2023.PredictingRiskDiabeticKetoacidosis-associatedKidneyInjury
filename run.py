import shutil
import subprocess
import sys
from constants import TEMP_PATH


def cleanTempPath():
    BAK_PATH = TEMP_PATH / "bak"

    if BAK_PATH.exists():
        shutil.rmtree(BAK_PATH)
        pass

    BAK_PATH.mkdir(parents=True, exist_ok=True)

    for item in TEMP_PATH.iterdir():
        if item != BAK_PATH:
            shutil.move(str(item), str(BAK_PATH))

    print(f"All files and folders in '{TEMP_PATH}' have been moved to '{BAK_PATH}'.")


def run():
    process = subprocess.Popen(
        "jupyter nbconvert --execute --inplace main.ipynb",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    output, error = process.communicate()
    return output, error


if __name__ == "__main__":
    if any("clear" in argv for argv in sys.argv):
        cleanTempPath()
        pass
    if "run" in sys.argv:
        run()
        pass
    pass
