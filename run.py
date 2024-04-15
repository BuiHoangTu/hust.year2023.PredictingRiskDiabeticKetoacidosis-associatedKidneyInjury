import shutil
# import subprocess
import sys
from constants import TEMP_PATH
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError




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
    # process = subprocess.Popen(
    #     "jupyter nbconvert --execute --inplace main.ipynb",
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     shell=True,
    # )
    # output, error = process.communicate()
    # return output, error
    
    notebookPath = "./main.ipynb"
    
    with open(notebookPath) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')  # Set timeout to -1 for no timeout
    try:
        out = ep.preprocess(nb, {'metadata': {}})
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook "%s".\n\n' % notebookPath
        msg += 'See notebook "%s" for the traceback.' % notebookPath
        print(msg)
        raise
    finally:
        with open(notebookPath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)


if __name__ == "__main__":
    if any("clean" in argv for argv in sys.argv):
        cleanTempPath()
        pass
    if "run" in sys.argv:
        run()
        pass
    pass
