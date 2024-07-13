from pathlib import Path
import re
import shutil

# import subprocess
import sys

import numpy as np
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


def run(modelNames: list[str] | None = None):
    from notebook_wrapper import NotebookWrapper

    from constants import ARCHIVED_NOTEBOOKS_PATH

    HOWS = ["first", "last", "avg", "max", "min", "med"]

    def _getModels():
        pattern = re.compile(r"tabular_model_(.*?)_template\.ipynb")

        for file in ARCHIVED_NOTEBOOKS_PATH.iterdir():
            if file.is_file():
                modelName = pattern.match(file.name)
                if modelName:
                    if modelNames is None or modelName.group(1) in modelNames:
                        yield modelName.group(1)

    for modelName in _getModels():
        mlNb = NotebookWrapper(
            ARCHIVED_NOTEBOOKS_PATH / f"tabular_model_{modelName}_template.ipynb",
            ["how"],
            [
                "auc_score_list",
                "accuracy_score_list",
                "precision_score_list",
                "recall_score_list",
                "auc_score_list_knn",
                "accuracy_score_list_knn",
                "precision_score_list_knn",
                "recall_score_list_knn",
                "auc_score_list_val",
                "accuracy_score_list_val",
                "precision_score_list_val",
                "recall_score_list_val",
                "auc_score_list_val_knn",
                "accuracy_score_list_val_knn",
                "precision_score_list_val_knn",
                "recall_score_list_val_knn",
            ],
            allowError=True,
            nbContext=Path(__file__).parent,
        )
        for how in HOWS:
            (
                auc_score_list,
                accuracy_score_list,
                precision_score_list,
                recall_score_list,
                auc_score_list_knn,
                accuracy_score_list_knn,
                precision_score_list_knn,
                recall_score_list_knn,
                auc_score_list_val,
                accuracy_score_list_val,
                precision_score_list_val,
                recall_score_list_val,
                auc_score_list_val_knn,
                accuracy_score_list_val_knn,
                precision_score_list_val_knn,
                recall_score_list_val_knn,
            ) = mlNb.export(
                ARCHIVED_NOTEBOOKS_PATH / modelName / f"tabular-model_{modelName}_{how}.ipynb",
                how=how,
            )

            def calculate_mean_and_error(array):
                mean = np.mean(array)
                standard_error = np.std(array) / np.sqrt(len(array))
                return mean, standard_error

            print(f"Model: {modelName}, How: {how}")
            print("==== RAW ====")
            print(f"AUC: {calculate_mean_and_error(auc_score_list)}")
            print(f"Accuracy: {np.mean(accuracy_score_list)}")
            print(f"Precision: {np.mean(precision_score_list)}")
            print(f"Recall: {np.mean(recall_score_list)}")

            print("==== KNN ====")
            print(f"AUC: {calculate_mean_and_error(auc_score_list_knn)}")
            print(f"Accuracy: {np.mean(accuracy_score_list_knn)}")
            print(f"Precision: {np.mean(precision_score_list_knn)}")
            print(f"Recall: {np.mean(recall_score_list_knn)}")

            print("==== WITH VALIDATE ====")
            print(f"AUC: {calculate_mean_and_error(auc_score_list_val)}")
            print(f"Accuracy: {np.mean(accuracy_score_list_val)}")
            print(f"Precision: {np.mean(precision_score_list_val)}")
            print(f"Recall: {np.mean(recall_score_list_val)}")

            print("==== KNN WITH VALIDATE ====")
            print(f"AUC: {calculate_mean_and_error(auc_score_list_val_knn)}")
            print(f"Accuracy: {np.mean(accuracy_score_list_val_knn)}")
            print(f"Precision: {np.mean(precision_score_list_val_knn)}")
            print(f"Recall: {np.mean(recall_score_list_val_knn)}")

            print("=============================")
    pass

def runGui():
    import utils.gui 


if __name__ == "__main__":
    if any("run" in argv for argv in sys.argv):
        paramId = sys.argv.index("run")
        if len(sys.argv) > paramId + 1:
            mode = sys.argv[paramId + 1]
            if mode == "--gui":
                runGui()
                exit()
    
    if any("clean" in argv for argv in sys.argv):
        cleanTempPath()
        pass
    if "train" in sys.argv:
        paramId = sys.argv.index("train")
        if len(sys.argv) > paramId + 1:
            modelNames = sys.argv[paramId + 1:]
            run(modelNames)
            pass
        else:
            run()
        pass
    if "--copy-tabular-template" in sys.argv or "-ctt" in sys.argv:
        paramId = (
            sys.argv.index("--copy-tabular-template")
            if "--copy-tabular-template" in sys.argv
            else sys.argv.index("-ctt")
        )
        if len(sys.argv) > paramId + 1:
            modelName = sys.argv[paramId + 1]
            from constants import ARCHIVED_NOTEBOOKS_PATH

            shutil.copy(
                "./machine_learning.ipynb",
                ARCHIVED_NOTEBOOKS_PATH / f"tabular_model_{modelName}_template.ipynb",
            )
            pass
        else:
            print("Please provide the model name.")
    pass
