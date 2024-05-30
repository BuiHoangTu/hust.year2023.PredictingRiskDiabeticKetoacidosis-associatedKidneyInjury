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


HOWS = ["first", "last", "avg", "max", "min", "std"]
# Define models
modelFactories = {}


## XGBoost
def _createXGBoostModel():
    import xgboost as xgb

    return xgb.XGBClassifier(objective="binary:logistic", device="gpu", n_jobs=-1)


modelFactories["XGBoost"] = _createXGBoostModel


## GRANDE
def _createGrandeModel():
    import os
    from GRANDE import GRANDE

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    params = {
        "depth": 5,  # tree depth
        "n_estimators": 2048,  # number of estimators / trees
        "learning_rate_weights": 0.005,  # learning rate for leaf weights
        "learning_rate_index": 0.01,  # learning rate for split indices
        "learning_rate_values": 0.01,  # learning rate for split values
        "learning_rate_leaf": 0.01,  # learning rate for leafs (logits)
        "optimizer": "adam",  # optimizer
        "cosine_decay_steps": 0,  # decay steps for lr schedule (CosineDecayRestarts)
        "loss": "crossentropy",  # loss function (default 'crossentropy' for binary & multi-class classification and 'mse' for regression)
        "focal_loss": False,  # use focal loss {True, False}
        "temperature": 0.0,  # temperature for stochastic re-weighted GD (0.0, 1.0)
        "from_logits": True,  # use logits for weighting {True, False}
        "use_class_weights": True,  # use class weights for training {True, False}
        "dropout": 0.0,  # dropout rate (here, dropout randomly disables individual estimators of the ensemble during training)
        "selected_variables": 0.8,  # feature subset percentage (0.0, 1.0)
        "data_subset_fraction": 1.0,  # data subset percentage (0.0, 1.0)
    }

    args = {
        "epochs": 1_000,  # number of epochs for training
        "early_stopping_epochs": 25,  # patience for early stopping (best weights are restored)
        "batch_size": 64,  # batch size for training
        "cat_idx": [],  # put list of categorical indices
        "objective": "binary",  # objective / task {'binary', 'classification', 'regression'}
        "random_seed": 42,
        "verbose": 1,
    }

    return GRANDE(params=params, args=args)


modelFactories["GRANDE"] = _createGrandeModel


## TabPFNClassifier
def _createTabPFNClassifier():
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(device="cuda", N_ensemble_configurations=32)


modelFactories["TabPFNClassifier"] = _createTabPFNClassifier


def run():
    from notebook_wrapper import NotebookWrapper

    from constants import ARCHIVED_NOTEBOOKS_PATH

    if __name__ == "__main__":
        mlNb = NotebookWrapper(
            "./machine_learning.ipynb",
            ["how", "createModel"],
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
        )
        for how in HOWS:
            for modelName, modelFactory in modelFactories.items():
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
                    ARCHIVED_NOTEBOOKS_PATH / f"tabular-model_{modelName}_{how}.ipynb",
                    how=how,
                    createModel=modelFactory,
                )

                print(f"Model: {modelName}, How: {how}")
                print("==== RAW ====")
                print(f"AUC: {np.mean(auc_score_list)}")
                print(f"Accuracy: {np.mean(accuracy_score_list)}")
                print(f"Precision: {np.mean(precision_score_list)}")
                print(f"Recall: {np.mean(recall_score_list)}")

                print("==== KNN ====")
                print(f"AUC: {np.mean(auc_score_list_knn)}")
                print(f"Accuracy: {np.mean(accuracy_score_list_knn)}")
                print(f"Precision: {np.mean(precision_score_list_knn)}")
                print(f"Recall: {np.mean(recall_score_list_knn)}")

                print("==== WITH VALIDATE ====")
                print(f"AUC: {np.mean(auc_score_list_val)}")
                print(f"Accuracy: {np.mean(accuracy_score_list_val)}")
                print(f"Precision: {np.mean(precision_score_list_val)}")
                print(f"Recall: {np.mean(recall_score_list_val)}")

                print("==== KNN WITH VALIDATE ====")
                print(f"AUC: {np.mean(auc_score_list_val_knn)}")
                print(f"Accuracy: {np.mean(accuracy_score_list_val_knn)}")
                print(f"Precision: {np.mean(precision_score_list_val_knn)}")
                print(f"Recall: {np.mean(recall_score_list_val_knn)}")

                print("=============================")
        pass


if __name__ == "__main__":
    if any("clean" in argv for argv in sys.argv):
        cleanTempPath()
        pass
    if "run" in sys.argv:
        run()
        pass
    pass
