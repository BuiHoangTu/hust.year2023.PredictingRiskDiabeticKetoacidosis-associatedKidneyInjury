# import pickle
# import tempfile
# from datetime import datetime
# from pathlib import Path
# from time import sleep
# from typing import Any, List

# import nbformat
# from nbconvert.preprocessors import ExecutePreprocessor
# from nbformat.v4 import nbbase


# class NotebookWrapper:
#     def __init__(self, notebookFile: str | Path, outputVariable: str | List[str]):
#         self.notebook = Path(notebookFile)
#         self.outputVariable = outputVariable

#     def run(self) -> Any | List[Any]:
#         nb = nbformat.read(self.notebook, as_version=nbformat.NO_CONVERT)

#         # add saving path
#         if isinstance(self.outputVariable, List):
#             folderStr = "-".join(self.outputVariable)
#         else:
#             folderStr = self.outputVariable

#         outputPath = Path(
#             tempfile.gettempdir(),
#             "BHTuNbWrapper",
#             self.notebook.stem,
#             folderStr,
#             datetime.now().__str__() + ".pkl",
#         )
#         outputPath.parent.mkdir(parents=True, exist_ok=True)

#         # add cell to nb
#         if isinstance(self.outputVariable, List):
#             requestVars = "[" + ",".join(self.outputVariable) + "]"
#         else:
#             requestVars = self.outputVariable

#         newCell = nbbase.new_code_cell(
#             source="""
#             from pathlib import Path
#             import pickle
            
#             outputVariable = %s
#             pickle.dump(outputVariable, Path("%s").open("wb+"))
#         """
#             % (requestVars, outputPath)
#         )

#         nb.cells.append(newCell)

#         ep = ExecutePreprocessor(timeout=None)
#         resultNb, _ = ep.preprocess(nb, {"metadata": {"path": self.notebook.parent}})

#         # wait for nb output
#         for _ in range(5):
#             if outputPath.exists():
#                 break
#             else:
#                 sleep(2)
#         else:
#             raise IOError(outputPath.__str__() + " took too much time to write.")

#         res = pickle.load(outputPath.open("rb"))

#         return res
