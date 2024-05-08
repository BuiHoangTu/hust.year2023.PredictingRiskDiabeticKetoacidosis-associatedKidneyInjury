from pathlib import Path


class NotebookWrapper:
    def __init__(self, notebook: str | Path):
        self.notebook = Path(notebook)
