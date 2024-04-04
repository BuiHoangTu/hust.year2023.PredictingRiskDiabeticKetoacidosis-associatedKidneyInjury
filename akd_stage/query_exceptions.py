class ResultEmptyException(Exception):
    def __init__(self) -> None:
        super().__init__("Query failed. Returned empty.")