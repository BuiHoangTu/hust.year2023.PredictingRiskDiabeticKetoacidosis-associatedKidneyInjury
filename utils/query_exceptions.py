class ResultEmptyException(Exception):
    def __init__(self) -> None:
        super().__init__("Query command does not contain main SELECT. Returned empty.")