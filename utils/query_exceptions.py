class ResultEmptyException(Exception):
    def __init__(self) -> None:
        super().__init__("Query returned empty. SQL may not contain main SELECT clause.")
