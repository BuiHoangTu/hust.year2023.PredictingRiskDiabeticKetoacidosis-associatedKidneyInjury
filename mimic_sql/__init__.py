from pathlib import Path
from typing import Any, Callable
import pandas as pd
from pandasql import PandaSQL

from utils.query_exceptions import ResultEmptyException


SQL_FOLDER = Path(__file__).parent


class SqlModifier:
    """Modify SQL text before running it. Default is to escape % character."""

    def __call__(self, sqlText: str) -> Any:
        return self.modify(sqlText)

    def modify(self, sqlText: str) -> str:
        return sqlText.replace("%", "%%")

    pass


class SqlWrapper:

    def __init__(
        self,
        pandasqlEngine: PandaSQL,
        sqlText: str | None = None,
        sqlPath: str | Path | None = None,
        sqlFileName: str | None = None,
        sqlModifier: Callable[[str], str] = SqlModifier(),
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            pandasqlEngine (PandaSQL): _description_
            sqlText (str | None, optional): Raw SQL query. If provided, ignore sqlPath, sqlFileName. Defaults to None.
            sqlPath (str | Path | None, optional): Path to sql file; or the folder that contains sql file, in this case sqlFileName must be provided. Defaults to None.
            sqlFileName (str | None, optional): name of sql file. If provided, search for this file in sqlPath, sqlPath must be a folder in this case. Defaults to None.
            kwargs: mapping of sql tables to pandas dataframes

        Raises:
            ValueError: if both sqlText and sqlPath are None
        """

        self.engine = pandasqlEngine

        if sqlText is not None:
            self.sqlText = sqlText
        elif sqlPath is not None:
            sqlPath = Path(sqlPath)
            if sqlFileName is not None:  # search for the first file in the path
                for path in sqlPath.glob("**/" + sqlFileName):
                    if path.is_file():
                        sqlPath = path
                        break
                    pass
                
                if not sqlPath.is_file():
                    raise Exception(f"{sqlPath} has no file {sqlFileName}.")
                pass
            self.sqlText = sqlPath.read_text()
        else:
            raise ValueError("sqlText or sqlFile must not be null")

        self.sqlText = sqlModifier(self.sqlText)

        self.tableMapping = kwargs

        pass

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        self.tableMapping[key] = value
        pass

    def runSQL(self, ignoreEmptyResult: bool = False) -> pd.DataFrame:
        result = self.engine(self.sqlText, self.tableMapping)
        if not ignoreEmptyResult and result is None:
            raise ResultEmptyException()
        return result # type: ignore
