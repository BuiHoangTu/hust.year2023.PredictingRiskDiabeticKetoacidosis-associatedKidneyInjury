from pathlib import Path
import pandas as pd
from pandasql import PandaSQL


SQL_PATH = Path(__file__).parent


class SqlRunner:

    def __init__(
        self,
        pandasqlEngine: PandaSQL,
        sqlText: str | None = None,
        sqlPath: str | Path | None = None,
        sqlFileName: str | None = None,
        cacheFile: str | Path | None = None,
    ) -> None:
        """_summary_

        Args:
            pandasqlEngine (PandaSQL): _description_
            sqlText (str | None, optional): _description_. Defaults to None.
            sqlPath (str | Path | None, optional): _description_. Defaults to None.
            sqlFileName (str | None, optional): _description_. Defaults to None.
            cacheFile (str | Path | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
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
                pass
            else:
                self.sqlText = sqlPath.read_text()
        else:
            raise ValueError("sqlText or sqlFile must not be null")

        self.cacheFile = cacheFile
        pass

    def runSQL(self):
        if self.cacheFile is not None and Path(self.cacheFile).exists():

            def parseDateColumn(column):
                try:
                    return pd.to_datetime(column, errors="raise")
                except Exception:
                    return column

            res = pd.read_csv(self.cacheFile)
            for col in res.columns:
                res[col] = parseDateColumn(res[col])

        res = self.engine
        return
