import numpy as np


class Outlier:
    def fit(self, df):
        self.Q1 = df.quantile(0.25)
        self.Q3 = df.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        
    def transform(self, df):
        df = df.copy()
        df[((df < (self.Q1 - 1.5 * self.IQR)) | (df > (self.Q3 + 1.5 * self.IQR)))] = np.nan
        return df
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)