import numpy as np
import pandas as pd


class DayFeatures:

    def __init__(self, thresholds: list [int] = [1,10,100]):
        self.thresholds = thresholds

    def _add_day_week_info(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Date" not in df.columns:
            raise KeyError("DayFeatures requires 'Date' column")
        
        first_date = df["Date"].min()
        df["Day"] = (df["Date"] - first_date).dt.days.astype("int32")
        df["WeekDay"] = df["Date"].dt.weekday

        return df

    def _add_days_since_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if not {"ConfirmedCases", "Fatalities"}.issubset(df.columns):
            return df
        
        for location_name, location_df in df.groupby(['Country/Region', 'Province/State']):
            for field in ['ConfirmedCases', 'Fatalities']:
                for threshold in self.thresholds:
                    first_day = location_df['Day'].loc[location_df[field] >= threshold].min()
                    if not np.isnan(first_day):
                        df.loc[location_df.index, 'Days_since_%s=%s' % (field, threshold)] = (
                            location_df['Day'].transform(lambda day: -1 if (day < first_day) else (day - first_day))
                        )                 
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._add_day_week_info(df)
        out = self._add_days_since_thresholds(out)
        
        return out
