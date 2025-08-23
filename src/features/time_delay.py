import numpy as np
import pandas as pd


class TimeDelayFeatures:

    def __init__(self, days_history_size: int = 30):
        self.days_history_size = days_history_size

    # not depend on instance
    @staticmethod
    def _is_cumulative(increment_series: np.ndarray) -> bool:

        for v in increment_series:
            if (not np.isnan(v)) and (v < 0):
                return False
        return True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # DEBUG: 
        print ('data size after removing bad data = ', len(df))

        # initialize log columns
        for field in ["LogNewConfirmedCases", "LogNewFatalities"]:
            df[field] = np.nan
            for prev_day in range(1, self.days_history_size + 1):
                df[f"{field}_prev_day_{prev_day}"] = np.nan

        # group by location
        for location_name, location_df in df.groupby(["Country/Region", "Province/State"]):
            for field in ["ConfirmedCases", "Fatalities"]:
                values = location_df[field].values.copy()
                # daily increment
                values[1:] -= values[:-1]

                if not self._is_cumulative(values):
                    print(f"{field} for {location_name} is not valid cumulative series, drop it")
                    df.drop(index=location_df.index, inplace=True)
                    break

                log_new = np.log1p(values)
                df.loc[location_df.index, "LogNew" + field] = log_new

                # create lag features
                for prev_day in range(1, self.days_history_size + 1):
                    df.loc[location_df.index[prev_day:], f"LogNew{field}_prev_day_{prev_day}"] = (
                        log_new[:-prev_day]
                    )
        
        return df