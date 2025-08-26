import pandas as pd
from typing import Tuple


class DataProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # train in config
        train_cfg = cfg.get("train", {})
        self.last_train_date = pd.Timestamp(train_cfg["last_train_date"])
        self.last_eval_date  = pd.Timestamp(train_cfg["last_eval_date"])
        self.cat_features    = train_cfg.get("cat_features", [])

        # test in config
        test_cfg = cfg.get("test", {})
        self.last_test_date = pd.Timestamp(test_cfg["last_test_date"])

    def split_by_date(self, main_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        df = main_df.copy()
        df["Date"] = pd.to_datetime(df["Date"]) # ensure Date is datetime

        train_df = df[df["Date"] <= self.last_train_date].copy()
        eval_df  = df[(df["Date"] > self.last_train_date) & (df["Date"] <= self.last_eval_date)].copy()
        test_df  = df[df["Date"] > self.last_eval_date].copy()
        return train_df, eval_df, test_df

    def preprocess_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        labels = df[["LogNewConfirmedCases", "LogNewFatalities"]].copy()

        DROP_COLS = [
            "Id", "ForecastId", "ConfirmedCases", "LogNewConfirmedCases",
            "Fatalities", "LogNewFatalities", "Date"
        ]
        features = df.drop(columns=DROP_COLS, errors="ignore").copy()

        for c in self.cat_features:
            if c in features.columns:
                features[c] = features[c].astype("category")

        return features, labels
