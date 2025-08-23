import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Type

from src.data.load_dataset import CovidDataLoader
from src.features.time_delay import TimeDelayFeatures
from src.features.day_feature import DayFeatures


FEATURE_REGISTRY = {
    "TimeDelayFeatures": TimeDelayFeatures,
    "DayFeatures": DayFeatures,
}

class FeatureExtraction:
    def __init__(self, registry: Dict[str, Type]):
        self.registry = registry

    def _swap_cruise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df["Province/State"].isin(["From Diamond Princess", "Grand Princess"])
        if mask.any():
            df.loc[mask, ["Province/State","Country/Region"]] = df.loc[mask, ["Country/Region","Province/State"]].values

        return df

    def _filling_null(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["Country/Region","Province/State"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
        return df

    def add_features(self, df: pd.DataFrame, recipes: List[Dict[str, Any]]) -> pd.DataFrame:

        out = df.copy() 
        # data processing
        out = self._swap_cruise(out)
        out = self._filling_null(out)
        
        # add features
        for r in recipes:
            name = r["name"]
            params = r.get("params", {}) or {}
            cls = self.registry.get(name)
            if cls is None:
                raise KeyError(f"Feature '{name}' is not registered.")
            transformer = cls(**params)
            out = transformer.transform(out)

        return out
    

def main(): 
    # load config
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    train_csv = cfg["paths"]["train_csv"]
    test_csv  = cfg["paths"]["test_csv"]

    print(list(cfg.keys()))

    # load dataset
    df = CovidDataLoader(train_csv, test_csv).load()

    # features params
    recipes = cfg["features_to_apply"]

    # apply features
    fx = FeatureExtraction(FEATURE_REGISTRY)
    df_feat = fx.add_features(df, recipes)
    print(df_feat.head(), df_feat.shape)

if __name__ == "__main__":
    main()