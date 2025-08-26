import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Type
import argparse

from src.data.load_dataset import CovidDataLoader
from src.features.time_delay import TimeDelayFeatures
from src.features.day_feature import DayFeatures
from src.features.distance_to_origin import DistanceToOriginFeatures
from src.features.country_area import CountryAreaFeatures
from src.features.country_population import CountryPopulationFeatures
from src.features.smoking import CountrySmokingRateFeatures
from src.features.hospital_beds import CountryHospitalBedsFeatures
from src.features.health_expenditure import CountryHealthExpenditureFeatures


FEATURE_REGISTRY: Dict[str, Type] = {
    "TimeDelayFeatures": TimeDelayFeatures,
    "DayFeatures": DayFeatures,
    "DistanceToOriginFeatures": DistanceToOriginFeatures,
    "CountryAreaFeatures": CountryAreaFeatures,
    "CountryPopulationFeatures": CountryPopulationFeatures,
    "CountrySmokingRateFeatures": CountrySmokingRateFeatures,
    "CountryHospitalBedsFeatures": CountryHospitalBedsFeatures,
    "CountryHealthExpenditureFeatures": CountryHealthExpenditureFeatures,
}

class FeatureExtraction:
    def __init__(self, registry: Dict[str, Type], params_map: Dict[str, Dict[str, Any]] | None = None):
        self.registry = registry
        self.params_map = params_map or {}

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
                df[col] = df[col].fillna("").astype(str)
        return df

    def add_features(self, df: pd.DataFrame, enabled_features: list[str]) -> pd.DataFrame:
        out = self._filling_null(self._swap_cruise(df))

        for name in enabled_features:  
            cls = self.registry.get(name)
            if cls is None:
                raise KeyError(f"Feature '{name}' is not registered in FEATURE_REGISTRY")
            params = self.params_map.get(name, {}) or {}
            transformer = cls(**params)
            out = transformer.transform(out)

        return out

def main(): 

    # handle args
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-file", type=str, default=None,
                        help="Path to save features CSV (default: from config)")
    args = parser.parse_args()

    # load config
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    df = CovidDataLoader(cfg["paths"]["train_csv"], cfg["paths"]["test_csv"]).load()

    # feature extraction
    params_map: Dict[str, Dict[str, Any]] = cfg.get("feature_params", {}) or {}
    enabled_features: list[str] = cfg.get("features_to_apply", [])

    fx = FeatureExtraction(FEATURE_REGISTRY, params_map)
    df_feat = fx.add_features(df, enabled_features)

    # DEBUG:
    # print(df_feat.head(), df_feat.shape)

    # save features
    if args.save_file is not None:
        # command arg
        save_dir = Path(cfg["features"]["save_df_dir"])
        save_filename = Path(args.save_file)
        save_path = save_dir / save_filename
    else:
        # default in config
        save_dir = Path(cfg["features"]["save_df_dir"])
        save_filename = Path(cfg["features"]["save_filename"])
        save_path = save_dir / save_filename

    save_path.parent.mkdir(exist_ok=True, parents=True)

    # DBUG:
    print(df_feat.shape)

    # save
    df_feat.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()