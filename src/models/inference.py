import argparse
import yaml
import pandas as pd
import logging
from pathlib import Path
import catboost as cb
from typing import Dict, Any, List, Type

from src.data.data_processing import DataProcessor
from src.features.main import FeatureExtraction, FEATURE_REGISTRY
from src.models.utils import predict_for_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=None, help="Path to test.csv")
    args = parser.parse_args()

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # load raw test
    if args.test is not None:
        test_csv = args.test
    else:
        test_csv = cfg["paths"]["test_csv"]

    df_raw = pd.read_csv(test_csv, parse_dates=["Date"])

    # feature extraction
    params_map: Dict[str, Dict[str, Any]] = cfg.get("feature_params", {}) or {}
    enabled_features: list[str] = cfg.get("features_to_apply", [])
    fx = FeatureExtraction(FEATURE_REGISTRY, params_map)
    df_feat = fx.add_features(df_raw, enabled_features)


if __name__ == "__main__":
    main()