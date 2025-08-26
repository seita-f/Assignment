import os
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Type
import argparse
import yaml
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import joblib

from src.data.load_dataset import CovidDataLoader
from src.data.data_processing import DataProcessor
from src.features.main import FeatureExtraction, FEATURE_REGISTRY
from src.models.utils import predict_for_dataset

# model registry
MODEL_REGISTRY = {
    "CatBoostRegressor": cb.CatBoostRegressor,
    "LGBMRegressor": lgb.LGBMRegressor,
    "XGBRegressor": xgb.XGBRegressor,
}

def setup_logger(log_dir: Path, model_name: str, params: dict) -> Path:
    """
    save log into model log folder 
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ],
    )
    logging.info(f"Model params:\n{yaml.dump(params)}")
    return log_file

def train_model(cfg: dict, df) -> Dict[str, Any]:

    # split data
    processor = DataProcessor(cfg)
    train_df, eval_df, _ = processor.split_by_date(df)
    train_X, train_y = processor.preprocess_df(train_df)
    eval_X,  eval_y  = processor.preprocess_df(eval_df)

    cat_features = cfg["train"].get("cat_features", [])

    # get the model you assinged in config
    save_model_dir = cfg["train"]["save_model_dir"]
    save_log_dir = cfg["train"]["save_log_dir"]
    chosen_model_key = cfg["train"]["model"]  
    model_info = cfg["models"][chosen_model_key]
    model_type = model_info["type"]
    params = model_info.get("params", {})

    # Convert the string paths to Path objects
    save_model_dir = Path(save_model_dir)
    save_log_dir = Path(save_log_dir)

    model_cls = MODEL_REGISTRY[model_type]
    setup_logger(save_log_dir, chosen_model_key, params)

    models = {}
    for target in ["LogNewConfirmedCases", "LogNewFatalities"]:
        if model_type == "CatBoostRegressor":
            model = model_cls(
                **params,
                logging_level="Verbose",   
                train_dir=""             
            )
            model.fit(
                train_X, train_y[target],
                eval_set=(eval_X, eval_y[target]),
                cat_features=cat_features,
                verbose=100
            )
        # else:
        #     model.fit(
        #         train_X, train_y[target],
        #         eval_set=[(eval_X, eval_y[target])],
        #         verbose=True
        #     )

        models[target] = model

        # save　model
        save_model_dir.mkdir(exist_ok=True, parents=True)
        if model_type == "CatBoostRegressor":
            save_path = save_model_dir / f"{chosen_model_key}_{target}.cbm"
            model.save_model(str(save_path))
        # elif model_type == "LGBMRegressor":
        #     save_path = save_model_dir / f"{chosen_model_key}_{target}.txt"
        #     model.booster_.save_model(str(save_path))
        # elif model_type == "XGBRegressor":
        #     save_path = save_model_dir / f"{chosen_model_key}_{target}.json"
        #     model.save_model(str(save_path))
        # else:
        #     save_path = save_model_dir / f"{chosen_model_key}_{target}.pkl"
        #     joblib.dump(model, save_path)

        logging.info(f"Finished training {chosen_model_key} for {target}, saved to {save_path}")

    # Evaluation
    last_train_date = pd.Timestamp(cfg["train"]["last_train_date"])
    last_eval_date = pd.Timestamp(cfg["train"]["last_eval_date"])
    prev_day_df = train_df.loc[train_df["Date"] == last_train_date]
    first_eval_date = last_train_date + pd.Timedelta(days=1)

    predict_for_dataset(
        eval_df, eval_X, prev_day_df,
        first_eval_date, last_eval_date,
        update_features_data=False,
        models=models,
        cat_features=cat_features
    )
    print(eval_df.head())
    logging.info(f"Eval prediction sample:\n{eval_df.head()}")

    return models


def main():

    # You can either run feature/main.py or choose created feature file
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=None, help="Path to precomputed features CSV")
    args = parser.parse_args()

    # load config
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    if args.features is not None and Path(args.features).exists():
        logging.info(f"Loading precomputed features from {args.features}")
        df_feat = pd.read_csv(args.features)
        df_feat = df_feat.fillna("")
    else:
        logging.info("No precomputed features provided, running feature extraction from scratch...")
        df_raw = CovidDataLoader(cfg["paths"]["train_csv"], cfg["paths"]["test_csv"]).load()
        params_map: Dict[str, Dict[str, Any]] = cfg.get("feature_params", {}) or {}
        enabled_features: list[str] = cfg.get("features_to_apply", [])
        fx = FeatureExtraction(FEATURE_REGISTRY, params_map)
        df_feat = fx.add_features(df_raw, enabled_features)

        # save features
        save_dir = Path(cfg["features"]["save_df_dir"])
        save_filename = Path(cfg["features"]["save_filename"])
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / save_filename
        df_feat.to_csv(save_path, index=False)
        logging.info(f"Features extracted and saved to {save_path}")

    # training model you assigned in config
    train_model(cfg, df_feat)

if __name__ == "__main__":
    main()
