import yaml
import pandas as pd
from pathlib import Path
# from dataclasses import dataclass


class CovidDataLoader:
    
    def __init__(self, train_data_path: Path, test_data_path: Path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
    
    def load(self) -> pd.DataFrame:
        # ensure data frame is successfully created
        try:
            train = pd.read_csv(self.train_data_path, parse_dates=["Date"])
            test  = pd.read_csv(self.test_data_path, parse_dates=["Date"])
        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
            raise ValueError(f"Failded loading CSV file: {e}") from e
    
        last_train_date = train["Date"].max()
        test = test[test["Date"] > last_train_date]

        print(f"DEBUG: dropped test shape = {test.shape}")

        df = pd.concat([train, test], ignore_index=True)

        return df.sort_values("Date").reset_index(drop=True)


def main():
    # load config
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    train_csv = cfg["paths"]["train_csv"]
    test_csv  = cfg["paths"]["test_csv"]

    # Load config file
    data = CovidDataLoader(train_csv, test_csv).load()
    
if __name__ == "__main__":
    main()
