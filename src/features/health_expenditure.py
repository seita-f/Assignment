from pathlib import Path
import zipfile, urllib.request
import os
import numpy as np
import pandas as pd


class CountryHealthExpenditureFeatures:
    def __init__(
        self,
        source_url: str = "http://api.worldbank.org/v2/en/indicator/SH.XPD.CHEX.PP.CD?downloadformat=csv",
        zip_filename: str = "health_expenditure.zip",
        known_filename: str = "API_SH.XPD.CHEX.PP.CD_DS2_en_csv_*.csv",
        out_dir: Path = "datasets/external_data/health_expenditure",
        right_on: str = "Country Name",
        left_on: str = "Country/Region",
    ):
        self.source_url = source_url
        self.zip_filename = zip_filename
        self.known_filename = known_filename
        self.out_dir = Path(out_dir)
        self.right_on = right_on
        self.left_on = left_on

    def _download_zip(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.out_dir / self.zip_filename
        if not zip_path.exists():
            urllib.request.urlretrieve(self.source_url, zip_path)
            print(f"Downloaded {zip_path}")
        else:
            print(f"{zip_path} already exists, skipping download.")
        return zip_path

    def _unzip_file(self, zip_path: Path) -> Path:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.out_dir)
        # search file (ignore version for now)
        candidates = list(self.out_dir.glob(self.known_filename))
        if not candidates:
            raise FileNotFoundError(f"No matching World Bank CSV found in {self.out_dir}")
        target_csv = candidates[0]
        print(f"[CountryArea] Using CSV: {target_csv}")
        return target_csv
    
    def _load_data(self, csv_path: str) -> pd.DataFrame:
        
        return pd.read_csv(csv_path, skiprows=4)

    def _get_health_expenditure(self, health_expenditure_df: pd.DataFrame) -> pd.DataFrame:

        recent_year_columns = [str(year) for year in range(2010, 2020)]
        health_expenditure_df['CountryHealthExpenditurePerCapitaPPP'] = health_expenditure_df[recent_year_columns].apply(
            lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
            axis='columns'    
        )
        health_expenditure_df = health_expenditure_df[['Country Name', 'CountryHealthExpenditurePerCapitaPPP']]
        return health_expenditure_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # print("DEBUG")
        # print(df.head(), df.shape)

        zip_path = self._download_zip()
        csv_path = self._unzip_file(zip_path)
        health_expenditure_df = self._load_data(csv_path)
        health_expenditure_df = self._get_health_expenditure(health_expenditure_df)

        merged = pd.merge(
            left=df,
            right=health_expenditure_df,
            how="left",
            left_on=self.left_on,
            right_on=self.right_on,
        )

        return merged.drop(columns=[self.right_on])
