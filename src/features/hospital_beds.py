from pathlib import Path
import zipfile, urllib.request
import os
import numpy as np
import pandas as pd


class CountryHospitalBedsFeatures:
    def __init__(
        self,
        source_url: str = "http://api.worldbank.org/v2/en/indicator/SH.MED.BEDS.ZS?downloadformat=csv",
        zip_filename: str = "hospital_beds.zip",
        known_filename: str = "API_SH.MED.BEDS.ZS_DS2_en_csv_*.csv",
        out_dir: Path = "datasets/external_data/hospital_beds",
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

    def _get_hospital_bed_rate(self, hospital_beds_df: pd.DataFrame) -> pd.DataFrame:
        hospital_beds_df = hospital_beds_df.copy()

        recent_year_columns = [str(year) for year in range(2010, 2020)]
        hospital_beds_df['CountryHospitalBedsRate'] = hospital_beds_df[recent_year_columns].apply(
            lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
            axis='columns'    
        )
        hospital_beds_df = hospital_beds_df[['Country Name', 'CountryHospitalBedsRate']]

        return hospital_beds_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        zip_path = self._download_zip()
        csv_path = self._unzip_file(zip_path)
        hospital_beds_df = self._load_data(csv_path)
        hospital_beds_df = self._get_hospital_bed_rate(hospital_beds_df)

        merged = pd.merge(
            left=df,
            right=hospital_beds_df,
            how="left",
            left_on=self.left_on,
            right_on=self.right_on,
        )
 
        return merged.drop(columns=[self.right_on])
