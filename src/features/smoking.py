from pathlib import Path
import zipfile, urllib.request
import os
import numpy as np
import pandas as pd

def remap_country_name_from_world_bank_to_main_df_name(country: str) -> str:
    return {
        'Bahamas, The': 'The Bahamas',
        'Brunei Darussalam': 'Brunei',
        'Congo, Rep.': 'Congo (Brazzaville)',
        'Congo, Dem. Rep.': 'Congo (Kinshasa)',
        'Czech Republic': 'Czechia',
        'Egypt, Arab Rep.': 'Egypt',
        'Iran, Islamic Rep.': 'Iran',
        'Korea, Rep.': 'Korea, South',
        'Kyrgyz Republic': 'Kyrgyzstan',
        'Russian Federation': 'Russia',
        'Slovak Republic': 'Slovakia',
        'St. Lucia': 'Saint Lucia',
        'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
        'United States': 'US',
        'Venezuela, RB': 'Venezuela',
    }.get(country, country)


class CountrySmokingRateFeatures:
    def __init__(
        self,
        source_url: str = "http://api.worldbank.org/v2/en/indicator/AG.LND.TOTL.K2?downloadformat=csv",
        zip_filename: str = "smoking.zip",
        known_filename: str = "API_SH.PRV.SMOK_DS2_en_csv_*.csv",
        out_dir: Path = "datasets/external_data/",
        right_on: str = "Country Name",
        left_on: str = "Country/Region",
        output_col: str = "Country",
    ):
        self.source_url = source_url
        self.zip_filename = zip_filename
        self.known_filename = known_filename
        self.out_dir = Path(out_dir)
        self.right_on = right_on
        self.left_on = left_on
        self.output_col = output_col

    def _download_zip(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.out_dir / self.zip_filename
        if not zip_path.exists():
            urllib.request.urlretrieve(self.source_url, zip_path)
            print(f"Downloaded {zip_path}")
        else:
            print(f"{zip_path} already exists, skipping download.")
        return zip_path

    def _unzip_file(self, zip_path: Path):
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

    def _get_smoking_rate(self, smoking_df: pd.DataFrame) -> pd.DataFrame:
        smoking_df = smoking_df.copy()

        recent_year_columns = [str(year) for year in range(2010, 2020)]
        smoking_df['CountrySmokingRate'] = smoking_df[recent_year_columns].apply(
            lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
            axis='columns'    
        )
        smoking_df = smoking_df[['Country Name', 'CountrySmokingRate']]

        return smoking_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        zip_path = self._download_zip()
        csv_path = self._unzip_file(zip_path)
        smoking_df = self._load_data(csv_path)
        smoking_df = self._get_smoking_rate(smoking_df)

        merged = pd.merge(
            left=df,
            right=smoking_df,
            how="left",
            left_on=self.left_on,
            right_on=self.right_on,
        )

        return merged.drop(columns=[self.right_on])
