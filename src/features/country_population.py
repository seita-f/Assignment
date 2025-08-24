from __future__ import annotations
from pathlib import Path
import zipfile, urllib.request
from typing import Union, List
import numpy as np
import pandas as pd
import re

def remap_country_name_from_un_wpp_to_main_df_name(country: str) -> str:
    return {
        'Bahamas': 'The Bahamas',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Brunei Darussalam': 'Brunei',
        'China, Taiwan Province of China': 'Taiwan*',
        'Congo' : 'Congo (Brazzaville)',
        "Côte d'Ivoire": "Cote d'Ivoire",
        'Democratic Republic of the Congo': 'Congo (Kinshasa)',
        'Gambia': 'The Gambia',
        'Iran (Islamic Republic of)': 'Iran',
        'Republic of Korea': 'Korea, South',
        'Republic of Moldova': 'Moldova',
        'Réunion': 'Reunion',
        'Russian Federation': 'Russia',
        'United Republic of Tanzania': 'Tanzania',
        'United States of America': 'US',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Viet Nam': 'Vietnam'
    }.get(country, country)


class CountryPopulationFeatures:

    def __init__(
        self,
        source_url: str = "https://github.com/ordinaryevidence/leep-cea/raw/refs/heads/master/WPP2019_PopulationByAgeSex_Medium.zip",
        zip_filename: str = "WPP2019_PopulationByAgeSex_Medium.zip",
        known_filename: str = "known_filename: WPP2019_PopulationByAgeSex_Medium.csv",
        out_dir: Union[str, Path] = "datasets/external_data/population",
        left_on: str = "Country/Region",                  
        right_on: str = "Location",                       
        time_from: str | None = "2014-01-01",
        time_to: str | None = "2019-01-01",    
    ):
        self.source_url = source_url
        self.zip_filename = zip_filename
        self.out_dir = Path(out_dir)
        self.known_filename = known_filename
        self.left_on = left_on
        self.right_on = right_on
        self.time_from = pd.to_datetime(time_from) if time_from else None
        self.time_to = pd.to_datetime(time_to) if time_to else None

    def _download_zip(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = (self.out_dir / self.zip_filename).resolve()
        if not zip_path.exists():
            urllib.request.urlretrieve(self.source_url, zip_path)
            print(f"[Population] Downloaded: {zip_path}")
        else:
            print(f"[Population] ZIP exists: {zip_path}")
        return zip_path

    def _unzip_and_find_csv(self, zip_path: Path) -> Path:

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.out_dir)
        # search file (ignore version for now)
        candidates = list(self.out_dir.glob(self.known_filename))
        if not candidates:
            raise FileNotFoundError(f"No matching WPP CSV found in {self.out_dir}")
        target_csv = candidates[0]
        print(f"[Population] Using CSV: {target_csv}")
        return target_csv
    
    def _load_data(self, csv_path: Path) -> pd.DataFrame:

        usecols = ["Location", "Time", "AgeGrp", "PopMale", "PopFemale", "PopTotal"]
        un_wpp_converters = {"Location": remap_country_name_from_un_wpp_to_main_df_name}
        df = pd.read_csv(csv_path, usecols=usecols, converters=un_wpp_converters)
      
        if np.issubdtype(df["Time"].dtype, np.number):
            df["Time"] = pd.to_datetime(df["Time"].astype(int), format="%Y")
        else:
            df["Time"] = pd.to_datetime(df["Time"])
        return df

    def _filter_time(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if self.time_from is not None:
            out = out.loc[out["Time"] >= self.time_from]
        if self.time_to is not None:
            out = out.loc[out["Time"] <= self.time_to]
        return out

    def _aggregate(self, population_df: pd.DataFrame) -> pd.DataFrame:
        df = population_df.copy()

        df["age_start"] = df["AgeGrp"].astype(str).str.extract(r"^(\d+)")[0].astype(int)
        df["bucket"] = np.minimum(df["age_start"] // 20, 4)

        pop = df.groupby(["Location", "Time", "bucket"], as_index=False)["PopTotal"].sum()
        pop_pivot = (
            pop.pivot(index=["Location", "Time"], columns="bucket", values="PopTotal")
            .rename(columns={
                0: "CountryPop_0-20",
                1: "CountryPop_20-40",
                2: "CountryPop_40-60",
                3: "CountryPop_60-80",
                4: "CountryPop_80+",
            })
            .reset_index()
        )

        for col in ["CountryPop_0-20","CountryPop_20-40","CountryPop_40-60","CountryPop_60-80","CountryPop_80+"]:
            if col not in pop_pivot.columns:
                pop_pivot[col] = 0.0

        sex = df.groupby(["Location", "Time"], as_index=False)[["PopMale","PopFemale"]].sum()
        sex = sex.rename(columns={"PopMale":"CountryPopMale","PopFemale":"CountryPopFemale"})
        sex["CountryPopTotal"] = sex["CountryPopMale"] + sex["CountryPopFemale"]

        wide = pop_pivot.merge(sex, on=["Location","Time"], how="inner")
        wide = wide.sort_values("Time").drop_duplicates(["Location"], keep="last")

        return wide.drop(columns=["Time"])

    def _add_density(self, df: pd.DataFrame) -> pd.DataFrame:
        df['CountryPopDensity'] = df['CountryPopTotal'] / df['CountryArea']
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        zip_path = self._download_zip()
        csv_path = self._unzip_and_find_csv(zip_path)
        population_df = self._load_data(csv_path)
        population_df = self._filter_time(population_df)
        agg_df = self._aggregate(population_df)

        merged = pd.merge(
            left=df,
            right=agg_df,
            how="left",
            left_on=self.left_on,
            right_on=self.right_on,
        )
        
        df = merged.drop(columns=[self.right_on])
        df = self._add_density(df)

        return df


