import numpy as np
import pandas as pd
import geopy.distance

class DistanceToOriginFeatures:

    def __init__(self, origin_province: str = 'Hubei'):
        self.origin_province = origin_province

    def _get_origin_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        for index, row in df.iterrows():
            if row['Province/State'] == self.origin_province:
                return (row['Lat'], row['Long'])

        raise Exception(f'{self.origin_province} not found in data')

    def _add_distance(self, df: pd.DataFrame, origin_coords: float) -> pd.DataFrame:
        df = df.copy()
        df['Distance_to_origin'] = df.apply(
            lambda row: geopy.distance.distance((row['Lat'], row['Long']), origin_coords).km,
            axis='columns')
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        out = df.copy()
        origin_coords = self._get_origin_coords(out)
        out = self._add_distance(out, origin_coords)

        return out