# Related to the logical handling of the data before and after AI handling


import pandas as pd
from typing import TypeVar
from sklearn.preprocessing import StandardScaler

TDataset = TypeVar("TDataset", bound="Dataset")


NORMAL = 1
NOT_NORMAL = 0


class Dataset:
    def __init__(self, csv_path: str, scale_by: TDataset = None):
        self._raw_data: pd.DataFrame = pd.read_csv(csv_path)
        self.normalized_domain: pd.DataFrame
        self.range: pd.Series
        self._scaler: StandardScaler = None

        # Split domain and range
        self.range = self._raw_data["label"]
        self.normalized_domain = self._raw_data.drop("label", axis=1)

        # Convert normal/not normal to 1/0
        self.range = self.range.map(lambda x: NORMAL if x == "normal" else NOT_NORMAL)

        # Convert other string columns to numbers
        string_cols = ["protocol_type", "service", "flag"]
        self.normalized_domain = pd.get_dummies(self.normalized_domain, columns=string_cols)

        # Normalize values
        if scale_by is None:
            self._scaler = StandardScaler()
            self.normalized_domain = self._scaler.fit_transform(self.normalized_domain)
        else:
            self._scaler = scale_by._scaler
            self.normalized_domain = self._scaler.transform(self.normalized_domain)
