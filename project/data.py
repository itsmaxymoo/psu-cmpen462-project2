# Related to the logical handling of the data before and after AI handling


import pandas as pd


NORMAL = 1
NOT_NORMAL = 0


class Dataset:
	def __init__(self, csv_path: str):
		self.raw_data: pd.DataFrame = pd.read_csv(csv_path)
		self.normalized_domain: pd.DataFrame
		self.range: pd.DataFrame

		self.normalized_domain["label"] = self.raw_data["label"].map(lambda x: NORMAL if x == "normal" else NOT_NORMAL)
		# todo: str -> int, normalize
		# todo: separate into range and domain (all attributes in domain, label is range)