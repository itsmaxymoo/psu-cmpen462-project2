# Classes to represent a model

from abc import ABC, abstractmethod
from typing import final
from .data import Dataset
from .data import NORMAL, NOT_NORMAL
from sklearn.base import ClassifierMixin


class Model(ABC):
	"""Class to represent a learning model.
	Each instance of Model must implement _train() and _predict()
	"""

	def __init__(self, name: str):
		"""Create a model.
		"""
		self.__did_train: bool = False
		self.name: str = name
		self.training_data: Dataset = None
		self.test_data: Dataset = None
	

	@abstractmethod
	def _train(self, training_data: Dataset) -> None:
		"""Internal implementation of the model training.

		Args:
			training_data (Dataset): The dataset to train on.
		"""
		pass
	

	@final
	def train(self, training_data: Dataset) -> None:
		"""Train the model with the given training set.

		Args:
			training_data (Dataset): Training data to use
		"""

		self.__did_train = True
		return self._train(training_data)
	

	@abstractmethod
	def _predict(self, testing_data: Dataset) -> list:
		"""Internal implementation of the model prediction.

		Args:
			testing_data (Dataset): The data to test/predict.

		Returns:
			list: List of integers, for each line, equivalent to data.NORMAL or data.NOT_NORMAL
		"""
		return []


	@final
	def predict(self, testing_data: Dataset) -> list:
		"""Create predictions for the testing_data based on the trained model.

		Args:
			testing_data (Dataset): The data to test/predict.

		Raises:
			RuntimeError: If the model has not been trained yet

		Returns:
			list: List of integers, for each line, equivalent to data.NORMAL or data.NOT_NORMAL
		"""

		if not self.__did_train:
			raise RuntimeError("You must train the model before using it.")

		return self._predict(testing_data)


class DummyModel(Model):
	"""Example model implementation
	"""


	def __init__(self):
		super().__init__("Dummy Model")


	def _train(self, training_data: Dataset):
		# Do Maths Here. Likely store the state internally somewhere.
		pass


	def _predict(self, testing_data):
		# Run some predict functions on the testing data.
		# Return a list of [NORMAL, NORMAL, NOT_NORMAL, etc...]
		return [NORMAL] * len(testing_data._raw_data)


class SKLearnModel(Model):
	def __init__(self, name, classifier: ClassifierMixin):
		super().__init__(name)
		self._model = None
		self._classifier = classifier
	

	def _train(self, training_data):
		self._model = self._classifier.fit(training_data.normalized_domain, training_data.range)
	

	def _predict(self, testing_data) -> list:
		return list(map(lambda x: int(x), list(self._model.predict(testing_data.normalized_domain))))