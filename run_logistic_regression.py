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


# Classes to represent a model

from abc import ABC, abstractmethod
from typing import final
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


# class TensorFlowModel(Model):
# 	import tensorflow as tf
# 	import keras

# 	def __init__(self):
# 		super().__init__("Tensor Flow")

# 		# Configure model structure
# 		self._model: self.keras.Sequential = self.keras.Sequential([
# 			self.keras.layers.Dense(16, activation="relu", input_shape=(45,)), # trial and error
# 			self.keras.layers.Dense(128, activation="relu"),
# 			self.keras.layers.Dense(128, activation="relu"),
# 			self.keras.layers.Dense(1, activation="sigmoid")
# 		])

# 		# Configure model parameters
# 		self._model.compile(
# 			loss=self.keras.losses.BinaryCrossentropy(),
# 			optimizer=self.keras.optimizers.Adam(learning_rate=0.1),
# 			metrics = [
# 				self.keras.metrics.BinaryAccuracy(name="accuracy")
# 			]
# 		)
# 		# Set history to None, will be set during training
# 		self.history = None


# 	def _train(self, training_data):
# 		self.history = self._model.fit(training_data.normalized_domain, training_data.range, epochs=16, batch_size=128)


# 	def _predict(self, testing_data):
# 		pred = self._model.predict(testing_data.normalized_domain)
# 		return list(map(lambda x: NORMAL if x[0] > 0.5 else NOT_NORMAL, pred))


# main() entrypoint

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- Define models
models = {
	"logistic_regression": SKLearnModel("Logistic Regression", LogisticRegression()),
	"svm": SKLearnModel("Support Vector Machine", SVC()),
	"random_forest": SKLearnModel("Random Forest", RandomForestClassifier()),
	# "tensorflow": TensorFlowModel()
}


# --- CLI Args
args = {
      "train": "train_kdd_small.csv",
      "test": "test_kdd_small.csv",
      "model": "logistic_regression"
}


# --- load datasets
training_dataset = Dataset(args["train"])
testing_dataset = Dataset(args["test"], training_dataset)

# Define function to plot training performance
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plot.png')
    print("\nTraining plots saved as 'plot.png'.")
    plt.show()
	
# --- Training
# In the future, we will have support for multiple models.
model: Model = models[args["model"]]
model.train(training_dataset)
predictions = model.predict(testing_dataset)

# --- Analysis and output
print("Analysis and predictions for model: " + model.name)
if True:
	print("Predictions, 0-indexed:")
	for i in range(0, len(predictions)):
		print("Line " + str(i) + " - " + ("normal" if predictions[i] == 1 else "NOT normal"))

accuracy = accuracy_score(testing_dataset.range, predictions)
precision = precision_score(testing_dataset.range, predictions)
recall = recall_score(testing_dataset.range, predictions)
f1 = f1_score(testing_dataset.range, predictions)

# Print metrics
print(f"\n{model.name} Metrics:")
print(f"Accuracy: {round(accuracy, 4)}")
print(f"Precision: {round(precision, 4)}")
print(f"Recall: {round(recall, 4)}")
print(f"F1 Score: {round(f1, 4)}")

# Print confusion matrix to show true/false positives/negatives
print("\nConfusion Matrix:")
print(confusion_matrix(testing_dataset.range, predictions))

# Plot training history if the model is TensorFlow, plot data
if args["model"] == "tensorflow" and hasattr(model, "history"):
    plot_training_history(model.history)