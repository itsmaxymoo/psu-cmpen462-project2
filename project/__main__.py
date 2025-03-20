# main() entrypoint

import argparse
from . import data
from .model import *

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# --- Define models
models = {
	"logistic_regression": SKLearnModel("Logistic Regression", LogisticRegression()),
	"svm": SKLearnModel("Support Vector Machine", SVC()),
	"random_forest": SKLearnModel("Random Forest", RandomForestClassifier())
}


# --- CLI Args
parser = argparse.ArgumentParser(
	prog="model-demo"
)
parser.add_argument("train", help="Path to the csv file containing training data")
parser.add_argument("test", help="Path to the csv file containing test data")
parser.add_argument("model", help="The model you wish to run. Available models: " + ", ".join(list(models.keys())), choices=list(models.keys()))
parser.add_argument("--omit-predictions", help="Use this flag to suppress the actual prediction output.", action="store_true")
args = parser.parse_args()


# --- load datasets
training_dataset = data.Dataset(args.train)
testing_dataset = data.Dataset(args.test, training_dataset)


# --- Training
# In the future, we will have support for multiple models.
model: Model = models[args.model]
model.train(training_dataset)
predictions = model.predict(testing_dataset)

# --- Analysis and output
print("Analysis and predictions for model: " + model.name)
if not args.omit_predictions:
	print("Predictions, 0-indexed:")
	for i in range(0, len(predictions)):
		print("Line " + str(i) + " - " + ("normal" if predictions[i] == 1 else "NOT normal"))