# main() entrypoint

import argparse
from . import data, model


# --- CLI Args
parser = argparse.ArgumentParser(
	prog="462 project 2"
)
parser.add_argument("train", help="Path to the csv file containing training data")
parser.add_argument("test", help="Path to the csv file containing test data")
args = parser.parse_args()


# --- load datasets
training_dataset = data.Dataset(args.train)
testing_dataset = data.Dataset(args.test, training_dataset)


# --- Training
# In the future, we will have support for multiple models.
m: model.DummyModel = model.DummyModel()
m.train(training_dataset)
predictions = m.predict(testing_dataset)