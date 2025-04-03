# main() entrypoint

import argparse
from . import data
from .model import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- Define models
models = {
	"logistic_regression": SKLearnModel("Logistic Regression", LogisticRegression()),
	"svm": SKLearnModel("Support Vector Machine", SVC()),
	"random_forest": SKLearnModel("Random Forest", RandomForestClassifier()),
	"tensorflow": TensorFlowModel()
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
model: Model = models[args.model]
model.train(training_dataset)
predictions = model.predict(testing_dataset)

# --- Analysis and output
print("Analysis and predictions for model: " + model.name)
if not args.omit_predictions:
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
if args.model == "tensorflow" and hasattr(model, "history"):
    plot_training_history(model.history)