import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Load data
data = pd.read_csv("C:\\Users\\yash2004\\Downloads\\creditcard.csv\\creditcard.csv")

# Check data
print(data.head())
print(data.shape)
print(data.describe())

# Fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))
print("Outlier Fraction:", outlier_fraction)
print('Fraud cases:', len(fraud))
print('Valid transactions:', len(valid))

# Amount details of the fraudulent transactions
print("Amount details of the fraudulent transactions")
print(fraud.Amount.describe())

# Details of valid transactions
print("Details of valid transactions")
print(valid.Amount.describe())



# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

# Prepare data for modeling
X = data.drop(['Class'], axis=1)
Y = data["Class"]
print(X.shape)
print(Y.shape)

# Split data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest Classifier model
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)

# Evaluation metrics
print("Random Forest Classifier Metrics:",yPred)
print("Accuracy:", accuracy_score(yTest, yPred))
print("Precision:", precision_score(yTest, yPred))
print("Recall:", recall_score(yTest, yPred))
print("F1 Score:", f1_score(yTest, yPred))
print("Matthews correlation coefficient:", matthews_corrcoef(yTest, yPred))

# Confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


