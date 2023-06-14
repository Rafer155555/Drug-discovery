import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the data
data = pd.read_csv('data/drug_discovery.csv')
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('activity', axis=1), data['activity'], test_size=0.2)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
