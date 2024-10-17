import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data.head()
data.describe()
data.info()

sns.heatmap(data.isnull())
plt.show()

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data.head()

data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
data.head()
data.info()

data["diagnosis"] = data['diagnosis'].astype("category", copy=False)
data["diagnosis"].value_counts().plot(kind="bar")
plt.show()

# divide into target varibles and predictors
X = data["diagnosis"] # our target varibles
y = data.drop(["diagnosis"], axis=1)

#### Normalize the data ####

from sklearn.preprocessing import StandardScaler

# Create a Scaler object
scaler = StandardScaler()

# Fit the scaler to the data and transfrom the data
X_scaled = scaler.fit_transfrom(X)

# X_scaled is now a numpy array with normalized data

# Split the data into Test and Train set
from sklearn.model.selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)


#### Train the model and make predictions ####

from sklearn.linear_model import LogisticRegression

# Create logistic regression model
lr = LogisticRegression()

# Train the model on the training data
lr.fit(X_train, y_train)

# Predict the target variable data on the test data
y_pred = lr.predict(X_test)


#### Evaluate the model ####

from sklearn.metrics import accuracy_score

# Evaluate the accuary of the model
accuracy = accuracy.score(y_test, y_pred)
print(f'Accuracy: {accuracy: .2f}')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))






