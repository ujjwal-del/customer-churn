#  import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Load the dataset
# data = pd.read_csv('data/customer_data.csv')

# # Identify columns with string values
# string_cols = data.select_dtypes(include=['object']).columns

# # Check which column contains the string value "P'an"
# for col in string_cols:
#     if data[col].eq("P'an").any():
#         print(f"Column '{col}' contains the string value 'P'an'")

# # Use LabelEncoder to convert string values to numerical values
# le = LabelEncoder()
# for col in string_cols:
#     data[col] = le.fit_transform(data[col])

# # Preprocess the dataset
# scaler = StandardScaler()
# numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
# data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# # Split the dataset into features (X) and target (y)
# X = data.drop(['Exited'], axis=1)
# y = data['Exited']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a random forest classifier model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Identify columns 
string_cols = data.select_dtypes(include=['object']).columns

# Use LabelEncoder 
le = LabelEncoder()
for col in string_cols:
    data[col] = le.fit_transform(data[col])

# Scaling
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split the dataset 
X = data.drop(['Exited'], axis=1)
y = data['Exited']

# Convert the target variable 
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))
print("Gradient Boosting Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))

# Model Evaluation
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "Gradient Boosting": gb
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("Best Model:", best_model)
print("Best Accuracy:", best_accuracy)

# Deployment
def predict_customer_churn(customer_data):
    customer_data = pd.DataFrame(customer_data)
    customer_data = le.transform(customer_data)
    customer_data = scaler.transform(customer_data)
    prediction = best_model.predict(customer_data)
    return prediction

