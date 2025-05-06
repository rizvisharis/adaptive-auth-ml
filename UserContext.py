import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

response = requests.get('http://localhost:8080/api/training-data-context')

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    print(df.head())
else:
    print("Failed to fetch data:", response.status_code)


X = df.drop(['email'], axis=1)
y = df['email']

# Categorical attributes to encode
cat_attribs = ['browser_name', 'browser_version','user_agent', 'color_depth', 'canvas_fingerprint', 'os', 'cpu_class', 'resolution']

# Preprocessing pipeline
preprocess_pipeline = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)
], remainder='passthrough')

# Build pipeline
model_pipeline = Pipeline([
    ('preprocess', preprocess_pipeline),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_pipeline.predict(X_test)
print("Context Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))