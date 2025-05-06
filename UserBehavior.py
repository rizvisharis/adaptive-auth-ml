import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Step 1: Fetch data
response = requests.get('http://localhost:8080/api/training-data-behavior')

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    print(df.head())
else:
    raise Exception(f"Failed to fetch data: {response.status_code}")

# Step 2: Drop non-feature columns
X = df.drop(['email'], axis=1)
y = df['email']

# Step 3: Identify all feature columns (assume all numeric for now)
numeric_features = X.columns.tolist()

# Step 4: Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocess_pipeline = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

# Step 5: Build the full pipeline
model_pipeline = Pipeline([
    ('preprocess', preprocess_pipeline),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42))
])

# Step 6: Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model_pipeline.predict(X_test)
print("Behavior Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
