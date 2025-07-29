import joblib
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.multiclass import unique_labels

# Step 1: Fetch context data
response = requests.get('http://localhost:8080/api/training-data-context')

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    print("Sample data:\n", df.head())
else:
    raise Exception(f"Failed to fetch context data: {response.status_code}")

# Step 2: Encode target labels (email as user ID)
le = LabelEncoder()
y = le.fit_transform(df['email'])
X = df.drop(['email'], axis=1)

joblib.dump(le, 'context_label_encoder.pkl')

# Step 3: Define categorical features
cat_attribs = [
    'browser_name', 'browser_version', 'user_agent',
    'color_depth', 'canvas_fingerprint', 'os',
    'cpu_class', 'resolution', 'ip', 'country_name',
    'country_code', 'region', 'city'
]

# Step 4: Preprocessing pipeline
preprocess_pipeline = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)
], remainder='passthrough')

# Step 5: Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Define models to evaluate
models = {
    'RandomForest': RandomForestClassifier(n_estimators=150, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
    'SVM': SVC(probability=True, kernel='rbf'),
    'KNN': KNeighborsClassifier()
}

# Step 7: Train and evaluate each model
best_model = None
best_score = 0
best_model_name = ""

for name, clf in models.items():
    pipeline = Pipeline([
        ('preprocess', preprocess_pipeline),
        ('classifier', clf)
    ])
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    labels = unique_labels(y_test, y_pred)
    print(classification_report(y_test, y_pred, labels=labels, target_names=le.classes_[labels]))

    if acc > best_score:
        best_score = acc
        best_model = pipeline
        best_model_name = name

# Step 8: Save best model
print(f"\n✅ Best Context Model: {best_model_name} with Accuracy: {best_score:.4f}")
joblib.dump(best_model, 'best_context_model.pkl')
