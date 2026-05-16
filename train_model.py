import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv(
    "data/diabetes_dataset.csv"
)

# Features and target
X = df.drop(
    "Diabetes_binary",
    axis=1
)

y = df["Diabetes_binary"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE
smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_smote
)

X_test_scaled = scaler.transform(
    X_test
)

# Train model
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train_smote
)

# Prediction
y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"Model Accuracy: {accuracy:.4f}"
)

# Save model
joblib.dump(
    model,
    "model/diabetes_model.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print(
    "Model Saved Successfully!"
)