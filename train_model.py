import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("data.csv")

# Fill missing values with 0
df = df.fillna(0)

SKILLS_LIST = [
    "python","java","sql","ml","excel",
    "communication","design","cloud",
    "marketing","management","aws",
    "docker","linux","sap","cyber","react","seo"
]

X = df[SKILLS_LIST]
y = df["career"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model + encoder
pickle.dump(model, open("career_model.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("✅ Model trained successfully!")
