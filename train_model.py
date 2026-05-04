import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv(r"C:\Users\user\Downloads\animalzip\cleaned_animal_disease_prediction.csv")

# -------------------------
# Feature Engineering
# -------------------------
symptom_cols = [
    'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
    'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
    'Labored_Breathing', 'Lameness', 'Skin_Lesions',
    'Nasal_Discharge', 'Eye_Discharge'
]

df[symptom_cols] = df[symptom_cols].fillna("")

df['symptoms'] = df[symptom_cols].astype(str).apply(lambda x: ' '.join(x), axis=1)
df['symptoms'] = df['symptoms'].str.lower()

# -------------------------
# Vectorization
# -------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['symptoms'])

# -------------------------
# Labels
# -------------------------
le = LabelEncoder()
y = le.fit_transform(df['Disease_Prediction'])

# -------------------------
# Model Training
# -------------------------
model = RandomForestClassifier(n_estimators=300,max_depth=20,random_state=42)
model.fit(X, y)

# -------------------------
# Save files
# -------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("✅ Model trained and saved successfully!")