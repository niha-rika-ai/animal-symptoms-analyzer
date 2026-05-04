import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def predict_disease(symptoms_list):
    input_text = " ".join(symptoms_list).lower().strip()
    
    X_input = vectorizer.transform([input_text])
    
    pred = model.predict(X_input)
    disease = le.inverse_transform(pred)[0]
    
    confidence = model.predict_proba(X_input).max()
    
    return disease, confidence


# Test locally
if __name__ == "__main__":
    symptoms = ["fever", "vomiting", "loss of appetite"]
    
    disease, confidence = predict_disease(symptoms)
    
    print("Disease:", disease)
    print("Confidence:", round(confidence * 100, 2), "%")