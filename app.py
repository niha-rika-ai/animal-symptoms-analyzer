from flask import Flask, request, jsonify, render_template
import pickle
import json
import random

app = Flask(__name__)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Load recommendations
with open("recommendations.json") as f:
    recommendations = json.load(f)


# 🔥 SMART RECOMMENDATION ENGINE
def generate_recommendation(disease, symptoms):
    base = recommendations.get(disease)

    # fallback if disease not found
    if not base:
        return {
            "do": ["Consult veterinarian immediately", "Observe symptoms closely"],
            "dont": ["Avoid self-medication"],
            "medicine": "Professional diagnosis required",
            "notes": ["Disease not clearly identified"]
        }

    final = {
        "do": [],
        "dont": [],
        "medicine": "",
        "notes": []
    }

    # Default recommendations
    if "default" in base:
        final["do"] += base["default"].get("do", [])
        final["dont"] += base["default"].get("dont", [])
        final["medicine"] = base["default"].get("medicine", "")

    # Symptom-based logic
    rules = base.get("rules", {})

    for s in symptoms:
        if s in rules:
            final["do"] += rules[s].get("do", [])
            final["dont"] += rules[s].get("dont", [])
            final["notes"] += rules[s].get("note", [])

            if "medicine" in rules[s]:
                final["medicine"] = rules[s]["medicine"]

    # Remove duplicates
    final["do"] = list(set(final["do"]))
    final["dont"] = list(set(final["dont"]))
    final["notes"] = list(set(final["notes"]))

    # 🔥 Shuffle to avoid repetition feeling
    random.shuffle(final["do"])
    random.shuffle(final["dont"])

    return final


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"message": "Please select symptoms"}), 400

    # Convert symptoms to text
    text = " ".join(symptoms).lower()
    X = vectorizer.transform([text])

    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[-3:][::-1]

    predictions = [
        {
            "disease": le.inverse_transform([i])[0],
            "confidence": round(float(probs[i]), 4)
        }
        for i in top_idx
    ]

    best_disease = predictions[0]["disease"]

    # Generate smart recommendation
    rec = generate_recommendation(best_disease, symptoms)

    return jsonify({
        "predictions": predictions,
        "recommendation": rec
    })


if __name__ == "__main__":
    app.run(debug=True)