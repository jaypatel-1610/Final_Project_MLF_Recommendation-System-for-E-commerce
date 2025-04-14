from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Sentiment Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review_text = data.get("text")

    if not review_text:
        return jsonify({"error": "No text provided"}), 400

    text_vectorized = vectorizer.transform([review_text])
    prediction = model.predict(text_vectorized)

    return jsonify({
        "text": review_text,
        "predicted_sentiment": prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)