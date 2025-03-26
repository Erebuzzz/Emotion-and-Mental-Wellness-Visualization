from flask import Flask, jsonify, request
from src.services.emotion_analysis import analyze_emotion

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Emotion and Mental Well-Being Monitoring System!"

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint to analyze emotions from an uploaded image.
    """
    image = request.files.get("image", None)
    if not image:
        return jsonify({"error": "No image provided"}), 400

    result = analyze_emotion(image)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)