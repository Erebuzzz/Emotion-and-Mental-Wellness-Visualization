from src.models.emotion_model import load_model, predict_emotion

model = load_model()

def analyze_emotion(image):
    if image is None:
        return {"error": "No image provided"}
    emotion = predict_emotion(model, image)
    return {"emotion": emotion, "confidence": 0.95}