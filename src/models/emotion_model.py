from tensorflow.keras.models import load_model
import numpy as np

# FER+ emotion labels
emotion_labels = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

def load_model():
    return load_model("emotion_model.h5")

def predict_emotion(model, image):
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    return emotion_labels[np.argmax(preds)]