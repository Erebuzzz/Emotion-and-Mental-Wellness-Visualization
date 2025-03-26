import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to preprocessed data
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(48, 48), batch_size=32, color_mode='grayscale', class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    VAL_DIR, target_size=(48, 48), batch_size=32, color_mode='grayscale', class_mode='categorical'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes for emotions
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=25)

# Save the model
model.save("emotion_model.h5")
print("Model training complete. Saved as emotion_model.h5")