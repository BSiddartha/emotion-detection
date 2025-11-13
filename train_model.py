import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset paths
train_data_path = "fer2013/train"
test_data_path = "fer2013/test"

# Image size
img_size = 48

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

# Building CNN Model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))   # 7 emotion classes

# Compile model
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
print("Training started...")
history = model.fit(train_set, validation_data=test_set, epochs=20)

# Save model
model.save("emotion_model.h5")
print("Model saved successfully as emotion_model.h5")
