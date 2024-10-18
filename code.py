#importing important libraries
import numpy as np
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Flatten, Dense
from keras.models import Model
import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for face classification
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define data generators for training and validation
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = data_generator.flow_from_directory(
    'img_for_deepfake_detection/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
     # Number of workers for parallel data loading
)

valid_data = data_generator.flow_from_directory(
    'img_for_deepfake_detection/valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
     # Number of workers for parallel data loading
)

# Train the model
model.fit(train_data, epochs=10, validation_data=valid_data)

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(valid_data)
print(f'Validation Accuracy: {accuracy*100:.2f}%')