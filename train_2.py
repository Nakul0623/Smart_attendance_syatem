import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Define hyperparameters
batch_size = 32
epochs = 50
input_shape = (128,)  # Input shape for face encodings

# Define the directory where your face encodings and labels are stored
data_dir = 'labels'  # Change this to your directory

# Load the face encodings and labels as strings
face_encodings = np.load(os.path.join(data_dir, 'face_encodings.npy'))
string_labels = np.load(os.path.join(data_dir, 'labels.npy'))

# Create a mapping from string labels to integer labels
label_to_int = {label: i for i, label in enumerate(np.unique(string_labels))}

# Convert string labels to integers
integer_labels = np.array([label_to_int[label] for label in string_labels])

# Split the data into training and validation sets (you can use other methods)
split_ratio = 0.8
split_idx = int(len(face_encodings) * split_ratio)

x_train, x_val = face_encodings[:split_idx], face_encodings[split_idx:]
y_train, y_val = integer_labels[:split_idx], integer_labels[split_idx:]

# Build a deeper and more complex face recognition model
model = Sequential([
    Dense(256, activation='relu', input_shape=input_shape),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),  # Additional hidden layer
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(integer_labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val)
)

# Save the trained model
model.save('face_recognition_model.h5')

# Optionally, save the class indices for later use (if you have class labels)
class_indices = {str(i): i for i in np.unique(integer_labels)}
np.save('class_indices.npy', class_indices)
