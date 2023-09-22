import os
import numpy as np
import boto3
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image


# Initialize AWS S3 client
s3 = boto3.client('s3', aws_access_key_id='AKIATROPCW2PY2GRJNQX', aws_secret_access_key='Cb+DiFmF0MiWMpukW11uZUTaRmnBJlTLeJF0n8Pv')

# Replace with your S3 bucket name
bucket_name = 'smartattendancesystem'

def load_data_from_s3():
    images = []
    labels = []

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix='images/')

    for obj in objects.get('Contents', []):
        file_name = obj['Key']
        if file_name.lower().endswith('.png'):
            image_data = s3.get_object(Bucket=bucket_name, Key=file_name)['Body'].read()
            image = Image.open(BytesIO(image_data))
            image = np.array(image.resize((128, 128))) / 255.0
            images.append(image)
            
            label = int(file_name.split('/')[1].split('_')[0])
            labels.append(label)

    return images, labels

# Load images and labels
images, labels = load_data_from_s3()

# Create a mapping of roll numbers to numerical labels
roll_to_label = {roll: idx for idx, roll in enumerate(np.unique(labels))}

# Convert roll numbers to numerical labels
numerical_labels = [roll_to_label[roll] for roll in labels]

X = np.array(images)
y = np.array(numerical_labels)

# Convert labels to one-hot encoded format
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

# Rest of the code remains the same


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("face_recognition_model.h5")
print("Model trained and saved.")
