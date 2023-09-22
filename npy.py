import face_recognition
import numpy as np
import os

# Specify the directory containing the images
image_directory = 'images'

# Initialize a list to store face encodings
face_encodings = []

# Loop through all files in the image directory
for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Load each image file
        image_path = os.path.join(image_directory, filename)
        image = face_recognition.load_image_file(image_path)

        # Extract face encodings (if a face is found)
        face_encoding = face_recognition.face_encodings(image)
        if len(face_encoding) > 0:
            # Append the first face encoding (assuming only one face per image)
            face_encodings.append(face_encoding[0])

# Check if any face encodings were collected
if len(face_encodings) == 0:
    print("No face encodings found in the directory.")
else:
    # Convert the list of face encodings to a NumPy array
    face_encodings_array = np.array(face_encodings)

    # Save face encodings as a .npy file
    np.save('face_encodings.npy', face_encodings_array)

    print(f"{len(face_encodings)} face encodings saved to 'face_encodings.npy'.")
