import cv2
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained face recognition model
model = load_model('face_recognition_model.h5')

# Load the class indices
class_indices = np.load('class_indices.npy', allow_pickle=True).item()

# Load the input photo
input_photo_path = 'photo.jpg'
input_photo = face_recognition.load_image_file(input_photo_path)

# Detect faces in the input photo
face_locations = face_recognition.face_locations(input_photo)
face_encodings = face_recognition.face_encodings(input_photo, face_locations)

# Create a list to store the names of recognized students
recognized_students = []

# Recognize faces in the input photo
for face_encoding in face_encodings:
    # Predict the class (student) for the face using the trained model
    predictions = model.predict(np.expand_dims(face_encoding, axis=0))
    predicted_class_index = np.argmax(predictions)
    
    # Get the student name based on the class index
    student_name = [name for name, index in class_indices.items() if index == predicted_class_index][0]
    
    # Store the recognized student's name
    recognized_students.append(student_name)

# Display the names of recognized students
if recognized_students:
    print("Recognized Students:")
    for student_name in recognized_students:
        print(student_name)
else:
    print("No students recognized in the photo.")

# Optionally, you can also display the names of students with their input face images
for student_name, face_encoding in zip(recognized_students, face_encodings):
    # Create an image with the recognized face
    recognized_face = face_recognition.face_encodings(input_photo, [face_recognition.face_locations(input_photo)[0]])[0]
    recognized_face_image = cv2.resize(input_photo, (250, 250))
    
    # Display the name and recognized face image
    cv2.putText(recognized_face_image, student_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(student_name, recognized_face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
