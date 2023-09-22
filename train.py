import cv2
import os
import numpy as np

# Define the directory to store student images
data_directory = "student_data"
os.makedirs(data_directory, exist_ok=True)

# Function to capture student images for training
def capture_images(student_name, student_id, num_images=20):
    folder_path = os.path.join(data_directory, f"Student_{student_id}")
    os.makedirs(folder_path, exist_ok=True)
    camera = cv2.VideoCapture(0)

    for i in range(num_images):
        ret, frame = camera.read()
        if not ret:
            continue
        img_path = os.path.join(folder_path, f"{student_name}_{i}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Captured image {i + 1}/{num_images}")

    camera.release()
    cv2.destroyAllWindows()

# Function to train the face recognition model
def train_face_recognition():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces = []
    labels = []

    for student_id, student_name in enumerate(os.listdir(data_directory), 1):
        student_path = os.path.join(data_directory, student_name)
        for image_file in os.listdir(student_path):
            image_path = os.path.join(student_path, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(student_id)

    recognizer.train(faces, np.array(labels))
    recognizer.save("face_recognizer.yml")

# Function to identify students and mark attendance
def mark_attendance():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("face_recognizer.yml")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            student_id, confidence = recognizer.predict(face_roi)

            if confidence < 100:
                student_name = os.listdir(data_directory)[student_id - 1]
                print(f"Recognized {student_name} - Confidence: {confidence}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment and run the functions as needed for each student.
    
    # Example usage for capturing images:
    # capture_images("Student_Name_1", student_id=1, num_images=20)
    # capture_images("Student_Name_2", student_id=2, num_images=20)
    
    # Train the face recognition model (run this once with all collected data)
    train_face_recognition()

    # Identify students and mark attendance (run this during attendance)
    # mark_attendance()
