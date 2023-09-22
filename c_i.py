import cv2
import os

# Create a directory to store student images
data_directory = "student_data"
os.makedirs(data_directory, exist_ok=True)

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

# Example usage:
# capture_images("Student_Name", student_id, num_images=20)
