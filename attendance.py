import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import date
import pandas as pd
import os


# Load the trained CNN model
model = load_model("face_recognition_model.h5")

# Initialize the camera
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Initialize Excel File for Attendance
attendance_file = "attendance_cnn.xlsx"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Roll", "Name"])
    df.to_excel(attendance_file, index=False)

# Get current date
current_date = date.today()

# Initialize a dictionary to track attendance
attendance_dict = {}
student_counter = 1  # Initialize a counter for generating student labels

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Check the depth of the frame
    if frame.dtype == np.float64:
        print("Unsupported depth of input image. Skipping frame.")
        continue

    # Check the number of color channels
    num_channels = frame.shape[-1] if frame.ndim == 3 else 1

    # Preprocess the frame (resize, normalization, etc.)
    frame = cv2.resize(frame, (128, 128))  # Resize to match the CNN model input size
    frame = frame / 255.0  # Normalize pixel values to the range [0, 1]

    # Convert to grayscale if it's a color image
    if num_channels == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Expand the dimensions to match the model's input shape
    input_frame = np.expand_dims(frame, axis=0)

    # Make predictions using the CNN model
    predictions = model.predict(input_frame)

    # Get the predicted label (student index)
    predicted_label = np.argmax(predictions)

    # Generate a student label based on the student_counter
    student_label = f"Student{student_counter}"

    # Increment the student_counter
    student_counter += 1

    # Draw a rectangle and display the recognized student's name on the frame
    cv2.putText(frame, f"Student: {student_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with recognized names
    cv2.imshow("Face Recognition Attendance", frame)

    # Update attendance if recognized
    attendance_dict[student_label] = current_date.strftime("%Y-%m-%d")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Save the updated attendance in the Excel file
df = pd.read_excel(attendance_file)

for student_name, date_marked in attendance_dict.items():
    if date_marked not in df.columns:
        df[date_marked] = ""
    if pd.notna(df.loc[df["Name"] == student_name, date_marked].values[0]):
        print(f"{student_name} already marked attendance on {date_marked}")
    else:
        df.loc[df["Name"] == student_name, date_marked] = "Present"
        print(f"{student_name} marked attendance on {date_marked}")

df.to_excel(attendance_file, index=False)

cam.release()
cv2.destroyAllWindows()
