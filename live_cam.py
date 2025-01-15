import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json


# Class labels
class_names = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

model = model_from_json(open(r"C:\Users\rajsa\OneDrive\Desktop\BTP\model.json", "r").read())
model.load_weights(r'C:\Users\rajsa\OneDrive\Desktop\BTP\model.weights.h5')

# Function to preprocess the image
def preprocess_single_image(img_path, target_size=(300, 300)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.cast(img_array, tf.float32)  # Convert to float32
    return img_array

# Function to predict the label for a single image
def predict_label(img_path):
    img_array = preprocess_single_image(img_path)
    predictions = model.predict(img_array)  # Get model predictions
    predicted_class_index = np.argmax(predictions, axis=-1)[0]  # Get the predicted class index
    predicted_class_name = class_names[predicted_class_index]  # Map to class name
    return predicted_class_name

# Capture and display camera feed
def update_camera_feed():
    global cap
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB format for tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    lbl_video.after(10, update_camera_feed)  # Refresh feed

# Capture image and predict emotion
def capture_image():
    global cap, face_cascade
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face detected!")
            return

        for (x, y, w, h) in faces:
            # Extract and save the face
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (300, 300))
            file_name = "captured_face.jpg"
            cv2.imwrite(file_name, cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

            # Predict emotion
            predicted_label = predict_label(file_name)

            # Display prediction
            lbl_prediction.config(text=f"Predicted Emotion: {predicted_label}", fg="#0a9396")

            messagebox.showinfo("Success", f"Face captured and emotion predicted as: {predicted_label}")
            break
    else:
        messagebox.showerror("Error", "Failed to capture image")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize GUI
root = tk.Tk()
root.title("Emotion Prediction Interface")
root.geometry("800x600")
root.configure(bg="#2d2f3e")

cap = cv2.VideoCapture(0)  # Start camera immediately

# Heading
lbl_heading = tk.Label(root, text="Emotion Prediction Application", font=("Arial", 20, "bold"), bg="#2d2f3e", fg="#f0a500")
lbl_heading.pack(pady=10)

# Video Feed
lbl_video = tk.Label(root, bg="#444654")
lbl_video.pack(pady=10, padx=10, fill="both", expand=True)

# Prediction Label
lbl_prediction = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#2d2f3e", fg="#f0a500")
lbl_prediction.pack(pady=10)

# Frame for buttons
controls_frame = tk.Frame(root, bg="#2d2f3e")
controls_frame.pack(side="bottom", pady=10)

# Capture & Predict Button
btn_capture = tk.Button(controls_frame, text="Capture & Predict", font=("Arial", 14), command=capture_image, bg="#bb3e03", fg="#ffffff", width=20)
btn_capture.pack(pady=10)

# Start updating the camera feed
update_camera_feed()

# Run GUI loop
root.mainloop()

# Release the camera when the application closes
cap.release()
cv2.destroyAllWindows()
