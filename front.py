import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import json

# Load the model and class indices
model = load_model('deepfake_detection_model.keras')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

# Function to preprocess image
def load_and_preprocess(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image():
    global img_display, uploaded_image_path
    if not uploaded_image_path:
        result_label.config(text="Please upload an image first.")
        return
    preprocessed_img = load_and_preprocess(uploaded_image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    result_label.config(text=f"Prediction: {predicted_class_name}")

    # Function to upload image
def upload_image():
    global img_display, uploaded_image_path
    uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if uploaded_image_path:
        img = Image.open(uploaded_image_path)
        img.thumbnail((300, 300))
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)
        image_label.image = img_display
        result_label.config(text="")

# Create GUI window
root = tk.Tk()
root.title("Deepfake Detection")

# Variables
uploaded_image_path = None
img_display = None

# GUI Widgets
upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_button.pack(pady=10)

image_label = Label(root)
image_label.pack()

predict_button = tk.Button(root, text="Predict", command=predict_image, font=("Arial", 14))
predict_button.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Run the application
root.mainloop()