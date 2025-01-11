import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import os
from tensorflow.keras.layers import Conv2D

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((256, 256))  
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Function to make predictions
def make_prediction(model, image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction

# Disease names mapping
disease_names = {
    0: "Bacterial Spot",
    1: "Early Blight",
    2: "Late Blight",
    3: "Leaf Mold",
    4: "Septoria Leaf Spot",
    5: "Spider Mites",
    6: "Target Spot",
    7: "Yellow Leaf Curl Virus",
    8: "Mosaic Virus",
    9: "Healthy"
}

# Function to load model and make prediction
def predict():
    try:
        selected_model = model_combo.get()
        if not selected_model:
            messagebox.showerror("Error", "Please select a model")
            return

        image_path = file_path_var.get()
        if not image_path:
            messagebox.showerror("Error", "Please upload an image")
            return

        model_path = os.path.join(model_folder, selected_model)
        model = load_model(model_path)
    
        prediction = make_prediction(model, image_path)
        predicted_index = np.argmax(prediction)
        predicted_disease = disease_names.get(predicted_index, "Unknown")
        confidence = np.max(prediction)

        result_label.config(text=f"Prediction: {predicted_disease} (Confidence: {confidence:.2f})")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to browse and upload an image
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        file_path_var.set(file_path)
        img = Image.open(file_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Initialize the main application window
app = tk.Tk()
app.title("Tomato Disease Detection")
app.geometry("500x600")

# Folder containing models
model_folder = "models"  # Replace with the actual folder where your models are stored
model_files = [f for f in os.listdir(model_folder) if f.endswith(".h5")]

# UI Components
model_label = tk.Label(app, text="Select Model:")
model_label.pack(pady=5)

model_combo = ttk.Combobox(app, values=model_files, state="readonly")
model_combo.pack(pady=5)

file_label = tk.Label(app, text="Upload an Image:")
file_label.pack(pady=5)

file_path_var = tk.StringVar()
file_button = tk.Button(app, text="Browse", command=browse_file)
file_button.pack(pady=5)

image_label = tk.Label(app)
image_label.pack(pady=5)

predict_button = tk.Button(app, text="Predict", command=predict)
predict_button.pack(pady=5)

result_label = tk.Label(app, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

# Run the application
app.mainloop()
