#%%
import os
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
#%%
def select_and_process_image(imgSize):
    # Hide the root Tkinter window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Bring file dialog to the front

    # Open file picker dialog
    file_path = askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    
    # If no file was selected, return None
    if not file_path:
        print("No file selected.")
        return None

    # Load and process the image
    img = cv2.imread(file_path)
    if img is None:
        print("Error: Could not read the selected file.")
        return None

    # Resize the image to match the input size expected by the model
    img = cv2.resize(img, (imgSize, imgSize))
    img = img / 255.0  # Normalize pixel values (optional, depending on your model)

    # Reshape the image to add the batch dimension
    img = np.expand_dims(img, axis=0)  # Shape becomes (1, imgSize, imgSize, 3)

    print(f"Image loaded and processed: {file_path}")
    return img

# Example usage:
imgSize = 128  # Set the size your model expects
processed_image = select_and_process_image(imgSize)

if processed_image is not None:
    print("Image is ready to be fed into the model.")


