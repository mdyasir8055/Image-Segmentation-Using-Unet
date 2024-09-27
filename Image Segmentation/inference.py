import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
# Function to preprocess a single image for inference
def preprocess_image(image_path, img_size=(256, 256)):
    image = Image.open(image_path).resize(img_size)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (for grayscale)
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image
# Function to display the input image and the predicted mask
def display_prediction(input_image, predicted_mask):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(input_image.squeeze(), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    
    plt.show()
# Load the trained U-Net model
model = tf.keras.models.load_model('unet_membrane.hdf5')  # Ensure you have saved your model as 'unet_model.h5'
# Path to the new image for inference
new_image_path = r'D:\ML\unet\data\membrane\test\20.png'
# Preprocess the new image
input_image = preprocess_image(new_image_path)
# Predict the segmentation mask
predicted_mask = model.predict(input_image)
# Display the input image and the predicted mask
display_prediction(input_image, predicted_mask)
print("Inference complete.")
