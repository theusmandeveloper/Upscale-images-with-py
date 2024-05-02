import main as tf
from tensorflow.keras.applications import SRGAN
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained SRGAN model
srgan = SRGAN(weights='imagenet')

# Load low-resolution image
img_path = 'low_res_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))  # Change target_size as per your requirement
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0  # Normalize pixel values

# Upscale image using SRGAN
upscaled_img = srgan.predict(x)

# Save the upscaled image
upscaled_img_path = r"C:\Users\Latitude\Downloads\202442583859. (1).png"
image.save_img(upscaled_img_path, upscaled_img[0])
