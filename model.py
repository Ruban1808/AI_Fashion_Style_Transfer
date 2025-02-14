import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load Pretrained Model (VGG19 for Feature Extraction)
def load_vgg_model():
    vgg = VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    model = Model(inputs=vgg.input, outputs=[vgg.get_layer("block1_conv1").output,
                                             vgg.get_layer("block2_conv1").output,
                                             vgg.get_layer("block3_conv1").output,
                                             vgg.get_layer("block4_conv1").output])
    return model

# Preprocess the image for VGG model
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0) / 255.0
    return img

# Generate styled image
def generate_style_transfer(content_path, style_path):
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)

    vgg = load_vgg_model()

    content_features = vgg(content_image)
    style_features = vgg(style_image)

    # Simple feature blending (for now)
    stylized_image = 0.6 * content_image + 0.4 * style_image

    return stylized_image

# Function to display the result
def display_image(image_array):
    plt.imshow(image_array[0])
    plt.axis("off")
    plt.show()
