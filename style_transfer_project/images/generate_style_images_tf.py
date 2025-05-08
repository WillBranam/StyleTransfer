import os
import numpy as np
import PIL.Image
import tensorflow as tf
from generator_model import Generator
import dnnlib.tflib as tflib  # Ensure this is in your PYTHONPATH

# --- CONFIG ---
model_path = "stylegan2_master/karras2019stylegan2-ffhq-1024x1024.pkl"  # path to pre-trained .pkl
output_dir = "images/style_generated"
num_images = 10
latent_dim = 512
output_resolution = 256

def save_image(img_array, filename):
    img = PIL.Image.fromarray(img_array, 'RGB')
    img = img.resize((output_resolution, output_resolution), PIL.Image.LANCZOS)
    img.save(filename)
    print(f"Saved {filename}")

def generate_images():
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TensorFlow
    tflib.init_tf()

    # Load the pretrained StyleGAN2 model
    print("Loading generator...")
    generator = Generator(model_path)
    generator.set_dlatents(np.zeros((1, latent_dim)))  # warm up

    for i in range(num_images):
        z = np.random.randn(1, latent_dim)
        generator.set_dlatents(z)
        img = generator.generate_images()[0]  # shape: (H, W, 3)

        filename = os.path.join(output_dir, f"stylegan2_style_{i}.jpg")
        save_image(img, filename)

if __name__ == "__main__":
    generate_images()
