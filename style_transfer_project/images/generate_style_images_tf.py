import os
import numpy as np
import PIL.Image
import tensorflow as tf

from generator_model import Generator
import dnnlib.tflib as tflib  # Make sure dnnlib is available in your environment

# --- CONFIGURATION ---
model_path = "stylegan2_master/karras2019stylegan2-ffhq-1024x1024.pkl"  # Path to pretrained .pkl file
output_dir = "images/style_generated"
num_images = 10
latent_dim = 512
truncation_psi = 0.5  # Controls variation: lower = smoother
image_size = 256

def save_image(img_array, filename):
    img = PIL.Image.fromarray(img_array, 'RGB')
    img = img.resize((image_size, image_size), PIL.Image.LANCZOS)
    img.save(filename)
    print(f"Saved: {filename}")

def generate_images():
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TensorFlow (TF1.x)
    tflib.init_tf()

    # Load generator model
    print("Loading StyleGAN2 generator...")
    generator = Generator(model_path)
    generator.set_dlatents(np.zeros((1, latent_dim)))  # Warm-up call

    # Generate images
    for i in range(num_images):
        z = np.random.randn(1, latent_dim) * truncation_psi
        generator.set_dlatents(z)
        image = generator.generate_images()[0]

        output_path = os.path.join(output_dir, f"stylegan2_style_{i}.jpg")
        save_image(image, output_path)

if __name__ == "__main__":
    generate_images()
