import torch
from torchvision.utils import save_image
import os

def generate_from_stylegan2(generator, latent_dim, num_images, output_dir, image_size=256):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = generator.to(device).eval()

    for i in range(num_images):
        z = torch.randn(1, latent_dim).to(device)
        with torch.no_grad():
            generated = generator(z)
        image = torch.nn.functional.interpolate(generated, size=(image_size, image_size), mode='bilinear')
        save_path = os.path.join(output_dir, f"stylegan2_style_{i}.jpg")
        save_image(image.clamp(0, 1), save_path)
        print(f"Saved {save_path}")

# Example usage (modify as needed):
if __name__ == "__main__":
    from stylegan2_master.training.networks_stylegan2 import Generator

    # These values need to match your training config
    latent_dim = 512
    image_size = 256
    model_path = "stylegan2_master/checkpoints/ffhq-config-f.pt"

    # Load pretrained StyleGAN2 model (ensure it exists)
    generator = Generator(z_dim=latent_dim, c_dim=0, w_dim=512, img_resolution=image_size, img_channels=3)
    generator.load_state_dict(torch.load(model_path, map_location="cpu"))
    generate_from_stylegan2(generator, latent_dim, num_images=10, output_dir="images/style_generated")
