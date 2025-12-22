import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Import the MiDaS model and processor
from transformers import DptForDepthEstimation, DptImageProcessor

def generate_demo(image_path, output_dir="images"):
    """
    Loads the MiDaS model, runs it on a single image,
    and saves the original and a clean depth map to the output directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # --- 1. Load Model and Processor ---
    print("Loading pre-trained MiDaS model...")
    model_name = "intel/dpt-hybrid-midas"
    model = DptForDepthEstimation.from_pretrained(model_name).to(device)
    processor = DptImageProcessor.from_pretrained(model_name)
    model.eval()

    # --- 2. Load and Prepare Image ---
    print(f"Loading image: {image_path}")
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
        
    image = Image.open(image_path).convert("RGB")
    
    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # --- 3. Run Inference ---
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth_relative = outputs.predicted_depth

    # Upsample prediction to match the original image size
    pred_upsampled = torch.nn.functional.interpolate(
        predicted_depth_relative.unsqueeze(1),
        size=image.size[::-1], # PIL size is (W, H), torch size is (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    output_depth = pred_upsampled.cpu().numpy()

    # --- 4. Save Clean Images ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # A. Save the original image
    original_save_path = os.path.join(output_dir, f"{base_name}_original.png")
    image.save(original_save_path)
    print(f"Original image saved to: {original_save_path}")

    # B. Save the clean, colormapped depth map
    depth_normalized = (output_depth - output_depth.min()) / (output_depth.max() - output_depth.min())
    colored_depth_map = (plt.cm.inferno(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    
    depth_image_to_save = Image.fromarray(colored_depth_map)
    depth_save_path = os.path.join(output_dir, f"{base_name}_depth.png")
    depth_image_to_save.save(depth_save_path)
    print(f"Clean depth map saved to: {depth_save_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("\nUsage: python generate_demo_images.py <path_to_your_image>")
        print("Example: python generate_demo_images.py 'C:/Users/darkc/Pictures/my_photo.jpg'")
    else:
        generate_demo(sys.argv[1])

