import torch
import cv2
import urllib.request
import matplotlib.pyplot as plt
import random

# Importing dataset
from nyu_dataset import NYUDataset, DATA_ROOT

def model():
    print("Loading model")
    print("Chill for a minute...")
    
    # Loads the model
    model_type = "MiDaS_small" 
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True) 

    # Move the model to the GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    print(f"--- Using device: {device} ---")

    #transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True) 
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

    #1. Load a random image from Test set
    print("\nLoading a random test image ")
    test_dataset = NYUDataset(root_dir=DATA_ROOT, split='nyu2_test')
    idx = random.randint(0, len(test_dataset) - 1)
    
    img_path = test_dataset.rgb_filenames[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB

    #2. Preprocess the image 
    input_batch = transform(img).to(device)

    with torch.no_grad():
        midas.forward = midas.forward
        prediction = midas(input_batch)

        # Resizing
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    
    output_depth = prediction.cpu().numpy()

    #3. Inspect the Raw Depth Data 
    
    print("\n--- Inspecting the Raw Depth Data ---")
    print(f"Shape of the depth map array: {output_depth.shape}")
    print(f"Data type of the array: {output_depth.dtype}")
    print(f"Minimum value in the map: {output_depth.min():.2f}")
    print(f"Maximum value in the map: {output_depth.max():.2f}")
    
    center_y, center_x = output_depth.shape[0] // 2, output_depth.shape[1] // 2
    center_depth_value = output_depth[center_y, center_x]
    print(f"Value at the center pixel ({center_y}, {center_x}): {center_depth_value:.2f}")

    #4. Visualize the Results
    print("\nDisplaying results...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(img)
    axes[0].set_title("Input RGB Image")
    axes[0].axis('off')

    axes[1].imshow(output_depth, cmap='inferno')
   
    axes[1].set_title("Depth Prediction") 
    axes[1].axis('off')
    
    plt.savefig('pretrained_model_result.png')
    print("\nResult saved to pretrained_model_result.png")
    plt.show()


if __name__ == '__main__':
    model()