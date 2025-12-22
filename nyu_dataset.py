# Standard imports
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#config
DATA_ROOT = './nyu_depth_v2_data/data'
IMAGE_SIZE = (256, 256)

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

class NYUDataset(Dataset):
    """
    NYU Depth V2 Dataset, updated to load the 'rgb_' and 'depth_' naming convention.
    """
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(self.root_dir, self.split)
        
        self.rgb_filenames = []
        self.depth_filenames = []
        
        if split == 'nyu2_test':
            print(f"DEBUG: Searching flat directory: {self.data_dir}")
            self.rgb_filenames = sorted(glob.glob(os.path.join(self.data_dir, '*_colors.png')))
            self.depth_filenames = sorted(glob.glob(os.path.join(self.data_dir, '*_depth.png')))
        
        elif split == 'nyu2_train':
            print(f"DEBUG: Searching for 'rgb_' prefixed files in: {self.data_dir}")
            
            rgb_files = glob.glob(os.path.join(self.data_dir, '**', 'rgb_*.jpg'), recursive=True)
            rgb_files += glob.glob(os.path.join(self.data_dir, '**', 'rgb_*.png'), recursive=True)

            ### FINAL FIX: Make the depth file matching logic robust to extensions ###
            for rgb_path in rgb_files:
                # Get the base filename without extension
                base_name, _ = os.path.splitext(os.path.basename(rgb_path))
                
                # Create the depth filename
                # We assume depth files are PNGs
                depth_filename = base_name.replace('rgb_', 'depth_') + '.png'
                
                # Getting the path to the folder containing the rgb image
                folder_path = os.path.dirname(rgb_path)
                
                # Construct the full path for the potential depth file
                depth_path = os.path.join(folder_path, depth_filename)
                
                # If the corresponding depth file exists, we have a valid pair.
                if os.path.exists(depth_path):
                    self.rgb_filenames.append(rgb_path)
                    self.depth_filenames.append(depth_path)

        print(f"Loaded {len(self.rgb_filenames)} samples for the {self.split} split.")

        if not self.rgb_filenames:
            raise FileNotFoundError(f"Could not load any data for split: {self.split}. Check data paths and naming.")

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_filenames[idx]).convert('RGB')
        depth_map = Image.open(self.depth_filenames[idx])

        # Case 1: A transform IS provided 
        if self.transform:
            # Apply spatial transforms and convert to Tensor
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            rgb_image = self.transform(rgb_image)
            
            torch.manual_seed(seed)
            depth_map = self.transform(depth_map)

            #tensor functions
            rgb_image = transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)(rgb_image)
            depth_map = depth_map.float() / 1000.0

            return rgb_image, depth_map
        
        else:
            # The evaluation script loads the RGB image on its own, but it still
            # needs the ground truth depth map as a tensor for comparison.
            depth_tensor = transforms.ToTensor()(depth_map).float() / 1000.0

            return None, depth_tensor

# Correct transform pipeline
data_transforms = transforms.Compose([
    transforms.Resize(280),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Test Block 
if __name__ == '__main__':
    print("--- Testing Dataset Availability ---")
    
    try:
        train_dataset = NYUDataset(root_dir=DATA_ROOT, split='nyu2_train', transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        if len(train_dataset) > 0:
            print(f"SUCCESS: Train split successfully loaded {len(train_dataset)} samples.")
            
            print("\nTesting DataLoader...")
            try:
                image_batch, depth_batch = next(iter(train_loader))
                print(f"Batch loaded: (Batch size: {image_batch.shape[0]})")
                print(f"RGB batch shape: {image_batch.shape}")
                print(f"Depth batch shape: {depth_batch.shape}")
            except Exception as e:
                print(f"ERROR: Failed to iterate through DataLoader: {e}")
        else:
            print("FAILURE: Train split failed to load.")
            
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")