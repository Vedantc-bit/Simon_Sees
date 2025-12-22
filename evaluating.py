import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor
from nyu_dataset import NYUDataset, DATA_ROOT

def align_and_calculate_metrics(pred, target):
    target = target.squeeze()

    mask = target > 1e-3
    if torch.sum(mask) == 0:
        return None

    pred_masked = pred[mask]
    target_masked = target[mask]
    
    A = torch.vstack([pred_masked, torch.ones_like(pred_masked)]).T
    b = target_masked
    
    x = torch.linalg.lstsq(A, b).solution
    
    scale, shift = x[0], x[1]

    pred_aligned = pred * scale + shift
    pred_aligned[pred_aligned <= 1e-3] = 1e-3

    abs_rel = torch.mean(torch.abs(pred_aligned[mask] - target_masked) / target_masked)
    rmse = torch.sqrt(torch.mean((pred_aligned[mask] - target_masked) ** 2))
    
    return abs_rel.item(), rmse.item()

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    #1. Load Model and Processor 
    print("Loading Model Please wait...")
    
    model = DPTForDepthEstimation.from_pretrained("intel/dpt-hybrid-midas").to(device)
    processor = DPTImageProcessor.from_pretrained("intel/dpt-hybrid-midas")
    
    model.eval()

    #2. Loading Test Dataset 
    test_dataset = NYUDataset(root_dir=DATA_ROOT, split='nyu2_test')
    
    #3. Calculating Metrics over the Full Test Set 
    total_abs_rel, total_rmse, count = 0, 0, 0
    with torch.no_grad():
       
        for i in tqdm(range(len(test_dataset)), desc="Calculating Metrics..."):
            # Loading raw RGB image and ground truth depth tensor
            rgb_path = test_dataset.rgb_filenames[i]
            image = Image.open(rgb_path).convert("RGB")
            _, gt_depth = test_dataset[i] # Get the processed ground truth tensor
            gt_depth = gt_depth.to(device)

            # Preparing the image for the model using the processor
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # model prediction
            outputs = model(pixel_values)
            predicted_depth_relative = outputs.predicted_depth

            # Upsampling prediction to match ground truth size
            pred_upsampled = torch.nn.functional.interpolate(
                predicted_depth_relative.unsqueeze(1),
                size=gt_depth.shape[-2:], # Get H, W from ground truth tensor
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # Alignment
            metrics = align_and_calculate_metrics(pred_upsampled, gt_depth)
            if metrics:
                abs_rel, rmse = metrics
                total_abs_rel += abs_rel
                total_rmse += rmse
                count += 1
    
    avg_abs_rel = total_abs_rel / count
    avg_rmse = total_rmse / count
    print("\nQuantitative Evaluation Results:")
    print(f"Average Absolute Relative Difference (Abs Rel): {avg_abs_rel:.4f}")
    print(f"Average Root Mean Squared Error (RMSE): {avg_rmse:.4f} meters")
    print("---------------------------------------------")

if __name__ == '__main__':
    evaluate_model()
    