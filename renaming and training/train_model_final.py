# Standard imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

# Your custom dataset script
from nyu_dataset import NYUDataset, data_transforms, DATA_ROOT

# Your model definition script
from monocular_depth_estimation import MonocularDepthEstimationModel

# Configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 40  # Adjust based on your GPU memory
NUM_EPOCHS = 15  # Start with 15 and increase if needed
MODEL_SAVE_PATH = './best_model.pth'

def main():
   
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    #1. Loading Datasets 
    print("\n--- Loading Datasets ---")
    # Training dataset
    train_dataset = NYUDataset(root_dir=DATA_ROOT, split='nyu2_train', transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Training samples: {len(train_dataset)}")

    # Validation/Test dataset
    # We use the 'nyu2_test' split for validation
    val_dataset = NYUDataset(root_dir=DATA_ROOT, split='nyu2_test', transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Validation samples: {len(val_dataset)}")

    #2. Initialize Model, Loss, and Optimizer 
    print("\n--- Initializing Model ---")
    model = MonocularDepthEstimationModel().to(device)
    
    # Loss Function: Mean Squared Error is a good starting point for depth estimation
    criterion = nn.MSELoss()
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop 
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for i, (images, depths) in enumerate(train_loader):
            images = images.to(device)
            depths = depths.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, depths)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)
        
        #Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (images,depths) in enumerate(val_loader):
                images = images.to(device)
                depths = depths.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, depths)
                
                running_val_loss += loss.item()
                
        epoch_val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)
    
        # Epoch Summary
        epoch_duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Duration: {epoch_duration:.2f}s | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}")

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # 4. Plotting Loss Curves 
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")
    plt.show()

if __name__ == '__main__':
    main()