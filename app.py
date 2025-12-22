import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from io import BytesIO

# Import the web server library
from flask import Flask, request, jsonify, render_template, send_from_directory

# Import the MiDaS model and processor
from transformers import DPTForDepthEstimation, DPTImageProcessor

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load the Model (Do this ONCE at the start) ---
print("--- Loading Model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "intel/dpt-hybrid-midas"
model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
processor = DPTImageProcessor.from_pretrained(model_name)
model.eval()
print(f"--- Model loaded. Using device: {device} ---")

# --- 3. Create a Function to Run the Model ---
def process_image(image_file):
    
    # Create a 'static/uploads' directory if it doesn't exist
    output_dir = "static/uploads"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # A. Save the original image
    image = Image.open(image_file.stream).convert("RGB")
    base_name = f"{hash(image.tobytes())}" # Create a unique name
    original_save_path = os.path.join(output_dir, f"{base_name}_original.png")
    image.save(original_save_path)
    
    # B. Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # C. Run Inference
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth_relative = outputs.predicted_depth

    # D. Process and save the depth map
    pred_upsampled = torch.nn.functional.interpolate(
        predicted_depth_relative.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
    output_depth = pred_upsampled.cpu().numpy()
    
    # E. Save the clean, colormapped depth map
    depth_normalized = (output_depth - output_depth.min()) / (output_depth.max() - output_depth.min())
    colored_depth_map = (plt.cm.inferno(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    depth_image_to_save = Image.fromarray(colored_depth_map)
    depth_save_path = os.path.join(output_dir, f"{base_name}_depth.png")
    depth_image_to_save.save(depth_save_path)
    
    # Return the *relative* paths for the website
    return f"{base_name}_original.png", f"{base_name}_depth.png"

# --- 4. Define the Website Routes ---

# This route serves your main index.html page
@app.route('/')
def home():
    # Renders the HTML file from the 'templates' folder
    # IMPORTANT: You must create a folder named 'templates' and put index.html inside it
    return render_template('index.html')

# This route handles the image upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Run the model
            original_fn, depth_fn = process_image(file)
            
            # Send the file paths back to the website
            return jsonify({
                'original_path': f'/uploads/{original_fn}',
                'depth_path': f'/uploads/{depth_fn}'
            })
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

# 5. Run the App 
if __name__ == '__main__':
    # Creates the necessary folders before starting
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    print("2. Open your web browser and go to: http://127.0.0.1:5000")
    print("----------------------\n")
    
    app.run(debug=True)
