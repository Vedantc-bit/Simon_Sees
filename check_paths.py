import os
import glob

DATA_ROOT = './nyu_depth_v2_data/data'
SPLIT = 'nyu2_train'

# 1. Constructing the path
data_dir = os.path.join(DATA_ROOT, SPLIT)

# 2. Getting the absolute path to be 100% sure about the path location
absolute_path = os.path.abspath(data_dir)
print(f"--- Path Debugger ---")
print(f"Current Directory: {os.getcwd()}")
print(f"Searching in Absolute Path: {absolute_path}")

# 3. Checking if this directory exists
if not os.path.exists(absolute_path):
    print("\n[ERROR] This directory does not exist!")
    print("Please verify your DATA_ROOT path and folder structure.")
else:
    print("\n[SUCCESS] Directory exists.")

    # 4. Try to find the files using the exact same glob pattern
    print("\nSearching for 'rgb_*.jpg' files recursively...")
    jpg_files = glob.glob(os.path.join(absolute_path, '**', 'rgb_*.jpg'), recursive=True)

    print(f"Found {len(jpg_files)} '.jpg' files.")
    if jpg_files:
        print("--- First 5 jpg files found: ---")
        for f in jpg_files[:5]:
            print(f"  - {f}")

    print("\nSearching for 'rgb_*.png' files recursively...")
    png_files = glob.glob(os.path.join(absolute_path, '**', 'rgb_*.png'), recursive=True)

    print(f"Found {len(png_files)} '.png' files.")
    if png_files:
        print("--- First 5 png files found: ---")
        for f in png_files[:5]:
            print(f"  - {f}")

    if not jpg_files and not png_files:
        print("\n--- [CRITICAL CONCLUSION] ---")
        print("No 'rgb_' prefixed files were found in the directory.")
        print("This means one of two things:")
        print("  1. The absolute path shown above is incorrect.")
        print("  2. The files were not actually renamed correctly.")
        print("\nPlease manually navigate to the absolute path and confirm the 'rgb_...' files are there.")
        